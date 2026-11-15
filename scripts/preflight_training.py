"""Read-only integrity and configuration checks before final model training.

Run from the repository root with:
    python scripts/preflight_training.py

By default, every manifest image is opened and decoded. Use --skip-image-decode
only when a faster path/file check is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "person_id", "eye", "label", "image_path", "label_idx", "training_class_idx"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def check_image(path: Path) -> tuple[bool, tuple[int, int] | None]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return False, None
    return True, (int(image.shape[1]), int(image.shape[0]))


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-image-decode", action="store_true",
        help="Check every image path and file size, but skip decoding pixels",
    )
    args = parser.parse_args()

    config_path = ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    split_dir = configured_path(config["dataset"].get(
        "subject_split_dir", "data/splits/subject_disjoint"
    ))
    metadata_path = split_dir / "split_metadata.json"
    if not metadata_path.is_file():
        fail(f"Split metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        manifest_path = split_dir / f"{split_name}.csv"
        if not manifest_path.is_file():
            fail(f"{split_name} manifest is missing: {manifest_path}")
        frame = pd.read_csv(manifest_path, dtype={"person_id": str})
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            fail(f"{split_name} manifest is missing columns: {sorted(missing)}")
        if frame.empty:
            fail(f"{split_name} manifest is empty")
        frames[split_name] = frame

    subject_sets: dict[str, set[str]] = {}
    class_index_maps: dict[str, dict[str, int]] = {}
    all_paths: dict[str, str] = {}
    image_paths: list[Path] = []
    decode_failures: list[str] = []
    dimensions: Counter[tuple[int, int]] = Counter()
    class_count_summary = {}

    for split_name, frame in frames.items():
        frame["person_id"] = frame["person_id"].astype(str)
        if frame[["person_id", "eye", "label", "image_path"]].isna().any().any():
            fail(f"{split_name} manifest contains empty identity or image-path values")
        if not set(frame["eye"].astype(str)).issubset({"L", "R"}):
            fail(f"{split_name} manifest contains an eye value outside L/R")
        expected_labels = frame["person_id"] + "_" + frame["eye"].astype(str)
        if not expected_labels.equals(frame["label"].astype(str)):
            fail(f"{split_name} manifest has labels inconsistent with person_id and eye")
        if frame["image_path"].duplicated().any():
            fail(f"{split_name} manifest contains duplicate image paths")

        label_indices = pd.to_numeric(frame["label_idx"], errors="coerce")
        if label_indices.isna().any() or not np.equal(label_indices, np.floor(label_indices)).all():
            fail(f"{split_name} manifest has missing or non-integer global labels")
        label_mapping = frame[["label", "label_idx"]].drop_duplicates()
        if label_mapping["label"].duplicated().any() or label_mapping["label_idx"].duplicated().any():
            fail(f"{split_name} global label mapping is not one-to-one")
        class_index_maps[split_name] = dict(zip(
            label_mapping["label"].astype(str), label_mapping["label_idx"].astype(int)
        ))

        image_counts = frame.groupby("label", sort=False).size()
        too_small = image_counts[image_counts < 2]
        if not too_small.empty:
            fail(f"{split_name} has person-eye classes with fewer than 2 images: {too_small.index[:5].tolist()}")
        eyes_per_person = frame.groupby("person_id")["eye"].agg(lambda values: set(values.astype(str)))
        missing_eye = eyes_per_person[eyes_per_person.map(lambda values: values != {"L", "R"})]
        if not missing_eye.empty:
            fail(f"{split_name} has people without both eye classes: {missing_eye.index[:5].tolist()}")
        class_count_summary[split_name] = {
            "images": int(len(frame)),
            "people": int(frame["person_id"].nunique()),
            "classes": int(frame["label"].nunique()),
            "min_images_per_class": int(image_counts.min()),
            "median_images_per_class": float(image_counts.median()),
            "max_images_per_class": int(image_counts.max()),
        }

        actual_people = set(frame["person_id"])
        expected_people = set(map(str, metadata.get("subjects", {}).get(split_name, [])))
        if actual_people != expected_people:
            fail(f"{split_name} identities do not match split_metadata.json")
        if len(frame) != metadata.get("split_image_counts", {}).get(split_name):
            fail(f"{split_name} image count does not match split_metadata.json")
        if len(actual_people) != metadata.get("split_subject_counts", {}).get(split_name):
            fail(f"{split_name} subject count does not match split_metadata.json")
        if frame["label"].nunique() != metadata.get("split_class_counts", {}).get(split_name):
            fail(f"{split_name} class count does not match split_metadata.json")

        subject_sets[split_name] = actual_people
        for relative_path in frame["image_path"].astype(str):
            image_path = Path(relative_path).expanduser()
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            image_path = image_path.resolve()
            if not image_path.is_file() or image_path.stat().st_size == 0:
                fail(f"Missing or empty image file: {image_path}")
            path_key = str(image_path)
            previous_split = all_paths.get(path_key)
            if previous_split is not None:
                fail(f"Image path appears in both {previous_split} and {split_name}: {path_key}")
            all_paths[path_key] = split_name
            image_paths.append(image_path)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = subject_sets[left] & subject_sets[right]
        if overlap:
            fail(f"Subject leakage between {left} and {right}: {len(overlap)} people")

    train_df = frames["train"]
    if train_df["training_class_idx"].isna().any():
        fail("Training manifest has empty training_class_idx values")
    training_mapping = train_df[["label", "training_class_idx"]].drop_duplicates()
    training_mapping["training_class_idx"] = pd.to_numeric(
        training_mapping["training_class_idx"], errors="coerce"
    )
    class_ids = training_mapping["training_class_idx"]
    if (class_ids.isna().any() or not np.equal(class_ids, np.floor(class_ids)).all()
            or training_mapping["label"].duplicated().any()
            or class_ids.duplicated().any()):
        fail("Training ArcFace class mapping is missing, duplicated, or invalid")
    indices = class_ids.astype(int).tolist()
    if sorted(indices) != list(range(len(indices))):
        fail("Training ArcFace class indices are not contiguous from zero")
    for split_name in ("val", "test"):
        if set(class_index_maps[split_name]) & set(training_mapping["label"].astype(str)):
            fail(f"{split_name} contains person-eye classes present in train")

    split_seed = int(config["split"].get("random_seed", metadata.get("seed", -1)))
    training_seed = int(config["training"].get("seed", -1))
    if split_seed != int(metadata.get("seed", -2)):
        fail(f"Configured split seed {split_seed} differs from split metadata seed {metadata.get('seed')}")

    training_cfg = config["training"]
    run_name = str(training_cfg.get("run_name", "subject_disjoint_run"))
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        fail("training.run_name must be a single, non-empty directory name")
    output_root = configured_path(training_cfg.get("output_root", "models/runs"))
    checkpoint_dir = output_root / run_name / "checkpoints"
    log_dir = configured_path(config.get("logging", {}).get("log_dir", "logs")) / run_name
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        fail(f"Training checkpoint output is not empty: {checkpoint_dir}; choose a new run_name")
    if (log_dir / "training_history.json").exists():
        fail(f"Training history already exists: {log_dir / 'training_history.json'}; choose a new run_name")

    active_checkpoint = ROOT / "models" / "checkpoints" / "best_model.pth"
    target_best = checkpoint_dir / "best_model.pth"
    if target_best.resolve() == active_checkpoint.resolve():
        fail("Configured training output targets the currently deployed checkpoint")

    print("IRISSECURE TRAINING PREFLIGHT")
    print(f"  Split directory: {split_dir.relative_to(ROOT)}")
    print(f"  Split seed / training seed: {split_seed} / {training_seed}")
    print(f"  Run name: {run_name}")
    print(f"  Future checkpoints: {checkpoint_dir.relative_to(ROOT)}")
    print(f"  Future logs: {log_dir.relative_to(ROOT)}")
    for name in ("train", "val", "test"):
        summary = class_count_summary[name]
        print(
            f"  {name:5}: {summary['people']} people, {summary['images']:,} images, "
            f"{summary['classes']} classes, images/class min/median/max "
            f"{summary['min_images_per_class']}/{summary['median_images_per_class']:.0f}/{summary['max_images_per_class']}"
        )
    print("  Identity overlap: none")
    print(f"  Unique manifest image paths: {len(image_paths):,}")

    if args.skip_image_decode:
        print("  Image decode: skipped by request (path and file-size checks passed)")
    else:
        print(f"  Decoding {len(image_paths):,} images...")
        for index, image_path in enumerate(image_paths, start=1):
            valid, dimensions_xy = check_image(image_path)
            if not valid:
                decode_failures.append(str(image_path))
            else:
                dimensions[dimensions_xy] += 1
            if index % 2000 == 0 or index == len(image_paths):
                print(f"    checked {index:,}/{len(image_paths):,}")
        if decode_failures:
            fail(f"Unreadable images ({len(decode_failures)}): {decode_failures[:10]}")
        print(f"  Image decode: all passed; common dimensions: {dimensions.most_common(3)}")

    if torch.cuda.is_available():
        device_summary = f"CUDA · {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_summary = "Apple MPS"
    else:
        device_summary = "No CUDA/MPS GPU visible to this process; training would use CPU"
    print(f"  PyTorch / torchvision: {torch.__version__} / {torchvision.__version__}")
    print(f"  Training device available: {device_summary}")

    model_cfg = config["model"]
    if model_cfg.get("pretrained", False):
        enum_by_backbone = {
            "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
            "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
            "mobilenet_v2": torchvision.models.MobileNet_V2_Weights.DEFAULT,
        }
        weights = enum_by_backbone.get(model_cfg.get("backbone"))
        cached_weight = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name if weights else None
        if cached_weight and cached_weight.is_file():
            print(f"  Pretrained backbone weights: cached ({cached_weight.name})")
        else:
            print("  WARNING: pretrained backbone weights are not cached; setup_model may download them")

    if active_checkpoint.is_file():
        print(f"  Existing deployed checkpoint preserved: {active_checkpoint.relative_to(ROOT)}")
        print(f"  Existing checkpoint SHA-256: {sha256_file(active_checkpoint)}")
        expected_artifact_bytes = active_checkpoint.stat().st_size * 5
        free_bytes = shutil.disk_usage(ROOT).free
        print(f"  Free disk: {free_bytes / (1024**3):.2f} GiB; estimated run artifacts: {expected_artifact_bytes / (1024**3):.2f} GiB")
        if free_bytes < expected_artifact_bytes:
            print("  WARNING: available disk may not fit best + three periodic checkpoints")

    print("PREFLIGHT PASSED — no training was started and no checkpoint was modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
