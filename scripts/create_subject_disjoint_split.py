"""Create a reproducible, person-disjoint train/validation/test copy of CASIA.

The source dataset and existing data/processed CSVs are left in place. Images are
hard-linked where supported (with a copy fallback) into data/splits/subject_disjoint.
All images and both eyes for a person stay in one split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "CASIA-Iris-Thousand"
OUTPUT = ROOT / "data" / "splits" / "subject_disjoint"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    test_fraction = 1.0 - args.train_fraction - args.validation_fraction
    if min(args.train_fraction, args.validation_fraction, test_fraction) <= 0:
        raise SystemExit("Train and validation fractions must be positive and sum to less than 1.")
    if not SOURCE.is_dir():
        raise SystemExit(f"Dataset directory not found: {SOURCE}")
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {OUTPUT}")

    people = sorted(path for path in SOURCE.iterdir() if path.is_dir() and path.name.isdigit())
    if len(people) < 3:
        raise SystemExit(f"Expected person directories under {SOURCE}")

    shuffled_people = [path.name for path in people]
    random.Random(args.seed).shuffle(shuffled_people)
    train_count = int(len(shuffled_people) * args.train_fraction)
    validation_count = int(len(shuffled_people) * args.validation_fraction)
    assignments = {
        "train": shuffled_people[:train_count],
        "val": shuffled_people[train_count:train_count + validation_count],
        "test": shuffled_people[train_count + validation_count:],
    }

    global_label_index: dict[str, int] = {}
    mapping_path = ROOT / "data" / "processed" / "label_mapping.csv"
    if mapping_path.is_file():
        with mapping_path.open(newline="", encoding="utf-8") as source:
            global_label_index = {row["label"]: int(row["label_idx"])
                                  for row in csv.DictReader(source)}

    train_classes = sorted(f"{person}_{eye}" for person in assignments["train"]
                           for eye in ("L", "R"))
    train_class_index = {label: index for index, label in enumerate(train_classes)}
    temp_output = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temp_output.exists():
        raise SystemExit(f"Temporary output already exists; inspect before removing: {temp_output}")

    manifest_fields = ["person_id", "eye", "label", "image_path", "image_filename",
                       "label_idx", "training_class_idx"]
    counts = {split: Counter() for split in assignments}
    classes_seen = {split: set() for split in assignments}
    linked_files = copied_files = 0
    try:
        for split, split_people in assignments.items():
            split_root = temp_output / split
            split_root.mkdir(parents=True)
            manifest_path = temp_output / f"{split}.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
                writer.writeheader()
                for person in sorted(split_people):
                    source_person = SOURCE / person
                    for eye in ("L", "R"):
                        source_eye = source_person / eye
                        if not source_eye.is_dir():
                            continue
                        for source_image in sorted(source_eye.iterdir()):
                            if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
                                continue
                            destination = split_root / person / eye / source_image.name
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                os.link(source_image, destination)
                                linked_files += 1
                            except OSError:
                                shutil.copy2(source_image, destination)
                                copied_files += 1
                            label = f"{person}_{eye}"
                            final_destination = OUTPUT / destination.relative_to(temp_output)
                            relative_path = final_destination.relative_to(ROOT).as_posix()
                            writer.writerow({
                                "person_id": person,
                                "eye": eye,
                                "label": label,
                                "image_path": relative_path,
                                "image_filename": source_image.name,
                                "label_idx": global_label_index.get(label, ""),
                                "training_class_idx": train_class_index.get(label, "") if split == "train" else "",
                            })
                            counts[split]["images"] += 1
                            classes_seen[split].add(label)
            counts[split]["subjects"] = len(split_people)

        summary = {
            "protocol": "subject-disjoint person split; both eyes and all images remain together",
            "seed": args.seed,
            "fractions": {"train": args.train_fraction,
                          "val": args.validation_fraction,
                          "test": test_fraction},
            "source": SOURCE.relative_to(ROOT).as_posix(),
            "output": OUTPUT.relative_to(ROOT).as_posix(),
            "link_method": "hard links where supported; otherwise independent file copies",
            "split_subject_counts": {name: len(subjects) for name, subjects in assignments.items()},
            "split_image_counts": {name: counts[name]["images"] for name in assignments},
            "split_class_counts": {name: len(classes_seen[name]) for name in assignments},
            "training_class_count": len(train_class_index),
            "global_label_mapping_available": bool(global_label_index),
            "subjects": assignments,
        }
        (temp_output / "split_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temp_output.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise

    print(f"Created subject-disjoint dataset at {OUTPUT}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {summary['split_subject_counts'][split]} people, "
              f"{summary['split_image_counts'][split]:,} images")
    print(f"  person-eye training classes: {summary['training_class_count']:,}")
    print(f"  linked {linked_files:,} files; copied {copied_files:,} files")
    print("Original dataset and existing processed manifests were preserved.")


if __name__ == "__main__":
    main()
