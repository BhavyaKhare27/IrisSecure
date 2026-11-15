"""Statistical threshold calibration from measured genuine/impostor summaries."""

from __future__ import annotations

import json
import os
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "database" / "embedding_stats.json"
with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
RUN_NAME = os.getenv("IRISSECURE_RUN_NAME", CONFIG.get("training", {}).get("run_name", "subject_disjoint_run"))
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
    raise ValueError("IRISSECURE_RUN_NAME must be a single directory name")
OUTPUT_ROOT = Path(CONFIG.get("training", {}).get("output_root", "models/runs"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
RUN_CHECKPOINT_PATH = OUTPUT_ROOT / RUN_NAME / "checkpoints" / "best_model.pth"
DEPLOYED_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
ACTIVE_CHECKPOINT_PATH = Path(os.getenv(
    "IRISSECURE_CHECKPOINT", str(RUN_CHECKPOINT_PATH)
)).expanduser()
if not ACTIVE_CHECKPOINT_PATH.is_absolute():
    ACTIVE_CHECKPOINT_PATH = PROJECT_ROOT / ACTIVE_CHECKPOINT_PATH
if ACTIVE_CHECKPOINT_PATH.resolve() not in {
    DEPLOYED_CHECKPOINT_PATH.resolve(), RUN_CHECKPOINT_PATH.resolve()
}:
    raise ValueError("IRISSECURE_CHECKPOINT must select the deployed checkpoint or configured training-run checkpoint")
EVALUATION_ROOT = Path(os.getenv(
    "IRISSECURE_EVALUATION_ROOT", str(PROJECT_ROOT / "outputs" / "evaluation" / RUN_NAME)
)).expanduser()
if not EVALUATION_ROOT.is_absolute():
    EVALUATION_ROOT = PROJECT_ROOT / EVALUATION_ROOT
SCORES_PATH = EVALUATION_ROOT / "validation_score_distributions.npz"
REPORT_PATH = EVALUATION_ROOT / "evaluation_report.json"


@lru_cache(maxsize=8)
def _cached_sha256(path_string: str, modified_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_matches_evaluation() -> bool:
    """Whether the candidate evaluation report belongs to the current checkpoint bytes."""
    if not RUN_CHECKPOINT_PATH.is_file() or not REPORT_PATH.is_file():
        return False
    try:
        with REPORT_PATH.open(encoding="utf-8") as source:
            report = json.load(source)
        expected = report.get("reproducibility", {}).get("checkpoint_sha256")
        stat = RUN_CHECKPOINT_PATH.stat()
        actual = _cached_sha256(str(RUN_CHECKPOINT_PATH.resolve()), stat.st_mtime_ns, stat.st_size)
        return bool(expected) and expected == actual
    except (OSError, ValueError, TypeError):
        return False


def load_reference_stats() -> dict:
    if not STATS_PATH.exists():
        return {}
    with STATS_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def optimize_threshold(stats: dict | None = None, target_far: float | None = None) -> dict:
    """Find the EER threshold from measured scores, with a marked estimate fallback."""
    # Calibration scores belong to the candidate model and are valid for live
    # inference only when that exact checkpoint is configured as the active model.
    candidate_is_active = ACTIVE_CHECKPOINT_PATH.resolve() == RUN_CHECKPOINT_PATH.resolve()
    if candidate_is_active and checkpoint_matches_evaluation() and SCORES_PATH.exists():
        with np.load(SCORES_PATH) as scores:
            genuine_scores = np.asarray(scores["genuine_scores"], dtype=np.float64)
            impostor_scores = np.asarray(scores["impostor_scores"], dtype=np.float64)
        if genuine_scores.size and impostor_scores.size:
            thresholds = np.unique(np.concatenate([genuine_scores, impostor_scores]))
            thresholds = np.concatenate([thresholds, [np.nextafter(thresholds[-1], np.inf)]])
            genuine_sorted = np.sort(genuine_scores)
            impostor_sorted = np.sort(impostor_scores)
            frr = np.searchsorted(genuine_sorted, thresholds, side="left") / genuine_sorted.size
            far = (impostor_sorted.size - np.searchsorted(impostor_sorted, thresholds, side="left")) / impostor_sorted.size
            eer_index = int(np.argmin(np.abs(far - frr)))
            if target_far is None:
                index = eer_index
                objective = "equal_error_rate"
                constraint_satisfied = True
            else:
                acceptable = np.flatnonzero(far <= target_far)
                index = int(acceptable[0]) if acceptable.size else len(thresholds) - 1
                objective = "minimize_frr_at_target_far"
                constraint_satisfied = bool(far[index] <= target_far)
            report = {}
            if REPORT_PATH.exists():
                with REPORT_PATH.open(encoding="utf-8") as source:
                    report = json.load(source)
            holdout_metrics = report.get("test_at_validation_eer")
            if target_far is not None:
                target_report = report.get("target_far_operations", {}).get(str(float(target_far)), {})
                holdout_metrics = target_report.get("test_metrics_at_threshold", holdout_metrics)
            return {
                "threshold": round(float(thresholds[index]), 4),
                "estimated_far": float(far[index]),
                "estimated_frr": float(frr[index]),
                "estimated_eer": float((far[eer_index] + frr[eer_index]) / 2),
                "eer_threshold": round(float(thresholds[eer_index]), 4),
                "method": "Validation-selected empirical EER threshold" if target_far is None else "Validation-selected minimum-FRR threshold at target FAR",
                "data_source": "empirical",
                "objective": objective,
                "target_far": target_far,
                "constraint_satisfied": constraint_satisfied,
                "split": report.get("calibration_split", "validation"),
                "evaluation_split": report.get("split", "unknown"),
                "holdout_test_metrics": holdout_metrics,
                "num_genuine_pairs": int(genuine_sorted.size),
                "num_impostor_pairs": int(impostor_sorted.size),
                "curve": [
                    {"threshold": round(float(thresholds[i]), 4), "far": float(far[i]), "frr": float(frr[i])}
                    for i in np.linspace(0, len(thresholds) - 1, min(501, len(thresholds)), dtype=int)
                ],
            }

    """Fallback uses Gaussian score assumptions on the aggregate reference file."""
    stats = stats or load_reference_stats()
    genuine = stats.get("intra_class_similarity", {})
    impostor = stats.get("inter_class_similarity", {})
    if not genuine or not impostor:
        raise ValueError("No genuine/impostor score summaries are available")

    g_mean, g_std = float(genuine["mean"]), max(float(genuine["std"]), 1e-6)
    i_mean, i_std = float(impostor["mean"]), max(float(impostor["std"]), 1e-6)
    thresholds = np.linspace(-1.0, 1.0, 2001)
    far = norm.sf(thresholds, loc=i_mean, scale=i_std)
    frr = norm.cdf(thresholds, loc=g_mean, scale=g_std)
    eer_index = int(np.argmin(np.abs(far - frr)))
    if target_far is None:
        index = eer_index
        objective = "equal_error_rate"
        constraint_satisfied = True
    else:
        acceptable = np.flatnonzero(far <= target_far)
        index = int(acceptable[np.argmin(frr[acceptable])]) if acceptable.size else int(np.argmin(far))
        objective = "minimize_frr_at_target_far"
        constraint_satisfied = bool(far[index] <= target_far)
    return {
        "threshold": round(float(thresholds[index]), 4),
        "estimated_far": float(far[index]),
        "estimated_frr": float(frr[index]),
        "estimated_eer": float((far[eer_index] + frr[eer_index]) / 2),
        "eer_threshold": round(float(thresholds[eer_index]), 4),
        "method": "Gaussian approximation from aggregate score statistics",
        "data_source": "reference_estimate",
        "objective": objective,
        "target_far": target_far,
        "constraint_satisfied": constraint_satisfied,
        "curve": [
            {"threshold": round(float(thresholds[i]), 3), "far": float(far[i]), "frr": float(frr[i])}
            for i in range(0, len(thresholds), 20)
        ],
    }
