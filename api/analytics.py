"""Statistical threshold calibration from measured genuine/impostor summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "database" / "embedding_stats.json"
SCORES_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "score_distributions.npz"
REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.json"


def load_reference_stats() -> dict:
    if not STATS_PATH.exists():
        return {}
    with STATS_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def optimize_threshold(stats: dict | None = None, target_far: float | None = None) -> dict:
    """Find the EER threshold from measured scores, with a marked estimate fallback."""
    if SCORES_PATH.exists():
        with np.load(SCORES_PATH) as scores:
            genuine_scores = np.asarray(scores["genuine_scores"], dtype=np.float64)
            impostor_scores = np.asarray(scores["impostor_scores"], dtype=np.float64)
        if genuine_scores.size and impostor_scores.size:
            thresholds = np.linspace(
                min(float(genuine_scores.min()), float(impostor_scores.min())) - 0.01,
                max(float(genuine_scores.max()), float(impostor_scores.max())) + 0.01,
                1001,
            )
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
                index = int(acceptable[np.argmin(frr[acceptable])]) if acceptable.size else int(np.argmin(far))
                objective = "minimize_frr_at_target_far"
                constraint_satisfied = bool(far[index] <= target_far)
            report = {}
            if REPORT_PATH.exists():
                with REPORT_PATH.open(encoding="utf-8") as source:
                    report = json.load(source)
            return {
                "threshold": round(float(thresholds[index]), 4),
                "estimated_far": float(far[index]),
                "estimated_frr": float(frr[index]),
                "estimated_eer": float((far[eer_index] + frr[eer_index]) / 2),
                "eer_threshold": round(float(thresholds[eer_index]), 4),
                "method": "Empirical EER threshold" if target_far is None else "Empirical minimum-FRR threshold at target FAR",
                "data_source": "empirical",
                "objective": objective,
                "target_far": target_far,
                "constraint_satisfied": constraint_satisfied,
                "split": report.get("split", "unknown"),
                "num_genuine_pairs": int(genuine_sorted.size),
                "num_impostor_pairs": int(impostor_sorted.size),
                "curve": [
                    {"threshold": round(float(thresholds[i]), 3), "far": float(far[i]), "frr": float(frr[i])}
                    for i in range(0, len(thresholds), 20)
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
