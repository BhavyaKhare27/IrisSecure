"""
Similarity Metrics for Iris Verification
============================================
Functions for computing pairwise similarities and generating
genuine/impostor score distributions for evaluation.

Metrics:
    - Cosine similarity (primary — embeddings are L2-normalized)
    - Euclidean distance (alternative)
    - FAR (False Accept Rate)
    - FRR (False Reject Rate)
    - EER (Equal Error Rate)
    - ROC / DET curves
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


# ============================================
# Core Similarity Functions
# ============================================

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized embeddings.
    Since embeddings are unit vectors: cos_sim = dot product.
    
    Returns: float in [-1, 1], higher = more similar
    """
    return float(np.dot(emb1, emb2))


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Euclidean distance between two embeddings.
    For unit vectors: dist = sqrt(2 - 2*cos_sim)
    
    Returns: float >= 0, lower = more similar
    """
    return float(np.linalg.norm(emb1 - emb2))


# ============================================
# Pair Generation for Evaluation
# ============================================

def generate_evaluation_pairs(embeddings: np.ndarray,
                               labels: np.ndarray,
                               max_genuine_per_class: int = 10,
                               max_impostor_pairs: int = 50000,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate genuine and impostor pairs from embeddings for evaluation.
    
    Genuine pairs: same identity, different images
    Impostor pairs: different identities
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) integer class labels
        max_genuine_per_class: Max genuine pairs per class (to avoid combinatorial explosion)
        max_impostor_pairs: Max number of impostor pairs to generate
        seed: Random seed for reproducibility
        
    Returns:
        genuine_scores: (G,) cosine similarities for genuine pairs
        impostor_scores: (I,) cosine similarities for impostor pairs
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    
    genuine_scores = []
    impostor_scores = []
    
    # --- Genuine pairs: same identity ---
    logger.info("Generating genuine pairs...")
    for label in unique_labels:
        mask = labels == label
        class_embs = embeddings[mask]
        
        if len(class_embs) < 2:
            continue
        
        # All combinations within this class
        indices = list(range(len(class_embs)))
        pairs = list(combinations(indices, 2))
        
        # Limit pairs per class
        if len(pairs) > max_genuine_per_class:
            pair_indices = rng.choice(len(pairs), max_genuine_per_class, replace=False)
            pairs = [pairs[i] for i in pair_indices]
        
        for i, j in pairs:
            sim = cosine_similarity(class_embs[i], class_embs[j])
            genuine_scores.append(sim)
    
    genuine_scores = np.array(genuine_scores)
    
    # --- Impostor pairs: different identities ---
    logger.info("Generating impostor pairs...")
    
    # Build per-class centroid for efficient sampling
    class_centroids = {}
    class_samples = {}
    for label in unique_labels:
        mask = labels == label
        class_embs = embeddings[mask]
        class_centroids[label] = class_embs.mean(axis=0)
        class_centroids[label] /= np.linalg.norm(class_centroids[label])
        class_samples[label] = class_embs
    
    # Random impostor pairs
    label_list = list(unique_labels)
    n_labels = len(label_list)
    count = 0
    
    while count < max_impostor_pairs:
        # Pick two random different classes
        idx1, idx2 = rng.choice(n_labels, 2, replace=False)
        l1, l2 = label_list[idx1], label_list[idx2]
        
        # Pick one random embedding from each class
        e1 = class_samples[l1][rng.randint(len(class_samples[l1]))]
        e2 = class_samples[l2][rng.randint(len(class_samples[l2]))]
        
        sim = cosine_similarity(e1, e2)
        impostor_scores.append(sim)
        count += 1
    
    impostor_scores = np.array(impostor_scores)
    
    logger.info(f"Generated {len(genuine_scores)} genuine pairs, "
                f"{len(impostor_scores)} impostor pairs")
    
    return genuine_scores, impostor_scores


# ============================================
# Biometric Metrics
# ============================================

def compute_far_frr(genuine_scores: np.ndarray,
                     impostor_scores: np.ndarray,
                     thresholds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR and FRR at various thresholds.
    
    FAR (False Accept Rate): fraction of impostors wrongly accepted
    FRR (False Reject Rate): fraction of genuines wrongly rejected
    
    Args:
        genuine_scores: Similarity scores for same-identity pairs
        impostor_scores: Similarity scores for different-identity pairs
        thresholds: Array of thresholds to evaluate (auto if None)
        
    Returns:
        Tuple of (thresholds, FAR_array, FRR_array)
    """
    if thresholds is None:
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.linspace(all_scores.min() - 0.01, all_scores.max() + 0.01, 1000)
    
    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))
    
    n_genuine = len(genuine_scores)
    n_impostor = len(impostor_scores)
    
    for i, t in enumerate(thresholds):
        # FAR: impostors accepted (score >= threshold)
        far[i] = np.sum(impostor_scores >= t) / n_impostor if n_impostor > 0 else 0
        # FRR: genuines rejected (score < threshold)
        frr[i] = np.sum(genuine_scores < t) / n_genuine if n_genuine > 0 else 0
    
    return thresholds, far, frr


def compute_eer(genuine_scores: np.ndarray,
                impostor_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) — the point where FAR == FRR.
    
    Lower EER = better system. State-of-the-art iris: EER < 1%
    
    Returns:
        Tuple of (eer_value, eer_threshold)
    """
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)
    
    # Find where FAR and FRR cross
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    
    eer = (far[eer_idx] + frr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]
    
    return float(eer), float(eer_threshold)


def compute_roc(genuine_scores: np.ndarray,
                impostor_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve (TPR vs FPR) and AUC.
    
    TPR = 1 - FRR (True Positive Rate = genuines correctly accepted)
    FPR = FAR (False Positive Rate = impostors wrongly accepted)
    
    Returns:
        Tuple of (fpr_array, tpr_array, auc_value)
    """
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)
    
    tpr = 1.0 - frr  # True Positive Rate
    fpr = far         # False Positive Rate
    
    # Sort by FPR for proper AUC calculation
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # AUC via trapezoidal rule (numpy 2.0+ uses trapezoid)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_fn(tpr_sorted, fpr_sorted))
    
    return fpr_sorted, tpr_sorted, auc


def compute_tar_at_far(genuine_scores: np.ndarray,
                        impostor_scores: np.ndarray,
                        target_far_values: List[float] = [0.01, 0.001, 0.0001]) -> Dict[str, float]:
    """
    Compute TAR (True Accept Rate) at specific FAR operating points.
    
    TAR@FAR=0.01 means: what fraction of genuine users are correctly accepted
    when the system is tuned to falsely accept only 1% of impostors?
    
    Args:
        genuine_scores: Genuine pair scores
        impostor_scores: Impostor pair scores
        target_far_values: FAR operating points to evaluate
        
    Returns:
        Dict: {'TAR@FAR=0.01': value, 'TAR@FAR=0.001': value, ...}
    """
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)
    tpr = 1.0 - frr
    
    results = {}
    for target_far in target_far_values:
        # Find threshold where FAR <= target_far
        valid = far <= target_far
        if np.any(valid):
            # Among valid thresholds, pick the one with highest TPR
            idx = np.argmax(tpr * valid)
            tar = tpr[idx]
            actual_far = far[idx]
            results[f'TAR@FAR={target_far}'] = round(float(tar), 4)
        else:
            results[f'TAR@FAR={target_far}'] = 0.0
    
    return results


def find_optimal_threshold(genuine_scores: np.ndarray,
                            impostor_scores: np.ndarray,
                            method: str = 'eer') -> Tuple[float, dict]:
    """
    Find the optimal verification threshold.
    
    Methods:
        'eer': threshold at Equal Error Rate point
        'min_cost': minimize weighted FAR + FRR
        'max_accuracy': maximize overall accuracy
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)
    
    if method == 'eer':
        diff = np.abs(far - frr)
        idx = np.argmin(diff)
    elif method == 'min_cost':
        # Equal cost for FAR and FRR
        cost = far + frr
        idx = np.argmin(cost)
    elif method == 'max_accuracy':
        n_gen = len(genuine_scores)
        n_imp = len(impostor_scores)
        total = n_gen + n_imp
        accuracy = ((1 - frr) * n_gen + (1 - far) * n_imp) / total
        idx = np.argmax(accuracy)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    threshold = thresholds[idx]
    metrics = {
        'threshold': round(float(threshold), 4),
        'FAR': round(float(far[idx]), 6),
        'FRR': round(float(frr[idx]), 6),
        'TAR': round(float(1 - frr[idx]), 4),
        'method': method,
    }
    
    return float(threshold), metrics
