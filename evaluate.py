"""
System Evaluation for Iris Recognition
==========================================
Comprehensive biometric evaluation using genuine/impostor pair analysis
on the test set embeddings.

Generates:
    1. ROC Curve with AUC
    2. FAR vs FRR plot with EER point
    3. Genuine/Impostor score distributions
    4. DET (Detection Error Tradeoff) curve
    5. Metrics report (EER, TAR@FAR, optimal threshold)

Run:
    python evaluate.py
    python evaluate.py --max_impostor 100000 --output_dir outputs/evaluation
"""

import os
import sys
import json
import argparse
import logging
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatter
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.similarity import (
    compute_far_frr,
    compute_eer,
    compute_roc,
    compute_tar_at_far,
    find_optimal_threshold,
)
from utils.preprocessing import IrisDataset, get_val_transforms
from generate_embeddings import load_trained_model, extract_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================
# Visualization Functions
# ============================================

def plot_score_distributions(genuine_scores, impostor_scores, eer_threshold, output_path):
    """Plot genuine vs impostor score distributions with EER threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(genuine_scores, bins=100, alpha=0.7, color='#2ecc71', label='Genuine (same identity)',
            density=True, edgecolor='white', linewidth=0.5)
    ax.hist(impostor_scores, bins=100, alpha=0.7, color='#e74c3c', label='Impostor (different identity)',
            density=True, edgecolor='white', linewidth=0.5)
    
    ax.axvline(x=eer_threshold, color='#f39c12', linewidth=2, linestyle='--',
               label=f'EER Threshold = {eer_threshold:.4f}')
    
    ax.set_xlabel('Cosine Similarity', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Genuine vs Impostor Score Distributions', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, alpha=0.3)
    
    # Add stats annotation
    stats_text = (f"Genuine: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}\n"
                  f"Impostor: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}\n"
                  f"Separation: {np.mean(genuine_scores) - np.mean(impostor_scores):.4f}")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Score distribution plot saved: {output_path}")


def plot_roc_curve(fpr, tpr, auc, output_path):
    """Plot ROC curve with AUC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale ROC
    ax1.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'ROC (AUC = {auc:.6f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    ax1.set_xlabel('False Positive Rate (FAR)', fontsize=13)
    ax1.set_ylabel('True Positive Rate (TAR)', fontsize=13)
    ax1.set_title('ROC Curve (Linear Scale)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    
    # Log scale ROC (zoomed into low-FAR region)
    ax2.semilogx(fpr + 1e-10, tpr, color='#e74c3c', linewidth=2, label=f'ROC (AUC = {auc:.6f})')
    ax2.set_xlabel('False Positive Rate (FAR) — Log Scale', fontsize=13)
    ax2.set_ylabel('True Positive Rate (TAR)', fontsize=13)
    ax2.set_title('ROC Curve (Log Scale — Low FAR Region)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([1e-5, 1.0])
    ax2.set_ylim([0.0, 1.01])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"ROC curve saved: {output_path}")


def plot_far_frr(thresholds, far, frr, eer, eer_threshold, output_path):
    """Plot FAR and FRR vs threshold with EER point."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, far, color='#e74c3c', linewidth=2, label='FAR (False Accept Rate)')
    ax.plot(thresholds, frr, color='#3498db', linewidth=2, label='FRR (False Reject Rate)')
    
    # EER point
    ax.scatter([eer_threshold], [eer], color='#f39c12', s=150, zorder=5, edgecolors='black',
               linewidth=1.5, label=f'EER = {eer*100:.4f}% @ t={eer_threshold:.4f}')
    
    # Annotate EER
    ax.annotate(f'EER = {eer*100:.4f}%\nThreshold = {eer_threshold:.4f}',
                xy=(eer_threshold, eer), xytext=(eer_threshold + 0.1, eer + 0.15),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='#f39c12'))
    
    ax.set_xlabel('Threshold', fontsize=13)
    ax.set_ylabel('Error Rate', fontsize=13)
    ax.set_title('FAR vs FRR — Equal Error Rate Analysis', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds.min(), thresholds.max()])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"FAR/FRR plot saved: {output_path}")


def plot_det_curve(far, frr, eer, output_path):
    """Plot DET (Detection Error Tradeoff) curve — log-log scale."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Filter out zeros for log scale
    mask = (far > 0) & (frr > 0)
    if np.sum(mask) > 10:
        far_plot = far[mask]
        frr_plot = frr[mask]
    else:
        far_plot = far + 1e-7
        frr_plot = frr + 1e-7
    
    ax.loglog(far_plot * 100, frr_plot * 100, color='#9b59b6', linewidth=2, label='DET Curve')
    ax.scatter([eer * 100], [eer * 100], color='#f39c12', s=150, zorder=5,
               edgecolors='black', linewidth=1.5, label=f'EER = {eer*100:.4f}%')
    
    # Diagonal (EER line)
    ax.plot([0.001, 100], [0.001, 100], 'k--', alpha=0.3)
    
    ax.set_xlabel('False Accept Rate (%)', fontsize=13)
    ax.set_ylabel('False Reject Rate (%)', fontsize=13)
    ax.set_title('DET Curve (Detection Error Tradeoff)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.001, 100])
    ax.set_ylim([0.001, 100])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"DET curve saved: {output_path}")


def plot_combined_dashboard(genuine_scores, impostor_scores, 
                             thresholds, far, frr, fpr, tpr,
                             eer, eer_threshold, auc, tar_at_far, output_path):
    """Create a combined 2x2 evaluation dashboard."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # --- 1. Score Distributions ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(genuine_scores, bins=80, alpha=0.7, color='#2ecc71', label='Genuine',
             density=True, edgecolor='white', linewidth=0.3)
    ax1.hist(impostor_scores, bins=80, alpha=0.7, color='#e74c3c', label='Impostor',
             density=True, edgecolor='white', linewidth=0.3)
    ax1.axvline(x=eer_threshold, color='#f39c12', linewidth=2, linestyle='--',
                label=f'EER Threshold = {eer_threshold:.3f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Score Distributions', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # --- 2. ROC Curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'AUC = {auc:.6f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.2)
    
    # --- 3. FAR/FRR ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(thresholds, far, color='#e74c3c', linewidth=2, label='FAR')
    ax3.plot(thresholds, frr, color='#3498db', linewidth=2, label='FRR')
    ax3.scatter([eer_threshold], [eer], color='#f39c12', s=120, zorder=5,
                edgecolors='black', linewidth=1.5, label=f'EER = {eer*100:.4f}%')
    ax3.set_xlabel('Threshold', fontsize=11)
    ax3.set_ylabel('Error Rate', fontsize=11)
    ax3.set_title('FAR vs FRR', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)
    
    # --- 4. Metrics Summary ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    metrics_text = [
        "IRIS RECOGNITION EVALUATION REPORT",
        "=" * 42,
        "",
        f"  Equal Error Rate (EER):   {eer*100:.4f}%",
        f"  EER Threshold:            {eer_threshold:.4f}",
        f"  ROC AUC:                  {auc:.6f}",
        "",
        "  TAR @ Operating Points:",
    ]
    for key, val in tar_at_far.items():
        metrics_text.append(f"    {key}: {val*100:.2f}%")
    
    metrics_text.extend([
        "",
        f"  Genuine Pairs:   {len(genuine_scores):,}",
        f"  Impostor Pairs:  {len(impostor_scores):,}",
        f"  Genuine Mean:    {np.mean(genuine_scores):.4f}",
        f"  Impostor Mean:   {np.mean(impostor_scores):.4f}",
        f"  Separation Gap:  {np.mean(genuine_scores) - np.mean(impostor_scores):.4f}",
    ])
    
    text = '\n'.join(metrics_text)
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.9))
    
    fig.suptitle('Iris Recognition System — Biometric Evaluation Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Evaluation dashboard saved: {output_path}")


# ============================================
# Main Evaluation Pipeline
# ============================================

def generate_protocol_scores(embeddings, labels, subject_ids, max_genuine_per_class=10,
                             max_impostor_pairs=50000, seed=42):
    """Create reproducible, unique image-pair scores with subject-cluster metadata."""
    from itertools import combinations

    rng = np.random.default_rng(seed)
    class_indices = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    class_subject = {}
    genuine, genuine_subject = [], []
    for label, indices in class_indices.items():
        # A person-eye class must refer to exactly one person.
        people = np.unique(subject_ids[indices])
        if len(people) != 1:
            raise ValueError(f"Class {label} maps to multiple people: {people.tolist()}")
        class_subject[label] = people[0]
        pairs = list(combinations(indices.tolist(), 2))
        if len(pairs) > max_genuine_per_class:
            pairs = [pairs[i] for i in rng.choice(len(pairs), max_genuine_per_class, replace=False)]
        for left, right in pairs:
            genuine.append(float(np.dot(embeddings[left], embeddings[right])))
            genuine_subject.append(people[0])

    # Sample distinct unordered image pairs, rather than sampling with replacement.
    n = len(embeddings)
    if n < 2 or len(class_indices) < 2:
        raise ValueError("At least two images from two person-eye classes are required")
    within_class_pairs = sum(len(indices) * (len(indices) - 1) // 2
                             for indices in class_indices.values())
    target = min(max_impostor_pairs, n * (n - 1) // 2 - within_class_pairs)
    chosen = set()
    while len(chosen) < target:
        need = target - len(chosen)
        batch = max(4096, need * 2)
        left = rng.integers(0, n, size=batch)
        right = rng.integers(0, n, size=batch)
        keep = (left != right) & (labels[left] != labels[right])
        left, right = np.minimum(left[keep], right[keep]), np.maximum(left[keep], right[keep])
        keys = np.unique(left.astype(np.int64) * n + right.astype(np.int64))
        for key in keys:
            chosen.add(int(key))
            if len(chosen) == target:
                break
    pair_keys = np.fromiter(chosen, dtype=np.int64, count=target)
    imp_left, imp_right = pair_keys // n, pair_keys % n
    impostor = np.einsum('ij,ij->i', embeddings[imp_left], embeddings[imp_right]).astype(np.float64)
    impostor_subjects = np.column_stack([subject_ids[imp_left], subject_ids[imp_right]])
    return (np.asarray(genuine, dtype=np.float64), impostor,
            np.asarray(genuine_subject), impostor_subjects)


def exact_curve(genuine, impostor):
    """Calculate empirical FAR/FRR at every observed score threshold."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.concatenate([thresholds, [np.nextafter(thresholds[-1], np.inf)]])
    gs, ins = np.sort(genuine), np.sort(impostor)
    frr = np.searchsorted(gs, thresholds, side='left') / max(1, len(gs))
    far = (len(ins) - np.searchsorted(ins, thresholds, side='left')) / max(1, len(ins))
    return thresholds, far, frr


def metrics_at_threshold(genuine, impostor, threshold):
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    return {"threshold": float(threshold), "far": far, "frr": frr,
            "tar": 1.0 - frr, "balanced_accuracy": 1.0 - (far + frr) / 2.0,
            "num_genuine_pairs": int(len(genuine)), "num_impostor_pairs": int(len(impostor))}


def select_eer_threshold(genuine, impostor):
    thresholds, far, frr = exact_curve(genuine, impostor)
    index = int(np.argmin(np.abs(far - frr)))
    return float(thresholds[index]), float((far[index] + frr[index]) / 2.0), thresholds, far, frr


def select_target_far_threshold(genuine, impostor, target_far):
    thresholds, far, frr = exact_curve(genuine, impostor)
    acceptable = np.flatnonzero(far <= target_far)
    index = int(acceptable[0]) if acceptable.size else len(thresholds) - 1
    return float(thresholds[index]), float(far[index]), float(frr[index])


def subject_cluster_bootstrap(genuine, impostor, genuine_subjects, impostor_subjects,
                              thresholds, repetitions=500, seed=2026):
    """Percentile intervals from resampling people, keeping each person's eye classes together."""
    people = np.unique(np.concatenate([genuine_subjects, impostor_subjects.ravel()]))
    person_index = {person: index for index, person in enumerate(people)}
    gs = np.asarray([person_index[p] for p in genuine_subjects], dtype=np.int32)
    ia = np.asarray([person_index[p] for p in impostor_subjects[:, 0]], dtype=np.int32)
    ib = np.asarray([person_index[p] for p in impostor_subjects[:, 1]], dtype=np.int32)
    rng = np.random.default_rng(seed)
    collected = {name: {metric: [] for metric in ('far', 'frr', 'tar')}
                 for name in thresholds}
    for _ in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, len(people), size=len(people)), minlength=len(people))
        gw = multiplicity[gs].astype(np.float64)
        iw = (multiplicity[ia] * multiplicity[ib]).astype(np.float64)
        same_person = ia == ib
        iw[same_person] = multiplicity[ia[same_person]]
        gden, iden = gw.sum(), iw.sum()
        if not gden or not iden:
            continue
        for name, threshold in thresholds.items():
            frr = float(gw[genuine < threshold].sum() / gden)
            far = float(iw[impostor >= threshold].sum() / iden)
            collected[name]['far'].append(far)
            collected[name]['frr'].append(frr)
            collected[name]['tar'].append(1.0 - frr)
    intervals = {}
    for name, values in collected.items():
        intervals[name] = {
            metric: [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
            for metric, samples in values.items() if samples
        }
    zero_event_thresholds = [name for name, threshold in thresholds.items()
                              if not np.any(impostor >= threshold)]
    return {"method": "subject-cluster bootstrap percentile interval",
            "repetitions": repetitions, "people": int(len(people)), "confidence": 0.95,
            "conditional_on_frozen_validation_thresholds": True, "intervals": intervals,
            "limitations": ("The percentile bootstrap is degenerate for FAR when no impostor was accepted; "
                            "a [0, 0] interval is not evidence that population FAR is zero. "
                            "Use the observed pair count and a larger, more diverse impostor test."),
            "zero_observed_false_accept_thresholds": zero_event_thresholds}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Evaluate IrisSecure on unseen identities")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to evaluate (default: configured run best model)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation/test embedding extraction')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers for local evaluation')
    parser.add_argument('--max_impostor', type=int, default=None,
                        help='Maximum unique impostor pairs per split')
    parser.add_argument('--max_genuine_per_class', type=int, default=None,
                        help='Maximum genuine pairs per person-eye class')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output root; results are placed in a run-specific subdirectory')
    parser.add_argument('--bootstrap_repetitions', type=int, default=500,
                        help='Subject-cluster bootstrap repetitions for test-rate intervals')
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_repetitions < 1:
        parser.error('batch_size and bootstrap_repetitions must be positive')

    import yaml

    with (PROJECT_ROOT / 'config.yaml').open(encoding='utf-8') as source:
        config = yaml.safe_load(source)
    run_name = os.getenv(
        'IRISSECURE_RUN_NAME', config['training'].get('run_name', 'subject_disjoint_run')
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        PROJECT_ROOT / config['training'].get('output_root', 'models/runs') /
        run_name / 'checkpoints' / 'best_model.pth'
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Subject-disjoint checkpoint not found: {checkpoint_path}. "
            "Complete the final training run first, or pass --checkpoint explicitly."
        )

    split_dir = PROJECT_ROOT / config['dataset'].get(
        'subject_split_dir', 'data/splits/subject_disjoint'
    )
    split_metadata_path = split_dir / 'split_metadata.json'
    if not split_metadata_path.is_file():
        raise FileNotFoundError(f"Subject-disjoint split metadata not found: {split_metadata_path}")
    split_metadata = json.loads(split_metadata_path.read_text(encoding='utf-8'))
    frames = {
        name: pd.read_csv(split_dir / f'{name}.csv', dtype={'person_id': str})
        for name in ('train', 'val', 'test')
    }
    for name, frame in frames.items():
        missing = {'person_id', 'label', 'label_idx', 'image_path'} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")
        actual_people = set(frame['person_id'].astype(str))
        metadata_people = set(map(str, split_metadata.get('subjects', {}).get(name, [])))
        if actual_people != metadata_people:
            raise ValueError(f"{name} manifest identities do not match split_metadata.json")

    subject_sets = {name: set(frame['person_id'].astype(str)) for name, frame in frames.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = subject_sets[left] & subject_sets[right]
        if overlap:
            raise ValueError(f"Identity leakage between {left} and {right}: {len(overlap)} people")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model, checkpoint_info = load_trained_model(str(checkpoint_path), device)
    checkpoint_split = checkpoint_info.get('split_metadata')
    if not checkpoint_split or checkpoint_split.get('subjects') != split_metadata.get('subjects'):
        raise ValueError(
            "Checkpoint split metadata does not match the current subject-disjoint manifests. "
            "Train a model using the configured split before evaluating it."
        )

    pp_cfg = config['preprocessing']
    image_size = tuple(pp_cfg['image_size'])

    def embeddings_for(frame, split_name):
        frame = frame.copy()
        frame['image_path'] = frame['image_path'].map(
            lambda value: str((PROJECT_ROOT / value).resolve())
            if not Path(value).is_absolute() else value
        )
        missing_paths = [path for path in frame['image_path'] if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"{split_name} manifest references missing image: {missing_paths[0]}"
            )
        dataset = IrisDataset(
            frame,
            transform=get_val_transforms(image_size),
            image_size=image_size,
            apply_clahe_flag=pp_cfg.get('apply_clahe', True),
            clahe_clip_limit=pp_cfg.get('clahe_clip_limit', 2.0),
            clahe_tile_grid_size=tuple(pp_cfg.get('clahe_tile_grid_size', [8, 8])),
            gaussian_blur_kernel=pp_cfg.get('gaussian_blur_kernel', 3),
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == 'cuda'
        )
        vectors, labels = extract_embeddings(model, loader, device, use_fp16=True)
        expected_labels = frame['label_idx'].to_numpy(dtype=np.int64)
        if not np.array_equal(labels.astype(np.int64), expected_labels):
            raise ValueError(f"{split_name} embedding order does not match its manifest")
        return vectors.astype(np.float32), expected_labels, frame['person_id'].astype(str).to_numpy()

    output_dir = PROJECT_ROOT / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    print("\n" + "=" * 65)
    print("   IrisSecure — Subject-Disjoint Biometric Evaluation")
    print("=" * 65)
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Device: {device} | batch size: {args.batch_size}")

    print(f"   Extracting validation embeddings ({len(frames['val']):,} images)...")
    val_embeddings, val_labels, val_subjects = embeddings_for(frames['val'], 'validation')
    print(f"   Extracting test embeddings ({len(frames['test']):,} images)...")
    test_embeddings, test_labels, test_subjects = embeddings_for(frames['test'], 'test')

    try:
        git_revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True
        ).stdout.strip()
        worktree_dirty = subprocess.run(
            ['git', 'diff', '--quiet'], cwd=PROJECT_ROOT
        ).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        git_revision, worktree_dirty = None, None

    validation_seed = int(config['training'].get('seed', 42))
    verification_cfg = config.get('verification', {})
    configured_genuine_limit = verification_cfg.get('validation_max_genuine_pairs_per_class', 10)
    genuine_limit = int(args.max_genuine_per_class or configured_genuine_limit)
    max_impostor_pairs = int(args.max_impostor or verification_cfg.get(
        'validation_max_impostor_pairs', 50000
    ))
    if genuine_limit < 1:
        parser.error('max_genuine_per_class must be positive')
    if max_impostor_pairs < 1:
        parser.error('max_impostor must be positive')

    def split_scores(embeddings, labels, subjects, seed):
        return generate_protocol_scores(
            embeddings, labels, subjects, genuine_limit, max_impostor_pairs, seed
        )

    print('   Generating validation pairs for threshold selection...')
    vg, vi, vgs, vis = split_scores(
        val_embeddings, val_labels, val_subjects, seed=validation_seed
    )
    val_threshold, val_eer, val_thresholds, val_far_curve, val_frr_curve = select_eer_threshold(vg, vi)
    validation_metrics = metrics_at_threshold(vg, vi, val_threshold)
    val_auc = float(roc_auc_score(
        np.r_[np.ones(len(vg)), np.zeros(len(vi))], np.r_[vg, vi]
    ))
    validation_path = output_dir / 'validation_score_distributions.npz'
    np.savez_compressed(validation_path, genuine_scores=vg.astype(np.float32),
                        impostor_scores=vi.astype(np.float32))

    print('   Generating held-out test pairs...')
    tg, ti, tgs, tis = split_scores(
        test_embeddings, test_labels, test_subjects, seed=validation_seed + 1
    )
    test_selected = metrics_at_threshold(tg, ti, val_threshold)
    test_fixed = metrics_at_threshold(tg, ti, 0.75)
    test_auc = float(roc_auc_score(
        np.r_[np.ones(len(tg)), np.zeros(len(ti))], np.r_[tg, ti]
    ))
    target_operations = {}
    for target_far in (0.01, 0.001, 0.0001):
        threshold, achieved_val_far, achieved_val_frr = select_target_far_threshold(vg, vi, target_far)
        target_operations[str(target_far)] = {
            'threshold_selected_on_validation': threshold,
            'validation_far': achieved_val_far,
            'validation_frr': achieved_val_frr,
            'test_metrics_at_threshold': metrics_at_threshold(tg, ti, threshold),
            'validation_impostor_pair_resolution': 1.0 / len(vi),
            'target_supported_by_pair_count': target_far >= 1.0 / len(vi),
        }

    bootstrap = subject_cluster_bootstrap(
        tg, ti, tgs, tis,
        {'validation_eer': val_threshold, 'fixed_0.75': 0.75},
        repetitions=args.bootstrap_repetitions,
    )
    test_path = output_dir / 'score_distributions.npz'
    np.savez_compressed(test_path, genuine_scores=tg.astype(np.float32),
                        impostor_scores=ti.astype(np.float32))

    report = {
        'evaluation_date': datetime.now().isoformat(),
        'split': 'test',
        'calibration_split': 'val',
        'threshold_selection': 'EER selected on validation; frozen for test reporting',
        'protocol': 'person-eye verification; unique unordered image pairs; subject-cluster bootstrap',
        'train_val_identity_overlap': False,
        'train_test_identity_overlap': False,
        'validation_test_identity_overlap': False,
        'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': checkpoint_info.get('epoch'),
        'checkpoint_validation_eer': checkpoint_info.get('val_eer'),
        'checkpoint_validation_threshold': checkpoint_info.get('val_threshold'),
        'training_seed': checkpoint_info.get('training_seed'),
        'reproducibility': {
            'checkpoint_sha256': sha256_file(checkpoint_path),
            'train_manifest_sha256': sha256_file(split_dir / 'train.csv'),
            'validation_manifest_sha256': sha256_file(split_dir / 'val.csv'),
            'test_manifest_sha256': sha256_file(split_dir / 'test.csv'),
            'split_metadata_sha256': sha256_file(split_metadata_path),
            'git_revision': git_revision,
            'git_worktree_dirty': worktree_dirty,
            'validation_pair_seed': validation_seed,
            'test_pair_seed': validation_seed + 1,
            'bootstrap_seed': 2026,
        },
        'num_train_persons': int(frames['train'].person_id.nunique()),
        'num_validation_images': int(len(frames['val'])),
        'num_test_images': int(len(frames['test'])),
        'num_validation_persons': int(frames['val'].person_id.nunique()),
        'num_test_persons': int(frames['test'].person_id.nunique()),
        'num_classes': int(frames['test'].label_idx.nunique()),
        'num_validation_genuine_pairs': int(len(vg)),
        'num_validation_impostor_pairs': int(len(vi)),
        'num_genuine_pairs': int(len(tg)),
        'num_impostor_pairs': int(len(ti)),
        'validation_score_distributions_file': validation_path.name,
        'score_distributions_file': test_path.name,
        'validation_roc_auc': val_auc,
        'validation_eer': val_eer,
        'validation_eer_threshold': val_threshold,
        'validation_metrics_at_eer': validation_metrics,
        'test_roc_auc': test_auc,
        'test_at_validation_eer': test_selected,
        'test_at_fixed_threshold': test_fixed,
        'target_far_operations': target_operations,
        'subject_cluster_bootstrap': bootstrap,
        'genuine_stats': {'mean': float(np.mean(tg)), 'std': float(np.std(tg)),
                          'min': float(np.min(tg)), 'max': float(np.max(tg))},
        'impostor_stats': {'mean': float(np.mean(ti)), 'std': float(np.std(ti)),
                           'min': float(np.min(ti)), 'max': float(np.max(ti))},
        'separation_gap': float(np.mean(tg) - np.mean(ti)),
        'interpretation': [
            'Train, validation, and test people are disjoint; validation selects the threshold and test estimates held-out performance.',
            'The threshold used for test reporting is frozen from validation scores.',
            'Pair scores reuse people and images; confidence intervals therefore resample people as clusters.',
            'Low-FAR estimates are limited by the impostor-pair count and pair dependence.',
            'This experiment measures identity verification, not presentation-attack detection.',
        ],
    }

    plot_score_distributions(vg, vi, val_threshold, output_dir / 'score_distributions.png')
    val_fpr, val_tpr, _ = compute_roc(vg, vi)
    plot_roc_curve(val_fpr, val_tpr, val_auc, output_dir / 'roc_curve.png')
    plot_far_frr(val_thresholds, val_far_curve, val_frr_curve, val_eer,
                 val_threshold, output_dir / 'far_frr_analysis.png')
    plot_det_curve(val_far_curve, val_frr_curve, val_eer, output_dir / 'det_curve.png')
    plot_combined_dashboard(
        vg, vi, val_thresholds, val_far_curve, val_frr_curve,
        val_fpr, val_tpr, val_eer, val_threshold, val_auc,
        {f'TAR@FAR={k}': 1 - select_target_far_threshold(vg, vi, k)[2]
         for k in (0.01, 0.001)},
        output_dir / 'evaluation_dashboard.png',
    )
    with (output_dir / 'evaluation_report.json').open('w', encoding='utf-8') as report_file:
        json.dump(report, report_file, indent=2)
    for private_output in output_dir.iterdir():
        if private_output.is_file():
            os.chmod(private_output, 0o600)

    print('\nVALIDATION-SELECTED / TEST-EVALUATED RESULTS')
    print(f"  Validation EER threshold: {val_threshold:.6f} (selected on validation)")
    print(f"  Test at frozen threshold: FAR={test_selected['far']:.4%}, FRR={test_selected['frr']:.4%}, TAR={test_selected['tar']:.4%}")
    print(f"  Test at fixed 0.75:       FAR={test_fixed['far']:.4%}, FRR={test_fixed['frr']:.4%}, TAR={test_fixed['tar']:.4%}")
    print(f"  Test ROC AUC:             {test_auc:.6f}")
    print(f"  Test pairs:               {len(tg):,} genuine / {len(ti):,} impostor")
    print(f"  Subject bootstrap:        {args.bootstrap_repetitions} repetitions / {bootstrap['people']} people")
    print(f"  Full report:              {output_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
