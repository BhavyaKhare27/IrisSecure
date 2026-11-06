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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.similarity import (
    generate_evaluation_pairs,
    compute_far_frr,
    compute_eer,
    compute_roc,
    compute_tar_at_far,
    find_optimal_threshold,
)

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate Iris Recognition System")
    parser.add_argument('--max_impostor', type=int, default=50000,
                        help='Max impostor pairs to generate')
    parser.add_argument('--max_genuine_per_class', type=int, default=10,
                        help='Max genuine pairs per class')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output directory for plots and reports')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'all'],
                        help='Which data split to evaluate on')
    args = parser.parse_args()
    
    # Setup
    import pickle
    import yaml
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print("   Iris Recognition System — Biometric Evaluation")
    print("=" * 65 + "\n")
    
    # Load config
    with open(PROJECT_ROOT / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load embeddings
    print("   Loading embeddings...")
    db_path = PROJECT_ROOT / "database" / "embeddings.pkl"
    if not db_path.exists():
        logger.error("Embeddings not found. Run generate_embeddings.py first!")
        sys.exit(1)
    
    with open(db_path, 'rb') as f:
        embedding_records = pickle.load(f)
    
    # Load appropriate split
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    
    if args.split == 'all':
        split_df = pd.read_csv(processed_dir / "iris_dataset.csv")
    else:
        split_df = pd.read_csv(processed_dir / f"{args.split}.csv")
    
    split_paths = set(split_df['image_path'].tolist())
    
    # Filter embeddings to the selected split
    embeddings_list = []
    labels_list = []
    
    for rec in embedding_records:
        if rec['image_path'] in split_paths:
            embeddings_list.append(rec['embedding'])
            labels_list.append(rec['label_idx'])
    
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    
    n_classes = len(np.unique(labels))
    print(f"   Split: {args.split}")
    print(f"   Embeddings: {len(embeddings):,}")
    print(f"   Classes: {n_classes}")
    
    # Generate evaluation pairs
    print(f"\n   Generating evaluation pairs...")
    genuine_scores, impostor_scores = generate_evaluation_pairs(
        embeddings, labels,
        max_genuine_per_class=args.max_genuine_per_class,
        max_impostor_pairs=args.max_impostor,
        seed=42
    )
    
    print(f"   Genuine pairs:  {len(genuine_scores):,}")
    print(f"   Impostor pairs: {len(impostor_scores):,}")
    
    # Compute metrics
    print("\n   Computing biometric metrics...")
    
    # EER
    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    
    # ROC
    fpr, tpr, auc = compute_roc(genuine_scores, impostor_scores)
    
    # FAR/FRR
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)
    
    # TAR@FAR
    tar_at_far = compute_tar_at_far(genuine_scores, impostor_scores,
                                     target_far_values=[0.1, 0.01, 0.001, 0.0001])
    
    # Optimal thresholds
    _, eer_metrics = find_optimal_threshold(genuine_scores, impostor_scores, method='eer')
    _, acc_metrics = find_optimal_threshold(genuine_scores, impostor_scores, method='max_accuracy')
    
    # Print report
    print("\n" + "=" * 65)
    print("   EVALUATION RESULTS")
    print("=" * 65)
    print(f"   Equal Error Rate (EER):     {eer*100:.4f}%")
    print(f"   EER Threshold:              {eer_threshold:.4f}")
    print(f"   ROC AUC:                    {auc:.6f}")
    print(f"   Optimal Threshold (acc):    {acc_metrics['threshold']:.4f}")
    print("-" * 65)
    print("   TAR @ Operating Points:")
    for key, val in tar_at_far.items():
        print(f"     {key}: {val*100:.2f}%")
    print("-" * 65)
    print(f"   Genuine:  mean={np.mean(genuine_scores):.4f}, "
          f"std={np.std(genuine_scores):.4f}, "
          f"min={np.min(genuine_scores):.4f}")
    print(f"   Impostor: mean={np.mean(impostor_scores):.4f}, "
          f"std={np.std(impostor_scores):.4f}, "
          f"max={np.max(impostor_scores):.4f}")
    print(f"   Separation Gap: {np.mean(genuine_scores) - np.mean(impostor_scores):.4f}")
    print("=" * 65)
    
    # Generate plots
    print("\n   Generating evaluation plots...")
    
    plot_score_distributions(
        genuine_scores, impostor_scores, eer_threshold,
        output_dir / "score_distributions.png"
    )
    
    plot_roc_curve(fpr, tpr, auc, output_dir / "roc_curve.png")
    
    plot_far_frr(
        thresholds, far, frr, eer, eer_threshold,
        output_dir / "far_frr_analysis.png"
    )
    
    plot_det_curve(far, frr, eer, output_dir / "det_curve.png")
    
    plot_combined_dashboard(
        genuine_scores, impostor_scores,
        thresholds, far, frr, fpr, tpr,
        eer, eer_threshold, auc, tar_at_far,
        output_dir / "evaluation_dashboard.png"
    )
    
    # Save metrics report
    score_path = output_dir / "score_distributions.npz"
    np.savez_compressed(
        score_path,
        genuine_scores=genuine_scores.astype(np.float32),
        impostor_scores=impostor_scores.astype(np.float32),
    )

    report = {
        'evaluation_date': datetime.now().isoformat(),
        'split': args.split,
        'num_embeddings': len(embeddings),
        'num_classes': n_classes,
        'num_genuine_pairs': len(genuine_scores),
        'num_impostor_pairs': len(impostor_scores),
        'score_distributions_file': score_path.name,
        'eer': round(eer, 6),
        'eer_percent': round(eer * 100, 4),
        'eer_threshold': round(eer_threshold, 4),
        'roc_auc': round(auc, 6),
        'tar_at_far': tar_at_far,
        'optimal_threshold_eer': eer_metrics,
        'optimal_threshold_accuracy': acc_metrics,
        'genuine_stats': {
            'mean': round(float(np.mean(genuine_scores)), 4),
            'std': round(float(np.std(genuine_scores)), 4),
            'min': round(float(np.min(genuine_scores)), 4),
            'max': round(float(np.max(genuine_scores)), 4),
        },
        'impostor_stats': {
            'mean': round(float(np.mean(impostor_scores)), 4),
            'std': round(float(np.std(impostor_scores)), 4),
            'min': round(float(np.min(impostor_scores)), 4),
            'max': round(float(np.max(impostor_scores)), 4),
        },
        'separation_gap': round(float(np.mean(genuine_scores) - np.mean(impostor_scores)), 4),
    }
    
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print(f"\n   Files saved to: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"     {f.name} ({size_kb:.1f} KB)")
    
    # Quality verdict
    print("\n" + "=" * 65)
    if eer < 0.005:
        verdict = "EXCELLENT — System is production-ready"
    elif eer < 0.01:
        verdict = "VERY GOOD — Near production quality"
    elif eer < 0.05:
        verdict = "GOOD — Acceptable for most applications"
    elif eer < 0.10:
        verdict = "FAIR — Needs improvement for critical applications"
    else:
        verdict = "POOR — Requires significant improvement"
    
    print(f"   VERDICT: EER = {eer*100:.4f}% --> {verdict}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
