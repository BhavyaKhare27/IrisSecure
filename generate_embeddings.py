"""
Embedding Generation Script
==============================
Loads the trained best model checkpoint and generates embeddings for
all images in the dataset. Saves embeddings to a pickle database for
the registration and verification pipeline.

This script:
    1. Loads the best trained model
    2. Processes ALL images (train + val + test) through the model
    3. Extracts 512-dim L2-normalized embeddings
    4. Saves per-image embeddings with metadata
    5. Computes and saves per-class mean embeddings (centroid gallery)
    6. Validates embedding quality with intra/inter class similarity stats

Output:
    database/embeddings.pkl      - Per-image embeddings with metadata
    database/gallery.pkl         - Per-class mean embeddings (verification gallery)
    database/embedding_stats.json - Quality statistics

Run:
    python generate_embeddings.py
    python generate_embeddings.py --checkpoint models/checkpoints/best_model.pth
"""

import os
import sys
import pickle
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.feature_extractor import IrisEmbeddingNet
from utils.preprocessing import IrisDataset, get_val_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: str, device: torch.device):
    """
    Load a trained IrisEmbeddingNet from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint
        device: Target device
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    
    # Build model with same architecture
    model = IrisEmbeddingNet(
        backbone=model_cfg.get('backbone', 'resnet50'),
        embedding_dim=model_cfg.get('embedding_dim', 512),
        pretrained=False,  # We'll load trained weights
        freeze_backbone=False
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
        'val_accuracy': checkpoint.get('val_accuracy', 'unknown'),
        'num_classes': checkpoint.get('num_classes', 'unknown'),
        'backbone': model_cfg.get('backbone', 'resnet50'),
        'embedding_dim': model_cfg.get('embedding_dim', 512),
    }
    
    logger.info(f"Model loaded: epoch={info['epoch']}, "
                f"val_acc={info['val_accuracy']}, "
                f"backbone={info['backbone']}")
    
    return model, info


@torch.no_grad()
def extract_embeddings(model: IrisEmbeddingNet, 
                       dataloader: DataLoader,
                       device: torch.device,
                       use_fp16: bool = True) -> tuple:
    """
    Extract embeddings for all images in a DataLoader.
    
    Args:
        model: Trained IrisEmbeddingNet
        dataloader: DataLoader with iris images
        device: Compute device
        use_fp16: Whether to use FP16 for faster inference
        
    Returns:
        Tuple of (embeddings_array, labels_array)
        - embeddings_array: numpy (N, embedding_dim) 
        - labels_array: numpy (N,) integer labels
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Extracting embeddings", ncols=100):
        images = images.to(device, non_blocking=True)
        
        if use_fp16 and device.type == 'cuda':
            with autocast('cuda'):
                embeddings = model(images, normalize=True)
        else:
            embeddings = model(images, normalize=True)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
    
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    logger.info(f"Extracted {len(embeddings_array)} embeddings of dim {embeddings_array.shape[1]}")
    
    return embeddings_array, labels_array


def compute_gallery(embeddings: np.ndarray, 
                    labels: np.ndarray,
                    label_names: dict) -> dict:
    """
    Compute per-class mean embeddings (gallery centroids).
    
    For verification, a query embedding is compared against these centroids.
    Averaging multiple embeddings per class reduces noise.
    
    Args:
        embeddings: (N, D) array of L2-normalized embeddings
        labels: (N,) integer labels
        label_names: Dict mapping label_idx -> label_name (e.g., '000_L')
        
    Returns:
        Gallery dict: {label_name: {'embedding': np.array, 'count': int, ...}}
    """
    gallery = {}
    
    unique_labels = np.unique(labels)
    
    for label_idx in unique_labels:
        mask = labels == label_idx
        class_embeddings = embeddings[mask]
        
        # Mean embedding (centroid) — re-normalize after averaging
        mean_embedding = class_embeddings.mean(axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        
        label_name = label_names.get(label_idx, str(label_idx))
        
        # Intra-class similarity: how consistent are embeddings within this class?
        if len(class_embeddings) > 1:
            similarities = class_embeddings @ mean_embedding
            intra_sim = float(np.mean(similarities))
        else:
            intra_sim = 1.0
        
        gallery[label_name] = {
            'embedding': mean_embedding,
            'count': int(len(class_embeddings)),
            'intra_class_similarity': round(intra_sim, 4),
            'label_idx': int(label_idx),
        }
    
    return gallery


def compute_embedding_stats(embeddings: np.ndarray, 
                            labels: np.ndarray,
                            gallery: dict) -> dict:
    """
    Compute statistics to assess embedding quality.
    
    Good embeddings have:
        - High intra-class similarity (same identity, similar embeddings)
        - Low inter-class similarity (different identity, different embeddings)
        - Large gap between intra and inter similarity
    """
    # Intra-class similarities (from gallery)
    intra_sims = [v['intra_class_similarity'] for v in gallery.values()]
    
    # Inter-class similarities (sample pairwise between random class centroids)
    centroids = np.array([v['embedding'] for v in gallery.values()])
    
    # Compute all pairwise cosine similarities between centroids
    sim_matrix = centroids @ centroids.T
    n = len(centroids)
    
    # Extract upper triangle (exclude diagonal = self-similarity)
    upper_indices = np.triu_indices(n, k=1)
    inter_sims = sim_matrix[upper_indices]
    
    stats = {
        'total_embeddings': len(embeddings),
        'total_classes': len(gallery),
        'embedding_dim': int(embeddings.shape[1]),
        'intra_class_similarity': {
            'mean': round(float(np.mean(intra_sims)), 4),
            'std': round(float(np.std(intra_sims)), 4),
            'min': round(float(np.min(intra_sims)), 4),
            'max': round(float(np.max(intra_sims)), 4),
        },
        'inter_class_similarity': {
            'mean': round(float(np.mean(inter_sims)), 4),
            'std': round(float(np.std(inter_sims)), 4),
            'min': round(float(np.min(inter_sims)), 4),
            'max': round(float(np.max(inter_sims)), 4),
        },
        'separation_gap': round(float(np.mean(intra_sims) - np.mean(inter_sims)), 4),
        'embedding_norm_mean': round(float(np.linalg.norm(embeddings, axis=1).mean()), 4),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate Iris Embeddings")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: best_model.pth)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference (can be larger than training)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Checkpoint path
    checkpoint_path = args.checkpoint or str(
        PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
    )
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run train.py first to train the model!")
        sys.exit(1)
    
    print("\n" + "=" * 65)
    print("   Iris Recognition System - Embedding Generation")
    print("=" * 65 + "\n")
    
    # Load trained model
    model, model_info = load_trained_model(checkpoint_path, device)
    
    # Load config for preprocessing
    import yaml
    with open(PROJECT_ROOT / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    pp_cfg = config['preprocessing']
    image_size = tuple(pp_cfg['image_size'])
    
    # Load the FULL dataset (all images, not just train)
    full_df = pd.read_csv(
        PROJECT_ROOT / config['dataset']['processed_dir'] / "iris_dataset.csv"
    )
    
    # Load label mapping
    label_mapping_df = pd.read_csv(
        PROJECT_ROOT / config['dataset']['processed_dir'] / "label_mapping.csv"
    )
    idx_to_label = dict(zip(label_mapping_df['label_idx'], label_mapping_df['label']))
    
    print(f"   Total images to process: {len(full_df):,}")
    print(f"   Embedding dimension:     {model_info['embedding_dim']}")
    print(f"   Model checkpoint:        epoch {model_info['epoch']}")
    print(f"   Batch size:              {args.batch_size}\n")
    
    # Create DataLoader (no augmentation for embedding extraction)
    dataset = IrisDataset(
        full_df,
        transform=get_val_transforms(image_size),
        image_size=image_size,
        apply_clahe_flag=pp_cfg['apply_clahe'],
        clahe_clip_limit=pp_cfg['clahe_clip_limit'],
        clahe_tile_grid_size=tuple(pp_cfg['clahe_tile_grid_size']),
        gaussian_blur_kernel=pp_cfg['gaussian_blur_kernel']
    )
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, dataloader, device)
    
    # Build per-image embedding database
    print("\nBuilding embedding database...")
    db_dir = PROJECT_ROOT / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-image embeddings with metadata
    embedding_records = []
    for i in range(len(full_df)):
        row = full_df.iloc[i]
        embedding_records.append({
            'person_id': row['person_id'],
            'eye': row['eye'],
            'label': row['label'],
            'label_idx': int(labels[i]),
            'image_path': row['image_path'],
            'image_filename': row['image_filename'],
            'embedding': embeddings[i],  # numpy array (512,)
        })
    
    # Save per-image embeddings
    embeddings_path = db_dir / "embeddings.pkl"
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embedding_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    embeddings_size = embeddings_path.stat().st_size / (1024 * 1024)
    print(f"   Per-image embeddings saved: {embeddings_path} ({embeddings_size:.1f} MB)")
    
    # Compute gallery (per-class mean embeddings)
    print("\nComputing gallery centroids...")
    gallery = compute_gallery(embeddings, labels, idx_to_label)
    
    gallery_path = db_dir / "gallery.pkl"
    with open(gallery_path, 'wb') as f:
        pickle.dump(gallery, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    gallery_size = gallery_path.stat().st_size / (1024 * 1024)
    print(f"   Gallery saved: {gallery_path} ({gallery_size:.1f} MB)")
    print(f"   Gallery entries: {len(gallery)} class centroids")
    
    # Compute and save quality statistics
    print("\nComputing embedding quality statistics...")
    stats = compute_embedding_stats(embeddings, labels, gallery)
    
    stats_path = db_dir / "embedding_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print quality report
    print("\n" + "=" * 65)
    print("   Embedding Quality Report")
    print("=" * 65)
    print(f"   Total Embeddings:      {stats['total_embeddings']:,}")
    print(f"   Total Classes:         {stats['total_classes']}")
    print(f"   Embedding Dimension:   {stats['embedding_dim']}")
    print(f"   Embedding Norm (mean): {stats['embedding_norm_mean']}")
    print("-" * 65)
    intra = stats['intra_class_similarity']
    inter = stats['inter_class_similarity']
    print(f"   Intra-Class Similarity:  {intra['mean']:.4f} +/- {intra['std']:.4f} "
          f"[{intra['min']:.4f}, {intra['max']:.4f}]")
    print(f"   Inter-Class Similarity:  {inter['mean']:.4f} +/- {inter['std']:.4f} "
          f"[{inter['min']:.4f}, {inter['max']:.4f}]")
    print(f"   Separation Gap:          {stats['separation_gap']:.4f}")
    
    # Quality assessment
    gap = stats['separation_gap']
    if gap > 0.5:
        quality = "EXCELLENT - Very well separated embeddings"
    elif gap > 0.3:
        quality = "GOOD - Well separated embeddings"
    elif gap > 0.1:
        quality = "FAIR - Some overlap between classes"
    else:
        quality = "POOR - Significant overlap, consider more training"
    
    print(f"   Quality Assessment:      {quality}")
    print("=" * 65)
    
    print(f"\n   Database files saved to: {db_dir}")
    for f in sorted(db_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"     {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n   Next Step: Start the IrisSecure dashboard to enroll and verify identities.\n")


if __name__ == "__main__":
    main()
