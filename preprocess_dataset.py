"""
Preprocess Dataset Script
==========================
Main entry point for Phase 1: Dataset preprocessing.

This script:
1. Loads the CASIA-Iris-Thousand dataset structure
2. Creates the dataset manifest CSV (catalogs all images with labels)
3. Saves label-to-index mapping
4. Performs train/val/test split
5. Saves split CSVs for reproducible training
6. Generates dataset statistics and visualizations
7. Verifies a sample batch from the DataLoader

Run:
    python preprocess_dataset.py
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_loader import CASIAIrisDatasetLoader
from utils.preprocessing import (
    create_data_splits,
    create_dataloaders,
    preprocess_single_image,
    get_val_transforms
)
from utils.visualization import (
    visualize_preprocessing_pipeline,
    visualize_sample_grid,
    plot_dataset_distribution,
    print_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from: {config_path}")
    return config


def main():
    """Execute the full preprocessing pipeline."""
    
    print("\n" + "=" * 60)
    print("  🔬 Iris Recognition System - Dataset Preprocessing")
    print("=" * 60 + "\n")
    
    # Step 0: Load Configuration
    
    config = load_config()
    dataset_root = PROJECT_ROOT / config['dataset']['root_dir']
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create other project directories
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    (PROJECT_ROOT / "database").mkdir(exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
    
    
    # Step 1: Load and Catalog Dataset
    
    print(" Step 1: Loading dataset structure...")
    loader = CASIAIrisDatasetLoader(str(dataset_root))
    df = loader.load()
    
    # Print statistics
    stats = loader.get_stats()
    print_stats(stats)
    
    # Save stats to JSON
    stats_path = processed_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats saved to: {stats_path}")
    
    
    # Step 2: Save Dataset CSV and Label Mapping
    
    print("💾 Step 2: Saving dataset manifest...")
    csv_path = processed_dir / "iris_dataset.csv"
    loader.save_csv(str(csv_path))
    
    label_map_path = processed_dir / "label_mapping.csv"
    loader.save_label_mapping(str(label_map_path))
    
    print(f"   Dataset CSV: {csv_path}")
    print(f"   Label mapping: {label_map_path}")
    print(f"   Total samples: {len(df):,}")
    print(f"   Total classes: {df['label'].nunique():,}")
    
    
    # Step 3: Create Train/Val/Test Split
    
    print(" Step 3: Creating train/val/test splits...")
    split_config = config['split']
    
    train_df, val_df, test_df = create_data_splits(
        df,
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio'],
        random_seed=split_config['random_seed'],
        stratify=split_config['stratify']
    )
    
    # Save split CSVs
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    print(f"   Train: {len(train_df):,} samples ({split_config['train_ratio']*100:.0f}%)")
    print(f"   Val:   {len(val_df):,} samples ({split_config['val_ratio']*100:.0f}%)")
    print(f"   Test:  {len(test_df):,} samples ({split_config['test_ratio']*100:.0f}%)")
    
    # Verify stratification
    train_classes = train_df['label'].nunique()
    val_classes = val_df['label'].nunique()
    test_classes = test_df['label'].nunique()
    print(f"   Classes in train/val/test: {train_classes}/{val_classes}/{test_classes}")
    
    
    # Step 4: Test Preprocessing on Sample Image
    
    print(" Step 4: Testing preprocessing pipeline...")
    sample_row = df.iloc[0]
    sample_path = sample_row['image_path']
    
    pp_config = config['preprocessing']
    preprocessed = preprocess_single_image(
        sample_path,
        image_size=tuple(pp_config['image_size']),
        apply_clahe_flag=pp_config['apply_clahe'],
        clahe_clip_limit=pp_config['clahe_clip_limit'],
        clahe_tile_grid_size=tuple(pp_config['clahe_tile_grid_size']),
        gaussian_blur_kernel=pp_config['gaussian_blur_kernel']
    )
    
    if preprocessed is not None:
        print(f"   Sample image preprocessed: {sample_path}")
        print(f"   Output shape: {preprocessed.shape}")
        print(f"   Pixel range: [{preprocessed.min()}, {preprocessed.max()}]")
        print(f"   Mean pixel value: {preprocessed.mean():.2f}")
    
    
    # Step 5: Test DataLoader
    
    print("\n Step 5: Testing DataLoader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        image_size=tuple(pp_config['image_size']),
        preprocessing_config=pp_config
    )
    
    # Get a sample batch
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"   Sample batch shape: {sample_batch.shape}")
    print(f"   Sample labels shape: {sample_labels.shape}")
    print(f"   Batch tensor dtype: {sample_batch.dtype}")
    print(f"   Tensor value range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    
    
    # Step 6: Generate Visualizations
    
    print("\n Step 6: Generating visualizations...")
    
    try:
        # Preprocessing pipeline visualization
        visualize_preprocessing_pipeline(
            sample_path,
            save_path=str(processed_dir / "preprocessing_pipeline.png")
        )
        print("    Preprocessing pipeline visualization saved")
    except Exception as e:
        logger.warning(f"Could not generate preprocessing visualization: {e}")
    
    try:
        # Dataset distribution plots
        plot_dataset_distribution(
            df,
            save_path=str(processed_dir / "dataset_distribution.png")
        )
        print("   Dataset distribution plots saved")
    except Exception as e:
        logger.warning(f"Could not generate distribution plots: {e}")
    
    try:
        # Sample grid
        visualize_sample_grid(
            df,
            n_persons=5,
            n_images=3,
            save_path=str(processed_dir / "sample_grid.png")
        )
        print("    Sample image grid saved")
    except Exception as e:
        logger.warning(f"Could not generate sample grid: {e}")
    
    
    # Summary
    
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n  Output directory: {processed_dir}")
    print(f"  Files generated:")
    for f in sorted(processed_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n   Next Step: Feature extraction & model training")
    print(f"     Run: python train.py")
    print()


if __name__ == "__main__":
    main()
