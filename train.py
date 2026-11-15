"""
Training Script for Iris Recognition System
==============================================
Trains an IrisEmbeddingNet with ArcFace loss on the CASIA-Iris-Thousand dataset.

Key Features:
    - Mixed precision training (FP16) for RTX 4060 efficiency
    - Cosine annealing LR scheduler with warm restarts
    - Early stopping on validation loss
    - Gradient clipping for training stability
    - Automatic model checkpointing (best + periodic)
    - TensorBoard-compatible logging
    - 2-phase training: frozen backbone -> full fine-tuning

GPU Memory Budget (RTX 4060 - 8GB):
    - ResNet50 + ArcFace (2000 classes): ~3.5GB
    - Batch 32 @ 224x224 FP16: ~2GB activations
    - Total: ~5.5GB with comfortable headroom

Run:
    python train.py
    python train.py --epochs 30 --batch_size 64 --backbone resnet18
"""

import os
import sys
import time
import random
import argparse
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.feature_extractor import IrisEmbeddingNet, get_model
from models.arcface_loss import ArcFaceLoss
from utils.preprocessing import (
    IrisDataset,
    get_train_transforms,
    get_val_transforms,
    create_dataloaders
)
from utils.similarity import generate_verification_scores, select_eer_threshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt when a minimized validation metric stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: {self.counter}/{self.patience} "
                           f"(best: {self.best_score:.4f}, current: {score:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Full training pipeline for iris embedding learning.
    
    Handles:
        - Model & loss setup
        - Optimizer & scheduler configuration  
        - Training loop with mixed precision
        - Validation with unseen-identity verification metrics
        - Checkpointing & early stopping
    """
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.start_time = datetime.now()

        # Seed all RNGs used by the training/data pipeline for a repeatable run.
        self.seed = int(self.config['training'].get('seed', 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        run_name = str(self.config['training'].get('run_name', 'subject_disjoint_run'))
        if not run_name or Path(run_name).name != run_name or run_name in {'.', '..'}:
            raise ValueError("training.run_name must be a single, non-empty directory name")

        # Keep future experiment checkpoints separate from the deployed checkpoint.
        output_root = PROJECT_ROOT / self.config['training'].get('output_root', 'models/runs')
        self.run_dir = output_root / run_name
        self.model_dir = self.run_dir / "checkpoints"
        self.log_dir = PROJECT_ROOT / self.config.get('logging', {}).get('log_dir', 'logs') / run_name

        # Never silently overwrite artifacts from a prior run with this name.
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            raise FileExistsError(
                f"Training output already exists at {self.model_dir}. "
                "Choose a new training.run_name to preserve prior run artifacts."
            )
        history_path = self.log_dir / "training_history.json"
        if history_path.exists():
            raise FileExistsError(
                f"Training history already exists at {history_path}. "
                "Choose a new training.run_name to preserve prior run artifacts."
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_eer = float('inf')
        self.training_history = []
        
    def setup_model(self):
        """Initialize model, loss, optimizer, and scheduler."""
        model_cfg = self.config['model']
        train_cfg = self.config['training']
        
        # --- Model ---
        self.model = get_model(
            backbone=model_cfg['backbone'],
            embedding_dim=model_cfg['embedding_dim'],
            pretrained=model_cfg['pretrained'],
            freeze_backbone=model_cfg.get('freeze_backbone', False),
            device=self.device
        )
        
        # --- ArcFace Loss ---
        # The training-only label map is created from the train manifest.
        if not getattr(self, 'num_classes', 0):
            raise RuntimeError("Load the subject-disjoint training split before building ArcFace loss")
        
        self.criterion = ArcFaceLoss(
            embedding_dim=model_cfg['embedding_dim'],
            num_classes=self.num_classes,
            scale=30.0,
            margin=0.5
        ).to(self.device)
        
        # --- Optimizer ---
        # Separate param groups: backbone (lower LR) + head + arcface (normal LR)
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.embedding_head.parameters())
        arcface_params = list(self.criterion.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': train_cfg['learning_rate'] * 0.1,
             'name': 'backbone'},
            {'params': head_params, 'lr': train_cfg['learning_rate'],
             'name': 'embedding_head'},
            {'params': arcface_params, 'lr': train_cfg['learning_rate'],
             'name': 'arcface'},
        ]
        
        if train_cfg['optimizer'] == 'adam':
            self.optimizer = optim.Adam(param_groups, weight_decay=1e-4)
        elif train_cfg['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        else:
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)
        
        # --- Scheduler ---
        if train_cfg['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif train_cfg['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        
        # --- Mixed Precision ---
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        
        # --- Early Stopping ---
        self.early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        
        logger.info(f"Training setup complete: {self.num_classes} classes, "
                    f"device={self.device}")
    
    def setup_data(self):
        """Load person-disjoint manifests and map ArcFace labels from train identities only."""
        split_dir = PROJECT_ROOT / self.config['dataset'].get(
            'subject_split_dir', 'data/splits/subject_disjoint'
        )
        pp_cfg = self.config['preprocessing']
        train_cfg = self.config['training']

        required_columns = {'person_id', 'eye', 'label', 'image_path', 'label_idx'}
        train_df = pd.read_csv(split_dir / 'train.csv', dtype={'person_id': str})
        val_df = pd.read_csv(split_dir / 'val.csv', dtype={'person_id': str})
        test_df = pd.read_csv(split_dir / 'test.csv', dtype={'person_id': str})
        for split_name, frame in (('train', train_df), ('val', val_df), ('test', test_df)):
            missing = required_columns - set(frame.columns)
            if missing:
                raise ValueError(f"{split_name} manifest is missing columns: {sorted(missing)}")

        # ArcFace class IDs are contiguous and derived only from training identities.
        train_index = pd.to_numeric(train_df['training_class_idx'], errors='raise').astype(int)
        train_classes = train_df[['label', 'training_class_idx']].drop_duplicates()
        train_classes['training_class_idx'] = pd.to_numeric(
            train_classes['training_class_idx'], errors='raise'
        ).astype(int)
        if train_classes['label'].duplicated().any() or train_classes['training_class_idx'].duplicated().any():
            raise ValueError("Training label mapping is not one-to-one")
        expected_indices = set(range(train_classes['training_class_idx'].nunique()))
        if set(train_classes['training_class_idx']) != expected_indices:
            raise ValueError("Training class indices must be contiguous, starting at zero")
        train_df['label_idx'] = train_index
        self.training_class_mapping = dict(zip(
            train_classes['label'], train_classes['training_class_idx']
        ))
        self.num_classes = len(self.training_class_mapping)

        # Validation and test retain their global identity-eye labels for verification.
        for split_name, frame in (('val', val_df), ('test', test_df)):
            frame['label_idx'] = pd.to_numeric(frame['label_idx'], errors='raise').astype(int)

        subject_sets = {
            name: set(frame['label'].astype(str).str.rsplit('_', n=1).str[0])
            for name, frame in (('train', train_df), ('val', val_df), ('test', test_df))
        }
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = subject_sets[left] & subject_sets[right]
            if overlap:
                raise ValueError(f"Subject leakage between {left} and {right}: {len(overlap)} people")

        # Resolve portable manifest paths against the project root before DataLoader use.
        for frame in (train_df, val_df, test_df):
            frame['image_path'] = frame['image_path'].map(
                lambda value: str((PROJECT_ROOT / value).resolve())
                if not Path(value).is_absolute() else value
            )

        self.split_metadata = json.loads((split_dir / 'split_metadata.json').read_text(encoding='utf-8'))
        
        image_size = tuple(pp_cfg['image_size'])
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_df, val_df, test_df,
            batch_size=train_cfg['batch_size'],
            num_workers=train_cfg['num_workers'],
            image_size=image_size,
            preprocessing_config=pp_cfg
        )
        
        logger.info(
            "Subject-disjoint data loaded: "
            f"train={len(train_df)} images/{self.num_classes} training classes, "
            f"val={len(val_df)} images/{val_df['label'].nunique()} unseen classes, "
            f"test={len(test_df)} images/{test_df['label'].nunique()} unseen classes"
        )
    
    def train_one_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch with mixed precision.
        
        Returns:
            Dict with epoch metrics (loss, accuracy, etc.)
        """
        self.model.train()
        self.criterion.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}",
                    ncols=120, leave=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                embeddings = self.model(images, normalize=True)
                loss = self.criterion(embeddings, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), max_norm=5.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            
            # Compute accuracy using ArcFace cosine similarities
            with torch.no_grad():
                cosine_sim = self.criterion.get_cosine_similarities(embeddings)
                _, predicted = cosine_sim.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            # Update progress bar
            current_loss = running_loss / total
            current_acc = 100. * correct / total
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.1f}%',
                'lr': f'{current_lr:.2e}'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """
        Measure genuine/impostor verification performance on unseen identities.
        
        Returns:
            Dict with EER, selected threshold, FAR/FRR, and pair counts.
        """
        self.model.eval()
        embeddings_all = []
        labels_all = []

        for images, labels in tqdm(self.val_loader, desc="Validating unseen identities",
                                   ncols=100, leave=False):
            images = images.to(self.device, non_blocking=True)
            with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                embeddings = self.model(images, normalize=True)
            embeddings_all.append(embeddings.float().cpu().numpy())
            labels_all.append(labels.numpy())

        embeddings_all = np.concatenate(embeddings_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        manifest_labels = self.val_loader.dataset.dataframe['label_idx'].to_numpy(dtype=np.int64)
        if not np.array_equal(labels_all.astype(np.int64), manifest_labels):
            raise ValueError("Validation loader order no longer matches its identity manifest")

        verification_cfg = self.config.get('verification', {})
        genuine_scores, impostor_scores = generate_verification_scores(
            embeddings_all,
            labels_all,
            max_genuine_per_class=verification_cfg.get(
                'validation_max_genuine_pairs_per_class', 10
            ),
            max_impostor_pairs=verification_cfg.get('validation_max_impostor_pairs', 50000),
            seed=self.seed,
        )
        metrics = select_eer_threshold(genuine_scores, impostor_scores)
        logger.info(
            "Unseen-identity validation: EER=%.4f, threshold=%.4f, FAR=%.4f, FRR=%.4f "
            "(%s genuine / %s impostor pairs)",
            metrics['eer'], metrics['threshold'], metrics['far'], metrics['frr'],
            metrics['num_genuine_pairs'], metrics['num_impostor_pairs']
        )
        return metrics
    
    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'arcface_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'validation_metrics': val_metrics,
            'config': self.config,
            'num_classes': self.num_classes,
            'training_seed': self.seed,
            'split_metadata': self.split_metadata,
        }
        
        # Save periodic checkpoint
        path = self.model_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: validation EER={val_metrics['eer']:.4f}, "
                        f"threshold={val_metrics['threshold']:.4f}")
        
        # Keep only last 3 periodic checkpoints to save disk space
        checkpoints = sorted(self.model_dir.glob("checkpoint_epoch_*.pth"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
    
    def train(self):
        """
        Full training loop with 2-phase strategy:
            Phase A (first 5 epochs): Frozen backbone, train only head + ArcFace
            Phase B (remaining epochs): Unfreeze backbone, fine-tune everything
        """
        epochs = self.config['training']['epochs']
        freeze_epochs = min(5, epochs // 4)  # Warm-up with frozen backbone
        
        print("\n" + "=" * 70)
        print("   Iris Recognition System - Training")
        print("=" * 70)
        print(f"   Device:      {self.device}")
        print(f"   Backbone:    {self.config['model']['backbone']}")
        print(f"   Embedding:   {self.config['model']['embedding_dim']}-dim")
        print(f"   Classes:     {self.num_classes}")
        print(f"   Epochs:      {epochs} (freeze backbone: first {freeze_epochs})")
        print(f"   Batch Size:  {self.config['training']['batch_size']}")
        print(f"   Optimizer:   {self.config['training']['optimizer']}")
        print(f"   Validation:  unseen-identity verification EER")
        print(f"   ArcFace:     scale=30.0, margin=0.5 rad")
        print(f"   Mixed Prec:  FP16 enabled")
        print("=" * 70 + "\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Phase transition: unfreeze backbone after warm-up
            if epoch == freeze_epochs and self.config['model'].get('freeze_backbone', False):
                logger.info(f"Epoch {epoch+1}: Unfreezing backbone for fine-tuning")
                self.model.unfreeze_backbone()
                print(f"\n   >> Backbone unfrozen at epoch {epoch+1} for full fine-tuning\n")
            
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_one_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Scheduler step
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['eer'])
            else:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Check if best
            is_best = val_metrics['eer'] < self.best_val_eer
            if is_best:
                self.best_val_eer = val_metrics['eer']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log
            record = {
                'epoch': epoch + 1,
                'train_loss': round(train_metrics['loss'], 4),
                'train_acc': round(train_metrics['accuracy'], 2),
                'val_eer': round(val_metrics['eer'], 6),
                'val_threshold': round(val_metrics['threshold'], 6),
                'val_far': round(val_metrics['far'], 6),
                'val_frr': round(val_metrics['frr'], 6),
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': round(epoch_time, 1),
                'best': is_best,
            }
            self.training_history.append(record)
            
            best_marker = " << BEST" if is_best else ""
            print(f"   Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.1f}% | "
                  f"Val EER: {val_metrics['eer']:.4f} | "
                  f"FAR: {val_metrics['far']:.4f} | "
                  f"FRR: {val_metrics['frr']:.4f} | "
                  f"Time: {epoch_time:.0f}s{best_marker}")
            
            # Early stopping check
            if self.early_stopping(val_metrics['eer']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                print(f"\n   >> Early stopping at epoch {epoch+1}")
                break
        
        # Save training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("   Training Complete!")
        print("=" * 70)
        print(f"   Best Validation EER: {self.best_val_eer:.4f}")
        print(f"   Total Time:        {total_time/60:.1f} minutes")
        print(f"   Best Model:        {self.model_dir / 'best_model.pth'}")
        print(f"   Training Log:      {history_path}")
        print("=" * 70 + "\n")
        
        return self.training_history


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    config_path = PROJECT_ROOT / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments (override config.yaml settings)."""
    parser = argparse.ArgumentParser(description="Train Iris Recognition Model")
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--backbone', type=str, default=None, 
                        choices=['resnet18', 'resnet50', 'mobilenet_v2'], help='Backbone')
    parser.add_argument('--embedding_dim', type=int, default=None, help='Embedding dim')
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone initially')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


def main():
    # Load config
    config = load_config()
    args = parse_args()
    
    # Override config with CLI args
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.backbone: config['model']['backbone'] = args.backbone
    if args.embedding_dim: config['model']['embedding_dim'] = args.embedding_dim
    if args.freeze: config['model']['freeze_backbone'] = True
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Apple GPU detected through PyTorch MPS")
    else:
        device = torch.device('cpu')
        logger.warning("No CUDA or Apple MPS GPU detected; training will use CPU.")
    
    # Initialize trainer
    trainer = Trainer(config, device)
    trainer.setup_data()
    trainer.setup_model()
    
    # Train
    history = trainer.train()
    
    print("Next step: Generate embeddings with generate_embeddings.py")


if __name__ == "__main__":
    main()
