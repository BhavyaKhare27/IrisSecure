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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: {self.counter}/{self.patience} "
                           f"(best: {self.best_loss:.4f}, current: {val_loss:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Full training pipeline for iris embedding learning.
    
    Handles:
        - Model & loss setup
        - Optimizer & scheduler configuration  
        - Training loop with mixed precision
        - Validation with accuracy tracking
        - Checkpointing & early stopping
    """
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.start_time = datetime.now()
        
        # Paths
        self.model_dir = PROJECT_ROOT / "models" / "checkpoints"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = PROJECT_ROOT / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
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
        # Load number of classes from label mapping
        label_mapping = pd.read_csv(
            PROJECT_ROOT / self.config['dataset']['processed_dir'] / "label_mapping.csv"
        )
        self.num_classes = len(label_mapping)
        
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
        """Load preprocessed data splits and create DataLoaders."""
        processed_dir = PROJECT_ROOT / self.config['dataset']['processed_dir']
        pp_cfg = self.config['preprocessing']
        train_cfg = self.config['training']
        
        # Load split CSVs from Phase 1
        train_df = pd.read_csv(processed_dir / "train.csv")
        val_df = pd.read_csv(processed_dir / "val.csv")
        test_df = pd.read_csv(processed_dir / "test.csv")
        
        image_size = tuple(pp_cfg['image_size'])
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_df, val_df, test_df,
            batch_size=train_cfg['batch_size'],
            num_workers=train_cfg['num_workers'],
            image_size=image_size,
            preprocessing_config=pp_cfg
        )
        
        logger.info(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
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
        Run validation and compute metrics.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        self.criterion.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating", 
                                    ncols=100, leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                embeddings = self.model(images, normalize=True)
                loss = self.criterion(embeddings, labels)
            
            running_loss += loss.item() * images.size(0)
            
            cosine_sim = self.criterion.get_cosine_similarities(embeddings)
            _, predicted = cosine_sim.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'arcface_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'config': self.config,
            'num_classes': self.num_classes,
        }
        
        # Save periodic checkpoint
        path = self.model_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: val_loss={val_metrics['loss']:.4f}, "
                       f"val_acc={val_metrics['accuracy']:.2f}%")
        
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
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Check if best
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log
            record = {
                'epoch': epoch + 1,
                'train_loss': round(train_metrics['loss'], 4),
                'train_acc': round(train_metrics['accuracy'], 2),
                'val_loss': round(val_metrics['loss'], 4),
                'val_acc': round(val_metrics['accuracy'], 2),
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': round(epoch_time, 1),
                'best': is_best,
            }
            self.training_history.append(record)
            
            best_marker = " << BEST" if is_best else ""
            print(f"   Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.1f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.1f}% | "
                  f"Time: {epoch_time:.0f}s{best_marker}")
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss']):
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
        print(f"   Best Val Loss:     {self.best_val_loss:.4f}")
        print(f"   Best Val Accuracy: {self.best_val_acc:.2f}%")
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
    trainer.setup_model()
    trainer.setup_data()
    
    # Train
    history = trainer.train()
    
    print("Next step: Generate embeddings with generate_embeddings.py")


if __name__ == "__main__":
    main()
