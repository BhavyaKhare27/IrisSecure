"""
Image Preprocessing Pipeline for Iris Recognition
===================================================
Handles all image transformations needed before feeding iris images
into a deep learning model:

1. Grayscale handling (CASIA images are already grayscale)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancing iris texture
3. Gaussian blur for noise reduction
4. Resizing to model input dimensions (224x224)
5. Normalization for pre-trained CNN compatibility
6. PyTorch Dataset and DataLoader creation with train/val/test splits

The preprocessing is designed to enhance iris texture patterns which are 
critical for discriminative feature extraction.
"""

import os
import cv2
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ============================================
# Image Preprocessing Functions
# ============================================

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE enhances local contrast in the iris region, making texture
    patterns more distinguishable for the feature extractor.
    
    Args:
        image: Grayscale image (H, W) as uint8
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced grayscale image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def preprocess_single_image(image_path: str, 
                             image_size: Tuple[int, int] = (224, 224),
                             apply_clahe_flag: bool = True,
                             clahe_clip_limit: float = 2.0,
                             clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                             gaussian_blur_kernel: int = 3) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for a single iris image.
    
    Pipeline:
        1. Load image in grayscale
        2. Apply Gaussian blur (noise reduction)
        3. Apply CLAHE (contrast enhancement)
        4. Resize to target dimensions
    
    Args:
        image_path: Path to the iris image
        image_size: Target (H, W) for resizing
        apply_clahe_flag: Whether to apply CLAHE
        clahe_clip_limit: CLAHE clip limit parameter
        clahe_tile_grid_size: CLAHE tile grid size
        gaussian_blur_kernel: Kernel size for Gaussian blur (0 to skip)
        
    Returns:
        Preprocessed image as numpy array (H, W), or None if loading fails
    """
    # Load as grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        logger.warning(f"Failed to load image: {image_path}")
        return None
    
    # Step 1: Gaussian blur for noise reduction
    if gaussian_blur_kernel > 0:
        # Ensure kernel size is odd
        k = gaussian_blur_kernel if gaussian_blur_kernel % 2 == 1 else gaussian_blur_kernel + 1
        image = cv2.GaussianBlur(image, (k, k), 0)
    
    # Step 2: CLAHE for contrast enhancement
    if apply_clahe_flag:
        image = apply_clahe(image, clahe_clip_limit, clahe_tile_grid_size)
    
    # Step 3: Resize to target dimensions
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return image


# ============================================
# PyTorch Dataset
# ============================================

class IrisDataset(Dataset):
    """
    PyTorch Dataset for CASIA-Iris-Thousand.
    
    Loads iris images on-the-fly with preprocessing and augmentation.
    Designed for use with pre-trained CNN models (ResNet, MobileNet).
    """
    
    def __init__(self, dataframe: pd.DataFrame, 
                 transform: Optional[transforms.Compose] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 apply_clahe_flag: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                 gaussian_blur_kernel: int = 3):
        """
        Args:
            dataframe: DataFrame with 'image_path' and 'label_idx' columns
            transform: Optional torchvision transforms for augmentation
            image_size: Target (H, W) for resizing
            apply_clahe_flag: Whether to apply CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_grid_size: CLAHE tile grid size
            gaussian_blur_kernel: Gaussian blur kernel (0 to skip)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.apply_clahe_flag = apply_clahe_flag
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.gaussian_blur_kernel = gaussian_blur_kernel
        
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load, preprocess, and return a single sample.
        
        Returns:
            Tuple of (image_tensor, label_idx)
            - image_tensor: (3, H, W) tensor (3 channels for pre-trained CNN)
            - label_idx: integer class label
        """
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = row['label_idx']
        
        # Preprocess the image
        image = preprocess_single_image(
            image_path,
            image_size=self.image_size,
            apply_clahe_flag=self.apply_clahe_flag,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid_size=self.clahe_tile_grid_size,
            gaussian_blur_kernel=self.gaussian_blur_kernel
        )
        
        if image is None:
            # Return a black image as fallback
            image = np.zeros(self.image_size, dtype=np.uint8)
        
        # Convert grayscale to 3-channel (required by pre-trained models)
        # Stack the grayscale channel 3 times
        image_rgb = np.stack([image, image, image], axis=-1)  # (H, W, 3)
        
        # Convert to PIL for torchvision transforms
        image_pil = Image.fromarray(image_rgb)
        
        # Apply transforms (augmentation + tensor conversion + normalization)
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)
        
        return image_tensor, label


# ============================================
# Transform Pipelines
# ============================================

def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Training transforms with data augmentation.
    
    Augmentations are chosen to be iris-appropriate:
    - Small rotations (eye tilt)
    - Horizontal flip (left/right invariance within same eye)
    - Brightness/contrast jitter (lighting variation)
    - NO vertical flip (unnatural for eye images)
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]     # ImageNet stds
        ),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Validation/test transforms (no augmentation, only normalization).
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ============================================
# Data Split and DataLoader Creation
# ============================================

def create_data_splits(dataframe: pd.DataFrame, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       random_seed: int = 42,
                       stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train/val/test sets with stratification.
    
    Stratification ensures each class (person_eye combination) is 
    proportionally represented in all splits.
    
    Args:
        dataframe: Full dataset DataFrame
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: For reproducibility
        stratify: Whether to stratify by label
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    stratify_col = dataframe['label_idx'] if stratify else None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        dataframe, 
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    stratify_col_temp = temp_df['label_idx'] if stratify else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=stratify_col_temp
    )
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (224, 224),
                       preprocessing_config: Optional[Dict] = None
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        train_df, val_df, test_df: Split DataFrames
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        image_size: Target image dimensions
        preprocessing_config: Optional dict with CLAHE/blur settings
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Default preprocessing config
    pp = preprocessing_config or {}
    
    # Common preprocessing kwargs
    pp_kwargs = {
        'image_size': image_size,
        'apply_clahe_flag': pp.get('apply_clahe', True),
        'clahe_clip_limit': pp.get('clahe_clip_limit', 2.0),
        'clahe_tile_grid_size': tuple(pp.get('clahe_tile_grid_size', [8, 8])),
        'gaussian_blur_kernel': pp.get('gaussian_blur_kernel', 3),
    }
    
    # Create datasets
    train_dataset = IrisDataset(train_df, transform=get_train_transforms(image_size), **pp_kwargs)
    val_dataset = IrisDataset(val_df, transform=get_val_transforms(image_size), **pp_kwargs)
    test_dataset = IrisDataset(test_df, transform=get_val_transforms(image_size), **pp_kwargs)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    logger.info(f"DataLoaders created - "
                 f"Train batches: {len(train_loader)}, "
                 f"Val batches: {len(val_loader)}, "
                 f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
