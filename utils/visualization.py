"""
Visualization Utilities for Iris Recognition System
=====================================================
Functions for visualizing iris images, preprocessing effects,
and dataset distribution analysis.
"""

import cv2
import numpy as np
import matplotlib

# These plots are saved as artifacts during preprocessing/evaluation; no GUI is needed.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def visualize_preprocessing_pipeline(image_path: str, 
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (20, 5)) -> None:
    """
    Show the effect of each preprocessing step on a sample iris image.
    
    Displays: Original -> Gaussian Blur -> CLAHE -> Resized
    
    Args:
        image_path: Path to an iris image
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    from utils.preprocessing import apply_clahe
    
    # Load original
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        logger.error(f"Cannot load image: {image_path}")
        return
    
    # Step 1: Gaussian blur
    blurred = cv2.GaussianBlur(original, (3, 3), 0)
    
    # Step 2: CLAHE
    clahe_img = apply_clahe(blurred, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Step 3: Resize
    resized = cv2.resize(clahe_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    titles = ['1. Original', '2. Gaussian Blur', '3. CLAHE Enhanced', '4. Resized (224×224)']
    images = [original, blurred, clahe_img, resized]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Iris Image Preprocessing Pipeline', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Pipeline visualization saved to: {save_path}")
    
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def visualize_sample_grid(dataframe: pd.DataFrame,
                           n_persons: int = 5,
                           n_images: int = 3,
                           save_path: Optional[str] = None) -> None:
    """
    Display a grid of iris images from multiple persons.
    Shows both left and right eye samples for each person.
    
    Args:
        dataframe: Dataset DataFrame
        n_persons: Number of persons to display
        n_images: Number of images per eye per person
        save_path: Optional path to save the figure
    """
    persons = sorted(dataframe['person_id'].unique())[:n_persons]
    
    fig, axes = plt.subplots(n_persons, n_images * 2, 
                              figsize=(n_images * 4, n_persons * 2.5))
    
    for i, person_id in enumerate(persons):
        person_data = dataframe[dataframe['person_id'] == person_id]
        
        for j, eye in enumerate(['L', 'R']):
            eye_data = person_data[person_data['eye'] == eye]
            sample_imgs = eye_data.head(n_images)
            
            for k, (_, row) in enumerate(sample_imgs.iterrows()):
                col_idx = j * n_images + k
                ax = axes[i, col_idx] if n_persons > 1 else axes[col_idx]
                
                img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap='gray')
                    
                if i == 0:
                    ax.set_title(f"{'Left' if eye == 'L' else 'Right'} Eye #{k+1}", fontsize=9)
                
                ax.axis('off')
        
        # Add person label on the left
        ax_left = axes[i, 0] if n_persons > 1 else axes[0]
        ax_left.set_ylabel(f"Person {person_id}", fontsize=10, fontweight='bold', rotation=90)
    
    plt.suptitle('Sample Iris Images from CASIA-Iris-Thousand', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sample grid saved to: {save_path}")
    
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def plot_dataset_distribution(dataframe: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of images across classes and eyes.
    
    Creates:
    1. Histogram of images per class
    2. Left vs Right eye count comparison
    3. Images per person distribution
    
    Args:
        dataframe: Dataset DataFrame
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Images per class histogram
    images_per_class = dataframe.groupby('label').size()
    axes[0].hist(images_per_class.values, bins=range(
        images_per_class.min(), images_per_class.max() + 2), 
        color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Images per Class', fontsize=11)
    axes[0].set_ylabel('Number of Classes', fontsize=11)
    axes[0].set_title('Distribution of Images per Class', fontsize=12, fontweight='bold')
    axes[0].axvline(images_per_class.mean(), color='red', linestyle='--', 
                     label=f'Mean: {images_per_class.mean():.1f}')
    axes[0].legend()
    
    # 2. Left vs Right eye counts
    eye_counts = dataframe['eye'].value_counts()
    colors = ['#4CAF50', '#FF9800']
    bars = axes[1].bar(eye_counts.index, eye_counts.values, color=colors, 
                        edgecolor='white', width=0.5)
    axes[1].set_xlabel('Eye', fontsize=11)
    axes[1].set_ylabel('Number of Images', fontsize=11)
    axes[1].set_title('Left vs Right Eye Distribution', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, eye_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                      str(count), ha='center', fontweight='bold')
    
    # 3. Images per person distribution
    images_per_person = dataframe.groupby('person_id').size()
    axes[2].hist(images_per_person.values, bins=20, 
                  color='#9C27B0', edgecolor='white', alpha=0.8)
    axes[2].set_xlabel('Images per Person', fontsize=11)
    axes[2].set_ylabel('Number of Persons', fontsize=11)
    axes[2].set_title('Distribution of Images per Person', fontsize=12, fontweight='bold')
    axes[2].axvline(images_per_person.mean(), color='red', linestyle='--',
                     label=f'Mean: {images_per_person.mean():.1f}')
    axes[2].legend()
    
    plt.suptitle('CASIA-Iris-Thousand Dataset Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Distribution plot saved to: {save_path}")
    
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def print_stats(stats: Dict) -> None:
    """Pretty-print dataset statistics."""
    print("\n" + "=" * 55)
    print("  CASIA-Iris-Thousand Dataset Statistics")
    print("=" * 55)
    print(f"  Total Images          : {stats['total_images']:,}")
    print(f"  Total Persons         : {stats['total_persons']:,}")
    print(f"  Total Classes (eyes)  : {stats['total_classes']:,}")
    print(f"  Left Eye Images       : {stats['total_left_eye_images']:,}")
    print(f"  Right Eye Images      : {stats['total_right_eye_images']:,}")
    print("-" * 55)
    print("  Images per Class:")
    ipc = stats['images_per_class']
    print(f"    Mean: {ipc['mean']}  |  Min: {ipc['min']}  |  Max: {ipc['max']}  |  Std: {ipc['std']}")
    print("  Images per Person:")
    ipp = stats['images_per_person']
    print(f"    Mean: {ipp['mean']}  |  Min: {ipp['min']}  |  Max: {ipp['max']}")
    print("=" * 55 + "\n")
