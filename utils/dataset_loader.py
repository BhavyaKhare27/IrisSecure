"""
Dataset Loader for CASIA-Iris-Thousand

Traverses the dataset directory structure, assigns labels (personID_eye),
and creates a comprehensive CSV manifest for downstream processing.

Directory Structure Expected:
    CASIA-Iris-Thousand/
        000/
            L/ -> S5000L00.jpg, S5000L01.jpg, ...
            R/ -> S5000R00.jpg, S5000R01.jpg, ...
        001/
            L/ -> ...
            R/ -> ...
        ...

Output CSV Format:
    person_id, eye, label, image_path, image_filename
    000, L, 000_L, data/CASIA-Iris-Thousand/000/L/S5000L00.jpg, S5000L00.jpg
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)


class CASIAIrisDatasetLoader:
    """
    Loads and catalogs the CASIA-Iris-Thousand dataset.
    
    Each eye of each person is treated as a separate class:
        - person 000, left eye  -> label '000_L'
        - person 000, right eye -> label '000_R'
    
    This gives 2000 unique classes (1000 persons × 2 eyes).
    """
    
    VALID_EYES = {'L', 'R'}
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: Path to the CASIA-Iris-Thousand root directory
                      (e.g., 'data/CASIA-Iris-Thousand')
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        
        self.dataframe: Optional[pd.DataFrame] = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
    def load(self) -> pd.DataFrame:
        """
        Scan the dataset directory and build a DataFrame of all samples.
        
        Returns:
            pd.DataFrame with columns:
                - person_id (str): e.g., '000', '001', ..., '999'
                - eye (str): 'L' or 'R'
                - label (str): e.g., '000_L', '000_R'
                - label_idx (int): numeric index for the label
                - image_path (str): full path to the image file
                - image_filename (str): just the filename
        """
        records = []
        
        # Get sorted list of person directories
        person_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        
        logger.info(f"Found {len(person_dirs)} person directories in {self.root_dir}")
        
        for person_dir in person_dirs:
            person_id = person_dir.name  # e.g., '000'
            
            for eye in self.VALID_EYES:
                eye_dir = person_dir / eye
                
                if not eye_dir.exists():
                    logger.warning(f"Missing eye directory: {eye_dir}")
                    continue
                
                # Get all valid image files
                image_files = sorted([
                    f for f in eye_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
                ])
                
                if len(image_files) == 0:
                    logger.warning(f"No images found in: {eye_dir}")
                    continue
                
                label = f"{person_id}_{eye}"  # e.g., '000_L'
                
                for img_file in image_files:
                    records.append({
                        'person_id': person_id,
                        'eye': eye,
                        'label': label,
                        'image_path': str(img_file),
                        'image_filename': img_file.name
                    })
        
        # Create DataFrame
        self.dataframe = pd.DataFrame(records)
        
        # Build label-to-index mapping (sorted for reproducibility)
        unique_labels = sorted(self.dataframe['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Add numeric label index
        self.dataframe['label_idx'] = self.dataframe['label'].map(self.label_to_idx)
        
        logger.info(f"Dataset loaded: {len(self.dataframe)} images, "
                     f"{len(unique_labels)} unique classes")
        
        return self.dataframe
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dictionary with dataset stats
        """
        if self.dataframe is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        df = self.dataframe
        
        images_per_class = df.groupby('label').size()
        images_per_person = df.groupby('person_id').size()
        
        stats = {
            'total_images': len(df),
            'total_persons': df['person_id'].nunique(),
            'total_classes': df['label'].nunique(),
            'total_left_eye_images': len(df[df['eye'] == 'L']),
            'total_right_eye_images': len(df[df['eye'] == 'R']),
            'images_per_class': {
                'mean': round(images_per_class.mean(), 2),
                'min': int(images_per_class.min()),
                'max': int(images_per_class.max()),
                'std': round(images_per_class.std(), 2),
            },
            'images_per_person': {
                'mean': round(images_per_person.mean(), 2),
                'min': int(images_per_person.min()),
                'max': int(images_per_person.max()),
            },
            'classes_with_min_images': int((images_per_class == images_per_class.min()).sum()),
            'classes_with_max_images': int((images_per_class == images_per_class.max()).sum()),
        }
        
        return stats
    
    def save_csv(self, output_path: str) -> None:
        """Save the dataset manifest to a CSV file."""
        if self.dataframe is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(output_path, index=False)
        logger.info(f"Dataset CSV saved to: {output_path}")
    
    def save_label_mapping(self, output_path: str) -> None:
        """Save the label-to-index mapping to a CSV file."""
        if not self.label_to_idx:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mapping_df = pd.DataFrame([
            {'label': label, 'label_idx': idx}
            for label, idx in self.label_to_idx.items()
        ])
        mapping_df.to_csv(output_path, index=False)
        logger.info(f"Label mapping saved to: {output_path}")


def load_dataset(root_dir: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Convenience function to load the CASIA-Iris-Thousand dataset.
    
    Args:
        root_dir: Path to dataset root
        
    Returns:
        Tuple of (DataFrame, label_to_idx mapping)
    """
    loader = CASIAIrisDatasetLoader(root_dir)
    df = loader.load()
    return df, loader.label_to_idx
