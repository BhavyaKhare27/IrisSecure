"""
Iris Embedding Network (Feature Extractor)
============================================
Builds a CNN-based feature extractor for iris recognition using transfer
learning from pre-trained ImageNet models.

Architecture:
    ResNet50 (pre-trained) --> Global Average Pool --> FC(2048 -> 512) 
    --> BatchNorm --> L2 Normalize --> 512-dim embedding

Why ResNet50?
    - Strong feature extraction capability for fine-grained texture (iris patterns)
    - Well-supported with pre-trained ImageNet weights
    - Good trade-off between performance and GPU memory on RTX 4060 (8GB)
    - Proven effective in biometric recognition literature

Why ArcFace (used in training)?
    - Classification-based metric learning with angular margin
    - Superior to triplet loss when class count is high (2000 classes)
    - Standard approach in face/iris recognition systems
    - Forces embeddings onto a hypersphere for clean cosine similarity matching

Embedding Design:
    - 512-dimensional L2-normalized vectors
    - Lie on unit hypersphere (||embedding|| = 1)
    - Cosine similarity becomes a simple dot product
    - Compact enough for efficient storage and retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IrisEmbeddingNet(nn.Module):
    """
    Iris feature extraction network based on pre-trained CNN backbone.
    
    Extracts fixed-size embeddings from iris images for verification.
    The embedding head projects CNN features to a compact, L2-normalized
    vector space where cosine similarity measures identity similarity.
    
    Supports backbones: resnet18, resnet50, mobilenet_v2
    """
    
    BACKBONE_DIMS = {
        'resnet18': 512,
        'resnet50': 2048,
        'mobilenet_v2': 1280,
    }
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 embedding_dim: int = 512,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.3):
        """
        Args:
            backbone: CNN backbone architecture name
            embedding_dim: Output embedding dimension
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: If True, freeze backbone weights (only train head)
            dropout_rate: Dropout rate in embedding head for regularization
        """
        super(IrisEmbeddingNet, self).__init__()
        
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        if backbone not in self.BACKBONE_DIMS:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Choose from: {list(self.BACKBONE_DIMS.keys())}")
        
        backbone_dim = self.BACKBONE_DIMS[backbone]
        
        # Build backbone (remove classification head)
        self.backbone = self._build_backbone(backbone, pretrained)
        
        # Freeze backbone if requested (fine-tune only embedding head)
        if freeze_backbone:
            self._freeze_backbone()
            logger.info(f"Backbone '{backbone}' frozen - only embedding head will be trained")
        
        # Embedding projection head
        # Maps backbone features -> compact embedding space
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Initialize embedding head weights
        self._init_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"IrisEmbeddingNet created: backbone={backbone}, "
                     f"embedding_dim={embedding_dim}, "
                     f"params={total_params:,} (trainable: {trainable_params:,})")
    
    def _build_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Build and return the CNN backbone without the classification head."""
        
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            # Remove the final FC layer, keep everything up to avgpool
            backbone = nn.Sequential(*list(base_model.children())[:-1])
            
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            backbone = nn.Sequential(*list(base_model.children())[:-1])
            
        elif backbone_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            # MobileNetV2: features + adaptive_avg_pool
            backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        return backbone
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_from: Optional[int] = None):
        """
        Unfreeze backbone parameters for fine-tuning.
        
        Args:
            unfreeze_from: If provided, only unfreeze layers from this index onwards.
                          If None, unfreeze all layers.
        """
        if unfreeze_from is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("All backbone layers unfrozen")
        else:
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers):
                if i >= unfreeze_from:
                    for param in layer.parameters():
                        param.requires_grad = True
            logger.info(f"Backbone layers from index {unfreeze_from} unfrozen")
    
    def _init_weights(self):
        """Initialize embedding head weights using Kaiming initialization."""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass: image -> embedding.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            normalize: If True, L2-normalize embeddings (for cosine similarity)
            
        Returns:
            Embeddings of shape (B, embedding_dim)
        """
        # Extract backbone features
        features = self.backbone(x)             # (B, backbone_dim, 1, 1)
        features = features.flatten(start_dim=1) # (B, backbone_dim)
        
        # Project to embedding space
        embeddings = self.embedding_head(features)  # (B, embedding_dim)
        
        # L2 normalize for cosine similarity
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw backbone features (before embedding head).
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Raw features (B, backbone_dim)
        """
        features = self.backbone(x)
        return features.flatten(start_dim=1)
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


def get_model(backbone: str = 'resnet50',
              embedding_dim: int = 512,
              pretrained: bool = True,
              freeze_backbone: bool = False,
              device: Optional[torch.device] = None) -> IrisEmbeddingNet:
    """
    Factory function to create and configure the iris embedding model.
    
    Args:
        backbone: CNN backbone name
        embedding_dim: Embedding vector dimension
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze backbone for transfer learning
        device: Target device (auto-detect if None)
        
    Returns:
        Configured IrisEmbeddingNet on the specified device
    """
    model = IrisEmbeddingNet(
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    return model
