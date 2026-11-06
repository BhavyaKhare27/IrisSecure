"""
ArcFace Angular Margin Loss for Iris Recognition

Implements ArcFace (Additive Angular Margin Loss) for learning discriminative
iris embeddings suited for verification via cosine similarity.

Why ArcFace over Triplet Loss?
    - Triplet loss requires careful hard mining, fragile with 2000 classes
    - ArcFace treats it as classification but enforces angular separation
    - Produces embeddings on a hypersphere — perfect for cosine similarity
    - State-of-the-art in face/iris recognition benchmarks

How it works:
    1. L2-normalize both embeddings AND classifier weights
    2. Compute cosine similarity (= dot product of normalized vectors)
    3. Add angular margin 'm' to the angle of the correct class
    4. Scale by factor 's' and apply softmax cross-entropy
    
    This forces the model to learn embeddings with a margin of separation
    between different iris classes on the unit hypersphere.

Reference:
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition", CVPR 2019. https://arxiv.org/abs/1801.07698
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) Loss.
    
    Computes: L = -log( exp(s * cos(theta_yi + m)) / 
                        (exp(s * cos(theta_yi + m)) + sum_j!=yi exp(s * cos(theta_j))) )
    
    Where:
        theta_yi = arccos(W_yi . x)  (angle between embedding and correct class center)
        m = additive angular margin (default 0.5 radians ≈ 28.6°)
        s = feature scale (default 30.0)
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 num_classes: int = 2000,
                 scale: float = 30.0,
                 margin: float = 0.5,
                 easy_margin: bool = False):
        """
        Args:
            embedding_dim: Dimension of input embeddings (must match model output)
            num_classes: Number of identity classes (2000 for CASIA per-eye)
            scale: Feature re-scaling factor 's' (higher = sharper logits)
            margin: Additive angular margin 'm' in radians
            easy_margin: If True, use easy margin variant (cos(theta+m) >= 0)
        """
        super(ArcFaceLoss, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class center weights (one vector per class on the hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)  # cos(pi - m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(pi - m) * m
        
        # Cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"ArcFaceLoss initialized: classes={num_classes}, "
                     f"embedding_dim={embedding_dim}, scale={scale}, "
                     f"margin={margin:.2f} rad ({math.degrees(margin):.1f} deg)")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Ground-truth class indices (B,)
            
        Returns:
            Scalar loss value
        """
        # Normalize weights to unit vectors (project class centers onto hypersphere)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity: cos(theta) = W_normalized . x_normalized
        # Since embeddings are already L2-normalized by the model
        cosine = F.linear(embeddings, normalized_weight)  # (B, num_classes)
        
        # Clamp for numerical stability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute sin(theta) from cos(theta): sin^2 + cos^2 = 1
        sine = torch.sqrt(1.0 - cosine.pow(2))
        
        # Compute cos(theta + m) using angle addition formula:
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        
        if self.easy_margin:
            # Easy margin: use phi only when cos(theta) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: use phi when cos(theta) > threshold
            phi = torch.where(cosine > self.threshold, phi, 
                            cosine - self.mm)
        
        # Create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)
        
        # Apply margin only to the correct class
        # For correct class: use cos(theta + m), for others: use cos(theta)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale logits
        logits = logits * self.scale
        
        # Cross-entropy loss
        loss = self.criterion(logits, labels)
        
        return loss
    
    def get_cosine_similarities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get cosine similarities without margin (for inference/evaluation).
        
        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            
        Returns:
            Cosine similarities (B, num_classes)
        """
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings, normalized_weight)


class CombinedLoss(nn.Module):
    """
    Combined loss: ArcFace + Center Loss (optional regularization).
    
    Center loss pulls embeddings toward their class center, complementing
    ArcFace's angular margin separation. This improves intra-class compactness.
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 num_classes: int = 2000,
                 arcface_scale: float = 30.0,
                 arcface_margin: float = 0.5,
                 center_loss_weight: float = 0.005):
        """
        Args:
            embedding_dim: Embedding dimension
            num_classes: Number of classes
            arcface_scale: ArcFace scale factor
            arcface_margin: ArcFace margin
            center_loss_weight: Weight for center loss (lambda)
        """
        super(CombinedLoss, self).__init__()
        
        self.arcface = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
        self.center_loss_weight = center_loss_weight
        
        # Class centers for center loss
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.centers)
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute combined loss.
        
        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Class indices (B,)
            
        Returns:
            Dict with 'total', 'arcface', and 'center' loss values
        """
        # ArcFace loss
        arcface_loss = self.arcface(embeddings, labels)
        
        # Center loss: pull embeddings toward class centers
        batch_centers = self.centers[labels]  # (B, embedding_dim)
        center_loss = F.mse_loss(embeddings, batch_centers)
        
        # Combined
        total_loss = arcface_loss + self.center_loss_weight * center_loss
        
        return {
            'total': total_loss,
            'arcface': arcface_loss.item(),
            'center': center_loss.item()
        }
