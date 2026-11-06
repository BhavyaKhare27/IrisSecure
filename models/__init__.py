"""
Iris Recognition System - Models Package

Contains:
    - feature_extractor.py: ResNet50-based iris embedding network
    - arcface_loss.py: ArcFace angular margin loss for metric learning
"""

from models.feature_extractor import IrisEmbeddingNet, get_model
from models.arcface_loss import ArcFaceLoss
