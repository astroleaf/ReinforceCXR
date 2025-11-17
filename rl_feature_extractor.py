# encoding: utf-8

"""
Feature Extractor for RL State Representation
Extracts features from CheXNet DenseNet121 before the classifier layer.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CheXNetFeatureExtractor(nn.Module):
    """
    Extracts feature vectors from CheXNet's DenseNet121 backbone.
    These features serve as state representation for the RL agent.
    """
    
    def __init__(self, checkpoint_path='model.pth.tar', feature_dim=1024):
        """
        Args:
            checkpoint_path: Path to pretrained CheXNet model
            feature_dim: Dimension of feature vector (DenseNet121 outputs 1024)
        """
        super(CheXNetFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        
        # Load DenseNet121 backbone
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        
        # Remove the classifier to extract features
        self.features = self.densenet121.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Load pretrained weights if available
        self.checkpoint_path = checkpoint_path
        self._load_weights()
        
        # Set to eval mode
        self.eval()
    
    def _load_weights(self):
        """Load pretrained CheXNet weights into feature extractor."""
        import os
        if os.path.isfile(self.checkpoint_path):
            print(f"=> Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Extract only feature layers from checkpoint
            state_dict = checkpoint['state_dict']
            feature_dict = {}
            for k, v in state_dict.items():
                # Remove 'module.' prefix and only take feature layers
                new_key = k.replace('module.densenet121.', '')
                if new_key.startswith('features'):
                    feature_dict[new_key] = v
            
            self.densenet121.load_state_dict(feature_dict, strict=False)
            print("=> Loaded pretrained features")
        else:
            print(f"=> No checkpoint found at {self.checkpoint_path}, using random initialization")
    
    def forward(self, x):
        """
        Extract feature vector from input image.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        return out
    
    def extract_from_image(self, image_path):
        """
        Extract features from a single image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Feature vector as numpy array
        """
        # Define image preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = self.forward(image_tensor)
        
        return features.squeeze(0).cpu().numpy()
    
    def extract_batch_features(self, image_tensor):
        """
        Extract features from a batch of preprocessed images.
        
        Args:
            image_tensor: Preprocessed image tensor (B, C, H, W)
        
        Returns:
            Feature tensor (B, feature_dim)
        """
        with torch.no_grad():
            features = self.forward(image_tensor)
        return features


class FeatureCache:
    """
    Cache for storing extracted features to avoid redundant computation.
    Useful when training RL agents with fixed feature representations.
    """
    
    def __init__(self, cache_dir='./feature_cache'):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_features(self, image_name, features, labels=None):
        """Save features and optional labels to cache."""
        import os
        cache_path = os.path.join(self.cache_dir, f"{image_name}.npz")
        if labels is not None:
            np.savez(cache_path, features=features, labels=labels)
        else:
            np.savez(cache_path, features=features)
    
    def load_features(self, image_name):
        """Load features from cache."""
        import os
        cache_path = os.path.join(self.cache_dir, f"{image_name}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data['features'], data.get('labels', None)
        return None, None
    
    def exists(self, image_name):
        """Check if features exist in cache."""
        import os
        cache_path = os.path.join(self.cache_dir, f"{image_name}.npz")
        return os.path.exists(cache_path)


if __name__ == '__main__':
    """Test feature extraction."""
    
    # Initialize feature extractor
    extractor = CheXNetFeatureExtractor()
    print(f"Feature extractor initialized with feature dimension: {extractor.feature_dim}")
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    features = extractor(dummy_input)
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature statistics - Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
