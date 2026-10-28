"""CNN Feature Extraction module.

Extracts deep feature representations from pretrained ImageNet architectures:
- ResNet-18 (512 features)
- ResNet-50 (2048 features)
- AlexNet (4096 features)
- VGG-16 (4096 features)

Models act as fixed feature extractors (all parameters frozen).
Features are extracted from penultimate global pooling / linear layers.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class DeepFeatureExtractor:
    def __init__(self, model_name="resnet18", device=None):
        self.model_name = model_name.lower().replace("-", "")
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.model, self.feature_dim = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _build_model(self):
        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
            feature_dim = base.fc.in_features # 512
            base.fc = nn.Identity()
            return base, feature_dim
            
        elif self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            feature_dim = base.fc.in_features # 2048
            base.fc = nn.Identity()
            return base, feature_dim
            
        elif self.model_name == "alexnet":
            weights = models.AlexNet_Weights.DEFAULT
            base = models.alexnet(weights=weights)
            feature_dim = 4096
            # Extract features after classifier[5] (ReLU before final fc)
            classifier_subset = nn.Sequential(*list(base.classifier.children())[:6])
            base.classifier = classifier_subset
            return base, feature_dim
            
        elif self.model_name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT
            base = models.vgg16(weights=weights)
            feature_dim = 4096
            # Extract features after classifier[5] (ReLU before final fc)
            classifier_subset = nn.Sequential(*list(base.classifier.children())[:6])
            base.classifier = classifier_subset
            return base, feature_dim
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Choose from resnet18, resnet50, alexnet, vgg16.")
            
    def extract_features(self, tensor_batch):
        """
        Extract features from a batch of normalized images.
        
        Parameters
        ----------
        tensor_batch : torch.Tensor of shape (B, 3, 224, 224) or list of tensors
        
        Returns
        -------
        features : 2D numpy array of shape (B, feature_dim)
        """
        if isinstance(tensor_batch, list):
            tensor_batch = torch.stack(tensor_batch, dim=0)
            
        tensor_batch = tensor_batch.to(self.device)
        with torch.no_grad():
            feat = self.model(tensor_batch)
            if feat.dim() > 2:
                feat = torch.flatten(feat, 1)
        return feat.cpu().numpy()
