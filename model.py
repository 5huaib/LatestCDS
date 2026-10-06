import os
import torch
import torch.nn as nn
import torchvision.models.video as video_models


class CrimeDetector3D(nn.Module):
    """
    3D-CNN temporal feature extractor and binary classification head.
    Uses ResNet3D-18 as a backbone, removes the original fc layer,
    and appends: FC (1024) -> ReLU -> Dropout(0.5) -> FC (512) -> ReLU -> Dropout(0.5) -> FC (1) -> Sigmoid.
    """
    def __init__(self, use_pretrained=True, weights_path=None):
        super().__init__()
        
        # Load 3D ResNet backbone
        try:
            if use_pretrained:
                try:
                    weights = video_models.R3D_18_Weights.DEFAULT
                    self.backbone = video_models.r3d_18(weights=weights)
                except AttributeError:
                    self.backbone = video_models.r3d_18(pretrained=True)
            else:
                self.backbone = video_models.r3d_18(weights=None)
        except Exception as e:
            print(f"[Warning] Could not load pre-trained weights ({e}). Initializing backbone with random weights.")
            self.backbone = video_models.r3d_18(weights=None)

        # Replace classification layer with Identity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Binary classification head: 512 -> 1024 -> 512 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        """Loads model weights from a .pth checkpoint file."""
        print(f"Loading custom fine-tuned weights from '{weights_path}'...")
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint)
        print("Successfully loaded model weights.")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, time_frames, height, width)
               e.g. (B, 3, 16, 112, 112)
        Returns:
            Tensor of shape (batch_size, 1) representing anomaly/crime probability in [0, 1].
        """
        features = self.backbone(x)
        return self.classifier(features)


def get_model(weights_path=None, device=None):
    """
    Convenience factory to initialize CrimeDetector3D and place it on target device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                              ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # Auto-check standard locations if no path specified
    if not weights_path:
        default_candidates = [
            os.path.join("weights", "best_crime_detector.pth"),
            "best_crime_detector.pth"
        ]
        for candidate in default_candidates:
            if os.path.exists(candidate):
                weights_path = candidate
                break

    use_pretrained = (weights_path is None or not os.path.exists(weights_path))
    model = CrimeDetector3D(use_pretrained=use_pretrained, weights_path=weights_path)
    model.to(device)
    model.eval()
    return model, device
