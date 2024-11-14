from Classify.Classify import CompactVisionNet
import torch
import torch.nn as nn

# Custom model with transfer learning for 5-class classification
class CompactVisionNetWithTransfer(CompactVisionNet):
    def __init__(self, num_classes: int, in_channels: int = 3, pretrained_path: str = "BASE_MODELS/best_model.pth"):
        super().__init__(num_classes=15, in_channels=in_channels)  # Initialize with original 15 classes

        # Load pretrained weights if provided
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Replace the classifier to match the new number of classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False