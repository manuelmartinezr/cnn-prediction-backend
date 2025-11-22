import torch
import torch.nn as nn
from torchvision import models


class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(BreastCancerCNN, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")  # replaces pretrained=True

        # Freeze earlier layers (transfer learning)
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace the FC head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
def load_model(model_path="weights/cnn.pth", device="cpu"):
    model = BreastCancerCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model