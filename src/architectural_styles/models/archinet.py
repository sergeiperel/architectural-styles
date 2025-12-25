import torch
from torch import nn


class ArchiNet(nn.Module):
    """
    Custom CNN architecture inspired by provided Keras template.
    Works with RGB images (3 channels).
    """

    def __init__(self, num_classes: int = 25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 256 → 128 → 64 → 32 → 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 16 * 16, 125),
            nn.ReLU(),
            nn.Linear(125, 75),
            nn.ReLU(),
            nn.Linear(75, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
