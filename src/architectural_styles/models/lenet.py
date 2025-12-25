import torch
from torch import nn


class LeNet5RGB(nn.Module):
    def __init__(self, num_classes: int = 25):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 62 * 62, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)

        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return self.fc3(x)
