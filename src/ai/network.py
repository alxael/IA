import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (32, 100, 100)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 50, 50)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 50, 50)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 25, 25)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (128, 12, 12)
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
