import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimplifiedCNN(nn.Module):
    def __init__(self):
        super(SimplifiedCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 100 * 100, 2 ** 14),
            nn.ReLU(),
            nn.Linear(2 ** 14, 2 ** 12),
            nn.ReLU(),
            nn.Linear(2 ** 12, 2 ** 10),
            nn.ReLU(),
            nn.Linear(2 ** 10, 2 ** 8),
            nn.ReLU(),
            nn.Linear(2 ** 8, 2 ** 6),
            nn.ReLU(),
            nn.Linear(2 ** 6, 2 ** 4),
            nn.ReLU(),
            nn.Linear(2 ** 4, 5),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class SimplifiedLinearNN(nn.Module):
    def __init__(self):
        super(SimplifiedLinearNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 100 * 100, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
