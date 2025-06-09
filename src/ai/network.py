import torch.nn as nn

# the final convolutional neural network structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 3 input channels to 64 filters
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), # nonlinear
            nn.MaxPool2d(2, 2), # downsizing 100x100 -> 50x50

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 to 128 filters
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # nonlinear
            nn.MaxPool2d(2, 2), # downsizing 50x50 -> 25x25

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 128 to 256 filters
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # nonlinear
            nn.AdaptiveAvgPool2d((4, 4)), # output always 4x4 spatial size
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # 50% dropout rate
            nn.Linear(256 * 4 * 4, 4096),
            # flattened layer into first linear layer of equal size 
            nn.ReLU(),
            nn.Dropout(0.3), # 30% dropout rate
            nn.Linear(4096, 512), # not too steep of a drop
            nn.ReLU(),
            nn.Dropout(0.1), # 10% dropout rate
            nn.Linear(512, 32),  # not too steep of a drop
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten output for linear layer
        x = self.classifier(x)
        return x