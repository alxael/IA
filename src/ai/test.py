import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import DataLoader

from dataset import TestDataset
from network import CNN

models = {
    'cnn': CNN,
}

batch_size = 50

transformers = {
    'image_net': transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])  # ImageNet Normalization
}

test_data = TestDataset("./data/test.csv", "./data/test", transform=transformers['image_net'])
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models['cnn']().to(device)
model.load_state_dict(torch.load('results/models/model_2.pth'))
model.eval()

answers = []
with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        for label, image_id in zip(predicted, image_ids):
            answers.append({
                'image_id': image_id,
                'label': label.item()
            })

test_file = pd.DataFrame(answers)
test_file.to_csv('results/test.csv', index=False)