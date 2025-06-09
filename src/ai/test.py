import torch
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import DataLoader

from dataset import TestDataset
from network import CNN

# same as main
models = {
    'cnn': CNN,
}

# fixed batch size
batch_size = 50

# same as main
transformers = {
    'image_net': transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])  # ImageNet Normalization
}

# load test dataset
test_data = TestDataset("./data/test.csv", "./data/test", transform=transformers['image_net'])
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

# check if GPU is available, if so, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of models
model_count = 2
final_models = []

# model ids
indices = [1, 2]
for index in indices:
    # create model and load state from file
    model = models['cnn']().to(device)
    model.load_state_dict(torch.load(f"results-1/models/model_{index}.pth"))
    # set model to evaluation mode
    model.eval()
    final_models.append(model)

# evaluation results
answers = []
with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device)

        # take sum of all outputs of the models
        outputs = None
        for model in final_models:
            if outputs is None:
                outputs = model(images)
            else:
                outputs = outputs + model(images)
        # average the results
        outputs = outputs / model_count
        
        # get the predicted values
        _, predicted = torch.max(outputs.data, 1)
        # append entries to results array
        for label, image_id in zip(predicted, image_ids):
            answers.append({
                'image_id': image_id,
                'label': label.item()
            })

# save results
test_file = pd.DataFrame(answers)
test_file.to_csv('results/test.csv', index=False)