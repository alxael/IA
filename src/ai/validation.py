import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

from dataset import TrainingDataset
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

# load validation data
validation_data = TrainingDataset("./data/validation.csv", "./data/validation", transform=transformers['image_net'])
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=0)

# check if GPU is available, if so, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of models
model_count = 2
final_models = []

# model ids
indices = [1, 2, 3, 4, 5]
# list of confusion matrices
conf_mats = []
for index in indices:
    # create model and load state from file
    model = models['cnn']().to(device)
    model.load_state_dict(torch.load(f"results/models/model_{index}.pth"))
    # set model to evaluation mode
    model.eval()
    final_models.append(model)
    # initialize confusion matrix with normalization
    conf_mats.append(MulticlassConfusionMatrix(num_classes=5, normalize='true').to(device))

answers = []
with torch.no_grad():
    for images, labels in validation_loader:
        # load validation data
        images, labels = images.to(device), labels.to(device)

        # run data through model then update the confusion matrix
        for model, conf_mat in zip(final_models, conf_mats):
            outputs = model(images)
            conf_mat.update(outputs, labels)

# generate confusion matrix image for each model
for index, conf_mat in zip(indices, conf_mats):
    conf_mat.compute()
    fig, ax = conf_mat.plot(cmap='Blues', add_text=True, labels=[0, 1, 2, 3, 4])
    fig.savefig(f"images/model_{index}.png", dpi=300)