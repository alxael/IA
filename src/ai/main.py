import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import pandas as pd

from torch.cuda import is_available
from torch.utils.data import DataLoader

from dataset import TrainingDataset
from network import CNN, SimplifiedCNN, LinearNN, SimplifiedLinearNN

models = {'cnn': CNN}

transformers = {
    'simple': transforms.ToTensor(),
    'image_net': transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])  # ImageNet Normalization
}

epochs = 300

batch_sizes = [32, 64]

optimizers = {
    'adam': {
        'lr': [0.0001],  # lower for larger batches
        'weight_decay': [0.001],  # higher values in case of overfitting
    },
    'adamw': {
        'lr': [0.0005],  # smaller for training from scratch
        'weight_decay': [0.01]  # smaller for CNNs
    },
    'sgd': {
        'lr': [0.05],  # try faster learning, watch carefully for instability
        'momentum': [0.9],  # test different convergence behaviours
        'weight_decay': [0.005],  # attempt to prevent overfitting
    }
}


def train_model(model_id, model, transformer, epochs, batch_size, optimizer_type, learning_rate, weight_decay, momentum):
    training_data = TrainingDataset("./data/train.csv", "./data/train", transform=transformer)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)

    validation_data = TrainingDataset("./data/validation.csv", "./data/validation", transform=transformer)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device(f'cuda:{model_id}' if torch.cuda.is_available() else 'cpu')

    model = model().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = None
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)

    best_accuracy = 0.0

    results = []

    training_data_size = len(training_data)
    validation_data_size = len(validation_data)

    for epoch in range(epochs):
        print(f"Training model {model_id} - epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Validation phase
        model.eval()

        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                validation_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        training_loss = running_loss / training_data_size
        validation_loss = validation_loss / validation_data_size
        validation_accuracy = correct / total

        results.append({
            'epoch': epoch,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy
        })

        # Save best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), f"results/models/model_{model_id}.pth")

    data_file = pd.DataFrame(results)
    data_file.to_csv(f"results/data/model_{model_id}.csv", index=False)


if __name__ == '__main__':
    models_data = []
    processes = []

    for model in models:
        for transformer in transformers:
            for batch_size in batch_sizes:
                for optimizer in optimizers:
                    lrs = optimizers[optimizer]['lr']
                    weight_decays = optimizers[optimizer]['weight_decay']
                    momentums = optimizers[optimizer]['momentum'] if 'momentum' in optimizers[optimizer] else [None]
                    for lr in lrs:
                        for weight_decay in weight_decays:
                            for momentum in momentums:
                                models_data.append({
                                    'model_id': len(models_data) + 1,
                                    'model': model,
                                    'transformer': transformer,
                                    'epochs': epochs,
                                    'batch_size': batch_size,
                                    'optimizer': optimizer,
                                    'learning_rate': lr,
                                    'weight_decay': weight_decay,
                                    'momentum': momentum
                                })
                                process = mp.Process(target=train_model, args=(
                                    len(models_data),
                                    models[model],
                                    transformers[transformer],
                                    epochs,
                                    batch_size,
                                    optimizer,
                                    lr,
                                    weight_decay,
                                    momentum
                                ))
                                processes.append(process)

    models_data_file = pd.DataFrame(models_data)
    models_data_file.to_csv('results/models.csv', index=False)

    for process in processes:
        process.start()

    for process in processes:
        process.join()
