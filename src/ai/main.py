import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import DataLoader

from dataset import TrainingDataset
from network import CNN

# models list
models = {
    'cnn': CNN,
}

# tansformers list
transformers = {
    'image_net': transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])  # ImageNet Normalization
}

# number of epochs
epochs = 300

# batch sizes
batch_sizes = [50]

# optimizers (SGD and ADAMW have been removed, but are supported)
optimizers = {
    'adam': {
        'lr': [0.001],
        'weight_decay': [0.0001, 0.0001],
    },
}


# parametrized model training
def train_model(model_id, model, transformer, epochs, batch_size, optimizer_type, learning_rate, weight_decay, momentum):
    # load training data, shuffled
    training_data = TrainingDataset("./data/train.csv", "./data/train", transform=transformer)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # load validation data, shuffled
    validation_data = TrainingDataset("./data/validation.csv", "./data/validation", transform=transformer)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # check if GPU is available, if so run on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize model and loss function
    model = model().to(device)
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer with given parameters
    optimizer = None
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)

    # initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=15)

    # the model with the best accuracy
    best_accuracy = 0.0

    # training data results
    results = []

    # length of training and validation datasets
    training_data_size = len(training_data)
    validation_data_size = len(validation_data)

    for epoch in range(epochs):
        print(f"Training model {model_id} - epoch {epoch+1}/{epochs}")

        # training phase
        model.train()
        running_loss = 0.0
        for index, [images, labels] in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # validation phase
        model.eval()

        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=5).float())

                validation_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # calculate metrics
        training_loss = running_loss / training_data_size
        validation_loss = validation_loss / validation_data_size
        validation_accuracy = correct / total

        # progress scheduler
        scheduler.step(validation_loss)

        # add training data to results array
        results.append({
            'epoch': epoch + 1,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # save the best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), f"results/models/model_{model_id}.pth")

            # update the models file with the best accuracy
            models_file = pd.read_csv('results/models.csv')
            models_file.loc[model_id - 1, 'performance'] = best_accuracy
            models_file.to_csv('results/models.csv', index=False)

        # save training results on each pass
        data_file = pd.DataFrame(results)
        data_file.to_csv(f"results/data/model_{model_id}.csv", index=False)


if __name__ == '__main__':
    models_data = []
    processes = []

    # loop inside of another loop for all parameters
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
                                # add data to models file
                                models_data.append({
                                    'model_id': len(models_data) + 1,
                                    'model': model,
                                    'transformer': transformer,
                                    'epochs': epochs,
                                    'batch_size': batch_size,
                                    'optimizer': optimizer,
                                    'learning_rate': lr,
                                    'weight_decay': weight_decay,
                                    'momentum': momentum,
                                    'performance': 0.0
                                })
                                # create training process
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

    # save models data to file
    models_data_file = pd.DataFrame(models_data)
    models_data_file.to_csv('results/models.csv', index=False)

    # start proesses
    for process in processes:
        process.start()

    # join processes
    for process in processes:
        process.join()
