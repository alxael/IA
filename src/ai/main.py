import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd

from torch.cuda import is_available
from torch.utils.data import DataLoader

from dataset import TrainingDataset
from network import CNN

print("Running on CUDA\n" if is_available() else "Running on CPU\n")

device = torch.device("cuda" if is_available() else "cpu")

batch_size = 64

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

print("Loading training data...")
training_data = TrainingDataset("./data/train.csv", "./data/train", transform=transform)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("Finished loading training data!\n")

print("Loading validation data...")
validation_data = TrainingDataset("./data/validation.csv", "./data/validation", transform=transform)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("Finished loading validation data!\n")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

def train_model(model, train_loader, validation_loader, epochs, folds):
    best_accuracy = 0.0

    results = []

    training_data_fold_size = len(training_data) // folds
    validation_data_size = len(validation_data)

    for epoch in range(epochs):
        for fold in range(folds):
            print(f"Epoch {epoch+1}/{epochs}, fold {fold + 1}/{folds}")

            # Training phase
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

                if index == training_data_fold_size:
                    break

            # Validation phase
            model.eval()

            validation_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for index, [images, labels] in enumerate(validation_loader, 1):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    validation_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate metrics
            training_loss = running_loss / training_data_fold_size
            validation_loss = validation_loss / validation_data_size
            validation_accuracy = correct / total

            print(f"Training Loss: {training_loss:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
            print("\n")

            results.append({
                'epoch': epoch,
                'fold': fold,
                'training_loss': training_loss,
                'validation_loss': validation_loss,
                'validation_accuracy': validation_accuracy
            })

            # Save best model
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                torch.save(model.state_dict(), 'results/best_model.pth')

    df = pd.DataFrame(results)
    df.to_csv('results/training_log.csv', index=False)


train_model(model, train_loader, validation_loader, 5, 4)
