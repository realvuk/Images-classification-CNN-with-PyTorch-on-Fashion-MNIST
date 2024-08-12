#Fashion MNIST Image Classification Project with PyTorch

## Overview

This project demonstrates how to build a Convolutional Neural Network (CNN) using PyTorch to classify images from the Fashion MNIST dataset. The dataset consists of grayscale images of various fashion items and is used to evaluate the performance of the model in categorizing these images into different classes.

The project incorporates several advanced techniques to improve model performance and prevent overfitting, including data augmentation, early stopping, and model checkpointing. Additionally, TensorBoard is utilized for visualizing the training process and evaluating the model.
Project Structure

Load Libraries
Check GPU Availability
Load and Prepare the Data
Define and Train the Model
Evaluate the Model
Close TensorBoard Writer

## 1. Load Libraries

First, import the necessary libraries and modules required for the project.

```
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
import copy
import matplotlib.pyplot as plt
import torchvision.utils
```

## 2. Check GPU Availability

Check if a GPU is available for training and set the device accordingly.

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

## 3. Load and Prepare the Data

Define data transformations and load the Fashion MNIST dataset.

Data Transformations

Apply transformations to the training and test datasets to improve model robustness and performance.

```
# Define transformations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
```

Load Datasets

Load the training and test datasets using torchvision.datasets.FashionMNIST.

```
# Load datasets
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transforms)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transforms)

# Get the number of classes
num_classes = len(train_data.classes)
print(num_classes)
```

## 4. Define and Train the Model

Define the CNN architecture, set up data loaders, and train the model with early stopping and checkpointing.

```
# Define the CNN
class MultiClassImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
```

Data Loaders

Create data loaders for training and testing.

```
# DataLoaders
batch_size = 10
dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)
```

Initialize TensorBoard Writer

Set up TensorBoard for logging and visualization.

```
# Initialize TensorBoard writer
writer = SummaryWriter()
```

Display Sample Images

Visualize some sample images from the dataset.

```
# Display some sample images
def display_sample_images(dataset, num_images=5):
    figure = plt.figure(figsize=(10, 10))
    for index in range(1, num_images + 1):
        img, label = dataset[index]
        plt.subplot(1, num_images, index)
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Label: {label}')
    plt.show()

# Display sample images from training data
display_sample_images(train_data)
```

Log Sample Images to TensorBoard

Log sample images to TensorBoard for visualization.

```
# Log sample images to TensorBoard
def log_sample_images(writer, dataloader, num_images=5):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    img_grid = torchvision.utils.make_grid(images[:num_images])
    writer.add_image('Sample Images', img_grid)

# Log sample images from training data
log_sample_images(writer, dataloader_train)
```

Train the Model

Define the training function with early stopping and model checkpointing.

```
# Define training function with early stopping and model saving
def train_model_with_early_stopping_and_save(optimizer, scheduler, net, dataloader_train, dataloader_val, num_epochs, patience, save_path):
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(net.state_dict())
    current_patience = patience
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0
        num_processed = 0
        
        for features, labels in dataloader_train:
            features, labels = features.to(device), labels.to(device)  # Move to GPU
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            num_processed += features.size(0)
    
        scheduler.step()
    
        epoch_loss = running_loss / num_processed
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
        # Log the training loss to TensorBoard
        writer.add_scalar('training loss', epoch_loss, epoch)
        
        # Early stopping check using validation set
        net.eval()
        val_running_loss = 0
        val_num_processed = 0
        
        with torch.no_grad():
            for val_features, val_labels in dataloader_val:
                val_features, val_labels = val_features.to(device), val_labels.to(device)  # Move to GPU
                val_outputs = net(val_features)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item() * val_features.size(0)
                val_num_processed += val_features.size(0)
        
        val_epoch_loss = val_running_loss / val_num_processed
        print(f'Validation Loss: {val_epoch_loss:.4f}')
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(net.state_dict())
            current_patience = patience
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss
            }, save_path)
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping triggered. No improvement in validation loss.')
                break  # Ensure this break is aligned with the if statement
    
    # Load the best model weights
    net.load_state_dict(best_model_weights)
    
    return net

# Train the model
net = MultiClassImageClassifier(num_classes).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Define early stopping parameters and save path
patience = 5  # Number of epochs with no improvement after which training will be stopped
save_path = 'fashion_mnist_model.pth'

# Train the model with early stopping and save the best model
net = train_model_with_early_stopping_and_save(optimizer, scheduler, net, dataloader_train, dataloader_test, num_epochs=20, patience=patience, save_path=save_path)
```
5. Evaluate the Model

Evaluate the model's performance on the test dataset using various metrics.
