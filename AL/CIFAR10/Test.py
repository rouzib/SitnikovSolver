import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset


def imshow(img, title):
    # Unnormalize the images for display
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))  # Make the figure bigger
    plt.title(title)
    plt.axis('off')  # Remove the axes
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_images_from_loaders(train_loader, test_loader):
    # Get a batch of training images and their labels
    train_images, _ = next(iter(train_loader))
    imshow(torchvision.utils.make_grid(train_images[:8]), "Training Images")  # Display 8 images with a title

    # Get a batch of test images and their labels
    test_images, _ = next(iter(test_loader))
    imshow(torchvision.utils.make_grid(test_images[:8]), "Test Images")  # Display 8 images with a title


# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformations for the training and testing data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define the classes you want
wanted_classes = ['airplane', 'automobile']
class_indices = [0, 1]  # Corresponding indices for 'airplane' and 'automobile'

n = 100  # Number of images you want for each class

# Load the CIFAR-10 dataset
train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Filter the datasets to retain only n images from the specified classes
train_indices = [i for i, (_, label) in enumerate(train_dataset_full) if label in class_indices]
test_indices = [i for i, (_, label) in enumerate(test_dataset_full) if label in class_indices]

# Now, select only n images from each class for training and testing datasets
train_indices = train_indices[:2 * n]

train_dataset = Subset(train_dataset_full, train_indices)
test_dataset = Subset(test_dataset_full, test_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

print(f"Size of the training dataset: {len(train_dataset)}")
print(f"Size of the test dataset: {len(test_dataset)}")


# To use the function:
# plot_images_from_loaders(train_loader, test_loader)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)
print("Loaded model")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)  # Move the data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move the data to GPU
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


accuracy = evaluate(model, test_loader)
print(f"Accuracy on test data: {accuracy:.2f}%")
