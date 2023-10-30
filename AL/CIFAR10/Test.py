import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset


def imshow(img, title):
    """
    Display an image with a given title.

    :param img: The image to display.
    :param title: The title for the image.
    """
    # Unnormalize the images for display
    img = img / 2 + 0.5
    npImg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(np.transpose(npImg, (1, 2, 0)))
    plt.show()


def plotImagesFromLoaders(trainLoader, testLoader):
    """
    Display a batch of training and test images.

    :param trainLoader: DataLoader for the training set.
    :param testLoader: DataLoader for the test set.
    """
    # Display training images
    trainImages, _ = next(iter(trainLoader))
    imshow(torchvision.utils.make_grid(trainImages[:8]), "Training Images")

    # Display test images
    testImages, _ = next(iter(testLoader))
    imshow(torchvision.utils.make_grid(testImages[:8]), "Test Images")


# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations for training and testing datasets
transformTrain = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transformTest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define classes of interest
wantedClasses = ['airplane', 'automobile']
classIndices = [0, 1]  # Indices corresponding to desired classes
n = 100  # Desired number of images per class

# Load CIFAR-10 dataset
trainDatasetFull = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformTrain)
testDatasetFull = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest)

# Filter datasets to retain only images of desired classes
trainIndices = [i for i, (_, label) in enumerate(trainDatasetFull) if label in classIndices]
testIndices = [i for i, (_, label) in enumerate(testDatasetFull) if label in classIndices]

# Select a subset of the data
trainIndices = trainIndices[:2 * n]

trainDataset = Subset(trainDatasetFull, trainIndices)
testDataset = Subset(testDatasetFull, testIndices)

# Create data loaders
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=128, shuffle=True, num_workers=0)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=100, shuffle=False, num_workers=0)

print(f"Size of the training dataset: {len(trainDataset)}")
print(f"Size of the test dataset: {len(testDataset)}")


class SimpleCNN(nn.Module):
    """
    A simple CNN model for image classification.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass through the model.

        :param x: Input tensor.
        :return: Model's output tensor.
        """
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

# Define loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    totalLoss = 0.0
    for i, (inputs, labels) in enumerate(trainLoader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {totalLoss / len(trainLoader)}")


def evaluate(model, testLoader):
    """
    Evaluate model's accuracy on a test dataset.

    :param model: The model to evaluate.
    :param testLoader: DataLoader for the test dataset.
    :return: Accuracy of the model on the test data.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


accuracy = evaluate(model, testLoader)
print(f"Accuracy on test data: {accuracy:.2f}%")
