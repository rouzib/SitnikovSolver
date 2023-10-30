import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset

"""# TO SLOW FOR AL
def ssim(img1, img2, window_size=11, size_average=True):
    # Parameters for SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create a Gaussian window
    window = torch.ones(window_size, window_size)
    window /= window.sum()

    window = window.view(1, 1, window_size, window_size).to(img1.device)
    window = window.repeat(img1.size(1), 1, 1, 1)  # Repeat the window for each channel

    # Compute SSIM between img1 and img2
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map.mean(1)  # Average across channels

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1)


def ssimDistance(X, Y):
    # Initialize an empty tensor to store SSIM values
    ssim_values = torch.empty((X.size(0), Y.size(0)), device=device)

    # Compute SSIM for each pair of images
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            ssim_value = ssim(x.unsqueeze(0), y.unsqueeze(0))
            ssim_values[i, j] = ssim_value

    # Convert SSIM to distance: distance = 1 - SSIM
    distances = 1 - ssim_values

    return torch.min(distances, dim=1)[0]
"""


# Extract all data from a given dataset
def getDataFromDataSet(dataSet):
    allData = []
    for i in range(len(dataSet)):
        sample = dataSet[i][0]
        allData.append(sample)
    allData = torch.stack(allData)
    return allData


# Add new samples and their ground truth to an existing dataset
def appendToDataSet(dataSet, samples, groundTruth, gtType=torch.float32):
    if gtType is not None:
        newDataSet = torch.utils.data.TensorDataset(samples, groundTruth.type(gtType))
    else:
        newDataSet = torch.utils.data.TensorDataset(samples, groundTruth)
    return torch.utils.data.ConcatDataset([dataSet, newDataSet])


# Compute the cosine distance between two sets of vectors
def dist(X, Y, batchSize=128):
    # Flatten the spatial dimensions
    XFlat = X.view(X.size(0), -1)
    YFlat = Y.view(Y.size(0), -1).to(device)

    # Initialize distance tensor
    distances = torch.zeros(XFlat.size(0), YFlat.size(0), device=device)

    # Compute cosine distance in batches
    for i in range(0, XFlat.size(0), batchSize):
        end = min(i + batchSize, XFlat.size(0))
        batchX = XFlat[i:end].to(device)

        # Compute the cosine similarity for the batch
        similarity = torch.nn.functional.cosine_similarity(batchX.unsqueeze(1), YFlat.unsqueeze(0), dim=2)

        # Convert similarity to distance and store in the distances tensor
        distances[i:end] = (1 - similarity) / 2

    return torch.min(distances, dim=1)[0]


# Compute the uncertainty of the model's predictions
def u(X, mod, subDiv=128):
    with torch.no_grad():
        res = torch.empty(0, device=device)
        for i in range(len(X) // subDiv + 1):
            end = min(len(X), (i + 1) * subDiv)
            probs = torch.softmax(mod(X[i * subDiv:end].to(device)), dim=1).max(dim=1)[0]
            pred = 2 * torch.abs(0.5 - probs)
            res = torch.cat((res, pred))
        return res


# Compute the active learning score for selecting instances
def score(X, Y, mod, scalar=1.0, uncertainty=None):
    a = len(X) / (len(X) + len(Y)) * scalar
    if uncertainty is None:
        term2 = (1 - a) * u(X, mod)
    else:
        term2 = (1 - a) * uncertainty
    term1 = a * (1 - dist(X, Y))
    return term1 + term2, term1 / term2


# Select instances to be labeled based on their active learning scores
def query(n, X, Y, mod, scalar=1.0):
    toLabel = torch.empty((0, 3, 32, 32))
    uncertainty = u(X, mod)

    for _ in range(n):
        xScore, bound = score(X, Y, mod, scalar, uncertainty)
        idx = torch.argmin(xScore)
        chosenValue = X[idx][None, :]
        toLabel = torch.cat((toLabel, chosenValue))
        Y = torch.cat((Y, chosenValue))
        X = torch.cat((X[:idx], X[idx + 1:]))
        uncertainty = torch.cat((uncertainty[:idx], uncertainty[idx + 1:]))

    return toLabel, X


# Add selected instances to the dataset
def queryToDataSet(n, X, dataSet, mod, scalar=1.0):
    samples, X = query(n, X, getDataFromDataSet(dataSet), mod, scalar)
    gt, samples, _ = oracle(samples)
    return appendToDataSet(dataSet, samples, gt, None), X


# Obtain the true labels for the selected instances
def oracle(X):
    labels = []

    for x in X:
        index = next((i for i, (img, _) in enumerate(fullDataset) if torch.equal(img, x)), None)

        if index is not None:
            # Append the label of the image x to the labels list
            labels.append(fullDataset[index][1])
        else:
            # Handle the case where the image is not found in the dataset
            print("Could not find the image in the dataset")
            labels.append(-1)  # using -1 as a placeholder label

    labels = torch.tensor(labels).long()
    return labels, X, None


# Randomly sample instances to be added to the dataset
def randomSampling(n, X, dataSet):
    # Randomly select indices
    indices = torch.randperm(X.size(0))[:n]

    # Get the sampled tensors
    samples = X[indices]
    gt, samples, _ = oracle(samples)

    mask = torch.ones(X.size(0), dtype=bool)
    mask[indices] = 0
    X = X[mask]

    return appendToDataSet(dataSet, samples, gt, None), X


# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformations for the training and testing data
"""transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])"""

# Define transformations for training data
transformTrain = transforms.Compose([
    transforms.ToTensor(),
])

# Define transformations for test data
transformTest = transforms.Compose([
    transforms.ToTensor(),
])

# Define the desired classes and their indices
classIndices = [0, 3]  # Corresponding indices for 'airplane' and 'cat'
n = 20  # Number of images for each class

# Load the CIFAR-10 dataset with the specified transformations
trainDatasetFull = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformTrain)
testDatasetFull = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest)

# Filter datasets to retain only n images from the specified classes
trainIndices = [i for i, (_, label) in enumerate(trainDatasetFull) if label in classIndices]
testIndices = [i for i, (_, label) in enumerate(testDatasetFull) if label in classIndices]

# Select only n images from each class for training and test datasets
trainIndicesN = trainIndices[:2 * n]

trainDataset = Subset(trainDatasetFull, trainIndicesN)
testDataset = Subset(testDatasetFull, testIndices)
fullDataset = Subset(trainDatasetFull, trainIndices)


# Function to remap labels from [3, 5] to [0, 1]
def remapLabels(sample):
    image, label = sample
    if label == classIndices[0]:
        return image, 0
    elif label == classIndices[1]:
        return image, 1


# Adjust labels using the remap function
trainDataset = [remapLabels(sample) for sample in trainDataset]
testDataset = [remapLabels(sample) for sample in testDataset]
fullDataset = [remapLabels(sample) for sample in fullDataset]

# Print the sizes of the datasets
print(f"Size of the training dataset: {len(trainDataset)}")
print(f"Size of the full training dataset: {len(fullDataset)}")
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


# Initialize the model and move it to the GPU if available
model = SimpleCNN().to(device)
print("Loaded model")


# Function to find nearest images to a target image
def findNearestImages(targetImage, dataset, topK=10):
    # Get representations of all images
    allImages = torch.stack([img for img, _ in dataset])

    # Compute distances between target image and all other images
    distances = dist(allImages, targetImage.unsqueeze(0))

    # Get indices of top_k images with smallest distances
    _, topIndices = distances.topk(topK, largest=False)

    # Return top_k nearest images
    nearestImages = [dataset[i][0] for i in topIndices]
    return nearestImages


# Function to visualize a set of images
def visualizeImages(images, title="Images"):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for img, ax in zip(images, axes):
        img = img / 2 + 0.5
        ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


"""targetImg = trainDataset[100][0]  # or any other index
nearestImgs = findNearestImages(targetImg, trainDataset, topK=10)

visualizeImages([targetImg] + nearestImgs, title="Target Image and its Nearest Images")"""

# ----------------------------------------------------
# ----------------------------------------------------

testLoader  = torch.utils.data.DataLoader(testDataset, batch_size=100, shuffle=False, num_workers=0)


# Define a function to evaluate the model's performance on a test set
def evaluate(model, testLoader):
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
    accuracy = correct / total
    return accuracy


def train(model, trainDataSet, epochs=5, learning_rate=0.001):
    # Define the loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a data loader from the dataset
    trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=1, shuffle=True, num_workers=0)

    model.train()
    # Training loop
    for epoch in range(epochs):
        runningLoss = 0.0
        for i, (inputs, labels) in enumerate(trainLoader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            runningLoss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {runningLoss:.6f}")


numIterations = 20  # or any desired number of iterations
desiredAccuracy = 0.90  # or any desired accuracy threshold

fullData = getDataFromDataSet(fullDataset)


def ALTraining(useAL, trainDataSet, fullData):
    model = SimpleCNN().to(device)

    accuracies = []
    dataSetSizes = []

    for iteration in range(numIterations):
        # Train the model on the current trainDataset
        train(model, trainDataSet, epochs=5)

        # Evaluate the model on a validation or test set (optional)
        accuracy = evaluate(model, testLoader)

        print(f"Iteration {iteration + 1}, Accuracy: {accuracy:.2f}, DataSetLength: {len(trainDataSet)}")
        accuracies.append(accuracy)
        dataSetSizes.append(len(trainDataSet))

        # Check if we've reached the desired accuracy
        if accuracy >= desiredAccuracy:
            print("Desired accuracy reached!")
            break

        if useAL:
            # Use the queryToDataSet function to select informative instances
            trainDataSet, fullData = queryToDataSet(5, fullData, trainDataSet, model, scalar=0.1)
        else:
            trainDataSet, fullData = randomSampling(5, fullData, trainDataSet)

    return dataSetSizes, accuracies


plt.plot(*ALTraining(False, trainDataset, fullData), label="No AL", color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(False, trainDataset, fullData), color="tab:green", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), label="AL", color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)
plt.plot(*ALTraining(True, trainDataset, fullData), color="tab:blue", alpha=0.3)

plt.xlabel("# Training data")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
