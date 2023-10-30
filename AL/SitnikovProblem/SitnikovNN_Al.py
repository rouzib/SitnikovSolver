import sys
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pathlib import Path
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Utils.SelectSims import inInterestZone
import Utils.SitnikovModels as SitnikovModels

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

useGPU = torch.cuda.is_available()
# useGPU = False
device = torch.device("cuda:0") if useGPU else torch.device("cpu")


def getDataFromDataSet(dataSet):
    allData = []
    for i in range(len(dataSet)):
        sample = dataSet[i][0]
        allData.append(sample)
    allData = torch.stack(allData)
    return allData


def appendToDataSet(dataSet, samples, groundTruth):
    newDataSet = torch.utils.data.TensorDataset(samples, groundTruth.type(torch.float32))
    return torch.utils.data.ConcatDataset([dataSet, newDataSet])


def dist(X, Y):
    return torch.min(torch.cdist(X.to(device), Y.to(device)), dim=1)[0]


def u(X, mod, subDiv=64):
    with torch.no_grad():
        res = torch.empty(0, device=device)
        for i in range(len(X) // subDiv + 1):
            end = min(len(X), (i + 1) * subDiv)
            pred: torch.Tensor = 2 * torch.abs(0.5 - mod(X[i * subDiv:end].to(device))[:, 0])
            res = torch.concat((res, pred))
        return res


def score(X, Y, mod, scalar=1.0, uncertainty=None):
    a = len(X) / (len(X) + len(Y)) * scalar
    if uncertainty is None:
        term2 = (1 - a) * u(X[:, :], mod)
    else:
        term2 = (1 - a) * uncertainty
    term1 = a * (1 - dist(X[:, 1:3], Y[:, 1:3]))
    return term1 + term2, term1 / term2


def query(n, X, Y, mod, scalar=1.0):
    toLabel = torch.empty((0, 5))
    uncertainty = u(X[:, 1:3], mod)

    for _ in range(n):
        xScore, bound = score(X, Y, mod, scalar, uncertainty)
        idx = torch.argmin(xScore)
        chosenValue = X[idx][None, :]
        toLabel = torch.concat((toLabel, chosenValue))
        Y = torch.concat((Y, chosenValue[:, :]))
        X = torch.cat((X[:idx], X[idx + 1:]))
        uncertainty = torch.cat((uncertainty[:idx], uncertainty[idx + 1:]))

    return toLabel, X


def queryToDataSet(n, X, dataSet, mod, scalar=1.0):
    samples, X = query(n, X, getDataFromDataSet(dataSet), mod, scalar)
    gt, samples, _ = oracle(samples)
    return appendToDataSet(dataSet, samples, gt), X


def oracle(X):
    return X[:, 3], X[:, :], X[:, 4]


def train(seed=0, lr=1e-3, batchSize=8, epochMax=200, startingTrainingSize=25, generateAmount=4, generatingEpoch=75,
          generatingAfterEpoch=25, modelName="sitnikovNN", c=0.01):
    generatingTotal = generatingEpoch - generatingAfterEpoch

    file = "Utils/data.npy"
    e = 0.5
    load = True
    if not load:
        data = []
        for zSpace in tqdm(np.linspace(-1, 1, 200)):
            data.append(np.load(f"Simulations/data_z={zSpace}.npy"))
        data = np.array(data)
        np.save(file, data)
    else:
        data = np.load(file)

    data = np.reshape(data, (200 * 500, 5))
    data = data[inInterestZone(data[:, 1], data[:, 2])]

    totData = torch.from_numpy(data).to(dtype=torch.float32)

    torch.manual_seed(seed)
    totData = totData[torch.randperm(totData.size(0))]

    # --- Make dataSets
    sizeTraining = startingTrainingSize
    sizeVal = 2000
    valDataSet = torch.utils.data.TensorDataset(totData[:sizeVal, :], totData[:sizeVal, 3])
    totData = totData[sizeVal:]
    trainSet = torch.utils.data.TensorDataset(totData[:sizeTraining, :], totData[:sizeTraining, 3])

    print(f"Validation dataset: {len(valDataSet)}, Training dataset: {len(trainSet)}, Remaining simulations: "
          f"{len(totData) - len(trainSet)}")

    # --- Initialize model
    model = SitnikovModels.models[modelName]()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, drop_last=False)

    valDataLoader = torch.utils.data.DataLoader(dataset=valDataSet, batch_size=batchSize, shuffle=False,
                                                drop_last=False)

    totLosses = []
    totValLosses = []
    totAccuracies = []
    minEvalLoss = 1000
    maxAccuracy = 0
    savedEpoch = -1
    with tqdm(range(0, epochMax), unit="epoch") as pbar:
        for epoch in pbar:
            totalLoss = 0

            if not epoch >= generatingEpoch and epoch >= generatingAfterEpoch:
                trainSet, totDat = queryToDataSet(generateAmount, totData, trainSet, model, scalar=c)
                dataLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True,
                                                         drop_last=False)

            model.train(mode=True)
            for data in dataLoader:
                x, y = data
                if useGPU:
                    x, y = x.to(device), y.to(device)
                prediction = model(x[:, 1:3])
                prediction = torch.select(prediction, 1, 0)
                # allEqual = torch.all(prediction == prediction[0])

                loss = criterion(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                totalLoss += loss / len(dataLoader)

            totLosses.append(totalLoss.detach())

            """if allEqual:
                print("All Equal")
                break"""

            with torch.no_grad():
                totalValLoss = 0
                accuracy = 0
                model.train(mode=False)
                for valData in valDataLoader:
                    x, y = valData
                    if useGPU:
                        x, y = x.to(device), y.to(device)
                    prediction = model(x[:, 1:3])
                    prediction = torch.select(prediction, 1, 0)
                    valLoss = criterion(prediction, y)
                    accuracy += torch.sum(((prediction > 0.5).int() == y))

                    totalValLoss += valLoss / len(valDataLoader)

                totValLosses.append(totalValLoss.detach())
                accuracy = accuracy / len(valDataSet)
                totAccuracies.append(accuracy)

                if accuracy > maxAccuracy:
                    maxAccuracy = accuracy
                    # torch.save(model.state_dict(), f"Models/CNNBest{'AL' if al else 'NoAL'}.pt")

                if totalValLoss < minEvalLoss and epoch > generatingAfterEpoch:
                    minEvalLoss = totalValLoss
                    savedEpoch = epoch
                    torch.save(model.state_dict(), f"Models/BestAL_model={modelName}_seed={seed}_train="
                                                   f"{startingTrainingSize + generatingTotal * generateAmount}"
                                                   f"_lr={lr}_batchSize={batchSize}_c={c}.pt")

            pbar.set_postfix(loss=totalLoss, evalLoss=totalValLoss,  # allEqual=allEqual,
                             accuracy=accuracy)

        torch.save(model.state_dict(), f"Models/AL_model={modelName}_seed={seed}_train="
                                       f"{startingTrainingSize + generatingTotal * generateAmount}"
                                       f"_lr={lr}_batchSize={batchSize}_c={c}.pt")

    if useGPU:
        losses = torch.tensor(totLosses).cpu().numpy()
        valLosses = torch.tensor(totValLosses).cpu().numpy()
        totAccuracies = torch.tensor(totAccuracies).cpu().numpy()
    else:
        losses = totLosses
        valLosses = totValLosses

    plt.axvline(x=savedEpoch, color="k", label="Save Epoch")
    plt.plot(np.arange(len(losses)), losses, label="Training loss")
    plt.plot(np.arange(len(valLosses)), valLosses, label="Validation loss")
    plt.plot(np.arange(len(totAccuracies)), totAccuracies, label="Validation accuracy")
    plt.axvline(generatingEpoch, color="red", label="End of AL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"Results/Training/AlLoss_model={modelName}_seed={seed}_train={startingTrainingSize + generatingTotal * generateAmount}"
                f"_lr={lr}_batchSize={batchSize}.png")
    plt.show()

    print(f"Validation dataset: {len(valDataSet)}, Training dataset: {len(trainSet)}, Remaining simulations: "
          f"{len(totData) - len(trainSet)}")


if __name__ == '__main__':
    train(seed=0, lr=1e-3, batchSize=8, epochMax=200, startingTrainingSize=25, generateAmount=4, generatingEpoch=75,
          generatingAfterEpoch=25)
