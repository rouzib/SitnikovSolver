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
from SitnikovNN_Al_WithSimulations import normalizeData

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

useGPU = torch.cuda.is_available()
device = torch.device("cuda:0") if useGPU else torch.device("cpu")


def getDataFromDataSet(dataSet):
    allData = []
    for i in range(len(dataSet)):
        sample = dataSet[i][0]
        allData.append(sample)
    allData = torch.stack(allData)
    return allData


def train(seed=0, lr=1e-3, batchSize=8, epochMax=200, startingTrainingSize=25, generateAmount=4, generatingEpoch=75,
          generatingAfterEpoch=25, modelName="SitnikovNN", fromDataset=""):
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

    totData = normalizeData(totData)

    # --- Make dataSets
    sizeTraining = startingTrainingSize + generatingTotal * generateAmount
    sizeVal = 2000
    valDataSet = torch.utils.data.TensorDataset(totData[:sizeVal, :], totData[:sizeVal, 3])
    totData = totData[sizeVal:]
    if fromDataset == "":
        dataSet = torch.utils.data.TensorDataset(totData[:sizeTraining, :], totData[:sizeTraining, 3])
    else:
        temp = torch.load(fromDataset)
        dataSet = torch.utils.data.TensorDataset(temp[:sizeTraining, :], temp[:sizeTraining, 3])

    print(f"Validation dataset: {len(valDataSet)}, Training dataset: {len(dataSet)}, Remaining simulations: "
          f"{len(totData) - len(dataSet)}")

    # --- Initialize model
    model = SitnikovModels.models[modelName]()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataLoader = torch.utils.data.DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=True, drop_last=False)

    valDataLoader = torch.utils.data.DataLoader(dataset=valDataSet, batch_size=batchSize, shuffle=False,
                                                drop_last=False)

    totLosses = []
    totValLosses = []
    totAccuracies = []
    minEvalLoss = 1000
    savedEpoch = -1

    with tqdm(range(0, epochMax), unit="epoch") as pbar:
        for epoch in pbar:
            totalLoss = 0

            model.train(mode=True)
            for data in dataLoader:
                x, y = data
                if useGPU:
                    x, y = x.to(device), y.to(device)
                prediction = model(x[:, 1:3])
                prediction = torch.select(prediction, 1, 0)
                loss = criterion(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                totalLoss += loss / len(dataLoader)

            totLosses.append(totalLoss.detach())

            with torch.no_grad():
                accuracy = 0
                totalValLoss = 0
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

                if totalValLoss < minEvalLoss:
                    minEvalLoss = totalValLoss
                    savedEpoch = epoch
                    torch.save(model.state_dict(), f"Models/BestNoAL_model={modelName}_seed={seed}_train="
                                                   f"{startingTrainingSize + generatingTotal * generateAmount}"
                                                   f"_lr={lr}_batchSize={batchSize}{'_fromAlDataset' if fromDataset != '' else ''}.pt")

            pbar.set_postfix(loss=totalLoss, evalLoss=totalValLoss, accuracy=accuracy)

        torch.save(model.state_dict(), f"Models/NoAL_model={modelName}_seed={seed}_train="
                                       f"{startingTrainingSize + generatingTotal * generateAmount}"
                                       f"_lr={lr}_batchSize={batchSize}{'_fromAlDataset' if fromDataset != '' else ''}.pt")

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
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        f"Results/Training/NoAlLoss_model={modelName}_seed={seed}_train={startingTrainingSize + generatingTotal * generateAmount}"
        f"_lr={lr}_batchSize={batchSize}{'_fromAlDataset' if fromDataset != '' else ''}.png")
    plt.show()

    print(f"Validation dataset: {len(valDataSet)}, Training dataset: {len(dataSet)}, Remaining simulations: "
          f"{len(totData) - len(dataSet)}")

    if fromDataset == "":
        torch.save(getDataFromDataSet(dataSet), f"Datasets/NoAL_model={modelName}_seed={seed}_train="
                                                f"{startingTrainingSize + generatingTotal * generateAmount}"
                                                f"_lr={lr}_batchSize={batchSize}.pt")


if __name__ == '__main__':
    train(seed=0, lr=1e-3, batchSize=64, epochMax=1000, startingTrainingSize=-1, generateAmount=0, generatingEpoch=75,
          generatingAfterEpoch=25, modelName="7Layers")
