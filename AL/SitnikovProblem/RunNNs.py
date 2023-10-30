import matplotlib as mpl
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import SitnikovNN_Al
import SitnikovNN_NoAL
from Utils.SelectSims import inInterestZone
import Utils.SitnikovModels as SitnikovModels

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

useGPU = torch.cuda.is_available()
# useGPU = False
device = torch.device("cuda:0") if useGPU else torch.device("cpu")

if __name__ == '__main__':
    for seed in [1]:
        train = True
        # seed = 3
        lr = 1e-3
        batchSize = 8
        epochMax = 100
        startingTrainingSize = 5
        generateAmount = 2
        generatingEpoch = 100
        generatingAfterEpoch = 0
        generatingTotal = generatingEpoch - generatingAfterEpoch
        modelName = "7Layers"
        c = 0.1

        if train:
            print("Training NoAl")
            SitnikovNN_NoAL.train(seed, lr, batchSize, epochMax, startingTrainingSize, generateAmount, generatingEpoch,
                                  generatingAfterEpoch, modelName)
            print("Training Al")
            SitnikovNN_Al.train(seed, lr, batchSize, epochMax, startingTrainingSize, generateAmount, generatingEpoch,
                                generatingAfterEpoch, modelName, c)

        file = "Utils/data.npy"
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

        model = SitnikovModels.models[modelName]()
        model.to(device)
        model.load_state_dict(torch.load(f"Models/AL_model={modelName}_seed={seed}_train="
                                         f"{startingTrainingSize + generatingTotal * generateAmount}"
                                         f"_lr={lr}_batchSize={batchSize}_c={c}.pt"))
        model1 = SitnikovModels.models[modelName]()
        model1.to(device)
        model1.load_state_dict(torch.load(f"Models/NoAL_model={modelName}_seed={seed}_train="
                                          f"{startingTrainingSize + generatingTotal * generateAmount}"
                                          f"_lr={lr}_batchSize={batchSize}.pt"))
        model2 = SitnikovModels.models[modelName]()
        model2.to(device)
        model2.load_state_dict(torch.load(f"Models/BestAL_model={modelName}_seed={seed}_train="
                                          f"{startingTrainingSize + generatingTotal * generateAmount}"
                                          f"_lr={lr}_batchSize={batchSize}_c={c}.pt"))
        model3 = SitnikovModels.models[modelName]()
        model3.to(device)
        model3.load_state_dict(torch.load(f"Models/BestNoAL_model={modelName}_seed={seed}_train="
                                          f"{startingTrainingSize + generatingTotal * generateAmount}"
                                          f"_lr={lr}_batchSize={batchSize}.pt"))

        models = [model, model1, model2, model3]

        thresholds = np.linspace(0, 1, int(1e2))

        totTns, totFps, totFns, totTps = [], [], [], []
        totAccuracies = []
        for model in models:
            tns, fps, fns, tps = [], [], [], []
            accuracies = []
            for threshold in tqdm(thresholds):
                data = totData[:, 1:3].to(device)
                labels = totData[0:, 3]
                predictions = (model(data)[:, 0] > threshold).detach().cpu()
                cm = confusion_matrix(labels, predictions, normalize="true")

                tn, fp, fn, tp = cm.ravel()
                tns.append(tn)
                fps.append(fp)
                fns.append(fn)
                tps.append(tp)
                accuracy = (tp + tn) / (tp + fp + tn + fn)
                accuracies.append(accuracy)

            totTns.append(tns)
            totFps.append(fps)
            totFns.append(fns)
            totTps.append(tps)
            totAccuracies.append(accuracies)

        totTns = np.array(totTns)
        totFps = np.array(totFps)
        totFns = np.array(totFns)
        totTps = np.array(totTps)

        colors = ["tab:blue", "tab:orange", "tab:green"]
        for i in range(len(models)):
            lineStyle = "dotted" if i // 2 == 0 else "solid"
            plt.plot(1 - totTns[i], totTps[i], linestyle=lineStyle, color=colors[i % 2])

        line1 = lines.Line2D([], [], color="k", label="Before Overfitting", linestyle="solid")
        line2 = lines.Line2D([], [], color="k", label="End of Training", linestyle="dotted")
        red_patch = patches.Patch(color="tab:blue", label="Active Learning")
        blue_patch = patches.Patch(color="tab:orange", label="No Active Learning")
        green_patch = patches.Patch(color="tab:green", label="Active Learning DataSet")
        plt.legend(handles=[red_patch, blue_patch, line1, line2], loc="lower right")
        plt.title("ROC curve for different models")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.savefig(
            f"Results/Plots/ROC_model={modelName}_seed={seed}_train={startingTrainingSize + generatingTotal * generateAmount}"
            f"_lr={lr}_batchSize={batchSize}_c={c}.png")
        plt.show()

        for i in range(len(models)):
            lineStyle = "dotted" if i // 2 == 0 else "solid"
            plt.plot(thresholds, totAccuracies[i], linestyle=lineStyle, color=colors[i % 2])

        plt.legend(handles=[red_patch, blue_patch, line1, line2], loc="lower center")
        plt.title("Accuracy curve for different models")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.savefig(
            f"Results/Plots/Accuracy_model={modelName}_seed={seed}_train={startingTrainingSize + generatingTotal * generateAmount}"
            f"_lr={lr}_batchSize={batchSize}_c={c}.png")
        plt.show()
