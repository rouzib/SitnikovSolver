import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

useGPU = torch.cuda.is_available()
# useGPU = False
device = torch.device("cuda:0") if useGPU else torch.device("cpu")


def inInterestZone(z0, vz0):
    # high
    a, c, = 2.8, 0.9
    vz0CutoffHigh = a * np.exp(-z0 ** 2 / (2 * c ** 2))
    # low
    vz0CutoffLow = - z0 ** 2 * 14 + 2.2

    return np.all([np.less_equal(vz0, vz0CutoffHigh), np.greater_equal(vz0, vz0CutoffLow)], axis=0)


if __name__ == '__main__':
    data = np.random.uniform([0.5, -1., 0.], [0.5, 1., 3.], (10_000_000, 3))
    dataInZone = data[inInterestZone(data[:, 1], data[:, 2])]
    print(f"Sampled data points: {len(data)}, Sampled data points of interest: {len(dataInZone)}")

    cmap = matplotlib.colormaps["plasma"]
    plt.scatter(dataInZone[:, 1], dataInZone[:, 2], label="""c=[cmap(min(sample, 1)) for sample in data[:, 3]]""", s=1)
    plt.scatter(data[:, 1], data[:, 2], label="""c=[cmap(min(sample, 1)) for sample in data[:, 3]]""", s=1, zorder=-100)
    plt.savefig("Plots/test1.png")
    plt.show()

