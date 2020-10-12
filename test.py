import os
import numpy as np
import models
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

z_dimension = 64


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    plt.figure(figsize=(4, 4))

    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1])
    plt.title(title)
    plt.show()


# 创建对象
AE = models.autoencoder(dimension=z_dimension).to(device)
AE.load_state_dict(torch.load('./AE.pth'))


feature = np.loadtxt("./data/feature.csv", delimiter=",")
label = np.loadtxt("./data/label.csv", delimiter=",")

data = feature[0:1000, :]
label = label[0:1000]

reduction_data, _ = AE(torch.from_numpy(data).clone().detach().type(
    torch.FloatTensor).view(1000, 1, 16, 16).to(device))

reduction_data = reduction_data.cpu().detach().numpy()

tsne_original = TSNE(n_components=2, init='pca', random_state=0)
original_tsne_data = tsne_original.fit_transform(data)
plot_embedding(original_tsne_data, label, "original")

tsne_reduction = TSNE(n_components=2, init='pca', random_state=0)
reduction_tsne_data = tsne_reduction.fit_transform(reduction_data)
plot_embedding(reduction_tsne_data, label, "reduction")
