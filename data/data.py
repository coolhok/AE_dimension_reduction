import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dim = 256
data_len = 50000


def normalization(x):
    return x/np.sqrt(np.sum(np.square(x)))


label = []
data = []

tem = None
tem_label = 0

for i in range(data_len):
    if i % 100 == 0:
        tem = np.random.rand(dim)
        tem_label = tem_label + 1
    data.append(normalization(tem + np.random.rand(dim)/8))
    label.append(tem_label)


np.savetxt("feature.csv", data, delimiter=",", fmt='%f')

np.savetxt("label.csv", label, delimiter=",")
