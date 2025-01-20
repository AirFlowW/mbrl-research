import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SimpleFnDataset(Dataset):
    """The simple function used in DUE/SNGP 1-d regression experiments.
    Based on the implementation presented in:
    https://github.com/y0ast/DUE/blob/main/toy_regression.ipynb
    """

    def __init__(self, num_samples):
        self.num_samples = int(num_samples)
        self.X, self.Y = self.get_data()

    def get_data(self, noise=0.05, seed=2):
        np.random.seed(seed)

        W = np.random.randn(30, 1)
        b = np.random.rand(30, 1) * 2 * np.pi

        x = 5 * np.sign(np.random.randn(self.num_samples)) + np.random.randn(self.num_samples).clip(-2, 2)
        y = np.cos(W * x + b).sum(0)/5. + noise * np.random.randn(self.num_samples)
        return torch.tensor(x[..., None]).float()/10, torch.tensor(y[..., None]).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def viz_data(dataset):
    plt.scatter(dataset.X, dataset.Y, color = 'k')
    plt.axis([-1.5, 1.5, -2, 2])
    plt.show()