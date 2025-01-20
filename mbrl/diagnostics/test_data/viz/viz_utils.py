

from typing import Tuple
from matplotlib import pyplot as plt
import torch
from vbll.layers.regression import VBLLReturn

def get_viz_pred(model, dataloader, stdevs = 1., title = None, save_path=None, dataset: Tuple[torch.tensor,torch.tensor] = None, axis_x_start = -5.5, axis_x_end = 1.5):
    """Visualize model predictions, including predictive uncertainty."""
    model.eval()

    X = torch.linspace(axis_x_start, axis_x_end, 1000)[..., None]
    Xp = X.detach().numpy().squeeze()

    pred = model(X)
    if isinstance(pred, tuple):
        Y_mean, Y_stdev = pred
    elif isinstance(pred, VBLLReturn):
        try:
            Y_mean = pred.predictive.mean
            Y_stdev = pred.predictive.covariance
        except:
            raise ValueError("model output must be either (mean, variance) or VBLLReturn object")
    elif isinstance(pred, torch.Tensor):
        Y_mean = pred
        Y_stdev = torch.zeros_like(pred)

    Y_mean = Y_mean.detach().numpy().squeeze()
    if isinstance(pred, torch.Tensor):
        Y_stdev = torch.sqrt(Y_stdev.squeeze()).detach().numpy()
    else:
        Y_stdev = torch.sqrt(torch.exp(Y_stdev.squeeze())).detach().numpy()

    return Xp, Y_mean, Y_stdev

def plot_normal_prediction(Xp, Y_mean, Y_stdev, stdevs =1.):
    plt.plot(Xp, Y_mean)
    plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2, color='b')
    plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color='b')

def plot_thompson_ensemble_pred(ensemble_model, Y_mean, Y_stdev, Xp, stdevs = 1.):
    for i in range(ensemble_model.no_thompson_heads):
        model_mean = Y_mean[i]
        model_stdev = Y_stdev[i]
        plt.plot(Xp, model_mean, label=f"Model {i+1}")
        plt.fill_between(Xp, model_mean - stdevs * model_stdev, model_mean + stdevs * model_stdev, alpha=0.2)
    

def plot_axis_and_save(dataloader, title = None, save_path=None, dataset: Tuple[torch.tensor,torch.tensor] = None, legend = False, axis_x_start = -1.5, axis_x_end = 1.5, axis_y_start = -2, axis_y_end = 2):
    if dataset is not None:
        (X, Y) = dataset
    else:
        (X, Y) = dataloader.dataset.X, dataloader.dataset.Y
    plt.scatter(X, Y, color='k')
    plt.axis([axis_x_start, axis_x_end, axis_y_start, axis_y_end])
    if not title == None:
        plt.title(title)
    if legend:
        plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show(block=True)