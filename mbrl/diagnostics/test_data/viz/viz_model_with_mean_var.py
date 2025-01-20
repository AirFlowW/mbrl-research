from typing import Tuple
import torch

import mbrl.diagnostics.test_data.viz.viz_utils as viz_utils

"""
model output either (mean, variance) or VBLLReturn object
"""
def viz_model(model, dataloader, stdevs = 1., title = None, save_path=None, dataset: Tuple[torch.tensor,torch.tensor] = None):
  """Visualize model predictions, including predictive uncertainty."""
  Xp, Y_mean, Y_stdev = viz_utils.get_viz_pred(model, dataloader, stdevs, title, save_path, dataset)
  viz_utils.plot_normal_prediction(Xp, Y_mean, Y_stdev, stdevs)
  viz_utils.plot_axis_and_save(dataloader, title, save_path, dataset)