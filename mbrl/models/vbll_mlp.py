# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import vbll # TODO in yaml activation fn aendern zu ELU

from .model import Model
from .util import truncated_normal_init


class VBLLMLP(Model):
    """Implements a Gaussian MLP model with Variational Bayesian Last Layer.

    This model is based on the
    2024 paper (VBLL) https://arxiv.org/abs/2404.11599v1

    It predicts per output mean and softplus variance. The variance is bounded between learned ``min_var``
    and ``max_var`` parameters.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        learn_logvar_bounds (bool): if ``True``, the logvar bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ELU when ``None``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        deterministic: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        regularization_weight_factor: int = 1,
        parameterization: str = 'dense',
        cov_rank: Optional[int] = None,
        prior_scale: float = 1.,
        wishart_scale: float = .1,
    ):
        super().__init__(device=device)
        self.deterministic = deterministic
        self.device = device
        self.regularization_weight_factor = regularization_weight_factor
        self.in_size = in_size
        self.out_size = out_size

        # define network / feature representation (MLP)
        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ELU()
            else:
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        hidden_layers = [nn.Sequential(nn.Linear(in_size, hid_size), create_activation())]
        for i in range(num_layers - 1):
            hidden_layers.append(nn.Sequential(nn.Linear(hid_size, hid_size), create_activation()))

        self.feature_extractor = nn.Sequential(*hidden_layers)

        # define output layer
        self.out_layer = vbll.Regression(hid_size, out_size, regularization_weight=regularization_weight_factor, parameterization=parameterization, 
            cov_rank=cov_rank, prior_scale = prior_scale, wishart_scale = wishart_scale)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def _default_forward_out(
        self, x: torch.Tensor, **_kwargs
    ) -> vbll.regression.VBLLReturn:
        x = self.feature_extractor(x)
        out = self.out_layer(x)
        return out

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.
        more documentation in interface class Model
        """
        out = self._default_forward_out(x)
        pred = out.predictive
        cov = pred.covariance
        last_dim = cov.dim() - 1
        var = torch.diagonal(cov, dim1=last_dim-1, dim2=last_dim)
        logvar = torch.log(var)

        return pred.mean, logvar

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """ computes vbll loss
        more documentation in interface class Model
        """
        out = self._default_forward_out(model_in, use_propagation=False)
        return out.train_loss_fn(target), {}
        
    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the negative log likelihood of the target given the input.
        """
        with torch.no_grad():
            pred_mean, pred_logvar = self.forward(model_in)
            nll = 0.5 * (pred_logvar + ((target - pred_mean) ** 2)
                                    / torch.exp(pred_logvar))
            mse = F.mse_loss(pred_mean, target, reduction="none")
            return nll, {"MSE":mse}
        
    def update_regularization_weight_from_dataset_length(self,dataset_length: int):
        self.out_layer.regularization_weight = self.regularization_weight_factor/dataset_length

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
