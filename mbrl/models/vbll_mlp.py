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

import vbll

from mbrl import util

from .model import Model
from .util import truncated_normal_init


class VBLLMLP(Model):
    """Implements a MLP model with Variational Bayesian Last Layer.

    This model is based on the
    2024 paper (VBLL) https://arxiv.org/abs/2404.11599v1

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        feature_dim (int): the size of the feature representation (e.g., size of l3 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        recursive_num_epochs (int): number of epochs to train the model recursively.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ELU when ``None``.
        regularization_weight_factor (int): the factor k to multiply the regularization weight by.
            Regularization weight is set to k/dataset_length and used in ELBO term.
        parameterization (str): Parameterization of covariance matrix.
            Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}.
        cov_rank (int, optional): the rank of the covariance matrix.
        prior_scale (float): Scale of prior covariance matrix.
        wishart_scale (float): Scale of Wishart prior on noise covariance.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        feature_dim: int = 200,
        deterministic: bool = False,
        recursive_num_epochs: int = 0,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        regularization_weight_factor: int = 1,
        parameterization: str = 'dense',
        cov_rank: Optional[int] = None,
        prior_scale: float = 1.,
        wishart_scale: float = .1,
        init_noise_logdiag: str = "random"
    ):
        super().__init__(device=device)
        self.deterministic = deterministic
        self.device = device
        self.regularization_weight_factor = regularization_weight_factor
        self.in_size = in_size
        self.out_size = out_size
        self.recursive_num_epochs = recursive_num_epochs

        # define network / feature representation (MLP)
        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ELU()
            else:
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2.")
        hidden_layers = [nn.Sequential(nn.Linear(in_size, hid_size), create_activation())]
        for i in range(num_layers - 2):
            hidden_layers.append(nn.Sequential(nn.Linear(hid_size, hid_size), create_activation()))
        hidden_layers.append(nn.Sequential(nn.Linear(hid_size, feature_dim), create_activation()))

        self.feature_extractor = nn.Sequential(*hidden_layers)

        # define output layer
        self.out_layer = vbll.Regression(feature_dim, out_size, regularization_weight=regularization_weight_factor, parameterization=parameterization, 
            cov_rank=cov_rank, prior_scale = prior_scale, wishart_scale = wishart_scale, init_noise_logdiag=init_noise_logdiag)

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
        var = pred.var
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
        out = self._default_forward_out(model_in)
        return out.train_loss_fn(target), {}
    
    def _train_recursively_auto_epoch(self, model_in, target, eval_model_in, eval_target):
        """Trains the model recursively until the validation score does not improve anymore."""
        def eval_fn(eval_model_in, eval_target):
                    squared = torch.pow(self.eval_score(eval_model_in, eval_target)[0],2)
                    mean = torch.mean(squared)
                    return mean.item()
                
        recursive_iterations = 0
        equal_counter = 0
        last_loss_sum = eval_fn(eval_model_in, eval_target)

        current_loss_sum = last_loss_sum
        while last_loss_sum >= current_loss_sum and equal_counter < 2 and recursive_iterations < self.recursive_num_epochs:
            recursive_iterations += 1
            if last_loss_sum == current_loss_sum:
                equal_counter += 1
            else:
                equal_counter = 0
            last_loss_sum = current_loss_sum
            
            out = self._default_forward_out(model_in)
            out.train_loss_fn(target, recursive_update=True)

            current_loss_sum =  eval_fn(eval_model_in, eval_target)
        return recursive_iterations
    
    def train_recursively(self, model_in, target, eval_model_in = None, eval_target = None, mode = 2) -> int:
        """Trains the model recursively for a given number of epochs.
        
        :param model_in: Input tensor for recursive updates
        :param target: Target tensor
        :param eval_model_in: Input to calculate the eval score to find a suitable number of epochs
        :param eval_target: Target for calculating the eval score
        :param mode: 0: no training, 1: train recursively for a fixed number of epochs, 2: train recursively until the validation score does not improve anymore
        
        :return: number of executed recursive iterations"""

        if mode == 0:
            return 0
        
        self.train() # gradients are not required but model has to be in train mode
        with torch.no_grad():
            if mode == 1:
                for _ in range(self.recursive_num_epochs):
                    out = self._default_forward_out(model_in)
                    out.train_loss_fn(target, recursive_update=True)
                return self.recursive_num_epochs

            if mode == 2:
                return self._train_recursively_auto_epoch(model_in, target, eval_model_in, eval_target)
        
        return 0
        
    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None, uncertainty = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the negative log likelihood of the target given the input.
        """
        with torch.no_grad():
            out = self._default_forward_out(model_in)
            pred_mean, pred_logvar = out.predictive.mean, torch.log(out.predictive.var)
            VBLL_val_loss = out.val_loss_fn(target)/self.out_size # divide by out_size to get rid of sum over last dim
            VBLL_train_loss = out.train_loss_fn(target)
            VBLL_val_loss_non_reduced = -out.predictive.log_prob(target) # same validation score as the val_loss_fn but non_reduced
            nll = util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            loss = nn.MSELoss(reduction="none")
            meta = {"NLL": nll.mean(), "VBLL_val_loss": VBLL_val_loss, "MSE": loss(pred_mean, target).mean(), "VBLL_train_loss": VBLL_train_loss}

            if uncertainty:
                batch_size, obs_dim = pred_mean.shape
                var = out.predictive.var
                overall_var = var

                aleatoric_var = self.get_aleatoric_var() # obs_dim - homoscedastic
                aleatoric_var_expanded = torch.unsqueeze(aleatoric_var, 0)
                aleatoric_var_expanded = aleatoric_var_expanded.expand(batch_size, obs_dim)

                epistemic_var = overall_var - aleatoric_var

                mean_gap = torch.abs(target - pred_mean)

                uncertainty = {}
                uncertainty["mean_aleatoric_var"] = aleatoric_var
                uncertainty["max_aleatoric"] = aleatoric_var
                uncertainty["min_aleatoric"] = aleatoric_var
                uncertainty["mean_epistemic_var"] = epistemic_var
                uncertainty["max_epistemic_var"] = epistemic_var
                uncertainty["min_epistemic_var"] = epistemic_var
                uncertainty["mean_variance"] = overall_var
                uncertainty["predicted_mean"] = pred_mean
                uncertainty["mean_gap_to_real_state"] = mean_gap
                uncertainty["gap_subtracted_aleatoric"] = mean_gap - aleatoric_var
                uncertainty["gap_subtracted_epistemic_var"] = mean_gap - epistemic_var
                uncertainty["gap_subtracted_variance"] = mean_gap - overall_var
                uncertainty["nll"] = nll
                meta["uncertainty"] = uncertainty
                
            return VBLL_val_loss_non_reduced, meta
        
    def get_aleatoric_var(self):
        return torch.exp(self.out_layer.noise_logdiag)
        
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
