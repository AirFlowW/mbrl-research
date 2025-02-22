# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import itertools
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch import optim as optim

from mbrl import constants
import mbrl.util.checks as checks
from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import BootstrapIterator, TransitionIterator

from .model import Model

MODEL_LOG_FORMAT = constants.MODEL_LOG_FORMAT
MODEL_LOG_FORMAT_VBLL_EXTENSION = constants.MODEL_LOG_FORMAT_VBLL_EXTENSION


class ModelTrainer:
    """Trainer for dynamics models.

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
        logger (:class:`mbrl.util.Logger`, optional): the logger to use.
    """

    _LOG_GROUP_NAME = constants.TRAIN_LOG_GROUP_NAME
    _LOG_GROUP_NAME_VBLL_EXTENSION = constants.TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION

    def __init__(
        self,
        model: Model,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self._train_iteration = 0
        self.train_time = 0
        self.recursive_train_time = 0
        self.is_vbll_dynamics_model = checks.is_VBLL_dynamics_model(self.model)

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME,
                MODEL_LOG_FORMAT,
                color="blue",
                dump_frequency=1,
            )
            if self.is_vbll_dynamics_model:
                self.logger.register_group(
                    self._LOG_GROUP_NAME_VBLL_EXTENSION,
                    MODEL_LOG_FORMAT_VBLL_EXTENSION,
                    color="blue",
                    dump_frequency=1,
                )
                self.logger.register_group(
                    constants.RECURSIVE_LOG_NAME,
                    constants.RECURSIVE_LOG_FORMAT,
                    color="blue",
                    dump_frequency=1,
                )
            else:
                self.logger.register_group(
                    constants.TRAIN_EXTENSION_LOG_GROUP_NAME,
                    constants.TRAIN_EXTENSION_LOG_FORMAT,
                    color="blue",
                    dump_frequency=1,
                )
        param_list = [
            # All parameters except the VBLL.Regression layer get the specified weight decay
            {'params': [param for name, param in model.named_parameters() if 'out_layer' not in name], 'weight_decay': weight_decay},
            
            # The VBLL.Regression layer (out_layer) has weight decay set to zero
            {'params': [param for name, param in model.named_parameters() if 'out_layer' in name], 'weight_decay': 0.},
        ]
        self.optimizer = optim.Adam(
            param_list,
            lr=optim_lr,
            weight_decay=weight_decay,
            eps=optim_eps,
        )

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method iterates over the stored train dataset, one batch of transitions at a time,
        updates the model.

        If a validation dataset is provided in the constructor, this method will also evaluate
        the model over the validation data once per training epoch. The method will keep track
        of the weights with the best validation score, and after training the weights of the
        model will be set to the best weights. If no validation dataset is provided, the method
        will keep the model with the best loss over training data.

        Args:
            dataset_train (:class:`mbrl.util.TransitionIterator`): the iterator to
                use for the training data.
            dataset_val (:class:`mbrl.util.TransitionIterator`, optional):
                an iterator to use for the validation data.
            num_epochs (int, optional): if provided, the maximum number of epochs to train for.
                Default is ``None``, which indicates there is no limit.
            patience (int, optional): if provided, the patience to use for training. That is,
                training will stop after ``patience`` number of epochs without improvement.
                Ignored if ``evaluate=False`.
            improvement_threshold (float): The threshold in relative decrease of the evaluation
                score at which the model is seen as having improved.
                Ignored if ``evaluate=False`.
            callback (callable, optional): if provided, this function will be called after
                every training epoch with the following positional arguments::

                    - the model that's being trained
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - validation score (for ensembles, factored per member)
                    - best validation score so far

            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.update()`` (during training),
                and ``model.eval_score()`` (during evaluation). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is one of ``"train"`` or ``"eval"``, indicating if the callback
                was called during training or evaluation.

            evaluate (bool, optional): if ``True``, the trainer will use ``model.eval_score()``
                to keep track of the best model. If ``False`` the model will not compute
                an evaluation score, and simply train for some number of epochs. Defaults to
                ``True``.

            silent (bool): if ``True`` logging and progress bar are deactivated. Defaults
                to ``False``.

        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        eval_dataset = dataset_train if dataset_val is None else dataset_val

        training_losses, val_scores = [], []
        best_weights: Optional[Dict] = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        (best_val_score, meta) = self.evaluate(eval_dataset) if evaluate else (None, None)
        # only enable tqdm if training for a single epoch,
        # otherwise it produces too much output
        disable_tqdm = silent or (num_epochs is None or num_epochs > 1)
        eval_meta_list = []

        early_stopping = False
        start_time = time.time()
        for epoch in epoch_iter:
            if batch_callback:
                batch_callback_epoch = functools.partial(batch_callback, epoch)
            else:
                batch_callback_epoch = None
            batch_losses: List[float] = []
            for batch in tqdm.tqdm(dataset_train, disable=disable_tqdm):
                loss, meta = self.model.update(batch, self.optimizer)
                batch_losses.append(loss)
                if batch_callback_epoch:
                    batch_callback_epoch(loss, meta, "train")
            total_avg_loss = np.mean(batch_losses).mean().item()
            training_losses.append(total_avg_loss)

            eval_score = None
            model_val_score = 0
            if evaluate:
                eval_score, eval_meta = self.evaluate(
                    eval_dataset, batch_callback=batch_callback_epoch
                )
                eval_meta_list.append(eval_meta)
                val_scores.append(eval_score.mean().item())

                maybe_best_weights = self.maybe_get_best_weights(
                    best_val_score, eval_score, improvement_threshold
                )
                if maybe_best_weights:
                    best_val_score = torch.minimum(best_val_score, eval_score)
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1
                model_val_score = eval_score.mean()

            if patience and epochs_since_update >= patience:
                early_stopping = True

            if self.logger and not silent:
                if early_stopping or (num_epochs and num_epochs - 1 == epoch):
                    train_time = time.time() - start_time
                else:
                    train_time = 0

                self.logger.log_data(
                    self._LOG_GROUP_NAME,
                    {
                        "train_iteration": self._train_iteration,
                        "epoch": epoch,
                        "train_dataset_size": dataset_train.num_stored,
                        "val_dataset_size": dataset_val.num_stored
                        if dataset_val is not None
                        else 0,
                        "model_loss": total_avg_loss,
                        "model_val_score": model_val_score,
                        "model_best_val_score": best_val_score.mean()
                        if best_val_score is not None
                        else 0,
                        "train_time": train_time,
                    },
                )
            if callback:
                callback(
                    self.model,
                    self._train_iteration,
                    epoch,
                    total_avg_loss,
                    eval_score,
                    best_val_score,
                )

            if early_stopping:
                break
        
        self.train_time += train_time

        # saving the best models:
        if evaluate:
            self._maybe_set_best_weights_and_elite(best_weights, best_val_score)
        if self.logger:
            if self.is_vbll_dynamics_model:
                VBLL_val_loss_tensor = torch.tensor([entry['VBLL_val_loss'] for entry in eval_meta_list])
                NLL_tensor = torch.tensor([entry['NLL'] for entry in eval_meta_list])
                MSE_tensor = torch.tensor([entry['MSE'] for entry in eval_meta_list])
                VBLL_train_loss_tensor = torch.tensor([entry['VBLL_train_loss'] for entry in eval_meta_list])
                self.logger.log_data(
                    self._LOG_GROUP_NAME_VBLL_EXTENSION,
                    {
                        "train_iteration": self._train_iteration,
                        "model_avg_val_MSE_score": torch.mean(MSE_tensor).item(),
                        "model_best_val_MSE_score": torch.min(MSE_tensor).item(),
                        "model_avg_val_nll_score": torch.mean(NLL_tensor).item(),
                        "model_best_val_nll_score": torch.min(NLL_tensor).item(),
                        "model_avg_val_vbll_score": torch.mean(VBLL_val_loss_tensor).item(),
                        "model_best_val_vbll_score": torch.min(VBLL_val_loss_tensor).item(),
                        "model_avg_vbll_train_loss_score": VBLL_train_loss_tensor.mean().item(),
                        "model_best_vbll_train_loss_score":  VBLL_train_loss_tensor.min().item(),
                    },
                )
            else:
                NLL_tensor = torch.stack([entry['NLL'] for entry in eval_meta_list])
                self.logger.log_data(
                    constants.TRAIN_EXTENSION_LOG_GROUP_NAME,
                    {
                        "train_iteration": self._train_iteration,
                        "model_best_val_nll_score": torch.min(NLL_tensor).item(),
                        "model_avg_val_nll_score": torch.mean(NLL_tensor).item(),
                    },
                )

        self._train_iteration += 1
        return training_losses, val_scores
    
    def train_vbll_recursively(self, cfg, update_transition_batch, eval_transition_batch, mode=2):
        """Trains the model recursively for a given number of epochs.

        :param cfg: configuration object (highest level)
        :param update_samples: Input TransitionBatch for recursive updates
        :param eval_samples: Input TransitionBatch to calculate the eval score to find a suitable number of epochs
        :param mode: 0: no training, 1: train recursively for a fixed number of epochs, 2: train recursively until the validation score does not improve anymore

        :return: numpy array of executed recursive iterations per member. None if recursive train not possible.
        """
        # check if recursive training is possible
        if not (checks.is_VBLL_dynamics_model(cfg) \
                and 'dense_precision' == cfg.dynamics_model.member_cfg.get("parameterization", 'False')):
            return None
        
        if checks.is_thompson_sampling_active(cfg):
            self.model.model.reset_thompson_mlps()
        
        start_time = time.time()
        recursive_updates_list = np.array([])
        recursive_update_model_in, recursive_update_target = self.model._process_batch(update_transition_batch)
        eval_transition_batch.add_transition_batch(update_transition_batch)
        eval_model_in, eval_target = self.model._process_batch(eval_transition_batch)

        for member in self.model.model.members:
            if member.recursive_num_epochs is not None:
                recursive_updates = member.train_recursively(recursive_update_model_in, 
                        recursive_update_target, eval_model_in, eval_target, mode=mode)
                recursive_updates_list = np.append(recursive_updates_list, recursive_updates)
        train_time = time.time() - start_time
        self.recursive_train_time += train_time
        if self.logger and self.is_vbll_dynamics_model:
            self.logger.log_data(
                    self._LOG_GROUP_NAME_VBLL_EXTENSION,
                    {"recursive_train_time": train_time},
                )
        return recursive_updates_list

    def evaluate(
        self, dataset: TransitionIterator, batch_callback: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Evaluates the model on the validation dataset.

        Iterates over the dataset, one batch at a time, and calls
        :meth:`mbrl.models.Model.eval_score` to compute the model score
        over the batch. The method returns the average score over the whole dataset.

        Args:
            dataset (bool): the transition iterator to use.
            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.eval_score()`` (the score will
                be passed as a float, reduced using mean()). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is the string ``"eval"``.

        Returns:
            (tensor): The average score of the model over the dataset (and for ensembles, per
                ensemble member).
        """
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        sum_meta = {}
        no_batches = 0
        batch_scores_list = []
        meta_list = []
        for batch in dataset:
            batch_score, meta = self.model.eval_score(batch)
            no_batches += 1
            if self.is_vbll_dynamics_model:
                sum_meta = {
                    model_name: {
                        key: sum_meta.get(model_name, {}).get(key, 0) + meta.get(model_name, {}).get(key, 0)
                        for key in set(sum_meta.get(model_name, {})) | set(meta.get(model_name, {}))
                    }
                    for model_name in set(meta)
                }
            batch_scores_list.append(batch_score)
            meta_list.append(meta)
            if batch_callback:
                batch_callback(batch_score.mean(), meta, "eval")
        try:
            batch_scores = torch.cat(
                batch_scores_list, dim=batch_scores_list[0].ndim - 2
            )
        except RuntimeError as e:
            print(
                f"There was an error calling ModelTrainer.evaluate(). "
                f"Note that model.eval_score() should be non-reduced. Error was: {e}"
            )
            raise e
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        mean_axis = 1 if batch_scores.ndim == 2 else (1, 2)
        batch_scores = batch_scores.mean(dim=mean_axis)
        if self.is_vbll_dynamics_model:    
            meta = {model: {key: value / no_batches for key, value in metrics.items()} for model, metrics in sum_meta.items()}
            meta = {
                key: sum(model_metrics.get(key, 0) for model_metrics in meta.values()) / len(meta)
                for key in {k for metrics in meta.values() for k in metrics}
            }
        else:
            meta = {key: torch.stack([meta_i[key] for meta_i in meta_list]) for key in meta.keys()}
        return batch_scores, meta

    def maybe_get_best_weights(
        self,
        best_val_score: torch.Tensor,
        val_score: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:
        """Return the current model state dict  if the validation score improves.

        For ensembles, this checks the validation for each ensemble member separately.

        Args:
            best_val_score (tensor): the current best validation losses per model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_score - val_score) / torch.abs(best_val_score)
        improved = (improvement > threshold).any().item()
        return copy.deepcopy(self.model.state_dict()) if improved else None

    def _maybe_set_best_weights_and_elite(
        self, best_weights: Optional[Dict], best_val_score: torch.Tensor
    ):
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        if len(best_val_score) > 1 and hasattr(self.model, "num_elites"):
            sorted_indices = np.argsort(best_val_score.tolist())
            elite_models = sorted_indices[: self.model.num_elites]
            self.model.set_elite(elite_models)
