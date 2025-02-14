import os
import pathlib
import shutil
from typing import Mapping, Union

import omegaconf
import torch
import wandb
from mbrl.util.logger import Logger
from mbrl.util.logger import LogTypes
import mbrl.constants as constants

class WANDBLogger(Logger):
    """extends the Logger class to log data to Weights & Biases.

    Args:
        log_dir, enable_back_compatible: see Logger class
        cfg: the configuration object to log to Weights & Biases
    """

    def __init__(
        self, log_dir: Union[str, pathlib.Path], cfg: omegaconf.DictConfig, enable_back_compatible: bool = False,
    ):
        super().__init__(log_dir=log_dir, enable_back_compatible=enable_back_compatible)
        wnb_cfg = omegaconf.OmegaConf.to_container(
            cfg, resolve=True,
        )

        wandb.init(
            project=cfg.wandb_project,
            config=wnb_cfg,
            group=cfg.algorithm.name + "_" + cfg.overrides.env,
        )
        define_vbllts_wandb_metrices()

    def log_data(self, group_name: str, data: Mapping[str, LogTypes]):
        super().log_data(group_name, data, dump=False)
        wb_data = {f"{group_name}/{key}": value for key, value in data.items()}
        if "model_train/train_time" in wb_data.keys() and wb_data["model_train/train_time"] == 0:
            del wb_data["model_train/train_time"]
        wandb.log(wb_data)

    def log_uncertainty(self, uncertainty, key):
        """logs the uncertainty of a model to Weights & Biases.
        logs for each key the mean, max, min of the dims.
        in the keys the first mean/max/min means over the model members, the second mean/max/min means over the output dim.
        'mean_variance_dim_min' - is the mean over the members and the min value of the output dim

        Args:
            uncertainty (dict): keys - aleatoric_var_mean, max_aleatoric, min_aleatoric, epistemic_var_mean, max_epistemic_var, 
                min_epistemic_var, log_variance, pred_mean, mean_gap, gap_subtracted_aleatoric,
                gap_subtracted_epistemic_var, gap_subtracted_overall_var - min max mean means over the model members
            key (str): either initial_data or last_data as a key so that it gets tracked at the right plot
        """
        uncertainty = {k: torch.mean(v, dim=0) if v.dim() > 1 else v for k, v in uncertainty.items()}
        wb_data = {}
        for uncertainty_name, value in uncertainty.items():
            tracking_id = f"uncertainty_{key}/{uncertainty_name}"
            wb_data[f"{tracking_id}_dim_mean"] = value.mean().item()
            wb_data[f"{tracking_id}_dim_max"] = value.max().item()
            wb_data[f"{tracking_id}_dim_min"] = value.min().item()

        wandb.log(wb_data)

    def upload_model(self, cfg_env_name, dynamics_model, env_steps, remove_after_upload=True):
        try:
            wd = os.getcwd()
            temp_model_dir = os.path.join(wd, "models", f"step{env_steps}")
            os.makedirs(temp_model_dir, exist_ok=True)
            dynamics_model.model.save(temp_model_dir)

            model_file = os.path.join(temp_model_dir, "model.pth")
            if not os.path.isfile(model_file):
                raise Exception(f"Model file {model_file} not found")

            artifact_name = f"{cfg_env_name}-{wandb.run.name}-step{env_steps}"
            artifact = wandb.Artifact(artifact_name, type='model')
            artifact.add_file(model_file)
            wandb.log_artifact(artifact)
            if remove_after_upload:
                shutil.rmtree(temp_model_dir)
        except Exception as e:
            print(f"Failed to upload model to wandb: {e}")
    
def define_vbllts_wandb_metrices(): # this mehtod just does not seem to work 
        EVAL_LOG_NAME = constants.RESULTS_LOG_NAME
        STEP_LOG_NAME = constants.STEP_LOG_NAME
        MODEL_LOG_NAME = constants.TRAIN_LOG_GROUP_NAME
        TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION = constants.TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION

        # define RL metrices
        env_step_metric = wandb.define_metric(f"{STEP_LOG_NAME}/env_step")
        wandb.define_metric(f"{EVAL_LOG_NAME}/step_reward", step_metric=env_step_metric)

        wandb.define_metric(f"{EVAL_LOG_NAME}/episode")
        wandb.define_metric(f"{EVAL_LOG_NAME}/env_step", step_metric=f"{EVAL_LOG_NAME}/episode")
        wandb.define_metric(f"{EVAL_LOG_NAME}/episode_reward", step_metric=f"{EVAL_LOG_NAME}/env_step", summary="max")
        wandb.define_metric(f"{EVAL_LOG_NAME}/episode_length", step_metric=f"{EVAL_LOG_NAME}/env_step", summary="mean")
        # define model metrices
        wandb.define_metric(f"{MODEL_LOG_NAME}/train_iteration")
        wandb.define_metric(f"{MODEL_LOG_NAME}/epoch")
        wandb.define_metric(f"{MODEL_LOG_NAME}/train_dataset_size", step_metric=f"{MODEL_LOG_NAME}/train_iteration")
        wandb.define_metric(f"{MODEL_LOG_NAME}/val_dataset_size", step_metric=f"{MODEL_LOG_NAME}/train_iteration")
        wandb.define_metric(f"{MODEL_LOG_NAME}/model_loss", summary="mean")
        wandb.define_metric(f"{MODEL_LOG_NAME}/model_val_score", summary="mean")
        wandb.define_metric(f"{MODEL_LOG_NAME}/model_best_val_score", step_metric=f"{MODEL_LOG_NAME}/train_iteration")

        # define extended model metrices
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_avg_val_MSE_score", summary="mean", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_best_val_MSE_score", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_avg_val_nll_score", summary="mean", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_best_val_nll_score", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_avg_val_vbll_score", summary="mean", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_best_val_vbll_score", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_avg_vbll_train_loss_score", summary="mean", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")
        wandb.define_metric(f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/model_best_vbll_train_loss_score", step_metric=f"{TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION}/train_iteration")