import os
import pathlib
import shutil
from typing import Mapping, Union

import omegaconf
import wandb
from mbrl.util.logger import Logger
from mbrl.util.logger import LogTypes
from mbrl.util.logger import LogFormatType
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
        wandb.log(wb_data)

    def upload_model(self, cfg_env_name, dynamics_model, env_steps, remove_after_upload=True):
        wd = os.getcwd()
        temp_model_dir = f"{wd}/models/step{env_steps}"
        os.makedirs(temp_model_dir, exist_ok=True)
        dynamics_model.model.save(temp_model_dir)
        artifact = wandb.Artifact(f"{cfg_env_name}-{wandb.run.name}-step{env_steps}", type='model')
        artifact.add_file(f"{temp_model_dir}/model.pth")
        wandb.log_artifact(artifact)
        if remove_after_upload:
            shutil.rmtree(temp_model_dir)
    
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
        wandb.define_metric(f"{EVAL_LOG_NAME}/episode_reward", step_metric=f"{EVAL_LOG_NAME}/episode", summary="max")
        wandb.define_metric(f"{EVAL_LOG_NAME}/episode_length", step_metric=f"{EVAL_LOG_NAME}/episode", summary="mean")
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