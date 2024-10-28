# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
RESULTS_LOG_NAME = "results"

EVAL_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("env_step", "GS", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
]

STEP_LOG_NAME = "step"

STEP_LOG_FORMAT = [
    ("env_step", "GS", "int"),
    ("step_reward", "R", "float"),
]

TRAIN_LOG_GROUP_NAME = "model_train"
MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]

TRAIN_LOG_GROUP_NAME_VBLL_EXTENSION = "model_train_validation_vbll_extension"
MODEL_LOG_FORMAT_VBLL_EXTENSION = [
    ("train_iteration", "I", "int"),
    ("model_avg_val_MSE_score", "AVGMSESCORE", "float"),
    ("model_best_val_MSE_score", "BESTMSESCORE", "float"),
    ("model_avg_val_nll_score", "AVGNLLSCORE", "float"),
    ("model_best_val_nll_score", "BESTNLLSCORE", "float"),
    ("model_avg_val_vbll_score", "AVGVBLLSCORE", "float"),
    ("model_best_val_vbll_score", "BESTVBLLSCORE", "float"),
    ("model_avg_vbll_train_loss_score", "AVGVBLLTRAINLOSS", "float"),
    ("model_best_vbll_train_loss_score", "BESTVBLLTRAINLOSS", "float"),
]