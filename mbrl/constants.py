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