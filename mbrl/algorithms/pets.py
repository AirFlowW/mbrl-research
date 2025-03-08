# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
from mbrl.util import time_keeping
import mbrl.util.common
import mbrl.util.math

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    start_overall_runtime = time.time()
    debug_mode = cfg.get("debug_mode", False)
    is_vbll_dynamics_model = mbrl.util.checks.is_VBLL_dynamics_model(cfg)
    is_thompson_sampling_enabled = mbrl.util.checks.is_thompson_sampling_active(cfg)

    # model specific context
    if is_vbll_dynamics_model:
        recursive_updates_list = None

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        if cfg.logger == "wandb":
            logger = mbrl.util.WANDBLogger(work_dir, cfg=cfg)
        else:
            logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )
        logger.register_group(
            mbrl.constants.OVERALL_LOG_NAME, mbrl.constants.OVERALL_LOG_FORMAT , color="green"
        )
        logger.register_group(
            mbrl.constants.STEP_LOG_NAME, mbrl.constants.STEP_LOG_FORMAT, color="yellow"
        )
        if is_vbll_dynamics_model:
            logger.register_group(
                mbrl.constants.VBLL_LOG_NAME, mbrl.constants.VBLL_LOG_FORMAT, color="yellow"
            )

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    replay_buffer.save(work_dir)

    env_data_for_analysis = {}
    env_data_for_analysis["initial_data"] = replay_buffer.get_all()

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    while env_steps < cfg.overrides.num_steps:
        obs, _ = env.reset()
        agent.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps_trial = 0
        while not terminated and not truncated:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:                        
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )
            # save model to wandb
            freq_log_model = cfg.overrides.get("freq_log_model", False)
            if freq_log_model and (env_steps+1) % freq_log_model == 0 and cfg.logger == "wandb":
                logger.upload_model(cfg.overrides.env, dynamics_model, env_steps+1)

            # --- Doing env step using the agent and adding to model dataset ---
            if is_thompson_sampling_enabled:
                dynamics_model.model.set_thompson_sampling_active()
                
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            if is_thompson_sampling_enabled:
                dynamics_model.model.set_thompson_sampling_inactive()

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1

            # update vbll model recursively
            if cfg.overrides.get("recursive_update", 0) > 0 and is_vbll_dynamics_model and \
                    env_steps % cfg.overrides.get("no_recursive_update_data", 5) == 0:
                recursive_updates_list = model_trainer.train_vbll_recursively(
                    cfg, replay_buffer.get_last_n_samples(cfg.overrides.get("no_recursive_update_data", 5)),
                    replay_buffer.sample(cfg.overrides.get("no_recursive_update_eval_data", 2500)),
                    mode= cfg.overrides.get("recursive_update", 0),
                    old_data=replay_buffer.sample(cfg.overrides.get("no_recursive_update_old_data", 0))
                    )

            if env_steps % cfg.overrides.get("track_uncertainty_freq", 250) == 1 and cfg.logger == "wandb":    
                env_data_for_analysis["last_data"] = replay_buffer.get_last_n_samples(n=cfg.overrides.trial_length)
                for key, batch in env_data_for_analysis.items():
                    _, meta = dynamics_model.eval_score(batch, uncertainty=True)
                    logger.log_uncertainty(meta["uncertainty"], key)

            if logger is not None:
                logger.log_data(
                    mbrl.constants.STEP_LOG_NAME,
                    {"env_step": env_steps, "planning_time": time_keeping.last_planning_time, "step_reward": reward},
                )
                if is_vbll_dynamics_model:
                    avg_recursive_update = np.mean(recursive_updates_list) if recursive_updates_list is not None and len(recursive_updates_list) > 0 else 0
                    logger.log_data(
                        mbrl.constants.VBLL_LOG_NAME,
                        {"avg_no_recursive_updates": avg_recursive_update},
                    )
                    recursive_updates_list = None

        current_trial += 1
        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"episode":current_trial, "env_step": env_steps, "episode_reward": total_reward, "episode_length": steps_trial},
            )
        
        max_total_reward = max(max_total_reward, total_reward)
    logger.log_data(
        mbrl.constants.OVERALL_LOG_NAME,
        {"planning_time":time_keeping.accumulated_planning_time, "train_time": model_trainer.train_time, "recursive_train_time":model_trainer.recursive_train_time, "overall_time": time.time() - start_overall_runtime},
    )
    return np.float32(max_total_reward)
