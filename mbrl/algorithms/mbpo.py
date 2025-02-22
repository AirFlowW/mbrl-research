# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
from typing import Optional, Sequence, cast

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
from mbrl.util import time_keeping
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder

MBPO_RESULTS_LOG_NAME = 'mbpo_eval'
MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        truncateds = np.zeros_like(pred_dones, dtype=bool)
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
            truncateds[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
) -> float:
    avg_episode_reward = 0.0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        video_recorder.init(enabled=(episode == 0))
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        (
            obs,
            action,
            next_obs,
            reward,
            terminated,
            truncated,
        ) = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, terminated, truncated)
        return new_buffer
    return sac_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
    reward_fn: mbrl.types.RewardFnType = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    start_overall_runtime = time.time()
    debug_mode = cfg.get("debug_mode", False)

    # to not run into half initialized configs that are not used anyway
    omegaconf.OmegaConf.set_struct(cfg, False)
    del cfg["action_optimizer"]
    omegaconf.OmegaConf.set_struct(cfg, True)

    is_vbll_dynamics_model = mbrl.util.checks.is_VBLL_dynamics_model(cfg)
    if is_vbll_dynamics_model:
        recursive_updates_list = None

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    if cfg.logger == "wandb":
        logger = mbrl.util.WANDBLogger(work_dir, cfg=cfg, enable_back_compatible=True)
    else:
        logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        MBPO_RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME, mbrl.constants.EVAL_LOG_FORMAT, color="green"
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

    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
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
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )
    env_data_for_analysis = {"last_data": replay_buffer.get_last_n_samples(n=cfg.algorithm.initial_exploration_steps)}

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None
    current_trial = 1
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )

        steps_epoch = 0
        episode_reward = 0
        episode_length = 0
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while True:
        # for steps_epoch in range(cfg.overrides.epoch_length): nicht ienfach for sondern man muss die episode noch zuende bringen?
            if terminated or truncated:
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {"episode":current_trial, "env_step": env_steps, "episode_reward": episode_reward, "episode_length": episode_length},
                )
                current_trial += 1
                episode_reward = 0
                episode_length = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False

                if (env_steps >= cfg.overrides.num_steps or steps_epoch >= cfg.overrides.epoch_length):
                    # ------ Epoch ended (evaluate and save model) ------
                    avg_reward = evaluate(
                        test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                    )
                    logger.log_data(
                        MBPO_RESULTS_LOG_NAME,
                        {
                            "epoch": epoch,
                            "episode": current_trial,
                            "env_step": env_steps,
                            "episode_reward": avg_reward,
                            "episode_length": episode_length,
                            "rollout_length": rollout_length,
                        },
                    )
                    if avg_reward > best_eval_reward:
                        video_recorder.save(f"{epoch}.mp4")
                        best_eval_reward = avg_reward
                        agent.sac_agent.save_checkpoint(
                            ckpt_path=os.path.join(work_dir, "sac.pth")
                        )
                    epoch += 1
                    break
            # --- Doing env step and adding to model dataset ---
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

                # --------- track the uncertainty of the model
                if cfg.logger == "wandb":
                    env_data_for_analysis["last_data"] = replay_buffer.get_last_n_samples(n=cfg.algorithm.initial_exploration_steps)
                    for key, batch in env_data_for_analysis.items():
                        _, meta = dynamics_model.eval_score(batch, uncertainty=True)
                        logger.log_uncertainty(meta["uncertainty"], key)

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # update vbll model recursively
            elif cfg.overrides.get("recursive_update", 0) > 0 and is_vbll_dynamics_model and \
                    env_steps % cfg.overrides.get("no_recursive_update_data", 5) == 0:
                recursive_updates_list = model_trainer.train_vbll_recursively(
                    cfg, replay_buffer.get_last_n_samples(cfg.overrides.get("no_recursive_update_data", 5)),
                    replay_buffer.sample(cfg.overrides.get("no_recursive_update_eval_data", 2500)),
                    mode = cfg.overrides.get("recursive_update", 0)
                    )
                
            # save model to wandb
            freq_log_model = cfg.overrides.get("freq_log_model", False)
            if freq_log_model and (env_steps+1) % freq_log_model == 0 and cfg.logger == "wandb":
                logger.upload_model(cfg.overrides.env, dynamics_model, env_steps+1)
                
            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            env_steps += 1
            steps_epoch += 1
            episode_reward += reward
            episode_length += 1
            obs = next_obs
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
    logger.log_data(
        mbrl.constants.OVERALL_LOG_NAME,
        {"planning_time":time_keeping.accumulated_planning_time, "train_time": model_trainer.train_time, "recursive_train_time":model_trainer.recursive_train_time, "overall_time": time.time() - start_overall_runtime},
    )
    return np.float32(best_eval_reward)
