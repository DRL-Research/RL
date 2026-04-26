import logging
import os

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from src.model.master_model import MasterModel
from src.model.model_handler import load_models, save_models
from src.plotting_utils.plotting_utils import plot_training_results
from src import project_globals
from src.project_globals import rollout_buffers
from src.training.episode_utils import process_episode
from src.training.general_utils import (
    setup_experiment_dirs,
    initialize_models,
    setup_loggers,
    close_everything,
    ensure_tensor,
)
from src.training.training_loop_utils import (
    init_training_results,
    prepare_models_for_cycle,
    perform_training_phase,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##########################################
# Training Loop
##########################################

def _make_master_buffer(experiment):
    """Create a RolloutBuffer sized for the shared master model (25-D obs, 4-D action)."""
    return RolloutBuffer(
        buffer_size=experiment.N_STEPS,
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(experiment.MASTER_OBS_DIM,),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=-1.0, high=1.0,
            shape=(experiment.EMBEDDING_SIZE,),
            dtype=np.float32,
        ),
        gamma=getattr(experiment, 'GAMMA', 0.99),
        gae_lambda=getattr(experiment, 'GAE_LAMBDA', 0.95),
        n_envs=1,
    )


def training_loop(experiment, env, agent_model, master_model):
    """
    Main training loop — 4 cycles:
      1. train everything
      2. local masters only
      3. agents only
      4. global master only
    """
    # In-place clear — same list object as all `import rollout_buffers` sites
    rollout_buffers.clear()
    project_globals.local_master_rollout_buffers.clear()
    project_globals.global_master_rollout_buffer = None

    # ── one agent buffer per controlled car ───────────────────────────────────
    for _ in env.env.config["controlled_cars"]:
        rollout_buffers.append(RolloutBuffer(
            buffer_size=experiment.N_STEPS,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(experiment.STATE_INPUT_SIZE,),
            ),
            action_space=spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
            ),
            gamma=getattr(experiment, 'GAMMA', 0.99),
            gae_lambda=getattr(experiment, 'GAE_LAMBDA', 0.95),
            n_envs=1,
        ))

    # ── two local-master buffers + one global-master buffer ───────────────────
    for _ in range(experiment.NUM_LOCAL_MASTERS):
        project_globals.local_master_rollout_buffers.append(
            _make_master_buffer(experiment)
        )
    project_globals.global_master_rollout_buffer = _make_master_buffer(experiment)

    collision_counter, episode_counter, total_steps = 0, 0, 0
    results = init_training_results()

    # ── Warm-up: pure random actions for the first WARMUP_EPISODES episodes ───
    # NOTE: we now use episode_counter (not total_steps) for the comparison so
    # that the warmup duration is exact regardless of episode length.
    warmup_ep = getattr(experiment, 'WARMUP_EPISODES', 0)
    if warmup_ep > 0:
        print(f"[Warmup] random agent actions for first {warmup_ep} episodes")

    # ── Peak-lock state (persists across all episodes) ────────────────────────
    peak_threshold = getattr(experiment, 'PEAK_ARRIVAL_THRESHOLD', 0.0)
    peak_locked = False

    # ── Best-model checkpointing ──────────────────────────────────────────────
    # Save models whenever rolling-50 arrival rate hits a new maximum.
    best_arrival_so_far = -1.0
    best_model_dir = os.path.join(experiment.EXPERIMENT_PATH, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    for cycle_num in range(1, experiment.CYCLES + 1):
        print(f"Cycle {cycle_num}/{experiment.CYCLES}")
        cotrain = getattr(experiment, 'COTRAIN_CYCLES', True)
        full_joint = getattr(experiment, 'FULL_JOINT_TRAINING', False)
        (train_both, training_local_master,
         training_agent, training_global_master) = prepare_models_for_cycle(
            cycle_num, experiment.CYCLES, master_model, agent_model,
            cotrain_cycles=cotrain,
            full_joint=full_joint,
        )

        for _ in range(experiment.EPISODES_PER_CYCLE):
            episode_counter += 1
            print(
                f"  Episode {episode_counter} / "
                f"{experiment.EPISODES_PER_CYCLE * experiment.CYCLES}"
            )
            episode_rewards, actions, steps, crashed, arrival_rate = process_episode(
                episode_counter, total_steps, env, master_model, agent_model,
                experiment,
                train_both=train_both,
                training_local_master=training_local_master,
                training_agent=training_agent,
                training_global_master=training_global_master,
            )
            if crashed:
                collision_counter += 1

            total_steps += steps
            results["episode_rewards"].append(episode_rewards)
            results["arrival_rates"].append(arrival_rate)
            results["all_actions"].append(actions)

            # ── Best-model checkpoint (rolling-50 arrival) ───────────────────
            if episode_counter >= 50:
                recent_arrivals = [v for v in results["arrival_rates"][-50:] if v is not None]
                rolling_arrival = float(np.mean(recent_arrivals)) if recent_arrivals else 0.0
                if rolling_arrival > best_arrival_so_far:
                    best_arrival_so_far = rolling_arrival
                    save_models(agent_model, master_model,
                                os.path.join(best_model_dir, "checkpoint"))
                    print(f"  [Checkpoint] New best arrival {rolling_arrival:.1f}% at ep {episode_counter}")

            metrics_csv = os.path.join(experiment.EXPERIMENT_PATH, "episode_metrics.csv")
            if episode_counter == 1:
                with open(metrics_csv, "w", encoding="utf-8") as f:
                    f.write("episode,reward,arrival_pct,collision\n")
            with open(metrics_csv, "a", encoding="utf-8") as f:
                f.write(
                    f"{episode_counter},{float(episode_rewards):.6f},"
                    f"{float(arrival_rate):.6f},{1 if crashed else 0}\n"
                )

            # Build bootstrap observations for terminal-value estimation
            env.reset()
            all_states = env.env.current_state  # (6, 4)
            zero_gm = np.zeros(experiment.EMBEDDING_SIZE, dtype=np.float32)

            from src.training.episode_utils import (
                _build_local_master_input,
                _build_global_master_input,
            )
            lm1_input = _build_local_master_input(
                zero_gm, all_states[0:experiment.AGENTS_PER_LOCAL_MASTER]
            )
            lm2_input = _build_local_master_input(
                zero_gm,
                all_states[experiment.AGENTS_PER_LOCAL_MASTER:experiment.CARS_AMOUNT],
            )
            lm1_emb, _, _ = master_model.get_proto_action(lm1_input)
            lm2_emb, _, _ = master_model.get_proto_action(lm2_input)
            gm_input = _build_global_master_input(lm1_emb, lm2_emb)

            # 8-D bootstrap obs for agents
            local_embeddings = (
                [lm1_emb] * experiment.AGENTS_PER_LOCAL_MASTER
                + [lm2_emb] * experiment.AGENTS_PER_LOCAL_MASTER
            )
            agent_obs_list = env.env.build_full_obs(local_embeddings)
            # One bootstrap obs per car — using only car 0 for all six corrupted GAE for agents 1–5.
            last_agent_obs_list = [
                np.asarray(agent_obs_list[i], dtype=np.float32).reshape(-1)
                for i in range(experiment.CARS_AMOUNT)
            ]

            # Training runs during warmup too — log_probs stored are from the
            # CURRENT NETWORK (not uniform random), so PPO importance ratios are
            # valid. Warmup only controls ACTION SELECTION (random vs policy).
            if episode_counter % experiment.EPISODE_AMOUNT_FOR_TRAIN == 0:
                # ── Entropy schedule: linear anneal + optional peak-lock ──────
                ent_start = getattr(experiment, 'ENT_COEF', 0.01)
                ent_end   = getattr(experiment, 'ENT_COEF_FINAL', ent_start)

                if peak_locked:
                    current_ent_coef = 0.0
                else:
                    # Check peak-lock trigger on rolling-20 arrival rate
                    if peak_threshold > 0.0:
                        recent = [v for v in results["arrival_rates"][-20:] if v is not None]
                        rolling_arr = float(np.mean(recent)) if recent else 0.0
                        if rolling_arr >= peak_threshold:
                            peak_locked = True
                            current_ent_coef = 0.0
                            print(
                                f"[Peak Lock] ep={episode_counter}  "
                                f"rolling-20 arrival={rolling_arr:.1f}% "
                                f">= {peak_threshold}%  → ent_coef locked at 0"
                            )
                        else:
                            total_episodes = experiment.EPISODES_PER_CYCLE * experiment.CYCLES
                            frac = min(1.0, episode_counter / max(1, total_episodes))
                            current_ent_coef = ent_start + (ent_end - ent_start) * frac
                    else:
                        # No peak-lock — plain linear anneal
                        total_episodes = experiment.EPISODES_PER_CYCLE * experiment.CYCLES
                        frac = min(1.0, episode_counter / max(1, total_episodes))
                        current_ent_coef = ent_start + (ent_end - ent_start) * frac

                perform_training_phase(
                    train_both=train_both,
                    training_local_master=training_local_master,
                    training_agent=training_agent,
                    training_global_master=training_global_master,
                    master_model=master_model,
                    agent_model=agent_model,
                    last_agent_obs_8d=last_agent_obs_list,
                    last_lm_obs_25d=lm1_input,
                    last_lm2_obs_25d=lm2_input,
                    last_gm_obs_25d=gm_input,
                    results=results,
                    ent_coef=current_ent_coef,
                    clip_range=getattr(experiment, 'CLIP_RANGE', 0.2),
                    vf_coef=getattr(experiment, 'VF_COEF', 0.5),
                    n_ppo_epochs=getattr(experiment, 'N_PPO_EPOCHS', 1),
                    n_value_epochs=getattr(experiment, 'N_VALUE_EPOCHS', 0),
                )

    print("Training completed.")
    return (
        agent_model, master_model, collision_counter,
        results["episode_rewards"], results["all_actions"], results,
        best_model_dir,
    )

#
# def training_loop(experiment, env, agent_model, master_model):
#     """
#     Main training loop that orchestrates cycles and episodes.
#     """
#
#     collision_counter, episode_counter, total_steps = 0, 0, 0
#
#     results = init_training_results()
#
#     for cycle_num in range(1, experiment.CYCLES + 1):
#         print('Cycle', cycle_num,'out of ', experiment.CYCLES)
#         train_both, training_master, training_agent = prepare_models_for_cycle(cycle_num, experiment.CYCLES,
#                                                                                master_model, agent_model)
#         for _ in range(experiment.EPISODES_PER_CYCLE):
#
#             episode_counter += 1
#             print('This is the ',episode_counter,'Out of',experiment.EPISODES_PER_CYCLE* experiment.CYCLES, 'episodes')
#             episode_rewards, actions, steps, crashed = process_episode(episode_counter, total_steps, env, master_model,
#                                                               agent_model, experiment, train_both, training_master)
#             if crashed:
#                 collision_counter += 1
#
#             total_steps += steps
#             results["episode_rewards"].append(episode_rewards)
#             results["all_actions"].append(actions)
#
#             # Prepare state for training
#             with torch.no_grad():
#                 _, _, _ = env.reset()
#                 full_state = env.env.current_state
#                 state_tensor = ensure_tensor(full_state)
#
#             if episode_counter % Experiment.EPISODE_AMOUNT_FOR_TRAIN == 0 :
#                 perform_training_phase(train_both, training_master, training_agent, master_model, agent_model, full_state,
#                                        state_tensor, results)
#
#     print("Training completed.")
#     return agent_model, master_model, collision_counter, results["episode_rewards"], results["all_actions"], results


##########################################
# Held-out test evaluation
##########################################

def run_held_out_test(experiment_config, env, agent_model, master_model,
                      best_model_dir: str, n_episodes: int = 100):
    """
    Load the best checkpoint saved during training and run n_episodes test
    episodes exclusively on the held-out scenarios (use_held_out_scenarios=True).

    Returns a dict with per-scenario and overall metrics.
    """
    from src.experiment.scenarios import HELD_OUT_SCENARIO_INDICES

    checkpoint_path = os.path.join(best_model_dir, "checkpoint")
    loaded = load_models(agent_model, master_model, checkpoint_path)
    if not loaded:
        print("[HeldOut] WARNING: could not load best checkpoint — using final weights.")

    # Tell the env to sample only from held-out scenarios.
    # Must set on the ACTUAL IntersectionEnv config (not the Driver wrapper config),
    # since IntersectionEnv._reset reads self.config which was created separately
    # by gym.make — modifying Driver.config would have no effect.
    _inner = env.env._get_unwrapped_env()
    _inner.config["use_held_out_scenarios"] = True
    print(f"[HeldOut] held-out mode ON — env id={id(_inner.config)}")

    # Initialise rollout buffers (needed by process_episode; training=False so no grad)
    rollout_buffers.clear()
    project_globals.local_master_rollout_buffers.clear()
    from gymnasium import spaces as _spaces
    for _ in range(experiment_config.CARS_AMOUNT):
        rollout_buffers.append(RolloutBuffer(
            buffer_size=experiment_config.N_STEPS,
            observation_space=_spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(experiment_config.STATE_INPUT_SIZE,),
            ),
            action_space=_spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            gamma=getattr(experiment_config, 'GAMMA', 0.99),
            gae_lambda=getattr(experiment_config, 'GAE_LAMBDA', 0.95),
            n_envs=1,
        ))
    for _ in range(experiment_config.NUM_LOCAL_MASTERS):
        project_globals.local_master_rollout_buffers.append(
            _make_master_buffer(experiment_config)
        )
    project_globals.global_master_rollout_buffer = _make_master_buffer(experiment_config)

    # Override warmup for test episodes — episode_counter must be > WARMUP_EPISODES
    # so get_action uses the loaded model instead of random actions.
    _orig_warmup = getattr(experiment_config, 'WARMUP_EPISODES', 0)
    experiment_config.WARMUP_EPISODES = 0

    arrivals, collisions = [], []
    for ep in range(1, n_episodes + 1):
        _, _, _, crashed, arrival_rate = process_episode(
            ep, 0, env, master_model, agent_model,
            experiment_config,
            train_both=False,
            training_local_master=False,
            training_agent=False,
            training_global_master=False,
        )
        arrivals.append(float(arrival_rate))
        collisions.append(1 if crashed else 0)

    experiment_config.WARMUP_EPISODES = _orig_warmup  # restore
    _inner.config["use_held_out_scenarios"] = False   # restore

    overall_arrival = float(np.mean(arrivals))
    overall_crash_rate = float(np.mean(collisions)) * 100.0
    print(
        f"[HeldOut] {n_episodes} eps | arrival={overall_arrival:.1f}% "
        f"| crash_rate={overall_crash_rate:.1f}%"
    )
    return {
        "held_out_arrival_pct": overall_arrival,
        "held_out_crash_rate_pct": overall_crash_rate,
        "held_out_n_episodes": n_episodes,
        "held_out_n_crashes": int(sum(collisions)),
    }


##########################################
# Evaluation
##########################################

def run_evaluation(experiment_config, env, agent_model, master_model=None):
    """Run evaluation episodes using the full hierarchical inference path."""
    eval_rewards = []
    eval_actions = []
    eval_episodes = getattr(experiment_config, 'EVAL_EPISODES', 5)

    # Initialise rollout buffers so process_episode can fill them (they are
    # discarded afterwards — no training phase is triggered).
    rollout_buffers.clear()
    project_globals.local_master_rollout_buffers.clear()
    for _ in env.env.config["controlled_cars"]:
        rollout_buffers.append(RolloutBuffer(
            buffer_size=experiment_config.N_STEPS,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(experiment_config.STATE_INPUT_SIZE,),
            ),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            gamma=getattr(experiment_config, 'GAMMA', 0.99),
            gae_lambda=getattr(experiment_config, 'GAE_LAMBDA', 0.95),
            n_envs=1,
        ))
    for _ in range(experiment_config.NUM_LOCAL_MASTERS):
        project_globals.local_master_rollout_buffers.append(
            _make_master_buffer(experiment_config)
        )
    project_globals.global_master_rollout_buffer = _make_master_buffer(experiment_config)

    for episode in range(eval_episodes):
        print(f"Evaluation episode {episode + 1}/{eval_episodes}")
        ep_reward, actions, _, crashed, arrival_rate = process_episode(
            episode + 1, 0, env, master_model, agent_model,
            experiment_config,
            train_both=False,
            training_local_master=False,
            training_agent=False,
            training_global_master=False,
        )
        eval_rewards.append(ep_reward)
        eval_actions.append(actions)
        print(f"  Reward: {ep_reward:.4f}  Arrived: {arrival_rate:.1f}%  "
              f"Crashed: {crashed}")

    print(f"Average evaluation reward: {np.mean(eval_rewards):.4f}")
    return eval_rewards, eval_actions


##########################################
# Inference & Training Mode Handlers
##########################################

def run_inference_mode(experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger):
    """Run inference episodes only."""
    if experiment_config.LOAD_PREVIOUS_WEIGHT and experiment_config.LOAD_MODEL_DIRECTORY:
        loaded = load_models(agent_model, master_model, experiment_config.LOAD_MODEL_DIRECTORY)
        if loaded:
            print("Models will be trained from loaded weights!")
        else:
            print("Starting inference with untrained models.")
    else:
        print("Starting inference with untrained models (No previous weights loaded)")

    eval_rewards, eval_actions = run_evaluation(experiment_config, wrapped_env, agent_model, master_model)
    close_everything(wrapped_env, agent_logger, master_logger)
    return agent_model, master_model, 0


def run_training_mode(experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger):
    """Run the training loop and handle saving/logging."""
    agent_model, master_model, collision_counter, all_rewards, all_actions, training_results = (
        training_loop(experiment=experiment_config, env=wrapped_env, agent_model=agent_model, master_model=master_model))
    save_models(agent_model, master_model, experiment_config.SAVE_MODEL_DIRECTORY)
    plot_training_results(experiment_config, training_results, show_plots=True)
    #log_training_results_to_neptune(experiment_config.logger, training_results)
    print("Training completed.")
    print("Total collisions:", collision_counter)
    #close_everything(wrapped_env, agent_logger, master_logger)
    return agent_model, master_model, collision_counter


##########################################
# Main experiment entrypoint
##########################################

def run_experiment(experiment_config, env_config):
    print(
        f"Environment configuration: {len(env_config['controlled_cars'])} controlled cars, {len(env_config['static_cars'])} static cars"
    )
    setup_experiment_dirs(experiment_config.EXPERIMENT_PATH)
    master_model, agent_model, wrapped_env = initialize_models(experiment_config, env_config)
    agent_logger, master_logger = setup_loggers(experiment_config.EXPERIMENT_PATH)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    if experiment_config.ONLY_INFERENCE:
        print("Running in inference-only mode")
        return run_inference_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )
    else:
        print("Running in training mode")
        return run_training_mode(
            experiment_config, wrapped_env, agent_model, master_model, agent_logger, master_logger
        )
