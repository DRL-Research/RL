import os

import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from src.model_handler import Model

from src.master_handler import *
from agent_handler import Agent
from src.airsim_manager import AirsimManager


def inference_loop(p_agent_loss, p_master_loss, p_episode_counter, experiment, env, agent_model, master_model, agent):
    """
    inference loop for a two-vehicle setup.
    In ONLY_INFERENCE mode:
      - agent_model is expected to be a dictionary with keys 'agent1' and 'agent2',
        each holding a loaded agent model.
      - Each vehicle builds its own observation (its local state concatenated with the proto embedding)
        and uses its corresponding model.
    In training mode, a single agent model is used for both vehicles.
    """
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []
    for episode in range(experiment.EPISODES_PER_CYCLE * experiment.CYCLES):
        episode_counter += 1
        p_episode_counter.append(episode_counter)
        print(f"  @ Episode {episode_counter} INFERENCE @")
        # Run an inference episode that uses separate agent models for each vehicle.
        episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, episode_states = run_episode_inference(
            env, agent_model['agent1'], agent_model['agent2'], master_model, experiment
        )
        total_steps += steps
        all_rewards.append(episode_rewards)
        all_actions.append(episode_actions)
    print("Inference completed.")
    return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions


def run_episode_inference(env, agent1_model, agent2_model, master_model, experiment):
    """
    Runs one inference episode for both vehicles.

    Each vehicle:
      - Retrieves its local 4-dim state.
      - The master network receives the concatenated local states (8-dim) and produces a 4-dim proto embedding.
      - Each vehicle’s observation is built by concatenating its own local state (4-dim) with the proto embedding (4-dim).
      - Each loaded agent model is used (deterministically) to choose its action.

    Returns:
      episode_sum: Total reward over the episode.
      actions_list: List of tuples (action1, action2) for each step.
      steps_counter: Number of steps.
      p_agent_loss: (Empty list in inference mode)
      p_master_loss: (Empty list in inference mode)
      all_states: List of master inputs from each step.
    """
    episode_sum, steps_counter = 0, 0
    actions_list, all_states, p_agent_loss, p_master_loss = [], [], [], []
    resume_experiment_simulation(env)
    while True:
        steps_counter += 1
        # Retrieve local states for Car1 and Car2 (each 4-dim)
        state_car1 = env.envs[0].airsim_manager.get_car1_state()
        state_car2 = env.envs[0].airsim_manager.get_car2_state()
        # Build master input: concatenated local states (8-dim vector)
        master_input = torch.tensor(np.concatenate((state_car1, state_car2))[np.newaxis, :], dtype=torch.float32)
        # Get proto embedding (4-dim) from master network
        proto_embedding = master_model.get_proto_action(master_input)
        # Build separate observations for each vehicle:
        obs_agent1 = np.concatenate((state_car1, proto_embedding))
        obs_agent2 = np.concatenate((state_car2, proto_embedding))
        # Get actions using Agent's static get_action method (set total_steps high for deterministic mode)
        action1 = Agent.get_action(agent1_model, obs_agent1, 100000, experiment.EXPLORATION_EXPLOTATION_THRESHOLD,
                                   False)
        action2 = Agent.get_action(agent2_model, obs_agent2, 100000, experiment.EXPLORATION_EXPLOTATION_THRESHOLD,
                                   False)
        print("agent 1 action :", action1)
        print("agent 2 action :", action2)
        print("proto action :", proto_embedding)
        combined_action = (action1, action2)
        # Step the environment using the tuple of actions
        next_state, reward, done, info = env.step(combined_action)
        all_states.append(master_input)
        episode_sum += reward
        actions_list.append(combined_action)
        if done:
            pause_experiment_simulation(env)
            break

    return episode_sum, actions_list, steps_counter, p_agent_loss, p_master_loss, all_states


# -------------------------------
# Main Experiment Runner
# -------------------------------
def run_inference(experiment_config):
    p_agent_loss = []
    p_master_loss = []
    p_episode_counter = []
    # Initialize AirSim Manager
    airsim_manager = AirsimManager(experiment_config)
    # Initialize Master Model and load pre-trained weights
    master_model = MasterModel(embedding_size=experiment_config.EMBEDDING_SIZE, airsim_manager=airsim_manager,
                               experiment=experiment_config)
    master_model.load(experiment_config.MASTER_TRAINED_MODEL)
    # Create two separate Agent instances (one for each car) with the same master model
    agent1 = Agent(experiment_config, airsim_manager, master_model)
    # Create the shared environment
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
    # Load separate pre-trained agent models for each vehicle
    agent1_model = Model.load(experiment_config.AGENT_TRAINED_MODEL, env,
                              experiment_config).model
    agent2_model = Model.load(experiment_config.AGENT_TRAINED_MODEL, env,
                              experiment_config).model
    # Configure loggers
    base_path = experiment_config.EXPERIMENT_PATH
    agent_logger_path = os.path.join(base_path, "agent_logs")
    agent_logger = configure(agent_logger_path, ["stdout", "csv", "tensorboard"])
    master_logger_path = os.path.join(base_path, "master_logs")
    master_model_logger = configure(master_logger_path, ["stdout", "csv", "tensorboard"])
    agent1_model.set_logger(agent_logger)
    master_model.set_logger(master_model_logger)
    # Create a dictionary for the agent models
    agent_models = {'agent1': agent1_model, 'agent2': agent2_model}
    # Run the inference loop; note that the inference branch uses the two-agent-model version.
    # all arrays are passing from and to functions, marked with p (ie p_episode_counter)
    p_episode_counter, p_agent_loss, p_master_loss, _, collision_counter, all_rewards, all_actions = inference_loop(
        p_agent_loss, p_master_loss, p_episode_counter,
        experiment=experiment_config,
        env=env,
        agent_model=agent_models,  # dictionary with keys 'agent1' and 'agent2'
        master_model=master_model,
        agent=agent1  # Pass one of the agent instances; the env is shared.
    )
    print(p_master_loss)
    # Optionally re-save the loaded models
    agent1_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_agent1_inference")
    agent2_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_agent2_inference")
    master_model.save(experiment_config.SAVE_MODEL_DIRECTORY + "_master_inference")
    agent_logger.close()
    master_model_logger.close()
    print("Models loaded and inference completed")
    print("Total collisions:", collision_counter)


# -------------------------------
# Override GYM Environment Functions
# -------------------------------
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()


def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
