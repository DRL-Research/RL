from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from experiment.experiment_constants import CarName
from utils.model.model_handler import Model
from utils.agent_handler import Agent
from utils.airsim_manager import AirsimManager
from utils.plotting_utils import PlottingUtils
from master_handler import MasterModel
import torch

def training_loop(experiment, env, agent, master_model, agent_model):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")

        if cycle % 2 == 0:
            # Master Network Training
            print("Training Master Network (Agent Network is frozen)")
            master_model.unfreeze()
            agent_model.policy.set_training_mode(False)  # Freeze PPO by toggling training mode

            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                print(f"  @ Episode {episode_counter} (Master Training) @")
                episode_rewards, episode_actions, steps = run_episode(experiment,total_steps,
                    env, agent, master_model, agent_model, training_master=True
                )
                total_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)

        else:
            # Agent Network Training
            print("Training Agent Network (Master Network is frozen)")
            master_model.freeze()
            agent_model.policy.set_training_mode(True)  # Unfreeze PPO by toggling training mode

            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                print(f"  @ Episode {episode_counter} (Agent Training) @")
                episode_rewards, episode_actions, steps = run_episode(
                    env, agent, master_model, agent_model, training_master=False
                )
                total_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)

    print("Training completed.")
    return agent_model, collision_counter, all_rewards, all_actions


def run_episode(experiment,total_steps,env, agent, master_model, agent_model, training_master):
    '''
    Run one episode of the simulation.
    :param env: The environment instance.
    :param agent: The agent instance.
    :param master_model: The master network model.
    :param agent_model: The agent network model.
    :param training_master: Whether the master network is being trained.
    :return: Episode rewards, actions, and steps.
    '''
    # Retrieve states
    state_car1 = env.envs[0].airsim_manager.get_car1_state()
    state_car2 = env.envs[0].airsim_manager.get_car2_state()

    # Concatenate states for master network input
    master_input = torch.tensor(
        [state_car1.tolist() + state_car2.tolist()], dtype=torch.float32
    )

    done = False
    episode_sum_of_rewards, steps_counter = 0, 0
    actions_per_episode = []

    resume_experiment_simulation(env)

    while not done:
        steps_counter += 1

        # Retrieve states of both cars
        state_car1 = env.envs[0].airsim_manager.get_car1_state()
        state_car2 = env.envs[0].airsim_manager.get_car2_state()

        # Concatenate states for master network input
        master_input = torch.tensor(
            [state_car1.tolist() + state_car2.tolist()], dtype=torch.float32
        )

        # Get embedding from master network
        environment_embedding = master_model.inference(master_input).squeeze(0)

        # Combine car1 state with environment embedding for agent input
        state_car1 = np.concatenate((state_car1, environment_embedding))



        # Determine action based on which network is being trained
        if training_master:
            action = master_model.inference(master_input)
        else:
            action = agent.get_action(agent_model, state_car1, total_steps,exploration_threshold=experiment.EXPLORATION_EXPLOTATION_THRESHOLD)

        # Step in environment
        _, reward, done, _ = env.step(action)
        episode_sum_of_rewards += reward
        actions_per_episode.append(action)

        if done:
            pause_experiment_simulation(env)
            break

    return episode_sum_of_rewards, actions_per_episode, steps_counter


def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def run_experiment(experiment_config):
    # Initialize MasterModel and Agent model
    master_model = MasterModel(input_size=16, embedding_size=8, learning_rate=1e-3)

    # Initialize AirSim manager
    airsim_manager = AirsimManager(experiment_config)

    # Initialize Agent model
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager, master_model)])
    agent_model = Model(env, experiment_config).model

    # Configure logger
    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(logger)

    # Run the training loop
    agent_model, collision_counter, all_rewards, all_actions = training_loop(
        experiment=experiment_config,
        env=env,
        agent=Agent(experiment_config, airsim_manager, master_model),
        master_model=master_model,
        agent_model=agent_model  # This line fixes the error
    )

    # Save the final trained models
    master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")
    agent_model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    logger.close()

    print("Models saved")
    print("Total collisions:", collision_counter)

    # Plot results
    if not experiment_config.ONLY_INFERENCE:
        plot_results(experiment=experiment_config, all_rewards=all_rewards, all_actions=all_actions)


# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
