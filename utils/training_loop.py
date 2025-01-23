from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from experiment.experiment_constants import CarName
from utils.model.model_handler import Model
from utils.agent_handler import Agent
from utils.airsim_manager import AirsimManager
from utils.plotting_utils import PlottingUtils
from master_handler import *
import torch
import pandas as pd
import matplotlib.pyplot as plt

def training_loop(p_agent_loss,p_master_loss,p_episode_counter,experiment, env, agent_model, master_model,agent,):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")

        if cycle ==0:
            # Master Network Training
            print("Training Master Network (Agent Network is frozen)")
            master_model.unfreeze()
            agent_model.policy.set_training_mode(False)  # Freeze PPO by toggling training mode]
            episode_data = []
            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                p_episode_counter.append(episode_counter)
                print(f"  @ Episode {episode_counter} (Master Training) @")
                episode_rewards, episode_actions, steps,p_agent_loss,p_master_loss,episode_states = run_episode(p_master_loss,p_agent_loss,experiment,total_steps,
                    env, master_model, agent_model,agent, training_master=True
                )
                total_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)
                episode_data.append({
                    'states': episode_states,
                    'rewards': episode_rewards
                })
                if len(episode_data) >= 4:
                    master_episode_states = [ep['states'] for ep in episode_data]
                    master_episode_rewards = [ep['rewards'] for ep in
                                              episode_data]
                    loss = master_model.train_master(master_episode_states, master_episode_rewards)
                    p_master_loss.append(loss)
                    episode_data = []

        else:
            # Agent Network Training
            print("Training Agent Network (Master Network is frozen)")
            master_model.freeze()
            agent_model.policy.set_training_mode(True)  # Unfreeze PPO by toggling training mode
            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                p_episode_counter.append(episode_counter)
                print(f"  @ Episode {episode_counter} (Agent Training) @")
                episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, episode_states = run_episode(
                    p_master_loss, p_agent_loss, experiment, total_steps,
                    env, master_model, agent_model, agent, training_master=False
                    )
                if episode % 5 == 0:
                    agent_model.learn(total_timesteps=total_steps, log_interval=1)
                total_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)

    print("Training completed.")
    return p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions


def run_episode(p_master_loss,p_agent_loss,experiment,total_steps,env, master_model, agent_model,agent, training_master):
    '''
    Run one episode of the simulation.
    :param env: The environment instance.
    :param agent: The agent instance.
    :param master_model: The master network model.
    :param agent_model: The agent network model.
    :param training_master: Whether the master network is being trained.
    :return: Episode rewards, actions, and steps.
    '''
    all_states=[]
    all_rewards=[]
    done = False
    episode_sum_of_rewards, steps_counter = 0, 0
    actions_per_episode = []
    # Start/reset the environment
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

        # Get embedding from master network (squeeze make it 8 and not 1,8)
        environment_embedding = master_model.get_proto_action(master_input)
        # print('************************************************************')
        # print('environment_embedding',environment_embedding)
        # print('************************************************************')
        # Combine car1 state with environment embedding for agent input
        state_car1_with_embedding = np.concatenate((state_car1, environment_embedding))
        # print('************************************************************')
        # print('car1_state_after_embedding', state_car1_with_embedding)
        # print('************************************************************')
        # Determine action for the agent
        agent_action = agent.get_action(
            agent_model,
            state_car1_with_embedding,
            total_steps,
            exploration_threshold=experiment.EXPLORATION_EXPLOTATION_THRESHOLD,
        )
        # Step in the environment
        next_state, reward, done, info = env.step(agent_action)
        all_states.append(master_input)
        all_rewards.append(reward)
        # Update rewards and actions
        episode_sum_of_rewards += reward
        actions_per_episode.append(agent_action)
        # Check if the episode is done
        if done:
            all_rewards.append(reward)
            pause_experiment_simulation(env)
            if not training_master:
                print(steps_counter)
                if len(p_master_loss) != 0:
                    p_master_loss.append(p_master_loss[-1])
                else:
                    p_master_loss.append(0)
            break

    return episode_sum_of_rewards, actions_per_episode, steps_counter,p_agent_loss,p_master_loss,all_states

def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def run_experiment(experiment_config):
    p_total_rewards=[]
    p_agent_loss=[]
    p_master_loss=[]
    p_master_entropy=[]
    p_episode_counter=[]
    # Initialize MasterModel and Agent model
    master_model = MasterModel(input_size=8, embedding_size=4)

    # Initialize AirSim manager and env

    airsim_manager = AirsimManager(experiment_config)
    agent = Agent(experiment_config, airsim_manager,master_model)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager,master_model)])

    # Initialize Agent model
    agent_model = Model(env, experiment_config).model
    # for name, param in agent_model.policy.named_parameters():
    #     print(name, param.mean().item(), param.std().item())

    # Configure logger
    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(logger)

    #run training loop
    p_episode_counter, p_agent_loss, p_master_loss, agent_model, collision_counter, all_rewards, all_actions = training_loop(
        p_agent_loss, p_master_loss, p_episode_counter,
        experiment=experiment_config,
        env=env,
        agent_model=agent_model,
        master_model=master_model,
        agent=agent,
    )
    print(p_master_loss)
    # Plot results
    if not experiment_config.ONLY_INFERENCE:
        plot_results(experiment=experiment_config, all_rewards=all_rewards, all_actions=all_actions)
        plt.figure(figsize=(10, 6))
        plt.plot(p_master_loss)
        plt.title("Master Loss over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Master Loss")
        plt.legend()
        plt.grid()
        plt.show()

    # Save the final trained models
    agent_model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    # master_model.save_metrics(f"{experiment_config.EXPERIMENT_PATH}/master_metrics.csv")
    logger.close()
    #master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")

    print("Models saved")
    print("Total collisions:", collision_counter)



        # plt.figure(figsize=(10, 6))
        # plt.plot(p_episode_counter, p_agent_loss)
        # plt.title("Agent Loss over Episodes")
        # plt.xlabel("Episode")
        # plt.ylabel("Agent Loss")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.plot(p_episode_counter, p_total_rewards)
        # plt.title("Rewards over Episodes")
        # plt.xlabel("Episode")
        # plt.ylabel("Rewards Loss")
        # plt.legend()
        # plt.grid()
        # plt.show()

# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()

## dont mad- only for development and debugging
def plot_metrics(metrics_path):
    # Load metrics
    metrics = pd.read_csv(metrics_path)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["loss"], label="Loss")
    plt.title("Loss over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["entropy"], label="Entropy", color="orange")
    plt.title("Entropy over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Return
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["return"], label="Return", color="green")
    plt.title("Return over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.show()