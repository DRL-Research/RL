from stable_baselines3.common.logger import configure
from utils.model.model_handler_for_training import Model
from utils.agent_handler_for_training import Agent
from utils.airsim_manager_for_training import AirsimManager
from utils.plotting_utils_for_training import *
from master_handler_for_training import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def training_loop(p_agent_loss,p_master_loss,p_episode_counter,experiment, env, agent_model, master_model,agent,):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []
    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")
        if cycle % 2 == 1:
            # Master Network Training
            print("Training Master Network (Agent Network is frozen)")
            master_model.unfreeze()
            agent_model.policy.set_training_mode(False)  # Freeze PPO by toggling training mode]
            master_training_steps = 0
            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                p_episode_counter.append(episode_counter)
                print(f"  @ Episode {episode_counter} (Master Training) @")
                episode_rewards, episode_actions, steps,p_agent_loss,p_master_loss,episode_states = run_episode(p_master_loss,p_agent_loss,experiment,total_steps,
                    env, master_model, agent_model,agent, training_master=True)
                total_steps += steps
                master_training_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)
                if episode_counter % 4 == 0:
                    master_model.train_master(master_training_steps)
                    master_training_steps = 0

        if cycle % 2 == 0 and cycle != 0:
            # Agent Network Training
            print("Training Agent Network (Master Network is frozen)")
            master_model.freeze()
            agent_model.policy.set_training_mode(True)  # Unfreeze PPO by toggling training mode
            agent_training_steps=0
            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                p_episode_counter.append(episode_counter)
                print(f"  @ Episode {episode_counter} (Agent Training) @")
                episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, episode_states = run_episode(
                    p_master_loss, p_agent_loss, experiment, total_steps,
                    env, master_model, agent_model, agent, training_master=False)
                agent_training_steps += steps
                if episode % 5 == 0:
                    agent_model.learn(total_timesteps=agent_training_steps, log_interval=1)
                    agent_training_steps = 0
                total_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)

        if cycle == 0:
            # train master and agent together for the first cycle
            print("Training Master and Agent Networks together")
            master_model.unfreeze() # Unfreeze the master network
            agent_model.policy.set_training_mode(True)  # Unfreeze PPO by toggling training mode
            master_training_steps = 0
            agent_training_steps = 0
            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                p_episode_counter.append(episode_counter)
                print(f"  @ Episode {episode_counter} (Master and Agent Training)  @")
                episode_rewards, episode_actions, steps, p_agent_loss, p_master_loss, episode_states = run_episode(
                    p_master_loss, p_agent_loss, experiment, total_steps,
                    env, master_model, agent_model, agent, training_master=True)
                total_steps += steps
                master_training_steps += steps
                agent_training_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)
                if episode_counter % 4 == 0:
                    pause_experiment_simulation(env) # pause simulation to train master and agent network
                    master_model.train_master(master_training_steps)
                    agent_model.learn(total_timesteps=agent_training_steps, log_interval=1)
                    agent_training_steps = 0
                    master_training_steps = 0



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
        # Combine car1 state with environment embedding for agent input
        state_car1_with_embedding = np.concatenate((state_car1, environment_embedding))
        # Determine action for the agent
        agent_action = agent.get_action(
            agent_model,
            state_car1_with_embedding,
            total_steps,
            exploration_threshold=experiment.EXPLORATION_EXPLOTATION_THRESHOLD,training_master=training_master
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


def run_experiment(experiment_config):
    p_agent_loss=[]
    p_master_loss=[]
    p_episode_counter=[]
    # Initialize MasterModel and Agent model,Initialize AirSim manager and environment
    airsim_manager = AirsimManager(experiment_config)
    master_model = MasterModel(embedding_size=4,airsim_manager=airsim_manager,experiment=experiment_config)
    agent = Agent(experiment_config, airsim_manager,master_model)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager,master_model)])
    # Initialize Agent model
    agent_model = Model(env, experiment_config).model
    # Configure logger
    base_path = experiment_config.EXPERIMENT_PATH
    agent_logger_path = os.path.join(base_path, "agent_logs")
    agent_logger = configure(agent_logger_path, ["stdout", "csv", "tensorboard"])
    master_logger_path = os.path.join(base_path, "master_logs")
    master_model_logger = configure(master_logger_path, ["stdout", "csv", "tensorboard"])
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_model_logger)
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
        plot_results(path=agent_logger_path, all_rewards=all_rewards, all_actions=all_actions)
        plot_results(path=master_logger_path, all_rewards=all_rewards, all_actions=all_actions)
    # Save the final trained models
    agent_model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    # master_model.save_metrics(f"{experiment_config.EXPERIMENT_PATH}/master_metrics.csv")
    agent_logger.close()
    master_model_logger.close()
    master_model.save(f"{experiment_config.SAVE_MODEL_DIRECTORY}_master.pth")
    print("Models saved")
    print("Total collisions:", collision_counter)

# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
