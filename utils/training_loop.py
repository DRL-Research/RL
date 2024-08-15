from utils.NN_utils import NN_handler
from utils.airsim_manager import AirsimManager
from utils.logger import Logger
from utils.rl import RL
from stable_baselines3 import PPO, DQN
from gym_enviroment import AirSimGymEnv
import airsim_manager
from torch.utils.tensorboard import SummaryWriter
import gym_enviroment
import time
def training_loop_ariel(config):
    ''' Training loop based on gym and stable baselines3 - changed by Ariel Harush Aug2024 '''
    # Classes initialization
    logger = Logger(config)
    airsim_manager = AirsimManager(config)
    nn_handler = NN_handler(config)
    rl = RL(config, logger, airsim_manager, nn_handler)
    env = AirSimGymEnv(config, airsim_manager)
    model = PPO('MlpPolicy', env, verbose=1)

    if config.ONLY_INFERENCE:
        rl.network = nn_handler.load_weights_to_network(rl.network) # TODO: optimize the network from 8 inputs (at the training process) to 4 inputs at ONLY_INFERENCE state
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    # Main loop for each episode
    for episode in range(config.MAX_EPISODES):
        episode_counter += 1
        print(f"Episode {episode_counter}/{config.MAX_EPISODES}")
        episode_sum_of_rewards = 0
        rl.current_trajectory = []
        # Reset the environment and extract the observation
        obs, _ = env.reset()
        # loop for each step in an episode
        for step in range(config.MAX_STEPS):
            steps_counter += 1
            print(f"Step {step + 1}/{config.MAX_STEPS} in Episode {episode_counter}")
            # step using the SB Model (ppo)
            action, _states = model.predict(obs, deterministic=False)
            # Take a step in the environment
            #TODO: check differnces between next_obs and obs- maybe with plot (continus "EDA")
            next_obs, reward, done, truncated, info = env.step(action)
            episode_sum_of_rewards += reward
            #printing for debug
            print(f"Action taken: {action}, Reward: {reward}", f"Episode sum of rewards: {episode_sum_of_rewards}")
            # Add current step to trajectory
            rl.current_trajectory.append((obs, action, reward, next_obs, done))
            # Update the observation
            obs = next_obs
            # Check for done conditions
            if done or truncated:
                airsim_manager.reset_cars_to_initial_positions()
                logger.log_scaler("episode_sum_of_rewards", episode_counter, episode_sum_of_rewards)
                if info.get("collision_occurred"):
                    collision_counter += 1
                break
            if airsim_manager.collision_occurred():
                collision_counter += 1
                break
            elif airsim_manager.has_reached_target(env.state):
                break
        # Update epsilon for epsilon-greedy
        rl.updateEpsilon()
        print(f"epsilon value: {rl.epsilon}")
        # Store the trajectory in memory
        rl.memory.append(rl.current_trajectory)
        # Handle trajectory-based training
        if config.TRAIN_OPTION == 'trajectory' and not config.ONLY_INFERENCE:
            trajectory_loss = rl.replay()
            rl.update_target_model()
            # Log the trajectory loss if applicable
            if trajectory_loss is not None:
                logger.log_scaler("loss_per_trajectory", episode_counter, trajectory_loss)
            if config.LOG_SAME_ACTION_SELECTED_IN_TRAJECTORY:
                logger.log_same_action_selected_in_trajectory()
    # Save the PPO model after the experiment
    nn_handler.save_network_weights(model)
    # Reset cars to initial settings file position before the next experiment
    airsim_manager.reset_cars_to_initial_settings_file_positions()
    print("---- Experiment Ended ----")
    print("The amount of collisions:")
    print(collision_counter)

def training_loop_ariel_step(config):
    env = AirSimGymEnv(config, AirsimManager(config))
    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, batch_size=64)
    if config.ONLY_INFERENCE:
        model = PPO.load(config.LOAD_WEIGHT_DIRECTORY)
        print("Loaded weights for inference.")
    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    for episode in range(config.MAX_EPISODES):
        print(f"@ Episode {episode + 1} @")
        obs = env.reset()
        done = False
        episode_sum_of_rewards = 0
        episode_counter += 1
        while not done:
            print(f"Before step: Done={done}, State={obs}")
            action, _ = model.predict(obs, deterministic=True)
            print(f"Action: {action}")
            obs, reward, done, _ = env.step(action)
            print(f"After step: Done={done}, State={obs}")
            steps_counter += 1
            episode_sum_of_rewards += reward
            print('episode counter:', episode_counter)
            if done:
                print(
                    '***********************************************************************************************************************************************')
                print(f"Episode {episode_counter} done. Resetting cars...")
                break

        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        if not config.ONLY_INFERENCE:
            model.learn(total_timesteps=config.MAX_STEPS)
            model.save(config.SAVE_WEIGHT_DIRECTORY)
            print(f"Weights saved to {config.SAVE_WEIGHT_DIRECTORY}")
    print("Training or inference finished.")
    print("Total collisions:", collision_counter)
