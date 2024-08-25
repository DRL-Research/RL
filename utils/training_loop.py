import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_enviroment import AirSimGymEnv
from airsim_manager import AirsimManager
from stable_baselines3.common.logger import configure
import pandas as pd
from config import Config
import random

def training_loop_ariel_step(config, path):
    all_rewards = []
    new_logger = configure(path, ["stdout", "csv", "tensorboard"])
    env = DummyVecEnv([lambda: AirSimGymEnv(config, AirsimManager(config))])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001, n_steps=160, batch_size=160)
    #model = PPO('MlpPolicy', env, verbose=1)
    model.set_logger(new_logger)

    def train_model():
        total_timesteps = 160
        print('Model learning')

        model.learn(total_timesteps=total_timesteps)
        print('Model learned ************')
        env.envs[0].resume_simulation()

    if config.ONLY_INFERENCE:
        model = PPO.load(config.LOAD_WEIGHT_DIRECTORY)
        print("Loaded weights for inference.")

    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    total_steps = 0

    for episode in range(config.MAX_EPISODES):
        print(f"@ Episode {episode + 1} @")
        car2_side = random.choice(["left", "right"])
        if car2_side == "left":
            config.CAR2_INITIAL_POSITION = [0, -30]
            config.CAR2_INITIAL_YAW = 90
        else:
            config.CAR2_INITIAL_POSITION = [0, 30]
            config.CAR2_INITIAL_YAW = 270

        print(f"Car 2 starts from {car2_side} with position {config.CAR2_INITIAL_POSITION} and yaw {config.CAR2_INITIAL_YAW}")


        env.envs[0].airsim_manager.reset_cars_to_initial_positions()

        obs = env.reset()
        done = False
        episode_sum_of_rewards = 0
        episode_counter += 1

        while not done:
            # print(f"Before step: Done={done}, State={obs}")
            if not config.ONLY_INFERENCE:
                if total_steps > 2500:
                    action, _ = model.predict(obs, deterministic=True)
                elif total_steps < 2500:
                    action, _ = model.predict(obs, deterministic=False)
                print(f"Action: {action}")
            elif config.ONLY_INFERENCE:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            #print(f"After step: Done={done}, State={obs}")
            steps_counter += 1
            total_steps += 1
            episode_sum_of_rewards += reward
            #print('Step:', steps_counter)
            #print('Reward:', reward)
            if done:
                if not config.ONLY_INFERENCE:
                    env.envs[0].pause_simulation()
                    model.learn(total_timesteps=steps_counter)
                    env.envs[0].resume_simulation()
                print('***********************************************************************************************************************************************')
                if env.get_attr('airsim_manager')[0].collision_occurred():
                    collision_counter += 1
                break
        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        all_rewards.append(episode_sum_of_rewards)
        steps_counter = 0
    model.save(path + '/model')
    print('Model saved')
    print("Total collisions:", collision_counter)
    loss_data = pd.read_csv(f"{path}/progress.csv")
    losses = loss_data["train/value_loss"]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(all_rewards, label='Cumulative Rewards')
    plt.title('Cumulative Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()
