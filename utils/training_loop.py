import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_enviroment import AirSimGymEnv
from airsim_manager import AirsimManager
from stable_baselines3.common.logger import configure
import pandas as pd
from config import Config

def training_loop_ariel_step(config,path):
    all_rewards = []
    new_logger = configure(path, ["stdout", "csv", "tensorboard"])
    env = DummyVecEnv([lambda: AirSimGymEnv(config, AirsimManager(config))])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001, n_steps=160, batch_size=160)
    model.set_logger(new_logger)
    def train_model():
        total_timesteps = 160
        print('Model learning')
        env.envs[0].pause_simulation()
        model.learn(total_timesteps=total_timesteps)
        print('Model learned ************')
        model.save(path + '/model')
        print('Model saved')
        env.envs[0].resume_simulation()
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
            if steps_counter > 500:
                action, _ = model.predict(obs, deterministic=True)
            elif steps_counter < 500:
                action, _ = model.predict(obs, deterministic=False)
            print(f"Action: {action}")
            obs, reward, done, _ = env.step(action)
            print(f"After step: Done={done}, State={obs}")
            steps_counter += 1
            episode_sum_of_rewards += reward
            print('Step:', steps_counter)
            print('Reward:', reward)
            if done:
                print(
                    '***********************************************************************************************************************************************')
                if env.get_attr('airsim_manager')[0].collision_occurred():
                    collision_counter += 1
                break
        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        all_rewards.append(episode_sum_of_rewards)
        if not config.ONLY_INFERENCE:
            train_model()
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

