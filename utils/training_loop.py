from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_enviroment import AirSimGymEnv
from airsim_manager import AirsimManager
from stable_baselines3.common.logger import configure
from plotting_utils import PlottingUtils


def model_training(config, path):
    all_rewards = []
    new_logger = configure(path, ["stdout", "csv", "tensorboard"])
    env = DummyVecEnv([lambda: AirSimGymEnv(config, AirsimManager(config))])

    model = PPO('MlpPolicy', env, verbose=1,
                learning_rate=config.LEARNING_RATE,
                n_steps=config.N_STEPS,
                batch_size=config.BATCH_SIZE)

    model.set_logger(new_logger)
    if config.ONLY_INFERENCE:
        model = PPO.load(config.LOAD_WEIGHT_DIRECTORY)
        print("Loaded weights for inference.")

    collision_counter = 0
    episode_counter = 0
    steps_counter = 0
    total_steps = 0

    for episode in range(config.MAX_EPISODES):
        print(f"@ Episode {episode + 1} @")
        # the next line place car2 in random position
        env.envs[0].airsim_manager.set_car2_initial_position_and_yaw()
        env.envs[0].airsim_manager.reset_cars_to_initial_positions()
        obs = env.reset()
        done = False
        episode_sum_of_rewards = 0
        episode_counter += 1
        while not done:
            steps_counter += 1
            total_steps += 1
            if not config.ONLY_INFERENCE:
                if total_steps > config.EXPLORATION_EXPLOTATION_THRESHOLD:
                    action, _ = model.predict(obs, deterministic=True)
                    print('Deterministic = True , ' , action)
                elif total_steps < config.EXPLORATION_EXPLOTATION_THRESHOLD:
                    action, _ = model.predict(obs, deterministic=False)
                    print('Deterministic = False ,', action)
                #print(f"Action: {action}")
            elif config.ONLY_INFERENCE:
                action, _ = model.predict(obs, deterministic=True)
                print('ONLY INFERNCE ' ,action)
            obs, reward, done, _ = env.step(action)
            episode_sum_of_rewards+=reward
            if reward < -20 or done:
                env.envs[0].pause_simulation()
                episode_sum_of_rewards += reward
                if reward <=-20:
                    collision_counter +=1
                    print('********* collision ***********')
                break
        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        all_rewards.append(episode_sum_of_rewards)
        print('Total Steps : ', total_steps , "Episode steps: ", steps_counter)
        if not config.ONLY_INFERENCE:
            model.learn(total_timesteps=steps_counter)
            print('Model learned on ', steps_counter , 'steps ')
        steps_counter = 0
        env.envs[0].resume_simulation()
    model.save(path + '/model')
    new_logger.close()
    print('Model saved')
    print("Total collisions:", collision_counter)
    PlottingUtils.plot_losses(path)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
