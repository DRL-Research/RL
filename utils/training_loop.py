
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_enviroment import AirSimGymEnv
from airsim_manager import AirsimManager
from stable_baselines3.common.logger import configure
from plotting_utils import PlottingUtils


def run_experiment(experiment, path):
    all_rewards = []
    new_logger = configure(path, ["stdout", "csv", "tensorboard"])
    env = DummyVecEnv([lambda: AirSimGymEnv(experiment, AirsimManager(experiment))])
    model = PPO('MlpPolicy', env, verbose=1,
                learning_rate=experiment.LEARNING_RATE,
                n_steps=experiment.N_STEPS,
                batch_size=experiment.BATCH_SIZE)

    if experiment.ONLY_INFERENCE:
        model.set_logger(new_logger)
        model = PPO.load(experiment.LOAD_WEIGHT_DIRECTORY)
        print("Loaded weights for inference.")

    collision_counter = 0
    episode_counter = 0
    total_steps = 0

    for episode in range(experiment.MAX_EPISODES):
        print(f"@ Episode {episode + 1} @")
        current_state = env.reset()
        done = False
        episode_sum_of_rewards = 0
        episode_counter += 1
        steps_counter = 0
        env.envs[0].resume_simulation()
        while not done:
            steps_counter += 1
            total_steps += 1
            if not experiment.ONLY_INFERENCE:
                if total_steps > experiment.EXPLORATION_EXPLOTATION_THRESHOLD:
                    action, _ = model.predict(current_state, deterministic=True)
                    #print('Deterministic = True , ' , action)
                elif total_steps < experiment.EXPLORATION_EXPLOTATION_THRESHOLD:
                    action, _ = model.predict(current_state, deterministic=False)
                    #print('Deterministic = False ,', action)
                #print(f"Action: {action}")
            elif experiment.ONLY_INFERENCE:
                action, _ = model.predict(current_state, deterministic=True)
                #print('ONLY INFERNCE ' ,action)
            current_state, reward, done, _ = env.step(action)
            episode_sum_of_rewards+=reward
            if reward < experiment.COLLISION_REWARD or done:
                env.envs[0].pause_simulation()
                episode_sum_of_rewards += reward
                if reward <=-20:
                    collision_counter +=1
                    print('********* collision ***********')
                break
        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        all_rewards.append(episode_sum_of_rewards)
        print('Total Steps : ', total_steps , "Episode steps: ", steps_counter)
        if not experiment.ONLY_INFERENCE:
            model.learn(total_timesteps=steps_counter)
            print('Model learned on ', steps_counter , 'steps ')
        env.envs[0].resume_simulation()
    model.save(path + '/model')
    new_logger.close()
    print('Model saved')
    print("Total collisions:", collision_counter)
    if not experiment.ONLY_INFERENCE:
        PlottingUtils.plot_losses(path)
        PlottingUtils.plot_rewards(all_rewards)
        PlottingUtils.show_plots()
