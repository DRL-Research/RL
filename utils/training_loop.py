from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.airsim_manager import AirsimManager
from utils.gym_enviroment import AirSimGymEnv
from utils.plotting_utils import PlottingUtils


def run_experiment(experiment):
    all_rewards = []
    logger = configure(experiment.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    env = DummyVecEnv([lambda: AirSimGymEnv(experiment, AirsimManager(experiment))])

    model = PPO('MlpPolicy', env, verbose=1,
                learning_rate=experiment.LEARNING_RATE,
                n_steps=experiment.N_STEPS,
                batch_size=experiment.BATCH_SIZE)

    if experiment.ONLY_INFERENCE:
        model.set_logger(logger)
        model = PPO.load(experiment.LOAD_WEIGHT_DIRECTORY)
        print(f"Loaded weights from {experiment.LOAD_WEIGHT_DIRECTORY} for inference.")

    collision_counter, episode_counter, total_steps = 0, 0, 0

    for episode in range(experiment.EPOCHS):
        print(f"@ Episode {episode + 1} @")
        current_state = env.reset()
        done = False
        episode_sum_of_rewards, steps_counter = 0, 0
        episode_counter += 1
        # TODO: make it transparent -> ___.resume_simulation()
        env.envs[0].resume_simulation()
        while not done:
            steps_counter += 1
            total_steps += 1
            if not experiment.ONLY_INFERENCE:
                if total_steps > experiment.EXPLORATION_EXPLOTATION_THRESHOLD:
                    # TODO: Create function -> action, _ = Agent.get_action
                    action, _ = model.predict(current_state, deterministic=True)
                    # print('Deterministic = True , ' , action)
                elif total_steps < experiment.EXPLORATION_EXPLOTATION_THRESHOLD:
                    action, _ = model.predict(current_state, deterministic=False)
                    # print('Deterministic = False ,', action)
                # print(f"Action: {action}")
            elif experiment.ONLY_INFERENCE:  # TODO: why elif and not else?
                action, _ = model.predict(current_state, deterministic=True)
                # print('ONLY INFERENCE ' ,action)
            current_state, reward, done, _ = env.step(action)
            episode_sum_of_rewards += reward

            if reward < experiment.COLLISION_REWARD or done:
                env.envs[0].pause_simulation()
                episode_sum_of_rewards += reward
                if reward <= -20:
                    collision_counter += 1
                    print('********* collision ***********')
                break
            # TODO: consider the following code:
            # if reward == experiment.COLLISION_REWARD or done:
            #     env.envs[0].pause_simulation()
            #     episode_sum_of_rewards += reward
            #     if reward == experiment.COLLISION_REWARD:
            #         collision_counter += 1
            #         print('********* collision ***********')
            #     break

        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        print(f"Total Steps: {total_steps}, Episode steps: {steps_counter}")
        all_rewards.append(episode_sum_of_rewards)
        if not experiment.ONLY_INFERENCE:
            model.learn(total_timesteps=steps_counter)
            print(f"Model learned on {steps_counter} steps")
        env.envs[0].resume_simulation()

    model.save(experiment.SAVE_MODEL_DIRECTORY)
    logger.close()
    print('Model saved')
    print("Total collisions:", collision_counter)

    if not experiment.ONLY_INFERENCE:
        # TODO: maybe this should be a notebook/other py file
        # PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)  # TODO: Where do you write to this file?
        PlottingUtils.plot_rewards(all_rewards)
        PlottingUtils.show_plots()
