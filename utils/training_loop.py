from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_enviroment import AirSimGymEnv  # TODO: fix typo
from airsim_manager import AirsimManager
from stable_baselines3.common.logger import configure
from plotting_utils import PlottingUtils


def run_experiment(experiment):

    # TODO: consider moving logger, env and model to experiment definition (they are not changing)
    logger = configure(experiment.SAVE_MODEL_DIRECTORY, ["stdout", "csv", "tensorboard"])
    env = DummyVecEnv([lambda: AirSimGymEnv(experiment, AirsimManager(experiment))])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=experiment.LEARNING_RATE, n_steps=experiment.N_STEPS,
                batch_size=experiment.BATCH_SIZE)
    model.set_logger(logger) # TODO: this line should be after only inference?

    if experiment.ONLY_INFERENCE:
        model = PPO.load(experiment.LOAD_WEIGHT_DIRECTORY)
        print(f"Loaded weights for inference, from {experiment.LOAD_WEIGHT_DIRECTORY}.")

    ####
    ####

    all_rewards = []
    collision_counter = 0
    episode_counter = 0
    total_steps = 0

    for episode in range(experiment.MAX_EPISODES):

        print(f"@ Episode {episode + 1} @")

        # Set cars in initial position
        # the next linse place car2 in random position
        # TODO: why do we need both lines?
        # TODO: what about setting initial position of car1 (up/down)?
        env.envs[0].airsim_manager.set_car2_initial_position_and_yaw()
        env.envs[0].airsim_manager.reset_cars_to_initial_positions()

        current_state = env.reset()
        done = False
        episode_sum_of_rewards = 0
        episode_counter += 1
        steps_counter = 0

        while not done:
            # choose action according to current_state
            if experiment.ONLY_INFERENCE:
                action, _ = model.predict(current_state, deterministic=True)
            elif not experiment.ONLY_INFERENCE:
                if total_steps > experiment.EXPLORATION_EXPLOTATION_THRESHOLD: # TODO: think about this exploration policy
                    action, _ = model.predict(current_state, deterministic=True)
                elif total_steps < experiment.EXPLORATION_EXPLOTATION_THRESHOLD:
                    action, _ = model.predict(current_state, deterministic=False)

            current_state, reward, done, _ = env.step(action)  # TODO: how this function works? what is the definition of done?

            steps_counter += 1
            total_steps += 1

            if reward == experiment.COLLISION_REWARD:  # TODO: is there a more generic way to do it?
                env.envs[0].pause_simulation()
                collision_counter += 1

            episode_sum_of_rewards += reward
            if done:
                # if env.envs[0].airsim_manager.collision_occurred():  # TODO: why is this commented?
                #     env.envs[0].pause_simulation()
                if not experiment.ONLY_INFERENCE:
                    model.learn(total_timesteps=steps_counter)
                break

        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        all_rewards.append(episode_sum_of_rewards)
        env.envs[0].resume_simulation()  # TODO: where is the pause? in the commented lines?


    model.save(experiment.SAVE_MODEL_DIRECTORY)
    logger.close()

    print('Model saved to ')
    print("Total collisions:", collision_counter)

    # TODO: where is plotting utils?
    PlottingUtils.plot_losses(experiment.SAVE_MODEL_DIRECTORY)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
