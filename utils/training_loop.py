import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from utils.plotting_utils import PlottingUtils
from matplotlib import pyplot as plt



def training_loop(experiment, env, agent, model):
    if experiment.ONLY_INFERENCE:
        print('Only Inference')
        model.load(experiment.LOAD_WEIGHT_DIRECTORY)
        print(f"Loaded weights from {experiment.LOAD_MODEL_DIRECTORY} for inference.")
    else:
        if experiment.LOAD_PREVIOUS_WEIGHT:
            model.load(experiment.LOAD_MODEL_DIRECTORY)
            print(f"Loaded weights from {experiment.LOAD_MODEL_DIRECTORY}, the model will be trained from this point! ")
        collision_counter, episode_counter, total_steps = 0, 0, 0
        all_rewards, all_actions = [], []

        for episode in range(experiment.EPOCHS):
            print(f"@ Episode {episode + 1} @")
            current_state = env.reset()
            done = False
            episode_sum_of_rewards, steps_counter = 0, 0
            episode_counter += 1
            resume_experiment_simulation(env)
            actions_per_episode = []

            while not done:
                steps_counter += 1
                total_steps += 1
                action = agent.get_action(model, current_state, total_steps,
                                          experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
                #print(f"Action: {action[0]}")
                current_state, reward, done, _ = env.step(action)
                episode_sum_of_rewards += reward
                if reward < experiment.COLLISION_REWARD or done:
                    pause_experiment_simulation(env)
                    episode_sum_of_rewards += reward
                    if reward <= experiment.COLLISION_REWARD:
                        collision_counter += 1
                        print('********* collision ***********')
                    break

            print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
            print(f"Total Steps: {total_steps}, Episode steps: {steps_counter}")
            all_rewards.append(episode_sum_of_rewards)
            all_actions.append(actions_per_episode)

            if not experiment.ONLY_INFERENCE:
                model.learn(total_timesteps=steps_counter, log_interval=1)
                print(f"Model learned on {steps_counter} steps")

        resume_experiment_simulation(env)

        return model, collision_counter, all_rewards, all_actions


def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def run_experiment(experiment_config):
    #env = gym.make("highway-v0", config={"vehicles_count": 2, "duration": 5},render_mode='rgb_array')
    env = gym.make("intersection-v0", config={"vehicles_count": 1, "duration": 5,"initial_vehicle_count": 1,"destination": "o2"},render_mode='rgb_array')
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=experiment_config.LEARNING_RATE,
        n_steps=experiment_config.N_STEPS,
        batch_size=experiment_config.BATCH_SIZE,
        verbose=1,
    )

    # logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    # model.set_logger(logger)

    for episode in range(experiment_config.EPOCHS):
        state, _ = env.reset()  # Unpack the tuple
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(state)
            result = env.step(action)
            #print(f"Step output: {result}")
            state, reward, done, info, __ = env.step(action)
            env.render()
            episode_reward += reward

            if done and info:
                print("Collision occurred!")
            elif done and not info:
                print("Goal reached!")

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        model.learn(total_timesteps=100)
        plt.imshow(env.render())
    plt.show()

    # model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    # print("Model saved!")
    # PlottingUtils.plot_rewards([episode_reward])

# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
