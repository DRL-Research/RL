import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.model.model_handler import Model

from utils.airsim_manager import AirsimManager
from utils.plotting_utils import PlottingUtils
from gym.envs.registration import register
import gymnasium as gym

def training_loop(experiment, env, model):
    if experiment.ONLY_INFERENCE:
        print('Only Inference')
        model.load(experiment.LOAD_WEIGHT_DIRECTORY)
        print(f"Loaded weights from {experiment.LOAD_MODEL_DIRECTORY} for inference.")
    else:
        if experiment.LOAD_PREVIOUS_WEIGHT:
            # model.load(experiment.LOAD_MODEL_DIRECTORY)
            print(f"Loaded weights from {experiment.LOAD_MODEL_DIRECTORY}, the model will be trained from this point! ")
        collision_counter, episode_counter, total_steps = 0, 0, 0
        all_rewards, all_actions = [], []
        for episode in range(experiment.EPOCHS):
            print(f"@ Episode {episode + 1} @")
            current_state,info = env.reset()
            done = False
            episode_sum_of_rewards, steps_counter = 0, 0
            episode_counter += 1
            actions_per_episode = []
            while not done:
                steps_counter += 1
                total_steps += 1
                ### need to fill action line here
                action,_state=model.predict(current_state,deterministic=False)
                env.render()
                print(f"Action: {action}")
                current_state, reward, done, truncated, info = env.step(action)
                print('current_state:',current_state)
                print('reward:',reward)
                print('done:',done)
                actions_per_episode.append(action)
                episode_sum_of_rewards += reward
                if done:
                    if reward == -20:
                        collision_counter += 1
                        print('Collision!')
                    else:
                        print('Success!')
                    episode_sum_of_rewards += reward
                    break

            print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
            print(f"Total Steps: {total_steps}, Episode steps: {steps_counter}")
            all_rewards.append(episode_sum_of_rewards)
            all_actions.append(actions_per_episode)

            if not experiment.ONLY_INFERENCE:
                model.learn(total_timesteps=steps_counter)
                print(f"Model learned on {steps_counter} steps")
        return model, collision_counter, all_rewards, all_actions


def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.plot_actions(all_actions)
    PlottingUtils.show_plots()



def run_experiment(experiment_config, config):

    env = gym.make('RELintersection-v0',render_mode="rgb_array", config=config)
    model = Model(env, experiment_config).model
    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    model, collision_counter, all_rewards, all_actions = training_loop(experiment=experiment_config, env=env,
                                                                       model=model)
    model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    logger.close()

    print('Model saved')
    print("Total collisions:", collision_counter)

    if not experiment_config.ONLY_INFERENCE:
       plot_results(experiment=experiment_config, all_rewards=all_rewards, all_actions=all_actions)

# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
