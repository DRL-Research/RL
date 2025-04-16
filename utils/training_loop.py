from stable_baselines3.common.logger import configure
from utils.model.model_handler import Model
from utils.plotting_utils import PlottingUtils
import gymnasium as gym
import os
from datetime import datetime
import json
import csv


def write_experiment_info(experiment_config, env_config):
    os.makedirs("logs", exist_ok=True)
    info_path = f"logs/experiment_{experiment_config.EXPERIMENT_ID}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_info.txt"
    with open(info_path, "w") as f:
        f.write("Experiment Config:\n")
        for attr in dir(experiment_config):
            if not attr.startswith("__") and not callable(getattr(experiment_config, attr)):
                f.write(f"  {attr}: {getattr(experiment_config, attr)}\n")
        f.write("\nEnvironment Config:\n")
        f.write(json.dumps(env_config, indent=4, default=str))

def write_experiment_steps(csv_path, vehicles_count, episode, step, action, current_state, reward, done):
    # Append step info to the CSV
    row = [episode, step, action]
    # Add one column per vehicle's state
    for i in range(vehicles_count):
        row.append(current_state[i])
    row.extend([reward, done])
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def training_loop(experiment, env, model, csv_path, vehicles_count):
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
                print("-------------------------------------------------------")
                steps_counter += 1
                total_steps += 1
                ### need to fill action line here
                action, _state = model.predict(current_state, deterministic=False)
                env.render()
                print(f"Action: {action}")
                current_state, reward, done, truncated, info = env.step(action)
                print('current_state:',current_state)
                print('reward:',reward)
                print('done:',done)

                write_experiment_steps(csv_path, vehicles_count, episode, steps_counter, action, current_state, reward, done)

                actions_per_episode.append(action)
                episode_sum_of_rewards += reward
                if done:
                    if info["crashed"]:
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
                model.learn(total_timesteps=1)
                print(f"Model learned on {steps_counter} steps")
                # print(model.rollout_buffer.rewards)
        return model, collision_counter, all_rewards, all_actions


def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH, experiment.EXPERIMENT_ID)
    PlottingUtils.plot_rewards(all_rewards, experiment.EXPERIMENT_ID)
    PlottingUtils.plot_actions(all_actions, experiment.EXPERIMENT_ID)
    PlottingUtils.show_plots()


def run_experiment(experiment_config, env_config):
    env = gym.make('RELintersection-v0', render_mode="rgb_array", config=env_config)
    model = Model(env, experiment_config).model
    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    write_experiment_info(experiment_config, env_config)
    csv_path = f"logs/experiment_{experiment_config.EXPERIMENT_ID}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_steps.csv"
    vehicles_count = len(env_config["static_cars"])+len(env_config["controlled_cars"])
    header = ["Episode", "Step", "Action"]
    # Add one column per vehicle's state
    for i in range(vehicles_count):
        header.append(f"Car{i + 1}_State")

    header.extend(["Reward", "Done"])
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    print(f"Starting experiment: {experiment_config.EXPERIMENT_ID}")
    model, collision_counter, all_rewards, all_actions = training_loop(experiment=experiment_config, env=env,
                                                                       model=model, csv_path=csv_path, vehicles_count=vehicles_count)
    model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    logger.close()

    print('Model saved')
    print("Total collisions:", collision_counter)

    if not experiment_config.ONLY_INFERENCE:
        plot_results(experiment=experiment_config, all_rewards=all_rewards, all_actions=all_actions)
