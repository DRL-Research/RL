from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from src.model_handler import Model
from agent_handler import Agent
from src.airsim_manager import AirsimManager
from utils.plotting_utils import *


def training_loop(experiment, env, agent, model):
    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions, all_speeds_car1, all_speeds_car2   = [], [], [], []
    episode_steps, collision_steps = [], []

    for episode in range(experiment.EPOCHS):
        print(f"@ Episode {episode + 1} @")
        current_state = env.reset()
        print(current_state)
        done = False
        episode_sum_of_rewards, episode_steps_counter = 0, 0
        episode_counter += 1

        episode_steps.append(total_steps)
        resume_experiment_simulation(env)
        actions_per_episode = []

        while not done:
            episode_steps_counter += 1
            total_steps += 1
            action = agent.get_action(model, current_state, total_steps, experiment.EXPLORATION_EXPLOTATION_THRESHOLD)

            print(f"Action: {('FAST' if action[0] == 0 else 'SLOW')} {action[0]}")

            experiment.logger.run["Actions"].append(action[0])
            actions_per_episode.append(action[0])

            # Step in environment
            current_state, reward, done, _ = env.step(action)

            # Get speed from AirSim
            speed_car1 = env.envs[0].airsim_manager.get_vehicle_speed(experiment.CAR1_NAME)
            speed_car2 = env.envs[0].airsim_manager.get_vehicle_speed(experiment.CAR2_NAME)

            all_speeds_car1.append(speed_car1)
            all_speeds_car2.append(speed_car2)

            # Log collision flag (1 if collision occurred at that step, else 0)
            is_collision = int(reward <= experiment.COLLISION_REWARD)
            experiment.logger.run["Speed"].append(
                {"speed_car1": speed_car1, "speed_car2": speed_car2, "episode": episode_counter, "collision_flag": is_collision * 12},
                step=total_steps
            )

            episode_sum_of_rewards += reward

            if reward < experiment.COLLISION_REWARD or done:
                if reward <= experiment.COLLISION_REWARD:
                    print("colisiion at" , total_steps)
                    collision_counter += 1
                    collision_steps.append(total_steps - 2)
                    print('********* collision ***********')
                pause_experiment_simulation(env)
                break
        print(f"Episode {episode_counter} finished with reward: {episode_sum_of_rewards}")
        print(f"Total Steps: {total_steps}, Episode steps: {episode_steps_counter}")
        all_rewards.append(episode_sum_of_rewards)
        all_actions.append(actions_per_episode)

        experiment.logger.log_actions_per_episode(actions_per_episode, experiment.CAR1_NAME)

        metrics = {
            "reward_episode": episode_sum_of_rewards,
            "steps_per_episode": episode_steps_counter,
            "collisions_episode": collision_counter,
        }
        experiment.logger.log_metrics(metrics, step=episode)

        if not experiment.ONLY_INFERENCE:
            model.learn(total_timesteps=episode_steps_counter, log_interval=1)
            print(f"Model learned on {episode_steps_counter} steps")

    resume_experiment_simulation(env)

    # TODO- I want to delete this entire graph, and I think the entire function, so we can just use Neptune.
    # plt = PlottingUtils.plot_vehicle_speeds(all_speeds_car1, all_speeds_car2, episode_steps, collision_steps)
    # plot_path = "vehicle_speeds.png"
    # plt.savefig(plot_path, bbox_inches='tight')
    #
    # plt.show()

    # experiment.logger.run["plots/vehicle_speeds"].upload(plot_path)

    # Save model and log training metrics
    if not experiment.ONLY_INFERENCE:
        experiment.logger.log_from_csv(
            path=f"{experiment.EXPERIMENT_PATH}/progress.csv",
            column_name="train/value_loss",
            metric_name="loss_episode"
        )

        experiment.logger.log_metric("total_collisions", collision_counter)

    return model, collision_counter, all_rewards, all_actions

# def plot_results(experiment, all_rewards, all_actions, all_speeds_car1, all_speeds_car2, episode_steps, collision_steps):
#     PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
#     PlottingUtils.plot_rewards(all_rewards)
#     PlottingUtils.show_plots()
#     PlottingUtils.plot_actions(all_actions)




def run_experiment(experiment_config):
    airsim_manager = AirsimManager(experiment_config)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager)])
    agent = Agent(experiment_config, airsim_manager)
    model = Model(env, experiment_config).model

    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    experiment_config.logger.log_hyperparameters(experiment_config)

    model, collision_counter, all_rewards, all_actions = training_loop(
        experiment=experiment_config,
        env=env,
        agent=agent,
        model=model
    )

    model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    logger.close()

    experiment_config.logger.upload_model(experiment_config.SAVE_MODEL_DIRECTORY, model_name="trained_model")
    experiment_config.logger.stop()

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