from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.model.model_handler import Model
from utils.agent_handler import Agent
from utils.airsim_manager import AirsimManager
from utils.plotting_utils import PlottingUtils


def training_loop(experiment, env, agent, model):

    if experiment.ONLY_INFERENCE:
        print('Only Inference')
        model.load(experiment.LOAD_MODEL_DIRECTORY)
        print(f"Loaded weights from {experiment.LOAD_MODEL_DIRECTORY} for inference.")
    else:
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
                # print(f"Action: {action[0]}")
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

            # Log episode metrics
            experiment.logger.log_metric("reward_episode", episode_sum_of_rewards, step=episode)
            experiment.logger.log_metric("steps_per_episode", steps_counter, step=episode)
            experiment.logger.log_metric("collisions_episode", collision_counter, step=episode)

            if not experiment.ONLY_INFERENCE:
                model.learn(total_timesteps=steps_counter, log_interval=1)
                print(f"Model learned on {steps_counter} steps")

        resume_experiment_simulation(env)

        # Save model and log training metrics
        if not experiment.ONLY_INFERENCE:
            experiment.logger.log_model(experiment.SAVE_MODEL_DIRECTORY, "trained_model")
            experiment.logger.log_from_csv(
                path=f"{experiment.EXPERIMENT_PATH}/progress.csv",
                column_name="train/value_loss",
                metric_name="loss_episode"
            )
            experiment.logger.log_metric("total_collisions", collision_counter)

        experiment.logger.stop()


        return model, collision_counter, all_rewards, all_actions


def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def run_experiment(experiment_config):

    airsim_manager = AirsimManager(experiment_config)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager)])
    agent = Agent(experiment_config, airsim_manager)
    model = Model(env, experiment_config).model
    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])

    model.set_logger(logger)

    hyperparams = {
        "experiment_id": experiment_config.EXPERIMENT_ID,
        "model_type": experiment_config.MODEL_TYPE,
        "epochs": experiment_config.EPOCHS,
        "learning_rate": experiment_config.LEARNING_RATE,
        "n_steps": experiment_config.N_STEPS,
        "batch_size": experiment_config.BATCH_SIZE,
        "seed": experiment_config.SEED,
        "policy_network": str(experiment_config.PPO_NETWORK_ARCHITECTURE['pi']),
        "value_network": str(experiment_config.PPO_NETWORK_ARCHITECTURE['vf']),
        "loss_function": experiment_config.LOSS_FUNCTION,
        "random_init": experiment_config.RANDOM_INIT,
        "time_between_steps": experiment_config.TIME_BETWEEN_STEPS,
        "exploration_threshold": experiment_config.EXPLORATION_EXPLOTATION_THRESHOLD,

        "success_reward": experiment_config.REACHED_TARGET_REWARD,
        "starvation_reward": experiment_config.STARVATION_REWARD,
        "collision_reward": experiment_config.COLLISION_REWARD,

        "car1_initial_position_options": str([
            experiment_config.CAR1_INITIAL_POSITION_OPTION_1,
            experiment_config.CAR1_INITIAL_POSITION_OPTION_2,
        ]),
        "car2_initial_position_options": str([
            experiment_config.CAR2_INITIAL_POSITION_OPTION_1,
            experiment_config.CAR2_INITIAL_POSITION_OPTION_2,
        ]),
        "car1_action_fast": experiment_config.THROTTLE_FAST,
        "car1_action_slow": experiment_config.THROTTLE_SLOW,
        "car2_action": (experiment_config.THROTTLE_FAST + experiment_config.THROTTLE_SLOW) / 2,
        "car1_state": "[x1, y1, Vx1, Vy1, x2, y2, Vx2, Vy2]",
        "car2_state": "[x2, y2, Vx2, Vy2, x1, y1, Vx1, Vy1]",
    }

    experiment_config.logger.log_hyperparameters(hyperparams)

    model, collision_counter, all_rewards, all_actions = training_loop(experiment=experiment_config, env=env, agent=agent,
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
