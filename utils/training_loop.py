from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from experiment.experiment_constants import CarName
from utils.model.model_handler import Model
from utils.agent_handler import Agent
from utils.airsim_manager import AirsimManager
from utils.plotting_utils import PlottingUtils


def training_loop(experiment, env, agent, model_training, model_inference):
    """
    Training loop for the experiment.
    :param experiment: The experiment configuration.
    :param env: The simulation environment.
    :param agent: The agent controlling the cars.
    :param model_training: The model currently being trained.
    :param model_inference: The model currently being used for inference.

    returns: model_training, collision_counter, all_rewards, all_actions
    """


    collision_counter, episode_counter, total_steps = 0, 0, 0
    all_rewards, all_actions = [], []

    for cycle in range(experiment.CYCLES):
        print(f"@ Cycle {cycle + 1}/{experiment.CYCLES} @")

        if cycle == 0:
            # Initial training phase: Car1 trains, Car2 drives at fixed speed
            experiment.ROLE = CarName.CAR1
            print("Initial phase: Car1 is training, Car2 is driving at fixed speed.")
            for episode in range(experiment.EPISODES_PER_CYCLE):
                episode_counter += 1
                print(f"  @ Episode {episode_counter} (Car1 Training) @")
                episode_rewards, episode_actions, steps = run_episode(
                    env, agent, model_training, experiment, training_car=CarName.CAR1
                )
                total_steps += steps
                all_rewards.append(episode_rewards)
                all_actions.append(episode_actions)
                # experiment.logger.log_actions_per_episode(episode_actions, ) # TODO: change episode actions to match 2 cars
                model_training.learn(total_timesteps=steps, log_interval=1)
                print(f"Car1 model learned on {steps} steps")

        else:
            # Alternate between training and inference roles every 20 episodes
            if cycle % 2 == 1:
                # Car1 trains, Car2 uses model_inference
                experiment.ROLE = CarName.CAR1
                print("Car1 is training, Car2 is using model_inference.")
                for episode in range(experiment.EPISODES_PER_CYCLE):
                    episode_counter += 1
                    print(f"  @ Episode {episode_counter} (Car1 Training) @")
                    episode_rewards, episode_actions, steps = run_episode(
                        env, agent, model_training, experiment,
                        training_car=CarName.CAR1,
                        inference_car=CarName.CAR2,
                        inference_model=model_inference
                    )
                    total_steps += steps
                    all_rewards.append(episode_rewards)
                    all_actions.append(episode_actions)
                    model_training.learn(total_timesteps=steps, log_interval=1)
                    print(f"Car1 model learned on {steps} steps")
            else:
                # Car2 trains, Car1 uses model_inference
                experiment.ROLE = CarName.CAR2
                print("Car2 is training, Car1 is using model_inference.")
                for episode in range(experiment.EPISODES_PER_CYCLE):
                    episode_counter += 1
                    print(f"  @ Episode {episode_counter} (Car2 Training) @")
                    episode_rewards, episode_actions, steps = run_episode(
                        env, agent, model_training, experiment,
                        training_car=CarName.CAR2,
                        inference_car=CarName.CAR1,
                        inference_model=model_inference
                    )
                    total_steps += steps
                    all_rewards.append(episode_rewards)
                    all_actions.append(episode_actions)
                    model_training.learn(total_timesteps=steps, log_interval=1)
                    print(f"Car2 model learned on {steps} steps")

        # Swap models at the end of the cycle
        temp_weights = model_training.get_parameters()
        print(temp_weights)
        model_inference.set_parameters(temp_weights)
        print(f"Swapped weights between training and inference models at the end of cycle {cycle + 1}")

    resume_experiment_simulation(env)
    print("Training completed.")
    return model_training, collision_counter, all_rewards, all_actions


def run_episode(env, agent, training_model, experiment, training_car, inference_car=None, inference_model=None):
    '''
    Run one episode of the simulation.
    :param env: The environment instance.
    :param agent: The agent instance.
    :param training_model: The model used for training.
    :param experiment: The experiment configuration.
    :param training_car: The car currently being trained.
    :param inference_car: The car performing inference only.
    :param inference_model: The model used for inference (if applicable).

    :return: episode_sum_of_rewards, actions_per_episode_training, steps_counter
    '''

    action_inference = np.random.choice([experiment.THROTTLE_SLOW, experiment.THROTTLE_FAST])
    if training_car == CarName.CAR1:
        current_state_training = env.envs[0].airsim_manager.get_car1_state()
        current_state_inference = None
    if training_car == CarName.CAR2:
        current_state_training = env.envs[0].airsim_manager.get_car2_state()
        current_state_inference = None
    if inference_car:
        if inference_car == CarName.CAR1:
            current_state_inference = env.envs[0].airsim_manager.get_car1_state()
        else:
            current_state_inference = env.envs[0].airsim_manager.get_car2_state()

    done = False
    episode_sum_of_rewards, steps_counter = 0, 0
    actions_per_episode_training = []
    actions_per_episode_inference = []
    resume_experiment_simulation(env)

    while not done:
        steps_counter += 1

        # Get actions for training and inference cars
        action_training = agent.get_action(
            training_model, current_state_training, steps_counter,
            experiment.EXPLORATION_EXPLOTATION_THRESHOLD
        )
        if inference_car:
            action_inference = agent.get_action(
                inference_model, current_state_inference, steps_counter,
                experiment.EXPLORATION_EXPLOTATION_THRESHOLD
            )
        print(f"Action for training car: {action_training}")
        print(f"Action for inference car: {action_inference}")
        # Perform steps in the environment
        next_state_training, reward, done, _ = env.step(action_training)  # Step for training car
        episode_sum_of_rewards += reward
        actions_per_episode_training.append(action_training)

        if inference_car:
            next_state_inference, _, _, _ = env.step(action_inference)  # Step for inference car
            actions_per_episode_inference.append(action_inference)

        # Update states
        current_state_training = next_state_training
        if inference_car:
            current_state_inference = next_state_inference

        if done:
            pause_experiment_simulation(env)
            if reward <= experiment.COLLISION_REWARD:
                print('********* collision ***********')
            break


    return episode_sum_of_rewards, actions_per_episode_training, steps_counter


def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def run_experiment(experiment_config):
    # Initialize AirSim manager and environment
    airsim_manager = AirsimManager(experiment_config)
    env = DummyVecEnv([lambda: Agent(experiment_config, airsim_manager)])
    agent = Agent(experiment_config, airsim_manager)

    # Initialize two models: one for training and one for inference
    #model_training = Model(env, experiment_config).model
    model_training=Model(env, experiment_config).model
    model_training.load('experiments/22_12_2024-10_42_16_Experiment3_infernce/trained_model.zip')
    model_inference = Model(env, experiment_config).model  # Separate model for inference

    # Configure logger
    logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])
    model_training.set_logger(logger)

    # Run the training loop
    model_training, collision_counter, all_rewards, all_actions = training_loop(
        experiment=experiment_config,
        env=env,
        agent=agent,
        model_training=model_training,
        model_inference=model_inference
    )



    # Save the final trained model
    model_training.save(experiment_config.SAVE_MODEL_DIRECTORY)
    logger.close()

    print("Model saved")
    print("Total collisions:", collision_counter)

    # Plot results if not in inference mode
    if not experiment_config.ONLY_INFERENCE:
        plot_results(experiment=experiment_config, all_rewards=all_rewards, all_actions=all_actions)


# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
