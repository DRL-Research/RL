import time
import random

import airsim
from stable_baselines3.common.vec_env import DummyVecEnv
from multiprocessing.dummy import Pool as ThreadPool
from turns.initialization.config_turns import CREATE_MAIN_PLOT, TURN_DIRECTION_STRAIGHT, TURN_DIRECTION_RIGHT, \
    TURN_DIRECTION_LEFT
from turns.initialization.setup_simulation_turns import SetupManager
from turns.path_planning import turn_points_generator
from turns.path_planning.path_following import following_loop_with_rl
from turns.utils import plots_utils_turns
from turns.utils.path_planning import turn_helper, path_control
from utils.model.model_handler import Model
from utils.agent_handler import Agent
from turns.utils.airsim_manager_merged_with_original import AirsimManager
from utils.plotting_utils import PlottingUtils

def process_car_turn(moving_car_name, experiment, current_state, env, agent, model, total_steps, collision_counter, episode_rewards, episode_actions):
    print(f"Generating turn for {moving_car_name}")
    airsim_client = airsim.CarClient()
    path_control.SteeringProcManager.create_steering_procedure()
    directions = [
        TURN_DIRECTION_STRAIGHT,
        TURN_DIRECTION_RIGHT,
        TURN_DIRECTION_LEFT]
    direction = random.choices(directions, k=1)[0]
    print(f"Generating turn for {moving_car_name} in direction {direction}")
    # Generate spline for the car
    tracked_points, _, _, _ = turn_points_generator.generate_points_for_turn(
        airsim_client, moving_car_name, direction
    )
    AirsimManager.stop_car(airsim_client, moving_car_name, 0.1)
    spline = turn_helper.filter_tracked_points_and_generate_spline(tracked_points, moving_car_name)

    # Follow the spline while integrating RL training
    print(f"Starting spline-following for {moving_car_name}")
    positions_lst = following_loop_with_rl(
        experiment=experiment,
        current_state=current_state,
        client=airsim_client,
        spline=spline,
        moving_car_name=moving_car_name,
        env=env,
        agent=agent,
        model=model,
        total_steps=total_steps,
        collision_counter=collision_counter,
        episode_rewards=episode_rewards,
        episode_actions=episode_actions,
    )
    return positions_lst, moving_car_name

def training_loop_turns(experiment, env, agent, model, cars_names):

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
            episode_rewards = []
            episode_actions = []

            episode_sum_of_rewards, steps_counter = 0, 0
            episode_counter += 1
            resume_experiment_simulation(env)
            actions_per_episode = []

            args_list = []
            for car_name in cars_names:
                args_list.append((
                    car_name, experiment, current_state, env, agent, model, total_steps, collision_counter,
                    episode_rewards, episode_actions))

            with ThreadPool(processes=len(cars_names)) as pool:
                car_location_by_name = pool.starmap(process_car_turn, args_list)

            all_cars_positions_list = []
            for positions_lst, car_name in car_location_by_name:
                all_cars_positions_list.append((positions_lst, car_name))

            print(f"Episode {episode + 1} finished with reward: {sum(episode_rewards)}")
            all_rewards.append(sum(episode_rewards))
            all_actions.append(episode_actions)

            if not experiment.ONLY_INFERENCE:
                    model.learn(total_timesteps=steps_counter, log_interval=1)
                    print(f"Model learned on {steps_counter} steps")

        return model, collision_counter, all_rewards, all_actions, all_cars_positions_list

def plot_results(experiment, all_rewards, all_actions):
    PlottingUtils.plot_losses(experiment.EXPERIMENT_PATH)
    PlottingUtils.plot_rewards(all_rewards)
    PlottingUtils.show_plots()
    PlottingUtils.plot_actions(all_actions)


def run_experiment_turns(experiment_config):
    setup_manager = SetupManager()
    time.sleep(0.2)
    cars_names = setup_manager.cars_names
    airsim_manager = AirsimManager(experiment_config, setup_manager_cars=setup_manager.cars)
    agent = Agent(experiment_config, airsim_manager)
    env = DummyVecEnv([lambda: agent])
    model = Model(env, experiment_config).model
    # logger = configure(experiment_config.EXPERIMENT_PATH, ["stdout", "csv", "tensorboard"])

    # model.set_logger(logger)

    # Training loop now includes running a single car simulation
    model, collision_counter, all_rewards, all_actions, all_cars_positions_list = training_loop_turns(
        experiment=experiment_config,
        env=env,
        agent=agent,
        model=model,
        cars_names=cars_names,  # Pass cars names to run them after training
    )

    model.save(experiment_config.SAVE_MODEL_DIRECTORY)
    # logger.close()

    print('Model saved')
    print("Total collisions:", collision_counter)

    # Plotting after the training and simulation
    if CREATE_MAIN_PLOT:
        plots_utils_turns.plot_vehicle_object_path(all_cars_positions_list[:-1])

    print('All cars have completed their tasks.')

# override the base function in "GYM" environment. do not touch!
def resume_experiment_simulation(env):
    env.envs[0].airsim_manager.resume_simulation()

# override the base function in "GYM" environment. do not touch!
def pause_experiment_simulation(env):
    env.envs[0].airsim_manager.pause_simulation()
