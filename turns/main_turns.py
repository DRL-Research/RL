import multiprocessing
import random
import time
import airsim
from turns.initialization.config_turns import *
from turns.utils import plots_utils_turns
from turns.initialization.setup_simulation_turns import SetupManager
from turns.path_planning import turn_points_generator, path_following
from turns.utils.path_planning import turn_helper, path_control
from turns.utils.airsim_manager_merged_with_original import AirsimManager

def run_for_single_car(moving_car_name):
    airsim_client = airsim.CarClient()
    path_control.SteeringProcManager.create_steering_procedure()  # Initialize shared memory

    directions = [TURN_DIRECTION_STRAIGHT, TURN_DIRECTION_RIGHT, TURN_DIRECTION_LEFT]
    direction = random.choices(directions, k=1)[0]
    # generate spline points, and return their location:
    tracked_points, execution_time, curr_vel, transition_matrix = turn_points_generator.generate_points_for_turn(
        airsim_client,
        moving_car_name,
        direction)

    # Stop until spline generation is complete:
    AirsimManager.stop_car(airsim_client, moving_car_name, 0.1)
    spline = turn_helper.filter_tracked_points_and_generate_spline(tracked_points, moving_car_name)
    # Follow the spline using Stanley's method:
    print(f'Starting variable speed spline following procedure for {moving_car_name}.')
    positions_lst = path_following.following_loop(airsim_client, spline,moving_car_name=moving_car_name)

    print(f'Full process complete for {moving_car_name}! Stopping vehicle.')
    AirsimManager.stop_car(airsim_client, moving_car_name)

    return positions_lst, moving_car_name



if __name__ == '__main__':
    """ Define the cars that will participate in the simulation: """
    setup_manager = SetupManager()
    time.sleep(0.2)
    cars_names = setup_manager.cars_names
    """ Run Simulation in MultiProcessing """
    number_of_processes = len(cars_names) + 1  # each car will have its own process
    # NOTE - single airsim manager for all processes.
    airsim_manager = AirsimManager(setup_manager_cars=setup_manager.cars)
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        car_location_by_name = pool.map(run_for_single_car, cars_names)

    """ Collect positions for each car """
    all_cars_positions_list = []
    for positions_lst, car_name in car_location_by_name:
        all_cars_positions_list.append((positions_lst, car_name))

    if CREATE_MAIN_PLOT:
        plots_utils_turns.plot_vehicle_object_path(all_cars_positions_list)

    print('All cars have completed their tasks.')
