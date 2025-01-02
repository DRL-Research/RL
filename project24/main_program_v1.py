import multiprocessing
import random
import time
import airsim
from project24.initialization.config_v1 import *
from project24.utils import plots_utils_v1
from project24.initialization.setup_simulation_v1 import SetupManager
from project24.path_planning import turn_mapping_v1, path_following_v1
from project24.utils.path_planning import turn_helper_v1, path_control_v1
from project24.utils.airsim_manager_v1 import AirsimManager


def run_for_single_car(moving_car_name):
    airsim_client = airsim.CarClient()
    # Note Our Line of Code: Ensure SteeringProcManager initialization
    path_control_v1.SteeringProcManager.create_steering_procedure()  # Initialize shared memory
    positions_lst = []
    try:
        directions = [TURN_DIRECTION_STRAIGHT, TURN_DIRECTION_RIGHT, TURN_DIRECTION_LEFT]
        direction = random.choices(directions, k=1)[0]
        # Detect the cones and spline points, and return their location:
        tracked_points, execution_time, curr_vel, transition_matrix = turn_mapping_v1.mapping_loop(
            airsim_client,
            moving_car_name,
            direction)

        # Stop until spline generation is complete:
        AirsimManager.stop_car(airsim_client, moving_car_name, 0.1)
        spline = turn_helper_v1.filter_tracked_points_and_generate_spline(tracked_points, moving_car_name)
        # Follow the spline using Stanley's method:
        print(f'Starting variable speed spline following procedure for {moving_car_name}.')
        positions_lst = path_following_v1.following_loop(airsim_client, spline, execution_time, curr_vel,
                                                         transition_matrix, moving_car_name=moving_car_name)

        print(f'Full process complete for {moving_car_name}! Stopping vehicle.')
        AirsimManager.stop_car(airsim_client, moving_car_name)

        return positions_lst, moving_car_name

    finally:
        return positions_lst, moving_car_name


if __name__ == '__main__':
    """ Define the cars that will participate in the simulation: """
    simulation_start_time = time.time()
    setup_manager = SetupManager()
    time.sleep(0.2)
    cars_names = setup_manager.cars_names
    """ Run Simulation in MultiProcessing """
    number_of_processes = len(cars_names) + 1  # each car will have its own process
    # NOTE - single airsim manager for all processes.
    airsim_manager = AirsimManager(setup_manager)
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        car_location_by_name = pool.map(run_for_single_car, cars_names)

    simulation_end_time = time.time()
    """ Collect positions for each car """
    all_cars_positions_list = []
    for positions_lst, car_name in car_location_by_name:
        all_cars_positions_list.append((positions_lst, car_name))
    # here i want to append the positions_lst that return from run_for_car of each procces to the all_cars_positions_list
    if CREATE_MAIN_PLOT:
        plots_utils_v1.plot_vehicle_object_path(all_cars_positions_list)

    simulation_time = round(simulation_end_time - simulation_start_time, 3)
    print(f'Simulation took: {simulation_time} seconds')
    print('All cars have completed their tasks.')
