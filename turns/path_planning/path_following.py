import logging

from turns.utils import spatial_utils, plots_utils_turns
import airsim
import time
from turns.utils.path_planning import turn_helper, path_control
from turns.utils.path_planning.pidf_controller import PidfControl
import struct
import os
from turns.initialization.config_turns import *


# def following_loop(client, spline=None,moving_car_name="Car1"):
#     """
#     Parameters:
#         spline - the spline to follow
#         moving_car_name - the name of the car
#
#     Output:
#         current_object_positions_lst - list of the positions while following the spline
#
#     """
#     data_dest = os.path.join(os.getcwd(), '../recordings')
#     os.makedirs(data_dest, exist_ok=True)
#     save_data = False
#
#     # Open access to shared memory blocks:
#     shmem_active, shmem_setpoint, shmem_output = path_control.SteeringProcManager.retrieve_shared_memories()
#
#     # Define Stanley-method parameters:
#     follow_handler = path_control.StanleyFollower(spline, MAX_VELOCITY, MIN_VELOCITY, LOOKAHEAD, K_STEER)
#     follow_handler.k_vel *= K_VEL
#
#     # Define speed controller:
#     speed_controller = PidfControl(0.01)
#     speed_controller.set_pidf(0.01, 0.01, 0.0, 0.043)
#     speed_controller.set_extrema(min_setpoint=0.01, max_integral=0.01)
#     speed_controller.alpha = 0.01
#
#     car_controls = airsim.CarControls(moving_car_name)
#     current_vehicle_positions_lst = []
#     current_object_positions_lst = []
#     start_time_lst = time.perf_counter()
#     max_run_time = 60
#     turn_completed = False
#     target_point = [spline.xi[-1], spline.yi[-1]]
#
#     while not turn_completed:
#         now = time.perf_counter()
#         vehicle_pose = client.simGetVehiclePose(moving_car_name)
#         car_state = client.getCarState(moving_car_name)
#         curr_vel = car_state.speed
#         curr_pos, curr_rot = spatial_utils.extract_pose_from_airsim(vehicle_pose)
#         rot_airsim = spatial_utils.extract_rotation_from_airsim(vehicle_pose.orientation)
#         current_yaw = rot_airsim[0]
#         current_position_airsim = [vehicle_pose.position.x_val, vehicle_pose.position.y_val,
#                                    vehicle_pose.position.z_val]
#         current_position_global = turn_helper.airsim_point_to_global(current_position_airsim)
#         current_object_pose_position = client.simGetObjectPose(moving_car_name).position
#         current_object_pose_position = [current_object_pose_position.x_val, current_object_pose_position.y_val,
#                                         current_object_pose_position.z_val]
#         distance_from_target_point = spatial_utils.calculate_distance_in_2d_from_array(current_position_global,
#                                                                                           target_point)
#
#         if now - start_time_lst >= max_run_time:
#             if CREATE_SUB_PLOTS:
#                 plots_utils_turns.plot_vehicle_relative_path(current_vehicle_positions_lst, moving_car_name)
#             return current_object_positions_lst
#
#         curr_heading = np.deg2rad(curr_rot[0])
#
#         current_vehicle_positions_lst.append(current_position_airsim)
#         current_object_positions_lst.append(current_object_pose_position)
#
#         yaw_is_0 = ZERO_YAW_LOW_BOUNDREY <= current_yaw <= ZERO_YAW_HIGH_BOUNDERY
#         yaw_is_180 = ONE_EIGHTY_YAW_LOW_BOUNDREY <= abs(current_yaw) <= ONE_EIGHTY_YAW_HIGH_BOUNDERY
#         yaw_is_90 = NINETY_YAW_LOW_BOUNDREY <= abs(current_yaw) <= NINETY_YAW_HIGH_BOUNDERY
#
#         if distance_from_target_point < 2.0 and (yaw_is_90 or yaw_is_0 or yaw_is_180):
#             set_car_controls_by_name(client, moving_car_name, desired_steer=0.0, throttle=0.2)
#
#             t = time.perf_counter()
#             while True:
#                 time_passed = time.perf_counter() - t
#                 if time_passed > TIME_TO_KEEP_STRAIGHT_AFTER_TURN:
#                     logging.info(f"{TIME_TO_KEEP_STRAIGHT_AFTER_TURN} seconds have passed. Exiting the loop.")
#                     break
#
#                 vehicle_position = client.simGetVehiclePose(moving_car_name).position
#                 current_position_airsim = [vehicle_position.x_val, vehicle_position.y_val, vehicle_position.z_val]
#                 current_position_global = turn_helper.airsim_point_to_global(current_position_airsim)
#                 current_object_pose_position = client.simGetObjectPose(moving_car_name).position
#                 current_object_pose_position = [current_object_pose_position.x_val, current_object_pose_position.y_val,
#                                                 current_object_pose_position.z_val]
#                 current_vehicle_positions_lst.append(current_position_airsim)
#                 current_object_positions_lst.append(current_object_pose_position)
#             turn_completed = True
#
#         if turn_completed:
#             break
#
#         desired_speed, desired_steer = follow_handler.calc_ref_speed_steering(current_position_global, curr_vel,
#                                                                               curr_heading)
#         desired_steer /= follow_handler.max_steering
#         desired_steer = np.clip(desired_steer, -1, 1)
#         shmem_setpoint.buf[:8] = struct.pack('d', desired_steer)
#         set_car_controls_by_name(client, moving_car_name, desired_steer)
#     #     env.step() - add here the training loop and change the training loop of the turns <----------------------
#     if CREATE_SUB_PLOTS:
#         plots_utils_turns.plot_vehicle_relative_path(current_vehicle_positions_lst, moving_car_name)
#         plots_utils_turns.combine_plot(spline.xi, spline.yi, current_vehicle_positions_lst, moving_car_name)
#     return current_object_positions_lst

# Helper functions

def setup_environment():
    """
    Sets up the environment by creating necessary directories.
    Input: None
    Output: Path to the data destination.
    """
    data_dest = os.path.join(os.getcwd(), '../recordings')
    os.makedirs(data_dest, exist_ok=True)
    return data_dest

def initialize_components(spline):
    """
    Initializes shared memory, Stanley follower, and speed controller.
    Input: Spline to follow.
    Output: Shared memory objects, Stanley follower, and PIDF speed controller.
    """
    shmem_active, shmem_setpoint, shmem_output = path_control.SteeringProcManager.retrieve_shared_memories()
    follow_handler = path_control.StanleyFollower(spline, MAX_VELOCITY, MIN_VELOCITY, LOOKAHEAD, K_STEER)
    follow_handler.k_vel *= K_VEL

    speed_controller = PidfControl(0.01)
    speed_controller.set_pidf(0.01, 0.01, 0.0, 0.043)
    speed_controller.set_extrema(min_setpoint=0.01, max_integral=0.01)
    speed_controller.alpha = 0.01

    return shmem_active, shmem_setpoint, shmem_output, follow_handler, speed_controller

def get_current_position(client, moving_car_name):
    """
    Retrieves the current vehicle and object positions.
    Input: AirSim client and the car name.
    Output: Current global position and object position.
    """
    vehicle_pose = client.simGetVehiclePose(moving_car_name)
    curr_pos_airsim = [vehicle_pose.position.x_val, vehicle_pose.position.y_val, vehicle_pose.position.z_val]
    curr_pos_global = turn_helper.airsim_point_to_global(curr_pos_airsim)

    obj_pose = client.simGetObjectPose(moving_car_name).position
    curr_obj_pos = [obj_pose.x_val, obj_pose.y_val, obj_pose.z_val]

    return curr_pos_airsim, curr_pos_global, curr_obj_pos

def turn_completed_check(distance, yaw, yaw_bounds, client, moving_car_name, positions_lst):
    """
    Checks if the turn is completed based on distance and yaw.
    Input: Distance, yaw angle, yaw bounds, client, car name, and position lists.
    Output: Boolean indicating if the turn is completed.
    """
    if distance < 2.0 and any(low <= abs(yaw) <= high for low, high in yaw_bounds):
        set_car_controls_by_name(client, moving_car_name, desired_steer=0.0, throttle=0.2)
        t = time.perf_counter()
        while True:
            if time.perf_counter() - t > TIME_TO_KEEP_STRAIGHT_AFTER_TURN:
                logging.info(f"{TIME_TO_KEEP_STRAIGHT_AFTER_TURN} seconds have passed. Exiting the loop.")
                break
            curr_pos_airsim, _, curr_obj_pos = get_current_position(client, moving_car_name)
            positions_lst.append(curr_pos_airsim)
        return True
    return False

def follow_spline_step(client, follow_handler, shmem_setpoint, curr_pos_global, curr_vel, curr_yaw,moving_car_name):
    """
    Calculates desired speed and steering, updates shared memory, and applies car controls.
    Input: Client, follower handler, shared memory, current position, velocity, and yaw.
    Output: None
    """
    desired_speed, desired_steer = follow_handler.calc_ref_speed_steering(curr_pos_global, curr_vel, curr_yaw)
    desired_steer /= follow_handler.max_steering
    desired_steer = np.clip(desired_steer, -1, 1)

    shmem_setpoint.buf[:8] = struct.pack('d', desired_steer)
    set_car_controls_by_name(client, moving_car_name, desired_steer)
#     TODO: this function gets the velocity from the original step function of RL (env.step)
#      function the determines the speed and steering
#      add a flag that change velocity when needed, not only when steering is changed


def plot_results(spline, current_vehicle_positions_lst, moving_car_name):
    """
    Creates plots to visualize the spline following process.
    Input: Spline, current vehicle positions, and car name.
    Output: None
    """
    if CREATE_SUB_PLOTS:
        plots_utils_turns.plot_vehicle_relative_path(current_vehicle_positions_lst, moving_car_name)
        plots_utils_turns.combine_plot(spline.xi, spline.yi, current_vehicle_positions_lst, moving_car_name)


# Main orchestrator function

def following_loop(client, spline=None, moving_car_name="Car1"):
    """
    Main function to follow a spline path with a car in AirSim.
    Input: AirSim client, spline to follow (default: None), and car name (default: "Car1").
    Output: List of object positions while following the spline.
    """
    # Environment setup
    setup_environment()

    # Initialize components
    shmem_active, shmem_setpoint, shmem_output, follow_handler, speed_controller = initialize_components(spline)

    # Initialize loop variables
    current_vehicle_positions_lst = []
    current_object_positions_lst = []
    start_time = time.perf_counter()
    max_run_time = 60
    turn_completed = False
    target_point = [spline.xi[-1], spline.yi[-1]]

    while not turn_completed:
        now = time.perf_counter()
        if now - start_time >= max_run_time:
            plot_results(spline, current_vehicle_positions_lst, moving_car_name)
            return current_object_positions_lst

        curr_pos_airsim, curr_pos_global, curr_obj_pos = get_current_position(client, moving_car_name)
        current_vehicle_positions_lst.append(curr_pos_airsim)
        current_object_positions_lst.append(curr_obj_pos)

        distance = spatial_utils.calculate_distance_in_2d_from_array(curr_pos_global, target_point)
        curr_yaw = spatial_utils.extract_rotation_from_airsim(
            client.simGetVehiclePose(moving_car_name).orientation
        )[0]

        yaw_bounds = [
            (ZERO_YAW_LOW_BOUNDREY, ZERO_YAW_HIGH_BOUNDERY),
            (NINETY_YAW_LOW_BOUNDREY, NINETY_YAW_HIGH_BOUNDERY),
            (ONE_EIGHTY_YAW_LOW_BOUNDREY, ONE_EIGHTY_YAW_HIGH_BOUNDERY),
        ]

        turn_completed = turn_completed_check(
            distance, curr_yaw, yaw_bounds, client, moving_car_name, current_vehicle_positions_lst
        )

        if not turn_completed:
            follow_spline_step(client, follow_handler, shmem_setpoint, curr_pos_global, client.getCarState(moving_car_name).speed, np.deg2rad(curr_yaw),moving_car_name)

    plot_results(spline, current_vehicle_positions_lst, moving_car_name)
    return current_object_positions_lst

def following_loop_with_rl(experiment,current_state, client, spline, moving_car_name, env, agent, model, total_steps, collision_counter, episode_rewards, episode_actions):
    """
    Modified following_loop that integrates RL training with spline-following.
    Input:
        - client: AirSim client instance.
        - spline: Spline to follow.
        - moving_car_name: Name of the car.
        - env: RL environment.
        - agent: RL agent.
        - model: RL model.
        - total_steps: Total steps taken so far.
        - collision_counter: Collision counter to update.
        - episode_rewards: List to store episode rewards.
        - episode_actions: List to store episode actions.
    Output:
        - positions_lst: List of positions for the car.
    """
    setup_environment()

    # Initialize components
    shmem_active, shmem_setpoint, shmem_output, follow_handler, speed_controller = initialize_components(spline)

    # Initialize loop variables
    current_vehicle_positions_lst = []
    current_object_positions_lst = []

    start_time = time.perf_counter()
    max_run_time = 60
    turn_completed = False
    target_point = [spline.xi[-1], spline.yi[-1]]

    while not turn_completed:
        now = time.perf_counter()
        if now - start_time >= max_run_time:
            print("Max runtime exceeded for spline-following.")
            break

        # RL action selection and environment step
        action = agent.get_action(model, current_state, total_steps,
                                  experiment.EXPLORATION_EXPLOTATION_THRESHOLD)
        # Spline-following logic
        curr_pos_airsim, curr_pos_global, curr_obj_pos = get_current_position(client, moving_car_name)
        current_vehicle_positions_lst.append(curr_pos_airsim)
        current_object_positions_lst.append(curr_obj_pos)

        distance = spatial_utils.calculate_distance_in_2d_from_array(curr_pos_global, target_point)
        curr_yaw = spatial_utils.extract_rotation_from_airsim(
            client.simGetVehiclePose(moving_car_name).orientation
        )[0]

        yaw_bounds = [
            (ZERO_YAW_LOW_BOUNDREY, ZERO_YAW_HIGH_BOUNDERY),
            (NINETY_YAW_LOW_BOUNDREY, NINETY_YAW_HIGH_BOUNDERY),
            (ONE_EIGHTY_YAW_LOW_BOUNDREY, ONE_EIGHTY_YAW_HIGH_BOUNDERY),
        ]

        turn_completed = turn_completed_check(
            distance, curr_yaw, yaw_bounds, client, moving_car_name, current_vehicle_positions_lst
        )

        if not turn_completed:
            follow_spline_step(client, follow_handler, shmem_setpoint, curr_pos_global, client.getCarState(moving_car_name).speed, np.deg2rad(curr_yaw), moving_car_name)


        current_state, reward, done, _ = env.step(action)

        # Update RL metrics
        # episode_rewards.append(reward)
        # episode_actions.append(action)
        # total_steps += 1
        #
        # if reward < experiment.COLLISION_REWARD or done:
        #     if reward <= experiment.COLLISION_REWARD:
        #         collision_counter += 1
        #         print('********* collision ***********')
        #     break


    plot_results(spline, current_vehicle_positions_lst, moving_car_name)
    return current_vehicle_positions_lst


def set_car_controls_by_name(airsim_client, car_name, desired_steer, throttle=0.4):
    car_controls = airsim.CarControls()
    car_controls.throttle = throttle
    car_controls.steering = desired_steer
    airsim_client.setCarControls(car_controls, car_name)


def quadratic_bezier(t, p0, p1, p2):
    u = 1 - t
    return u**2 * p0 + 2 * u * t * p1 + t**2 * p2


def generate_curve_points(p0, p1, p2, num_points=20000):
    curve_points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = quadratic_bezier(t, p0, p1, p2)
        curve_points.append(point)
    curve_points = [[p[0], p[1]] for p in curve_points]
    return curve_points

