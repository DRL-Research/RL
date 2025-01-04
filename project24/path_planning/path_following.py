import logging

from project24.utils import spatial_utils, plots_utils_turns
import airsim
import time
from project24.utils.path_planning import turn_helper, path_control
from project24.utils.path_planning.pidf_controller import PidfControl
import struct
import os
from project24.initialization.config_turns import *


def following_loop(client, spline=None, execution_time=None, curr_vel=None,
                   transition_matrix=None, moving_car_name="Car1"):
    data_dest = os.path.join(os.getcwd(), '../recordings')
    os.makedirs(data_dest, exist_ok=True)
    save_data = False

    # Open access to shared memory blocks:
    shmem_active, shmem_setpoint, shmem_output = path_control.SteeringProcManager.retrieve_shared_memories()

    # Define Stanley-method parameters:
    follow_handler = path_control.StanleyFollower(spline, MAX_VELOCITY, MIN_VELOCITY, LOOKAHEAD, K_STEER)
    follow_handler.k_vel *= K_VEL

    # Define speed controller:
    speed_controller = PidfControl(0.01)
    speed_controller.set_pidf(0.01, 0.01, 0.0, 0.043)
    speed_controller.set_extrema(min_setpoint=0.01, max_integral=0.01)
    speed_controller.alpha = 0.01

    car_controls = airsim.CarControls(moving_car_name)
    current_vehicle_positions_lst = []
    current_object_positions_lst = []
    start_time_lst = time.perf_counter()
    max_run_time = 60
    turn_completed = False
    target_point = [spline.xi[-1], spline.yi[-1]]

    while not turn_completed:
        now = time.perf_counter()
        vehicle_pose = client.simGetVehiclePose(moving_car_name)
        car_state = client.getCarState(moving_car_name)
        curr_vel = car_state.speed
        curr_pos, curr_rot = spatial_utils.extract_pose_from_airsim(vehicle_pose)
        rot_airsim = spatial_utils.extract_rotation_from_airsim(vehicle_pose.orientation)
        current_yaw = rot_airsim[0]
        current_position_airsim = [vehicle_pose.position.x_val, vehicle_pose.position.y_val,
                                   vehicle_pose.position.z_val]
        current_position_global = turn_helper.airsim_point_to_global(current_position_airsim)
        current_object_pose_position = client.simGetObjectPose(moving_car_name).position
        current_object_pose_position = [current_object_pose_position.x_val, current_object_pose_position.y_val,
                                        current_object_pose_position.z_val]
        distance_from_target_point = spatial_utils.calculate_distance_in_2d_from_array(current_position_global,
                                                                                          target_point)

        if now - start_time_lst >= max_run_time:
            if CREATE_SUB_PLOTS:
                plots_utils_turns.plot_vehicle_relative_path(current_vehicle_positions_lst, moving_car_name)
            return current_object_positions_lst

        curr_heading = np.deg2rad(curr_rot[0])

        current_vehicle_positions_lst.append(current_position_airsim)
        current_object_positions_lst.append(current_object_pose_position)

        yaw_is_0 = ZERO_YAW_LOW_BOUNDREY <= current_yaw <= ZERO_YAW_HIGH_BOUNDERY
        yaw_is_180 = ONE_EIGHTY_YAW_LOW_BOUNDREY <= abs(current_yaw) <= ONE_EIGHTY_YAW_HIGH_BOUNDERY
        yaw_is_90 = NINETY_YAW_LOW_BOUNDREY <= abs(current_yaw) <= NINETY_YAW_HIGH_BOUNDERY

        if distance_from_target_point < 2.0 and (yaw_is_90 or yaw_is_0 or yaw_is_180):
            set_car_controls_by_name(client, moving_car_name, desired_steer=0.0, throttle=0.2)

            t = time.perf_counter()
            while True:
                time_passed = time.perf_counter() - t
                if time_passed > TIME_TO_KEEP_STRAIGHT_AFTER_TURN:
                    logging.info(f"{TIME_TO_KEEP_STRAIGHT_AFTER_TURN} seconds have passed. Exiting the loop.")
                    break

                vehicle_position = client.simGetVehiclePose(moving_car_name).position
                current_position_airsim = [vehicle_position.x_val, vehicle_position.y_val, vehicle_position.z_val]
                current_position_global = turn_helper.airsim_point_to_global(current_position_airsim)
                current_object_pose_position = client.simGetObjectPose(moving_car_name).position
                current_object_pose_position = [current_object_pose_position.x_val, current_object_pose_position.y_val,
                                                current_object_pose_position.z_val]
                current_vehicle_positions_lst.append(current_position_airsim)
                current_object_positions_lst.append(current_object_pose_position)
            turn_completed = True

        if turn_completed:
            break

        desired_speed, desired_steer = follow_handler.calc_ref_speed_steering(current_position_global, curr_vel,
                                                                              curr_heading)
        desired_steer /= follow_handler.max_steering
        desired_steer = np.clip(desired_steer, -1, 1)
        shmem_setpoint.buf[:8] = struct.pack('d', desired_steer)
        set_car_controls_by_name(client, moving_car_name, desired_steer)
    if CREATE_SUB_PLOTS:
        plots_utils_turns.plot_vehicle_relative_path(current_vehicle_positions_lst, moving_car_name)
        plots_utils_turns.combine_plot(spline.xi, spline.yi, current_vehicle_positions_lst, moving_car_name)
    return current_object_positions_lst


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

