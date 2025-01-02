import logging
from project24.utils import spatial_utils_v1, plots_utils_v1
from project24.initialization.config_v1 import *
from project24.path_planning import bezier_v1
from project24.utils.path_planning import spline_utils_v1

from project24.initialization.config_v1 import CREATE_SUB_PLOTS


def airsim_point_to_global_full_version(airsim_point, execution_time, curr_vel, transition_matrix):
    # we didnt use this function since the transition matrix is crucial for the transformations.
    # in case of 4 cars, each has its own matrix.
    airsim_point_copy = airsim_point.copy()
    centroid_eng, dump = spatial_utils_v1.convert_eng_airsim(airsim_point_copy, [0, 0, 0])
    centroid_eng[0] -= execution_time * curr_vel * 2.0  # Compensate for sensor sync
    centroid_lidar = np.append(centroid_eng, 1)
    global_point = np.matmul(transition_matrix, centroid_lidar)[:3]
    return global_point


def airsim_point_to_global(airsim_point):
    airsim_point_copy = airsim_point.copy()
    centroid_eng, dump = spatial_utils_v1.convert_eng_airsim(airsim_point_copy, BASE_ROTATION)
    return centroid_eng


def get_points_for_bezier_curve(client, moving_car_name, initial_yaw, direction):
    # the destination and control points in this function are from the vehicle point of view ( assumes start at 0,0)
    vehicle_pose = client.simGetVehiclePose(moving_car_name)
    yaw_is_0 = ZERO_YAW_LOW_BOUNDREY <= abs(initial_yaw) <= ZERO_YAW_HIGH_BOUNDERY
    yaw_is_90 = NINETY_YAW_LOW_BOUNDREY <= initial_yaw <= NINETY_YAW_HIGH_BOUNDERY
    yaw_is_180 = ONE_EIGHTY_YAW_LOW_BOUNDREY <= initial_yaw <= ONE_EIGHTY_YAW_HIGH_BOUNDERY

    if yaw_is_0:
        destination_point_airsim, control_point_airsim = calculate_points_for_yaw_0(vehicle_pose, direction)
    elif yaw_is_180:
        destination_point_airsim, control_point_airsim = calculate_points_for_yaw_180(vehicle_pose, direction)
    elif yaw_is_90:
        destination_point_airsim, control_point_airsim = calculate_points_for_yaw_90(vehicle_pose, direction)
    else:  # yaw 270 (-90)
        destination_point_airsim, control_point_airsim = calculate_points_for_yaw_270(vehicle_pose, direction)

    destination_point_global = airsim_point_to_global(destination_point_airsim)
    destination_point_airsim_x, destination_point_airsim_y = destination_point_airsim[0], destination_point_airsim[
        1]  # just for monitoring

    destination_point_airsim = np.array(
        [destination_point_airsim_x, destination_point_airsim_y])  # no need - just for monitoring

    destination_point_global_x = destination_point_global[0]
    destination_point_global_y = destination_point_global[1]
    destination_point_global = np.array([destination_point_global_x, destination_point_global_y])
    logging.info(f"destination from airsim : {destination_point_airsim}")
    logging.info(f"destination from global : {destination_point_global}")

    control_point_global = airsim_point_to_global(control_point_airsim)
    control_point_global_x = control_point_global[0]
    control_point_global_y = control_point_global[1]
    control_point_global = np.array([control_point_global_x, control_point_global_y])

    return destination_point_global, control_point_global


def calculate_points_for_yaw_0(vehicle_pose, direction):
    if direction == TURN_DIRECTION_RIGHT:
        destination_point = [vehicle_pose.position.x_val + FORWARD_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.y_val + SIDE_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val + FORWARD_DISTANCE_RIGHT_TURN,
                         vehicle_pose.position.y_val,
                         vehicle_pose.position.z_val]
    elif direction == TURN_DIRECTION_LEFT:
        destination_point = [vehicle_pose.position.x_val + FORWARD_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.y_val - SIDE_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val + FORWARD_DISTANCE_LEFT_TURN,
                         vehicle_pose.position.y_val,
                         vehicle_pose.position.z_val]
    else:  # straight
        destination_point = [vehicle_pose.position.x_val + FORWARD_DISTANCE_STRAIGHT,
                             vehicle_pose.position.y_val,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val + FORWARD_DISTANCE_LEFT_TURN/2,
                         vehicle_pose.position.y_val,
                         vehicle_pose.position.z_val]
    return destination_point, control_point


def calculate_points_for_yaw_180(vehicle_pose, direction):
    if direction == TURN_DIRECTION_RIGHT:
        destination_point = [vehicle_pose.position.x_val - FORWARD_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.y_val - SIDE_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val - FORWARD_DISTANCE_RIGHT_TURN,
                         vehicle_pose.position.y_val,
                         vehicle_pose.position.z_val]
    elif direction == TURN_DIRECTION_LEFT:
        destination_point = [vehicle_pose.position.x_val - FORWARD_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.y_val + SIDE_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val - FORWARD_DISTANCE_LEFT_TURN,
                         vehicle_pose.position.y_val,
                         vehicle_pose.position.z_val]
    else:
        destination_point = [vehicle_pose.position.x_val - FORWARD_DISTANCE_STRAIGHT,
                             vehicle_pose.position.y_val,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val - FORWARD_DISTANCE_STRAIGHT/2,
                         vehicle_pose.position.y_val,
                         vehicle_pose.position.z_val]
    return destination_point, control_point


def calculate_points_for_yaw_90(vehicle_pose, direction):
    if direction == TURN_DIRECTION_RIGHT:
        destination_point = [vehicle_pose.position.x_val - SIDE_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.y_val + FORWARD_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val,
                         vehicle_pose.position.y_val + FORWARD_DISTANCE_RIGHT_TURN,
                         vehicle_pose.position.z_val]
    elif direction == TURN_DIRECTION_LEFT:
        destination_point = [vehicle_pose.position.x_val + SIDE_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.y_val + FORWARD_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val,
                         vehicle_pose.position.y_val + FORWARD_DISTANCE_LEFT_TURN,
                         vehicle_pose.position.z_val]
    else:
        destination_point = [vehicle_pose.position.x_val,
                             vehicle_pose.position.y_val + FORWARD_DISTANCE_STRAIGHT,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val,
                         vehicle_pose.position.y_val + FORWARD_DISTANCE_STRAIGHT/2,
                         vehicle_pose.position.z_val]
    return destination_point, control_point


def calculate_points_for_yaw_270(vehicle_pose, direction):
    if direction == TURN_DIRECTION_RIGHT:
        destination_point = [vehicle_pose.position.x_val + SIDE_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.y_val - FORWARD_DISTANCE_RIGHT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val,
                         vehicle_pose.position.y_val - FORWARD_DISTANCE_RIGHT_TURN,
                         vehicle_pose.position.z_val]
    elif direction == TURN_DIRECTION_LEFT:
        destination_point = [vehicle_pose.position.x_val - SIDE_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.y_val - FORWARD_DISTANCE_LEFT_TURN,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val,
                         vehicle_pose.position.y_val - FORWARD_DISTANCE_LEFT_TURN,
                         vehicle_pose.position.z_val]
    else:
        destination_point = [vehicle_pose.position.x_val,
                             vehicle_pose.position.y_val - FORWARD_DISTANCE_STRAIGHT,
                             vehicle_pose.position.z_val]
        control_point = [vehicle_pose.position.x_val,
                         vehicle_pose.position.y_val - FORWARD_DISTANCE_STRAIGHT/2,
                         vehicle_pose.position.z_val]
    return destination_point, control_point


def create_bezier_curve(client, initial_yaw, vehicle_pose, direction, moving_car_name):
    destination_point_global, control_point_global = \
        get_points_for_bezier_curve(client, moving_car_name, initial_yaw, direction)

    start_point_airsim = [vehicle_pose.position.x_val,
                          vehicle_pose.position.y_val,
                          vehicle_pose.position.z_val]
    start_point_global = airsim_point_to_global(start_point_airsim)
    start_point_x_coordinate = start_point_global[0]
    start_point_y_coordinate = start_point_global[1]
    start_point_global_np = np.array(
        [start_point_x_coordinate, start_point_y_coordinate])  # needed for creating the bezier

    global_curve_points = bezier_v1.generate_curve_points(start_point_global_np, control_point_global,
                                                          destination_point_global)
    global_curve_points_with_z_set_to_zero = []
    for point_as_tuple in global_curve_points:
        x = point_as_tuple[0]
        y = point_as_tuple[1]
        z = 0  # by definition
        point = [x, y, z]
        global_curve_points_with_z_set_to_zero.append(point)
    global_curve_points = global_curve_points_with_z_set_to_zero

    return global_curve_points


def create_keep_straight_bezier_curve(client, initial_yaw, vehicle_pose, direction, moving_car_name):
    destination_point_global, control_point_global = \
        get_points_for_bezier_curve(client, moving_car_name, initial_yaw, direction)

    start_point_airsim = [vehicle_pose.position.x_val,
                          vehicle_pose.position.y_val,
                          vehicle_pose.position.z_val]
    start_point_global = airsim_point_to_global(start_point_airsim)
    start_point_x_coordinate = start_point_global[0]
    start_point_y_coordinate = start_point_global[1]
    start_point_global_np = np.array(
        [start_point_x_coordinate, start_point_y_coordinate])  # needed for creating the bezier

    global_curve_points = bezier_v1.generate_curve_points(start_point_global_np, control_point_global,
                                                          destination_point_global)
    global_curve_points_with_z_set_to_zero = []
    for point_as_tuple in global_curve_points:
        x = point_as_tuple[0]
        y = point_as_tuple[1]
        z = 0  # by definition
        point = [x, y, z]
        global_curve_points_with_z_set_to_zero.append(point)
    global_curve_points = global_curve_points_with_z_set_to_zero

    return global_curve_points


def filter_tracked_points_and_generate_spline(tracked_points,moving_car_name):
    x = [sublist[0] for sublist in tracked_points[::2]]
    y = [sublist[1] for sublist in tracked_points[::2]]
    spline_obj = spline_utils_v1.PathSpline(x, y)
    spline_obj.generate_spline(amount=0.1, meters=True, smoothing=1, summation=len(x))
    if CREATE_SUB_PLOTS:
        plots_utils_v1.plot_the_spline(spline_obj.xi, -1 * spline_obj.yi, moving_car_name)
    return spline_obj
