
import os.path

from project24.utils.perception.car_detection_experiments_helper_v1 import *
from project24.utils.path_planning import turn_helper_v1, path_control_v1

from project24.initialization.setup_simulation_v1 import *
from project24.utils import spatial_utils_v1
import time
decimation = 30e9  # Used to save an output image every X iterations.


def mapping_loop(client, moving_car_name='Car1', direction=None):
    global decimation
    image_dest = os.path.join(os.getcwd(), '../images')
    data_dest = os.path.join(os.getcwd(), '../recordings')
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)
    save_data = False

    # Define pure pursuit parameters - need
    pursuit_follower = path_control_v1.PursuitFollower(2.0, 6.0)
    pursuit_follower.k_steer = 0.5

    # Open access to shared memory blocks:
    shmem_active, shmem_setpoint, shmem_output = path_control_v1.SteeringProcManager.retrieve_shared_memories()

    # Initialize vehicle starting point
    time.sleep(1.0)
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.2
    client.setCarControls(car_controls, vehicle_name=moving_car_name)
    execution_time = 0.0
    initial_car_position = spatial_utils_v1.get_car_settings_position(client, moving_car_name)
    initial_yaw, vehicle_pose, curr_vel, vehicle_to_map = None, None, None, None
    distance_from_initial_position = np.inf
    reached_start_turning_point = DISTANCE_BEFORE_START_TURNING > distance_from_initial_position
    while not reached_start_turning_point:
        vehicle_pose = client.simGetVehiclePose(moving_car_name)
        vehicle_to_map = spatial_utils_v1.tf_matrix_from_airsim_object(vehicle_pose)
        car_state = client.getCarState()
        curr_vel = car_state.speed
        current_car_position = spatial_utils_v1.get_car_settings_position(client, moving_car_name)
        distance_from_initial_position = spatial_utils_v1.calculate_distance_in_2d_from_3dvector(current_car_position,
                                                                                                 initial_car_position)
        vehicle_rotation = spatial_utils_v1.extract_rotation_from_airsim(vehicle_pose.orientation)  # return yaw,pitch,roll
        initial_yaw = vehicle_rotation[0]  # return the yaw
        # for start, the vehicle is moving straight a few meters
        reached_start_turning_point = DISTANCE_BEFORE_START_TURNING <= distance_from_initial_position

    try:
        tracked_points_bezier = turn_helper_v1.create_bezier_curve(client, initial_yaw, vehicle_pose,
                                                                   direction=direction, moving_car_name=moving_car_name)
        return tracked_points_bezier, execution_time, curr_vel, vehicle_to_map
    except:
        raise Exception("Turn Mapping Problem")


# def get_args_for_car_detection(client, moving_car_name, vehicle_pose, lidar_to_vehicle):
#     vehicle_to_map = spatial_utils_v0.tf_matrix_from_airsim_object(vehicle_pose)
#     lidar_to_map = np.matmul(vehicle_to_map, lidar_to_vehicle)
#     car_state = client.getCarState(vehicle_name=moving_car_name)
#     curr_vel = car_state.speed
#     lidar_data = client.getLidarData()
#     pointcloud = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
#     pointcloud = pointcloud.reshape((int(pointcloud.shape[0] / 3), 3))
#     return pointcloud, curr_vel, lidar_to_map