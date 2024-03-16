import airsim
import numpy as np
import random
from config import CAR1_INITIAL_POSITION, CAR2_INITIAL_POSITION, CAR1_INITIAL_YAW, CAR2_INITIAL_YAW, \
    CAR1_NAME, CAR2_NAME


class AirsimClientHandler:

    def __init__(self):

        airsim_client = airsim.CarClient()  # Create an AirSim client for car simulation
        airsim_client.confirmConnection()  # Confirm the connection to the AirSim simulator

        airsim_client.enableApiControl(True, CAR1_NAME)  # Enable API control for Car1
        airsim_client.enableApiControl(True, CAR2_NAME)  # Enable API control for Car2

        # Set car 1 throttle to 1:
        car_controls = airsim.CarControls()
        car_controls.throttle = 1
        airsim_client.setCarControls(car_controls, CAR1_NAME)

        # Set car 2 throttle to 1:
        car_controls = airsim.CarControls()
        car_controls.throttle = 1
        airsim_client.setCarControls(car_controls, CAR2_NAME)

        self.airsim_client = airsim_client

        # get initial positions according to settings offset
        # (later be used whenever we need to reset to initial position -> start of each epoch)
        # (arbitrary starting position in settings file as long as they are not spawned on top of each other)
        self.car1_x_offset = airsim_client.simGetObjectPose(CAR1_NAME).position.x_val
        self.car1_y_offset = airsim_client.simGetObjectPose(CAR1_NAME).position.y_val
        self.car2_x_offset = airsim_client.simGetObjectPose(CAR2_NAME).position.x_val
        self.car2_y_offset = airsim_client.simGetObjectPose(CAR2_NAME).position.y_val

        self.reset_cars_to_initial_positions()

    def reset_cars_to_initial_positions(self):

        self.airsim_client.reset()

        # pick at random (car 2 goes from left/right)
        left_or_right = random.choice([1, -1])

        car1_start_location_x = CAR1_INITIAL_POSITION[0] - self.car1_x_offset
        car1_start_location_y = left_or_right * CAR1_INITIAL_POSITION[1] - self.car1_y_offset
        car2_start_location_x = CAR2_INITIAL_POSITION[0] - self.car2_x_offset
        car2_start_location_y = left_or_right * CAR2_INITIAL_POSITION[1] - self.car2_y_offset
        car1_start_yaw = CAR1_INITIAL_YAW
        car2_start_yaw = left_or_right * CAR2_INITIAL_YAW

        # Set the reference_position for Car1 and Car2 (Do not change this code)
        reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
        self.airsim_client.simSetVehiclePose(reference_position, True, CAR1_NAME)
        self.airsim_client.simSetVehiclePose(reference_position, True, CAR2_NAME)

        # Convert yaw values from degrees to radians as AirSim uses radians
        car1_start_yaw_rad = np.radians(car1_start_yaw)
        car2_start_yaw_rad = np.radians(car2_start_yaw)

        # Set initial position and yaw of Car1
        initial_position_car1 = airsim.Vector3r(car1_start_location_x, car1_start_location_y, -1)
        initial_orientation_car1 = airsim.to_quaternion(0, 0, car1_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car1 = airsim.Pose(initial_position_car1, initial_orientation_car1)
        self.airsim_client.simSetVehiclePose(initial_pose_car1, True, CAR1_NAME)

        # Set initial position and yaw of Car2
        initial_position_car2 = airsim.Vector3r(car2_start_location_x, car2_start_location_y, -1)
        initial_orientation_car2 = airsim.to_quaternion(0, 0, car2_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car2 = airsim.Pose(initial_position_car2, initial_orientation_car2)
        self.airsim_client.simSetVehiclePose(initial_pose_car2, True, CAR2_NAME)

    def detect_and_handle_collision(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            # Handling the collision
            # Here, I'm just returning a fixed negative reward. You might want to include more complex logic.
            collision_reward = -1000
            return True, collision_reward
        else:
            # No collision occurred
            return False, 0
