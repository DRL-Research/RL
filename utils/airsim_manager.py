import random

import airsim
import numpy as np

from RL.config import CAR1_INITIAL_POSITION, CAR2_INITIAL_POSITION, CAR1_INITIAL_YAW, CAR2_INITIAL_YAW, \
    CAR1_NAME, CAR2_NAME, CAR1_DESIRED_POSITION


class AirsimManager:

    def __init__(self):
        self.airsim_client = airsim.CarClient()  # Create an AirSim client for car simulation
        self.airsim_client.confirmConnection()  # Confirm the connection to the AirSim simulator

        self.airsim_client.enableApiControl(True, CAR1_NAME)  # Enable API control for Car1
        self.airsim_client.enableApiControl(True, CAR2_NAME)  # Enable API control for Car2

        # Set car 1 throttle to 1:
        car_controls_car_1 = airsim.CarControls()
        car_controls_car_1.throttle = 1
        self.airsim_client.setCarControls(car_controls_car_1, CAR1_NAME)

        # Set car 2 throttle to 1:
        car_controls = airsim.CarControls()
        car_controls.throttle = 1
        self.airsim_client.setCarControls(car_controls, CAR2_NAME)

        # get initial positions according to settings offset
        # (later be used whenever we need to reset to initial position -> start of each epoch)
        # (arbitrary starting position in settings file as long as they are not spawned on top of each other)
        self.car1_x_offset = self.airsim_client.simGetObjectPose(CAR1_NAME).position.x_val
        self.car1_y_offset = self.airsim_client.simGetObjectPose(CAR1_NAME).position.y_val
        self.car2_x_offset = self.airsim_client.simGetObjectPose(CAR2_NAME).position.x_val
        self.car2_y_offset = self.airsim_client.simGetObjectPose(CAR2_NAME).position.y_val

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

    def collision_occurred(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        return collision_info.has_collided

    def get_car_controls(self, car_name):
        return self.airsim_client.getCarControls(car_name)

    def set_car_controls(self, updated_car_controls, car_name):
        self.airsim_client.setCarControls(updated_car_controls, car_name)

    def get_car_position_and_speed(self, car_name):
        car_position = self.airsim_client.simGetObjectPose(car_name).position
        car_position_and_speed = {
            "x": car_position.x_val,
            "y": car_position.y_val,
            "Vx": self.airsim_client.getCarState(car_name).kinematics_estimated.linear_velocity.x_val,
            "Vy": self.airsim_client.getCarState(car_name).kinematics_estimated.linear_velocity.y_val,
        }
        return car_position_and_speed

    def get_cars_distance(self):
        car1_position_and_speed = self.get_car_position_and_speed(CAR1_NAME)
        car2_position_and_speed = self.get_car_position_and_speed(CAR2_NAME)
        dist_c1_c2 = np.sum(np.square(
            np.array([[car1_position_and_speed["x"], car1_position_and_speed["y"]]]) -
            np.array([[car2_position_and_speed["x"], car2_position_and_speed["y"]]])))
        return dist_c1_c2

    def get_car1_state(self):
        car1_position_and_speed = self.get_car_position_and_speed(CAR1_NAME)
        car2_position_and_speed = self.get_car_position_and_speed(CAR2_NAME)
        car1_state = np.array([
            car1_position_and_speed["x"],
            car1_position_and_speed["y"],
            car1_position_and_speed["Vx"],
            car1_position_and_speed["Vy"],
            car2_position_and_speed["x"],
            car2_position_and_speed["y"],
            car2_position_and_speed["Vx"],
            car2_position_and_speed["Vy"],
            self.get_cars_distance()
        ])
        return car1_state

    def get_car2_state(self):
        car2_position_and_speed = self.get_car_position_and_speed(CAR2_NAME)
        car1_position_and_speed = self.get_car_position_and_speed(CAR1_NAME)
        car2_state = np.array([
            car2_position_and_speed["x"],
            car2_position_and_speed["y"],
            car2_position_and_speed["Vx"],
            car2_position_and_speed["Vy"],
            car1_position_and_speed["x"],
            car1_position_and_speed["y"],
            car1_position_and_speed["Vx"],
            car1_position_and_speed["Vy"],
            self.get_cars_distance()
        ])
        return car2_state

    @staticmethod
    def has_reached_target(car_state):
        # TODO: change car_state[0] to be more generic
        return car_state[0] > CAR1_DESIRED_POSITION[0]
