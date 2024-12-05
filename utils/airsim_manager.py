import random
import time

import airsim
import numpy as np


class AirsimManager:

    """
    AirsimManager is responsible for handling cars (API between Simulation and code)
    """

    def __init__(self, experiment):
        self.simulation_paused = False
        self.experiment = experiment
        self.airsim_client = airsim.CarClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)  # Enable API control for Car1
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)  # Enable API control for Car2

        # Set cars initial throttle:
        car_controls_car_1 = airsim.CarControls()
        car_controls_car_1.throttle = 1
        self.airsim_client.setCarControls(car_controls_car_1, self.experiment.CAR1_NAME)
        car_controls_car_2 = airsim.CarControls()
        car_controls_car_2.throttle = 1
        self.airsim_client.setCarControls(car_controls_car_2, self.experiment.CAR2_NAME)

        # Get initial positions according to settings offset
        self.car1_x_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR1_NAME).position.x_val
        self.car1_y_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR1_NAME).position.y_val
        self.car2_x_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME).position.x_val
        self.car2_y_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME).position.y_val

        self.car1_initial_position_saved = None
        self.simulation_paused = False

        self.reset_cars_to_initial_positions()

    def reset_cars_to_initial_positions(self):
        self.airsim_client.reset()
        # Pick random directions for car1 and car2
        car1_direction = random.choice([1, -1])
        car2_direction = random.choice([1, -1])
        # Apply random starting positions for Car1
        if car1_direction == 1:
            car1_start_location_x = self.experiment.CAR1_INITIAL_POSITION_OPTION_1[0] - self.car1_x_offset
            car1_start_location_y = self.experiment.CAR1_INITIAL_POSITION_OPTION_1[1] - self.car1_y_offset
            car1_start_yaw = self.experiment.CAR1_INITIAL_YAW_OPTION_1
        else:
            car1_start_location_x = self.experiment.CAR1_INITIAL_POSITION_OPTION_2[0] - self.car1_x_offset
            car1_start_location_y = self.experiment.CAR1_INITIAL_POSITION_OPTION_2[1] - self.car1_y_offset
            car1_start_yaw = self.experiment.CAR1_INITIAL_YAW_OPTION_2

        # Apply random starting positions for Car2
        if car2_direction == 1:
            car2_start_location_x = self.experiment.CAR2_INITIAL_POSITION_OPTION_1[0] - self.car2_x_offset
            car2_start_location_y = self.experiment.CAR2_INITIAL_POSITION_OPTION_1[1] - self.car2_y_offset
            car2_start_yaw = self.experiment.CAR2_INITIAL_YAW_OPTION_1
        else:
            car2_start_location_x = self.experiment.CAR2_INITIAL_POSITION_OPTION_2[0] - self.car2_x_offset
            car2_start_location_y = self.experiment.CAR2_INITIAL_POSITION_OPTION_2[1] - self.car2_y_offset
            car2_start_yaw = self.experiment.CAR2_INITIAL_YAW_OPTION_2

        # Set the new positions after reset
        car1_start_yaw_rad = np.radians(car1_start_yaw)  # Convert yaw from degrees to radians
        car2_start_yaw_rad = np.radians(car2_start_yaw)  # Convert yaw from degrees to radians

        # Set the reference_position for Car1 and Car2 (Do not change this code)
        reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
        self.airsim_client.simSetVehiclePose(reference_position, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(reference_position, True, self.experiment.CAR2_NAME)

        initial_position_car1 = airsim.Vector3r(car1_start_location_x, car1_start_location_y, -1)
        initial_orientation_car1 = airsim.to_quaternion(0, 0, car1_start_yaw_rad)
        initial_pose_car1 = airsim.Pose(initial_position_car1, initial_orientation_car1)

        initial_position_car2 = airsim.Vector3r(car2_start_location_x, car2_start_location_y, -1)
        initial_orientation_car2 = airsim.to_quaternion(0, 0, car2_start_yaw_rad)
        initial_pose_car2 = airsim.Pose(initial_position_car2, initial_orientation_car2)

        # Set the poses for both cars
        self.airsim_client.simSetVehiclePose(initial_pose_car1, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(initial_pose_car2, True, self.experiment.CAR2_NAME)
        time.sleep(1)

    def collision_occurred(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            print('************************************Colission!!!!!**********************************************')
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

    def get_car1_state(self, logger=None):
        car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
        car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
        car1_state = np.array([
            car1_position_and_speed["x"],
            car1_position_and_speed["y"],
            car1_position_and_speed["Vx"],
            car1_position_and_speed["Vy"],
            car2_position_and_speed["x"],
            car2_position_and_speed["y"],
            car2_position_and_speed["Vx"],
            car2_position_and_speed["Vy"]
        ])

        if logger is not None and self.experiment.LOG_CAR_STATES:
            logger.log_state(car1_state, self.experiment.CAR1_NAME)
        return car1_state

    def get_car2_state(self, logger=None):
        car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
        car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
        car2_state = np.array([
            car2_position_and_speed["x"],
            car2_position_and_speed["y"],
            car2_position_and_speed["Vx"],
            car2_position_and_speed["Vy"],
            car1_position_and_speed["x"],
            car1_position_and_speed["y"],
            car1_position_and_speed["Vx"],
            car1_position_and_speed["Vy"]
        ])

        if logger is not None and self.experiment.LOG_CAR_STATES:
            logger.log_state(car2_state, self.experiment.CAR2_NAME)

        return car2_state

    def get_car1_initial_position(self):
        if self.car1_initial_position_saved is None:
            car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
            self.car1_initial_position_saved = np.array([car1_position_and_speed["x"], car1_position_and_speed["y"]])
        return self.car1_initial_position_saved

    def reset_for_new_episode(self):
        self.car1_initial_position_saved = None

    def has_reached_target(self, car_state):
        car1_initial_position = self.get_car1_initial_position()
        if car1_initial_position[0] > 0:
            return car_state[0] < self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
        elif car1_initial_position[0] < 0:
            return car_state[0] > self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0]

    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    def is_simulation_paused(self):
        return self.simulation_paused

