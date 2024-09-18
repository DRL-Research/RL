import random
import airsim
import numpy as np


class AirsimManager:

    def __init__(self, experiment):

        self.simulation_paused = False

        self.experiment = experiment
        self.airsim_client = airsim.CarClient()  # Create an AirSim client for car simulation
        self.airsim_client.confirmConnection()  # Confirm the connection to the AirSim simulator

        self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)  # Enable API control for Car1
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)  # Enable API control for Car2

        # Set car 1 throttle to 1:
        car_controls_car_1 = airsim.CarControls()
        car_controls_car_1.throttle = 1
        self.airsim_client.setCarControls(car_controls_car_1, self.experiment.CAR1_NAME)

        # Set car 2 throttle to 1:
        car_controls = airsim.CarControls()
        car_controls.throttle = 1
        self.airsim_client.setCarControls(car_controls, self.experiment.CAR2_NAME)

        # get initial positions according to settings offset
        # TODO: is the following 2 lines still true?
        # (later be used whenever we need to reset to initial position -> start of each epoch)
        # (arbitrary starting position in settings file as long as they are not spawned on top of each other)
        self.car1_x_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR1_NAME).position.x_val
        self.car1_y_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR1_NAME).position.y_val
        self.car2_x_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME).position.x_val
        self.car2_y_offset = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME).position.y_val

        self.reset_cars_to_initial_positions()


    def reset_cars_to_initial_positions(self):

        self.airsim_client.reset()

        # TODO: we pick left/right twice in the code... see the function: set_car2_initial_position_and_yaw
        # pick at random (car 2 goes from left/right)
        left_or_right = random.choice([1, -1])

        car1_start_location_x = self.experiment.CAR1_INITIAL_POSITION[0] - self.car1_x_offset
        car1_start_location_y = left_or_right * self.experiment.CAR1_INITIAL_POSITION[1] - self.car1_y_offset
        car2_start_location_x = self.experiment.CAR2_INITIAL_POSITION[0] - self.car2_x_offset
        car2_start_location_y = left_or_right * self.experiment.CAR2_INITIAL_POSITION[1] - self.car2_y_offset
        car1_start_yaw = self.experiment.CAR1_INITIAL_YAW
        car2_start_yaw = left_or_right * self.experiment.CAR2_INITIAL_YAW


        # Set the reference_position for Car1 and Car2 (Do not change the following code)
        reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
        self.airsim_client.simSetVehiclePose(reference_position, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(reference_position, True, self.experiment.CAR2_NAME)
        car1_start_yaw_rad = np.radians(car1_start_yaw) # Convert yaw values from degrees to radians as AirSim uses radians
        car2_start_yaw_rad = np.radians(car2_start_yaw) # Convert yaw values from degrees to radians as AirSim uses radians
        # Set initial position and yaw of Car1
        initial_position_car1 = airsim.Vector3r(car1_start_location_x, car1_start_location_y, -1)
        initial_orientation_car1 = airsim.to_quaternion(0, 0, car1_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car1 = airsim.Pose(initial_position_car1, initial_orientation_car1)
        self.airsim_client.simSetVehiclePose(initial_pose_car1, True, self.experiment.CAR1_NAME)
        # Set initial position and yaw of Car2
        initial_position_car2 = airsim.Vector3r(car2_start_location_x, car2_start_location_y, -1)
        initial_orientation_car2 = airsim.to_quaternion(0, 0, car2_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car2 = airsim.Pose(initial_position_car2, initial_orientation_car2)
        self.airsim_client.simSetVehiclePose(initial_pose_car2, True, self.experiment.CAR2_NAME)

    # TODO: If this function is not used, delete it
    def reset_cars_to_initial_settings_file_positions(self):
        self.airsim_client.reset()

        # car1_start_location_x = self.config.CAR1_INITIAL_POSITION[0] - self.car1_x_offset
        # car1_start_location_y = self.config.CAR1_INITIAL_POSITION[1] - self.car1_y_offset
        # car2_start_location_x = self.config.CAR2_INITIAL_POSITION[0] - self.car2_x_offset
        # car2_start_location_y = self.config.CAR2_INITIAL_POSITION[1] - self.car2_y_offset
        car1_start_location_x = self.car1_x_offset
        car1_start_location_y = self.car1_y_offset
        car2_start_location_x = self.car2_x_offset
        car2_start_location_y = self.car2_y_offset
        car1_start_yaw = self.experiment.CAR1_INITIAL_YAW
        car2_start_yaw = self.experiment.CAR2_INITIAL_YAW

        # # Set the reference_position for Car1 and Car2 (Do not change this code)
        # reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
        # self.airsim_client.simSetVehiclePose(reference_position, True, self.config.CAR1_NAME)
        # self.airsim_client.simSetVehiclePose(reference_position, True, self.config.CAR2_NAME)

        # Convert yaw values from degrees to radians as AirSim uses radians
        car1_start_yaw_rad = np.radians(car1_start_yaw)
        car2_start_yaw_rad = np.radians(car2_start_yaw)

        # Set initial position and yaw of Car1
        initial_position_car1 = airsim.Vector3r(car1_start_location_x, car1_start_location_y, -1)
        initial_orientation_car1 = airsim.to_quaternion(0, 0, car1_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car1 = airsim.Pose(initial_position_car1, initial_orientation_car1)
        self.airsim_client.simSetVehiclePose(initial_pose_car1, True, self.experiment.CAR1_NAME)

        # Set initial position and yaw of Car2
        initial_position_car2 = airsim.Vector3r(car2_start_location_x, car2_start_location_y, -1)
        initial_orientation_car2 = airsim.to_quaternion(0, 0, car2_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car2 = airsim.Pose(initial_position_car2, initial_orientation_car2)
        self.airsim_client.simSetVehiclePose(initial_pose_car2, True, self.experiment.CAR2_NAME)

    def collision_occurred(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            print('************************************Colission!!!!!**********************************************')
            print(collision_info.has_collided)
        return collision_info.has_collided

    # TODO: If this function is not used, delete it
    def get_collision_occured_outside(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            return 1
        return 0

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
        car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
        car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
        dist_c1_c2 = np.sum(np.square(
            np.array([[car1_position_and_speed["x"], car1_position_and_speed["y"]]]) -
            np.array([[car2_position_and_speed["x"], car2_position_and_speed["y"]]])))
        return dist_c1_c2

    def get_car1_state(self, logger=None):
        car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
        car1_state = np.array([
            car1_position_and_speed["x"],
            car1_position_and_speed["y"],
            car1_position_and_speed["Vx"],
            car1_position_and_speed["Vy"],
        ])

        if logger is not None and self.experiment.LOG_CAR_STATES:
            logger.log_state(car1_state, self.experiment.CAR1_NAME)

        return car1_state

    def get_car2_state(self, logger):
        car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
        car2_state = np.array([
            car2_position_and_speed["x"],
            car2_position_and_speed["y"],
            car2_position_and_speed["Vx"],
            car2_position_and_speed["Vy"],
        ])

        if self.experiment.LOG_CAR_STATES:
            logger.log_state(car2_state, self.experiment.CAR2_NAME)

        return car2_state

    def has_reached_target(self, car_state):
        # TODO: change car_state[0] to be more generic
        return car_state[0] > self.experiment.CAR1_DESIRED_POSITION[0]


    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    # TODO: If this function is not used, delete it
    def is_simulation_paused(self):
        return self.simulation_paused

    def set_car2_initial_position_and_yaw(self):
        car2_side = random.choice(["left", "right"])
        if car2_side == "left":
            # TODO: if this is the function being used, CAR2_INITIAL_POSITION needs to be defined in experiment, not here.
            self.experiment.CAR2_INITIAL_POSITION = [0, -30]
            self.experiment.CAR2_INITIAL_YAW = 90
        else:
            self.experiment.CAR2_INITIAL_POSITION = [0, 30]
            self.experiment.CAR2_INITIAL_YAW = 270

        print(f"Car 2 starts from {car2_side} with position {self.experiment.CAR2_INITIAL_POSITION} and yaw {self.experiment.CAR2_INITIAL_YAW}")