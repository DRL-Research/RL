import random
import airsim
import numpy as np
from utils.experiment.experiment_constants import StartingLocation
from turns.initialization.config_turns import *
from turns.initialization.setup_simulation_turns import SetupManager


class AirsimManager:
    """
    AirsimManager is responsible for handling cars (API between Simulation and code)
    """

    def __init__(self, experiment=None, setup_manager_cars=None):
        self.simulation_paused = False
        self.experiment = experiment
        self.airsim_client = airsim.CarClient()
        self.airsim_client.confirmConnection()
        self.car_name_to_offset = {}
        self.cars = setup_manager_cars

        # Initialize cars for different contexts
        if self.experiment:
            self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)
            self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)

            # Set initial throttle
            car_controls_car_1 = airsim.CarControls()
            car_controls_car_1.throttle = 1
            self.airsim_client.setCarControls(car_controls_car_1, self.experiment.CAR1_NAME)

            car_controls_car_2 = airsim.CarControls()
            car_controls_car_2.throttle = 1
            self.airsim_client.setCarControls(car_controls_car_2, self.experiment.CAR2_NAME)

            # Offsets
            self.car1_x_offset = 0
            self.car1_y_offset = 0
            self.car2_x_offset = 0
            self.car2_y_offset = 5

            self.car1_initial_position_saved = None
            self.reset_cars_to_initial_positions()

        elif self.cars:
            self.enable_api_cars_control()

            # Set initial throttle for active cars
            for car_obj in self.cars.values():
                if car_obj.is_active:
                    self.set_car_throttle_by_name(car_obj.name)

                car_pos = self.airsim_client.simGetObjectPose(car_obj.name).position
                self.car_name_to_offset[car_obj.name] = {'x_offset': car_pos.x_val, 'y_offset': car_pos.y_val}

            self.reset_cars_to_initial_positions()

    def enable_api_cars_control(self):
        for car in self.cars.values():
            if not self.airsim_client.isApiControlEnabled(car.name):
                self.airsim_client.enableApiControl(is_enabled=True, vehicle_name=car.name)

    def set_car_throttle_by_name(self, car_name, throttle=0.4):
        car_controls = airsim.CarControls()
        car_controls.throttle = throttle
        self.airsim_client.setCarControls(car_controls, car_name)

    def get_start_location(self, car_name, side=None):
        if self.cars.get(car_name):
            car = self.cars[car_name]
            car_offset_x = self.car_name_to_offset[car_name]['x_offset']
            car_offset_y = self.car_name_to_offset[car_name]['y_offset']
            car_start_location_x = car.get_start_location_x(car_offset_x)
            car_start_location_y = car.get_start_location_y(car_offset_y, side)
            return car_start_location_x, car_start_location_y
        return None, None

    def reset_cars_to_initial_positions(self):
        self.airsim_client.reset()

        if self.experiment:
            def get_initial_position_and_yaw(car_direction, position_option_1, position_option_2, yaw_option_1,
                                             yaw_option_2, x_offset, y_offset):
                if car_direction == StartingLocation.RIGHT:
                    start_location_x = position_option_1[0] - x_offset
                    start_location_y = position_option_1[1] - y_offset
                    start_yaw = yaw_option_1
                else:
                    start_location_x = position_option_2[0] - x_offset
                    start_location_y = position_option_2[1] - y_offset
                    start_yaw = yaw_option_2
                return start_location_x, start_location_y, start_yaw

            if self.experiment.RANDOM_INIT:
                car1_direction = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])
                car2_direction = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])

            car1_start_location_x, car1_start_location_y, car1_start_yaw = get_initial_position_and_yaw(
                car1_direction,
                self.experiment.CAR1_INITIAL_POSITION_OPTION_1,
                self.experiment.CAR1_INITIAL_POSITION_OPTION_2,
                self.experiment.CAR1_INITIAL_YAW_OPTION_1,
                self.experiment.CAR1_INITIAL_YAW_OPTION_2,
                self.car1_x_offset,
                self.car1_y_offset
            )
            car2_start_location_x, car2_start_location_y, car2_start_yaw = get_initial_position_and_yaw(
                car2_direction,
                self.experiment.CAR2_INITIAL_POSITION_OPTION_1,
                self.experiment.CAR2_INITIAL_POSITION_OPTION_2,
                self.experiment.CAR2_INITIAL_YAW_OPTION_1,
                self.experiment.CAR2_INITIAL_YAW_OPTION_2,
                self.car2_x_offset,
                self.car2_y_offset
            )

            def create_initial_pose(x, y, yaw_rad):
                position = airsim.Vector3r(x, y, -1.0)
                orientation = airsim.to_quaternion(0.0, 0.0, yaw_rad)
                return airsim.Pose(position, orientation)

            initial_pose_car1 = create_initial_pose(car1_start_location_x, car1_start_location_y,
                                                    np.radians(car1_start_yaw))
            initial_pose_car2 = create_initial_pose(car2_start_location_x, car2_start_location_y,
                                                    np.radians(car2_start_yaw))

            self.airsim_client.simSetVehiclePose(initial_pose_car1, True, self.experiment.CAR1_NAME)
            self.airsim_client.simSetVehiclePose(initial_pose_car2, True, self.experiment.CAR2_NAME)

        elif self.cars:
            car1_start_location_x, car1_start_location_y = self.get_start_location(CAR1_NAME)
            car2_start_location_x, car2_start_location_y = self.get_start_location(CAR2_NAME, 1)

            reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
            for car_name in self.cars:
                self.airsim_client.simSetVehiclePose(reference_position, True, car_name)

            if self.cars.get(CAR1_NAME):
                self.set_initial_position_and_yaw(car_name=CAR1_NAME, start_location_x=car1_start_location_x,
                                                  start_location_y=car1_start_location_y,
                                                  car_start_yaw=CAR1_INITIAL_YAW)
            if self.cars.get(CAR2_NAME):
                self.set_initial_position_and_yaw(car_name=CAR2_NAME, start_location_x=car2_start_location_x,
                                                  start_location_y=car2_start_location_y,
                                                  car_start_yaw=CAR2_INITIAL_YAW)

    def set_initial_position_and_yaw(self, car_name, start_location_x, start_location_y, car_start_yaw):
        car_start_yaw_rad = np.radians(car_start_yaw)
        initial_position_car = airsim.Vector3r(start_location_x, start_location_y, -1)
        initial_orientation_car = airsim.to_quaternion(0, 0, car_start_yaw_rad)
        initial_pose_car = airsim.Pose(initial_position_car, initial_orientation_car)
        self.airsim_client.simSetObjectPose(car_name, initial_pose_car, True)

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

    def get_local_input_car1_perspective(self):
        car1_state = self.get_car_position_and_speed(CAR1_NAME)
        car2_state = self.get_car_position_and_speed(CAR2_NAME)
        dist_c1_c2 = self.get_cars_distance()
        local_input_car1_perspective = {
            "x_c1": car1_state["x"],
            "y_c1": car1_state["y"],
            "Vx_c1": car1_state["Vx"],
            "Vy_c1": car1_state["Vy"],
            "x_c2": car2_state["x"],
            "y_c2": car2_state["y"],
            "Vx_c2": car2_state["Vx"],
            "Vy_c2": car2_state["Vy"],
            "dist_c1_c2": dist_c1_c2
        }
        return local_input_car1_perspective

    def get_local_input_car2_perspective(self):
        car2_state = self.get_car_position_and_speed(CAR2_NAME)
        car1_state = self.get_car_position_and_speed(CAR1_NAME)
        dist_c1_c2 = self.get_cars_distance()
        local_input_car2_perspective = {
            "x_c2": car2_state["x"],
            "y_c2": car2_state["y"],
            "Vx_c2": car2_state["Vx"],
            "Vy_c2": car2_state["Vy"],
            "x_c1": car1_state["x"],
            "y_c1": car1_state["y"],
            "Vx_c1": car1_state["Vx"],
            "Vy_c1": car1_state["Vy"],
            "dist_c1_c2": dist_c1_c2
        }
        return local_input_car2_perspective

    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    def is_simulation_paused(self):
        return self.simulation_paused

    def has_reached_target(self, car_state):
        if self.experiment:
            car1_initial_position = self.get_car1_initial_position()
            if car1_initial_position[0] > 0:
                return car_state[0] < self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
            elif car1_initial_position[0] < 0:
                return car_state[0] > self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0]
        else:
            return car_state['x_c1'] > CAR1_DESIRED_POSITION[0]

    def get_car1_initial_position(self):
        if self.car1_initial_position_saved is None:
            car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
            self.car1_initial_position_saved = np.array([car1_position_and_speed["x"], car1_position_and_speed["y"]])
        return self.car1_initial_position_saved

    def reset_for_new_episode(self):
        self.car1_initial_position_saved = None

    @staticmethod
    def stop_car(airsim_client, moving_car_name, throttle=0.0):
        car_controls = airsim.CarControls()
        car_controls.throttle = throttle
        if throttle > 0:
            car_controls.brake = 1.0
        airsim_client.setCarControls(car_controls, vehicle_name=moving_car_name)