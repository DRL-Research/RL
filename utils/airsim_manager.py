import random
import torch
import airsim
import numpy as np
from utils.experiment.experiment_constants import StartingLocation,CarName


import random
import numpy as np
import airsim

class StartingLocation:
    LEFT = "LEFT"
    RIGHT = "RIGHT"


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
        self.car1_x_offset = 0  # Change only if settings.json is changed.
        self.car1_y_offset = 0
        self.car2_x_offset = 0
        self.car2_y_offset = 5

        self.car1_initial_position_saved = None
        self.car2_initial_position_saved = None
        self.simulation_paused = False
        if hasattr(self.experiment, "INIT_SERIAL") and self.experiment.INIT_SERIAL:
            self.serial_combinations = [
                (StartingLocation.LEFT, StartingLocation.RIGHT),
                (StartingLocation.RIGHT, StartingLocation.RIGHT),
                (StartingLocation.RIGHT, StartingLocation.LEFT),
                (StartingLocation.LEFT, StartingLocation.LEFT)
            ]
            self.serial_counter = 0

        self.reset_cars_to_initial_positions()

    def reset_cars_to_initial_positions(self):
        self.airsim_client.reset()

        # Helper function to calculate starting position and orientation
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

        if hasattr(self.experiment, "INIT_SERIAL") and self.experiment.INIT_SERIAL:
            car1_direction, car2_direction = self.serial_combinations[self.serial_counter]
            self.serial_counter = (self.serial_counter + 1) % len(self.serial_combinations)
        elif self.experiment.RANDOM_INIT:
            car1_direction = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])
            car2_direction = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])
        else:
            car1_direction = StartingLocation.RIGHT
            car2_direction = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])

        # Get starting positions and orientations for Car1 and Car2
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

        # Helper function to create initial pose
        def create_initial_pose(x, y, yaw_rad):
            position = airsim.Vector3r(x, y, -1.0)
            orientation = airsim.to_quaternion(0.0, 0.0, yaw_rad)
            return airsim.Pose(position, orientation)

        # Set the poses for Car1 and Car2
        initial_pose_car1 = create_initial_pose(car1_start_location_x, car1_start_location_y, np.radians(car1_start_yaw))
        initial_pose_car2 = create_initial_pose(car2_start_location_x, car2_start_location_y, np.radians(car2_start_yaw))

        self.airsim_client.simSetVehiclePose(initial_pose_car1, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(initial_pose_car2, True, self.experiment.CAR2_NAME)

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

    # def get_car1_state(self,):
    #     car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
    #     car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
    #     car1_state = np.array([
    #         car1_position_and_speed["x"],
    #         car1_position_and_speed["y"],
    #         abs(car1_position_and_speed["Vx"]),
    #         abs(car1_position_and_speed["Vy"]),
    #     ])
    #     #
    #     # if logger is not None and self.experiment.LOG_CAR_STATES:
    #     #     logger.log_state(car1_state, self.experiment.CAR1_NAME)
    #     return car1_state
    #
    # def get_car2_state(self):
    #     car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
    #     car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
    #     car2_state = np.array([
    #         car2_position_and_speed["x"],
    #         car2_position_and_speed["y"],
    #         abs(car2_position_and_speed["Vx"]),
    #         abs(car2_position_and_speed["Vy"]),
    #     ])
    #     #print(car2_position_and_speed["x"],car2_position_and_speed["y"])
    #
    #     # if logger is not None and self.experiment.LOG_CAR_STATES:
    #     #     logger.log_state(car2_state, self.experiment.CAR2_NAME)
    #
    #     return car2_state

    def get_car1_state(self):
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR1_NAME)
        position = pose.position
        orientation = pose.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        car_state = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
        Vx = car_state["Vx"]
        Vy = car_state["Vy"]
        v_global = np.array([Vx, Vy])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw), np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car1_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_pos_global = pos_global - init_pos
        rel_pos_local = R.dot(rel_pos_global)
        car1_state_local = np.array([rel_pos_local[0], rel_pos_local[1], v_local[0], v_local[1]])

        return car1_state_local

    def get_car2_state(self):
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME)
        position = pose.position
        orientation = pose.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)

        car_state = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
        Vx = car_state["Vx"]
        Vy = car_state["Vy"]
        v_global = np.array([Vx, Vy])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw), np.cos(-yaw)]])
        v_local = R.dot(v_global)

        init_pos = self.get_car2_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_pos_global = pos_global - init_pos
        rel_pos_local = R.dot(rel_pos_global)

        car2_state_local = np.array([rel_pos_local[0], rel_pos_local[1], v_local[0], v_local[1]])
        return car2_state_local

    def get_combined_states(self):
        """
        Get the concatenated states of Car1 and Car2 for input to the MasterNetwork.
        """
        car1_state = self.get_car1_state()
        car2_state = self.get_car2_state()
        return torch.tensor(np.concatenate((car1_state, car2_state)), dtype=torch.float32)

    def get_car1_initial_position(self):
        if self.car1_initial_position_saved is None:
            car1_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
            self.car1_initial_position_saved = np.array([car1_position_and_speed["x"], car1_position_and_speed["y"]])
        return self.car1_initial_position_saved

    def get_car2_initial_position(self):
        if self.car2_initial_position_saved is None:
            car2_position_and_speed = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
            self.car2_initial_position_saved = np.array([car2_position_and_speed["x"], car2_position_and_speed["y"]])
        return self.car2_initial_position_saved

    def reset_for_new_episode(self):
        self.car1_initial_position_saved = None

    def has_reached_target(self, car_state):
        if self.experiment.ROLE == CarName.CAR1:
            init_pos = self.get_car1_initial_position()

            if init_pos[0] > 0:
                desired_global_x = self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
            else:
                desired_global_x = self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0]
            required_distance = abs(desired_global_x - init_pos[0])
            return car_state[0] >= required_distance

        elif self.experiment.ROLE == CarName.CAR2:
            init_pos = self.get_car2_initial_position()  # [x, y]
            if init_pos[1] > 0:
                desired_global_y = self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
            else:
                desired_global_y = self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1]
            required_distance = abs(desired_global_y - init_pos[1])
            return car_state[0] >= required_distance

        return False


    # def has_reached_target(self, car_state):
    #     if self.experiment.ROLE is CarName.CAR1:
    #         car1_initial_position = self.get_car1_initial_position()
    #         if car1_initial_position[0] > 0:
    #             return car_state[0] < self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
    #         elif car1_initial_position[0] < 0:
    #             return car_state[0] > self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0]
    #
    #     if self.experiment.ROLE is CarName.CAR2:
    #         car2_initial_position = self.get_car2_initial_position()
    #         if car2_initial_position[1] > 0:
    #             return car_state[1] < self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
    #         elif car2_initial_position[1] < 0:
    #             return car_state[1] > self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1]
    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    def is_simulation_paused(self):
        return self.simulation_paused

    def get_proto_state(self,master_model):
        state_car1 = self.get_car1_state()
        state_car2 = self.get_car2_state()
        master_input = torch.tensor(
            [state_car1.tolist() + state_car2.tolist()], dtype=torch.float32
        )
        # Get embedding from master network (squeeze make it 8 and not 1,8)
        environment_embedding = master_model.get_proto_action(master_input)
        return environment_embedding