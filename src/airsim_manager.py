import random
import airsim
import numpy as np
import torch
from src.constants import StartingLocation, CarName

class AirsimManager:
    """
    AirsimManager is responsible for handling cars (API between Simulation and code)
    """

    def __init__(self, experiment):
        self.simulation_paused = False
        self.experiment = experiment
        self.airsim_client = airsim.CarClient()
        self.airsim_client.confirmConnection()

        # TODO
        # Enable API control for Car1, Car2 and Car3
        # self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)  # Car1
        # self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)  # Car2
        # self.airsim_client.enableApiControl(True, "Car3")                      # Car3

        # Enable API control for Car1, Car2 and Car3
        self.car_names = self.experiment.CAR_NAMES
        for car_name in self.car_names:
            self.airsim_client.enableApiControl(True, car_name)

        # Set cars initial throttle:
        car_controls_car_1 = airsim.CarControls()
        car_controls_car_1.throttle = 1
        self.airsim_client.setCarControls(car_controls_car_1, self.experiment.CAR1_NAME)
        car_controls_car_2 = airsim.CarControls()
        car_controls_car_2.throttle = 1
        self.airsim_client.setCarControls(car_controls_car_2, self.experiment.CAR2_NAME)
        # car_controls_car_3 = airsim.CarControls()
        # car_controls_car_3.throttle = 1
        # self.airsim_client.setCarControls(car_controls_car_3, "Car3")

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
        # Reset the simulation
        self.airsim_client.reset()

        # Re-enable API control for all vehicles after reset
        self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)
        # self.airsim_client.enableApiControl(True, "Car3")

        # Helper function to calculate starting position and orientation
        def get_initial_position_and_yaw(car_direction, pos_opt1, pos_opt2, yaw_opt1, yaw_opt2, x_offset, y_offset):
            if car_direction == StartingLocation.RIGHT:
                start_x = pos_opt1[0] - x_offset
                start_y = pos_opt1[1] - y_offset
                start_yaw = yaw_opt1
            else:
                start_x = pos_opt2[0] - x_offset
                start_y = pos_opt2[1] - y_offset
                start_yaw = yaw_opt2
            return start_x, start_y, start_yaw

        if hasattr(self.experiment, "INIT_SERIAL") and self.experiment.INIT_SERIAL:
            car1_dir, car2_dir = self.serial_combinations[self.serial_counter]
            self.serial_counter = (self.serial_counter + 1) % len(self.serial_combinations)
        elif self.experiment.RANDOM_INIT:
            car1_dir = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])
            car2_dir = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])
        else:
            car1_dir = StartingLocation.RIGHT
            car2_dir = random.choice([StartingLocation.LEFT, StartingLocation.RIGHT])

        # Calculate starting positions and orientations for Car1 and Car2
        car1_start_x, car1_start_y, car1_start_yaw = get_initial_position_and_yaw(
            car1_dir,
            self.experiment.CAR1_INITIAL_POSITION_OPTION_1,
            self.experiment.CAR1_INITIAL_POSITION_OPTION_2,
            self.experiment.CAR1_INITIAL_YAW_OPTION_1,
            self.experiment.CAR1_INITIAL_YAW_OPTION_2,
            self.car1_x_offset,
            self.car1_y_offset
        )
        car2_start_x, car2_start_y, car2_start_yaw = get_initial_position_and_yaw(
            car2_dir,
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
        initial_pose_car1 = create_initial_pose(car1_start_x, car1_start_y, np.radians(car1_start_yaw))
        initial_pose_car2 = create_initial_pose(car2_start_x, car2_start_y, np.radians(car2_start_yaw))
        self.airsim_client.simSetVehiclePose(initial_pose_car1, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(initial_pose_car2, True, self.experiment.CAR2_NAME)

        # Set the pose for Car3 based on Car2's position and orientation.
        # We place Car3 a fixed distance behind Car2 along the reverse direction of Car2's heading.
        # distance_offset = 5  # desired distance behind Car2
        # car2_yaw_rad = np.radians(car2_start_yaw)
        # car3_start_x = car2_start_x + distance_offset * np.cos(car2_yaw_rad + np.pi)
        # car3_start_y = car2_start_y + distance_offset * np.sin(car2_yaw_rad + np.pi)
        # car3_start_yaw = car2_start_yaw  # Same heading as Car2

        # initial_pose_car3 = create_initial_pose(car3_start_x, car3_start_y, np.radians(car3_start_yaw))
        # self.airsim_client.simSetVehiclePose(initial_pose_car3, True, "Car3")
        # print("car 1 initial position: ", car1_start_x, car1_start_y)
        # print("car 2 initial position: ", car2_start_x, car2_start_y)
        # print("car 3 initial position: ", car3_start_x, car3_start_y)

    def collision_occurred(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        if collision_info.has_collided:
            print('************************************Collision!!!!!**********************************************')
        return collision_info.has_collided

    def get_car_controls(self, car_name):
        return self.airsim_client.getCarControls(car_name)

    def set_car_controls(self, updated_car_controls, car_name):
        self.airsim_client.setCarControls(updated_car_controls, car_name)

    def get_vehicle_speed(self, car_name):
        """Returns the speed (magnitude of velocity vector) of the given vehicle."""
        car_state = self.airsim_client.getCarState(car_name)
        velocity = car_state.kinematics_estimated.linear_velocity
        speed = np.sqrt(velocity.x_val ** 2 + velocity.y_val ** 2)  # Compute speed magnitude
        return speed

    def get_car_position_and_speed(self, car_name):
        car_position = self.airsim_client.simGetObjectPose(car_name).position
        state = self.airsim_client.getCarState(car_name)
        return {
            "x": car_position.x_val,
            "y": car_position.y_val,
            "Vx": state.kinematics_estimated.linear_velocity.x_val,
            "Vy": state.kinematics_estimated.linear_velocity.y_val,
        }

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
        rel_pos = pos_global - init_pos
        rel_local = R.dot(rel_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

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
        rel_pos = pos_global - init_pos
        rel_local = R.dot(rel_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_combined_states(self):
        """
        Get the concatenated states of Car1 and Car2 for input to the MasterNetwork.
        """
        car1_state = self.get_car1_state()
        car2_state = self.get_car2_state()
        return torch.tensor(np.concatenate((car1_state, car2_state)), dtype=torch.float32)

    def get_car1_initial_position(self):
        if self.car1_initial_position_saved is None:
            state = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
            self.car1_initial_position_saved = np.array([state["x"], state["y"]])
        return self.car1_initial_position_saved

    def get_car2_initial_position(self):
        if self.car2_initial_position_saved is None:
            state = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
            self.car2_initial_position_saved = np.array([state["x"], state["y"]])
        return self.car2_initial_position_saved

    def reset_for_new_episode(self):
        self.car1_initial_position_saved = None

    def has_reached_target(self, car_state):
        if self.experiment.ROLE == CarName.CAR1:
            init_pos = self.get_car1_initial_position()
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos[0] > 0
                              else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos[0])
            return car_state[0] >= required_distance
        elif self.experiment.ROLE == CarName.CAR2:
            init_pos = self.get_car2_initial_position()
            desired_global = (self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
                              if init_pos[1] > 0
                              else self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos[1])
            return car_state[0] >= required_distance
        return False

    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    def is_simulation_paused(self):
        return self.simulation_paused

    def get_proto_state(self, master_model):
        state_car1 = self.get_car1_state()
        state_car2 = self.get_car2_state()
        master_input = torch.tensor(
            [state_car1.tolist() + state_car2.tolist()],
            dtype=torch.float32
        )
        # Get embedding from master network (squeeze makes it 8-dim instead of 1x8)
        environment_embedding = master_model.get_proto_action(master_input)
        return environment_embedding