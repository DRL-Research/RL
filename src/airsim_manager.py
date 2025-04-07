import time
import airsim
import numpy as np
import torch

class AirsimManager:
    """
    Manages the AirSim simulation including vehicle API control, resetting positions,
    retrieving state information, and computing proto embeddings for the Master network.

    Vehicles:
      - Car1: The agent-controlled vehicle.
      - Car2: An autonomous vehicle.
      - Car3: An additional autonomous vehicle (newly added).
    """
    def __init__(self, experiment):
        self.experiment = experiment
        self.simulation_paused = False

        # Connect to AirSim and confirm connection.
        self.airsim_client = airsim.CarClient()
        self.airsim_client.confirmConnection()

        # Enable API control for all vehicles.
        self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR3_NAME)

        # Set initial throttle for all vehicles.
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR1_NAME)
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR2_NAME)
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR3_NAME)

        # Variables for saving initial positions.
        self.car1_initial_position_saved = None
        self.car2_initial_position_saved = None
        self.car3_initial_position_saved = None

        self.reset_cars_to_initial_positions()

    def set_car_controls(self, controls, car_name):
        self.airsim_client.setCarControls(controls, car_name)

    def create_initial_pose(self, x, y, yaw_degrees):
        """
        Creates an initial Pose for a vehicle with the given (x,y) and yaw (degrees).
        Z is set to 0.1 to keep the vehicle above the ground.
        """
        yaw_rad = np.radians(yaw_degrees)
        position = airsim.Vector3r(x, y, 0.1)
        orientation = airsim.to_quaternion(0.0, 0.0, yaw_rad)
        return airsim.Pose(position, orientation)

    def reset_cars_to_initial_positions(self):
        """
        Resets the simulation and sets the vehicles to their starting poses.
        Example positions:
          - Car1: (-30, 0) with yaw=0.
          - Car2: (0, -30) with yaw=90.
          - Car3: (0, -35) with yaw=90.
        """
        self.airsim_client.reset()
        time.sleep(0.5)

        # Re-enable API control after reset.
        self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR3_NAME)

        car1_pose = self.create_initial_pose(-30, 0, 0)
        car2_pose = self.create_initial_pose(0, -30, 90)
        car3_pose = self.create_initial_pose(0, -35, 90)

        self.airsim_client.simSetVehiclePose(car1_pose, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(car2_pose, True, self.experiment.CAR2_NAME)
        self.airsim_client.simSetVehiclePose(car3_pose, True, self.experiment.CAR3_NAME)


    def reset_for_new_episode(self):
        """
        Clears saved initial positions for a new episode.
        """
        self.car1_initial_position_saved = None
        self.car2_initial_position_saved = None
        self.car3_initial_position_saved = None

    def get_car_position_and_speed(self, car_name):
        """
        Returns global position (x,y) and velocity (Vx,Vy) for the specified car.
        """
        pose = self.airsim_client.simGetVehiclePose(car_name)
        state = self.airsim_client.getCarState(car_name)
        position = pose.position
        return {
            "x": position.x_val,
            "y": position.y_val,
            "Vx": state.kinematics_estimated.linear_velocity.x_val,
            "Vy": state.kinematics_estimated.linear_velocity.y_val,
        }

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

    def get_car3_initial_position(self):
        if self.car3_initial_position_saved is None:
            state = self.get_car_position_and_speed(self.experiment.CAR3_NAME)
            self.car3_initial_position_saved = np.array([state["x"], state["y"]])
        return self.car3_initial_position_saved

    def get_car1_state(self):
        """
        Returns Car1's local state vector: [relative_x, relative_y, local_Vx, local_Vy].
        """
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR1_NAME)
        position = pose.position
        orientation = pose.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        car_state = self.get_car_position_and_speed(self.experiment.CAR1_NAME)
        v_global = np.array([car_state["Vx"], car_state["Vy"]])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw),  np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car1_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_local = R.dot(pos_global - init_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_car2_state(self):
        """
        Returns Car2's local state vector: [relative_x, relative_y, local_Vx, local_Vy].
        """
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME)
        position = pose.position
        orientation = pose.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        car_state = self.get_car_position_and_speed(self.experiment.CAR2_NAME)
        v_global = np.array([car_state["Vx"], car_state["Vy"]])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw),  np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car2_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_local = R.dot(pos_global - init_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_car3_state(self):
        """
        Returns Car3's local state vector: [relative_x, relative_y, local_Vx, local_Vy].
        """
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR3_NAME)
        position = pose.position
        orientation = pose.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        car_state = self.get_car_position_and_speed(self.experiment.CAR3_NAME)
        v_global = np.array([car_state["Vx"], car_state["Vy"]])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw),  np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car3_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_local = R.dot(pos_global - init_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_proto_state(self, master_model):
        """
        Constructs the proto state for the Master network by concatenating the local states of Car1, Car2, and Car3,
        passing the result through the master model, and returning a NumPy array embedding.
        """
        state_car1 = self.get_car1_state()
        state_car2 = self.get_car2_state()
        state_car3 = self.get_car3_state()
        # Concatenate the states (total dimension: 4+4+4 = 12)
        master_input = torch.tensor(np.concatenate((state_car1, state_car2, state_car3)),
                                      dtype=torch.float32).unsqueeze(0)
        environment_embedding = master_model.get_proto_action(master_input)
        # Ensure the output is a torch.Tensor and squeeze if needed.
        if not isinstance(environment_embedding, torch.Tensor):
            environment_embedding = torch.tensor(environment_embedding)
        if environment_embedding.dim() == 2 and environment_embedding.size(0) == 1:
            return environment_embedding.squeeze(0).cpu().numpy()
        else:
            return environment_embedding.cpu().numpy()

    def has_reached_target(self, car_state):
        """
        Checks if the vehicle has reached the target using its relative state.
        For Car1: compares relative x; for Car2: compares relative y.
        (You may need to extend this logic to include Car3 if desired.)
        """
        if self.experiment.ROLE == self.experiment.CAR1_NAME:
            init_pos = self.get_car1_initial_position()
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos[0] > 0
                              else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos[0])
            return abs(car_state[0]) >= required_distance
        elif self.experiment.ROLE == self.experiment.CAR2_NAME:
            init_pos = self.get_car2_initial_position()
            desired_global = (self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
                              if init_pos[1] > 0
                              else self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos[1])
            return abs(car_state[1]) >= required_distance
        return False

    def collision_occurred(self, car_name=None):
        """
        Checks whether a collision has occurred for the given car (default: Car1).
        """
        car = car_name if car_name is not None else self.experiment.CAR1_NAME
        collision_info = self.airsim_client.simGetCollisionInfo(car)
        if collision_info.has_collided:
            print(f"Collision detected for {car}!")
        return collision_info.has_collided

    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    def is_simulation_paused(self):
        return self.simulation_paused
