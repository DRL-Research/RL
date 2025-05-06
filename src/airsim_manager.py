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
      - Car2: An autonomous vehicle (perpendicular).
      - Car3: An additional autonomous vehicle (perpendicular).
      - Car4: A vehicle driving in front of Car1 at high speed.
      - Car5: A vehicle driving in the opposite direction to Cars 2 and 3.
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
        self.airsim_client.enableApiControl(True, self.experiment.CAR4_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR5_NAME)

        # Set initial throttle for all vehicles.
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR1_NAME)
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR2_NAME)
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR3_NAME)
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR4_NAME)
        self.set_car_controls(airsim.CarControls(throttle=1), self.experiment.CAR5_NAME)

        # Variables for saving initial positions.
        self.car1_initial_position_saved = None
        self.car2_initial_position_saved = None
        self.car3_initial_position_saved = None
        self.car4_initial_position_saved = None
        self.car5_initial_position_saved = None

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
        Positions:
          - Car1: (-30, 0) with yaw=0.
          - Car2: (0, -30) with yaw=90.
          - Car3: (0, -35) with yaw=90.
          - Car4: (-20, 0) with yaw=0 (in front of Car1).
          - Car5: (0, 30) with yaw=270 (opposite direction to Cars 2 and 3).
        """
        self.airsim_client.reset()

        # Re-enable API control after reset.
        self.airsim_client.enableApiControl(True, self.experiment.CAR1_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR3_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR4_NAME)
        self.airsim_client.enableApiControl(True, self.experiment.CAR5_NAME)

        car1_pose = self.create_initial_pose(-30, 0, 0)
        car2_pose = self.create_initial_pose(0, -30, 90)
        car3_pose = self.create_initial_pose(0, -45, 90)
        car4_pose = self.create_initial_pose(-40, 0, 0)  # In front of Car1
        car5_pose = self.create_initial_pose(0, 20, 270)  # Opposite direction to Cars 2 and 3

        self.airsim_client.simSetVehiclePose(car1_pose, True, self.experiment.CAR1_NAME)
        self.airsim_client.simSetVehiclePose(car2_pose, True, self.experiment.CAR2_NAME)
        self.airsim_client.simSetVehiclePose(car3_pose, True, self.experiment.CAR3_NAME)
        self.airsim_client.simSetVehiclePose(car4_pose, True, self.experiment.CAR4_NAME)
        self.airsim_client.simSetVehiclePose(car5_pose, True, self.experiment.CAR5_NAME)

        time.sleep(0.5)
    def reset_for_new_episode(self):
        """
        Clears saved initial positions for a new episode.
        """
        self.car1_initial_position_saved = None
        self.car2_initial_position_saved = None
        self.car3_initial_position_saved = None
        self.car4_initial_position_saved = None
        self.car5_initial_position_saved = None

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

    def get_car4_initial_position(self):
        if self.car4_initial_position_saved is None:
            state = self.get_car_position_and_speed(self.experiment.CAR4_NAME)
            self.car4_initial_position_saved = np.array([state["x"], state["y"]])
        return self.car4_initial_position_saved

    def get_car5_initial_position(self):
        if self.car5_initial_position_saved is None:
            state = self.get_car_position_and_speed(self.experiment.CAR5_NAME)
            self.car5_initial_position_saved = np.array([state["x"], state["y"]])
        return self.car5_initial_position_saved

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
                      [np.sin(-yaw), np.cos(-yaw)]])
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
                      [np.sin(-yaw), np.cos(-yaw)]])
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
                      [np.sin(-yaw), np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car3_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_local = R.dot(pos_global - init_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_car4_state(self):
        """
        Returns Car4's local state vector: [relative_x, relative_y, local_Vx, local_Vy].
        """
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR4_NAME)
        position = pose.position
        orientation = pose.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        car_state = self.get_car_position_and_speed(self.experiment.CAR4_NAME)
        v_global = np.array([car_state["Vx"], car_state["Vy"]])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw), np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car4_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_local = R.dot(pos_global - init_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_car5_state(self):
        """
        Returns Car5's local state vector: [relative_x, relative_y, local_Vx, local_Vy].
        """
        pose = self.airsim_client.simGetObjectPose(self.experiment.CAR5_NAME)
        position = pose.position
        orientation = pose.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        car_state = self.get_car_position_and_speed(self.experiment.CAR5_NAME)
        v_global = np.array([car_state["Vx"], car_state["Vy"]])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                      [np.sin(-yaw), np.cos(-yaw)]])
        v_local = R.dot(v_global)
        init_pos = self.get_car5_initial_position()
        pos_global = np.array([position.x_val, position.y_val])
        rel_local = R.dot(pos_global - init_pos)
        return np.array([rel_local[0], rel_local[1], v_local[0], v_local[1]])

    def get_proto_state(self, master_model):
        """
        Constructs the proto state for the Master network by concatenating the local states of all 5 cars,
        passing the result through the master model, and returning a NumPy array embedding.
        """
        state_car1 = self.get_car1_state()
        state_car2 = self.get_car2_state()
        state_car3 = self.get_car3_state()
        state_car4 = self.get_car4_state()
        state_car5 = self.get_car5_state()
        # Concatenate the states (total dimension: 4+4+4+4+4 = 20)
        master_input = torch.tensor(np.concatenate((state_car1, state_car2, state_car3, state_car4, state_car5)),
                                    dtype=torch.float32).unsqueeze(0)
        environment_embedding = master_model.get_proto_action(master_input)
        # Ensure the output is a torch.Tensor and squeeze if needed.
        if not isinstance(environment_embedding, torch.Tensor):
            environment_embedding = torch.tensor(environment_embedding)
        if environment_embedding.dim() == 2 and environment_embedding.size(0) == 1:
            return environment_embedding.squeeze(0).cpu().numpy()
        else:
            return environment_embedding.cpu().numpy()

    def has_reached_target(self, car_state, car_name=None):
        """
        Checks if the vehicle has reached the target using its relative state.
        When car_name is provided, checks for that specific car.
        """
        car_role = car_name if car_name is not None else self.experiment.ROLE

        if car_role == self.experiment.CAR1_NAME:
            init_pos = self.get_car1_initial_position()
            desired_global = (self.experiment.CAR1_DESIRED_POSITION_OPTION_1[0]
                              if init_pos[0] > 0
                              else self.experiment.CAR1_DESIRED_POSITION_OPTION_2[0])
            required_distance = abs(desired_global - init_pos[0])
            return abs(car_state[0]) >= required_distance
        elif car_role == self.experiment.CAR3_NAME:
            init_pos = self.get_car3_initial_position()
            desired_global = (self.experiment.CAR3_DESIRED_POSITION_OPTION_1[1]
                              if init_pos[1] > 0
                              else self.experiment.CAR3_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos[1])
            return abs(car_state[1]) >= required_distance
        elif car_role == self.experiment.CAR2_NAME:
            init_pos = self.get_car2_initial_position()
            desired_global = (self.experiment.CAR2_DESIRED_POSITION_OPTION_1[1]
                              if init_pos[1] > 0
                              else self.experiment.CAR2_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos[1])
            return abs(car_state[1]) >= required_distance
        return False

    def collision_occurred(self, car_name=None):
        """
        Checks whether a collision has occurred for the given car.
        """
        car = car_name if car_name is not None else self.experiment.CAR1_NAME
        collision_info = self.airsim_client.simGetCollisionInfo(car)
        if collision_info.has_collided:
            print(f"Collision detected for {car}!")
        return collision_info.has_collided

    def hard_reset_simulation(self):
        """
        Performs a complete, forceful reset of the simulation when anomalies are detected.
        """
        print("Performing hard reset of simulation...")

        # First, try a standard reset
        self.airsim_client.reset()
        time.sleep(0.5)

        # Re-enable API control for all vehicles
        for car_name in [self.experiment.CAR1_NAME, self.experiment.CAR2_NAME,
                         self.experiment.CAR3_NAME, self.experiment.CAR4_NAME,
                         self.experiment.CAR5_NAME]:
            self.airsim_client.enableApiControl(True, car_name)

        # Set zero controls first to ensure vehicles are stationary
        for car_name in [self.experiment.CAR1_NAME, self.experiment.CAR2_NAME,
                         self.experiment.CAR3_NAME, self.experiment.CAR4_NAME,
                         self.experiment.CAR5_NAME]:
            self.airsim_client.setCarControls(airsim.CarControls(throttle=0, brake=1.0), car_name)

        time.sleep(0.5)  # Let the controls take effect

        # Create and set poses with explicit physics override
        car1_pose = self.create_initial_pose(-30, 0, 0)
        car2_pose = self.create_initial_pose(0, -30, 90)
        car3_pose = self.create_initial_pose(0, -45, 90)
        car4_pose = self.create_initial_pose(-40, 0, 0)
        car5_pose = self.create_initial_pose(0, 20, 270)

        # Apply poses with physics override
        self.airsim_client.simSetObjectPose(self.experiment.CAR1_NAME, car1_pose, True)
        self.airsim_client.simSetObjectPose(self.experiment.CAR2_NAME, car2_pose, False)
        self.airsim_client.simSetObjectPose(self.experiment.CAR3_NAME, car3_pose, True)
        self.airsim_client.simSetObjectPose(self.experiment.CAR4_NAME, car4_pose, True)
        self.airsim_client.simSetObjectPose(self.experiment.CAR5_NAME, car5_pose, True)

        time.sleep(0.5)  # Allow poses to take effect

        # Verify position and reset again if needed
        for car_name, expected_pose in [
            (self.experiment.CAR1_NAME, car1_pose),
            (self.experiment.CAR2_NAME, car2_pose),
            (self.experiment.CAR3_NAME, car3_pose),
            (self.experiment.CAR4_NAME, car4_pose),
            (self.experiment.CAR5_NAME, car5_pose)
        ]:
            actual_pose = self.airsim_client.simGetObjectPose(car_name)
            print(f"{car_name} reset position: ({actual_pose.position.x_val}, {actual_pose.position.y_val})")

        # Reset initial positions in our tracking variables
        self.car1_initial_position_saved = None
        self.car2_initial_position_saved = None
        self.car3_initial_position_saved = None
        self.car4_initial_position_saved = None
        self.car5_initial_position_saved = None



        print("Hard reset completed.")
    def pause_simulation(self):
        self.simulation_paused = True
        self.airsim_client.simPause(True)

    def resume_simulation(self):
        self.simulation_paused = False
        self.airsim_client.simPause(False)

    def is_simulation_paused(self):
        return self.simulation_paused

    def reset_car2_specifically(self):
        """
        Function dedicated to resetting only Car2 which appears to be consistently corrupted.
        """
        print("Performing targeted reset of Car2...")

        # 1. Try to completely stop Car2 first
        self.airsim_client.setCarControls(airsim.CarControls(throttle=0, brake=1.0), self.experiment.CAR2_NAME)
        time.sleep(0.2)

        # 2. Disable and re-enable API control specifically for Car2
        self.airsim_client.enableApiControl(False, self.experiment.CAR2_NAME)
        time.sleep(0.2)
        self.airsim_client.enableApiControl(True, self.experiment.CAR2_NAME)

        # 3. Reset its pose with physics override
        car2_pose = self.create_initial_pose(0, -30, 90)

        # Try multiple positioning methods
        try:
            # Method 1: Standard vehicle pose
            self.airsim_client.simSetVehiclePose(car2_pose, True, self.experiment.CAR2_NAME)
            time.sleep(0.1)

            # Method 2: Object pose (more aggressive)
            self.airsim_client.simSetObjectPose(self.experiment.CAR2_NAME, car2_pose, True)
            time.sleep(0.1)

            # Method 3: Try teleporting the vehicle
            if hasattr(self.airsim_client, 'simTeleportVehicle'):
                self.airsim_client.simTeleportVehicle(self.experiment.CAR2_NAME, airsim.Vector3r(0, -30, 0))

            # Verify position
            actual_pose = self.airsim_client.simGetObjectPose(self.experiment.CAR2_NAME)
            print(f"Car2 reset position: ({actual_pose.position.x_val}, {actual_pose.position.y_val})")

            # Reset saved position
            self.car2_initial_position_saved = None

        except Exception as e:
            print(f"Failed to reset Car2: {str(e)}")

    def has_reached_target(self, car_state):
        """
        Checks if the vehicle has reached the target using its relative state.
        For Car1: compares relative x; for Car2/Car3: compares relative y.
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
        elif self.experiment.ROLE == self.experiment.CAR3_NAME:
            init_pos = self.get_car3_initial_position()
            desired_global = (self.experiment.CAR3_DESIRED_POSITION_OPTION_1[1]
                              if init_pos[1] > 0
                              else self.experiment.CAR3_DESIRED_POSITION_OPTION_2[1])
            required_distance = abs(desired_global - init_pos[1])
            return abs(car_state[1]) >= required_distance
        return False