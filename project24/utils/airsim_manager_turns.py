from project24.initialization.config_turns import *
import airsim
from project24.initialization.setup_simulation_turns import SetupManager


class AirsimManager:

    def __init__(self, setup_manager: SetupManager):
        self.airsim_client = airsim.CarClient()
        self.airsim_client.confirmConnection()  # Confirm the connection to the AirSim simulator
        self.car_name_to_offset = {}
        self.cars = setup_manager.cars
        self.enable_api_cars_control()

        # Set cars throttle to 1:
        for car_obj in self.cars.values():
            if car_obj.is_active:
                self.set_car_throttle_by_name(car_obj.name)  # default throttle is 1
            # get initial positions according to settings offset
            # (later be used whenever we need to reset to initial position -> start of each epoch)
            # (arbitrary starting position in settings file as long as they are not spawned on top of each other)
            car_pos = self.airsim_client.simGetObjectPose(car_obj.name).position
            self.car_name_to_offset[car_obj.name] = {'x_offset': car_pos.x_val, 'y_offset': car_pos.y_val}

        self.reset_cars_to_initial_positions()

    def set_car_throttle_by_name(self, car_name, throttle=0.4):
        car_controls = airsim.CarControls()
        car_controls.throttle = throttle
        self.airsim_client.setCarControls(car_controls, car_name)

    def enable_api_cars_control(self):
        for car in self.cars.values():
            if not self.airsim_client.isApiControlEnabled(car.name):
                self.airsim_client.enableApiControl(is_enabled=True, vehicle_name=car.name)

    def get_start_location(self, car_name, side=None):
        car_start_location_x, car_start_location_y = None, None
        if self.cars.get(car_name):
            car = self.cars[car_name]
            car_offset_x = self.car_name_to_offset[car_name]['x_offset']
            car_offset_y = self.car_name_to_offset[car_name]['y_offset']
            car_start_location_x = car.get_start_location_x(car_offset_x)
            car_start_location_y = car.get_start_location_y(car_offset_y, side)
        return car_start_location_x, car_start_location_y

    def reset_cars_to_initial_positions(self):
        self.airsim_client.reset()

        car1_start_location_x, car1_start_location_y = self.get_start_location(CAR1_NAME)

        # -24, 6
        car2_start_location_x, car2_start_location_y = self.get_start_location(CAR2_NAME, 1)

        # -6, -24
        # car3_start_location_x, car3_start_location_y = self.get_start_location(CAR3_NAME)

        # 24, -6
        # car4_start_location_x, car4_start_location_y = self.get_start_location(CAR4_NAME, -1)

        # Set the reference_position for Car1 and Car2 (Do not change this code)
        reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
        for car_name in self.cars:
            self.airsim_client.simSetVehiclePose(reference_position, True, car_name)

        # Set initial position and yaw of Car1
        if self.cars.get(CAR1_NAME):
            self.set_initial_position_and_yaw(car_name=CAR1_NAME, start_location_x=car1_start_location_x,
                                          start_location_y=car1_start_location_y, car_start_yaw=CAR1_INITIAL_YAW)
        # Set initial position and yaw of Car2
        if self.cars.get(CAR2_NAME):
            self.set_initial_position_and_yaw(car_name=CAR2_NAME, start_location_x=car2_start_location_x,
                                          start_location_y=car2_start_location_y, car_start_yaw=CAR2_INITIAL_YAW)
        # Set initial position and yaw of Car3
        # if self.cars.get(CAR3_NAME):
        #     self.set_initial_position_and_yaw(car_name=CAR3_NAME, start_location_x=car3_start_location_x,
        #                                   start_location_y=car3_start_location_y, car_start_yaw=CAR3_INITIAL_YAW)
        # # Set initial position and yaw of Car4
        # if self.cars.get(CAR4_NAME):
        #     self.set_initial_position_and_yaw(car_name=CAR4_NAME, start_location_x=car4_start_location_x,
        #                                   start_location_y=car4_start_location_y, car_start_yaw=CAR4_INITIAL_YAW)
    @staticmethod
    def stop_car(airsim_client, moving_car_name, throttle=0.0):
        car_controls = airsim.CarControls()
        car_controls.throttle = throttle
        if throttle > 0:
            car_controls.brake = 1.0
        airsim_client.setCarControls(car_controls, vehicle_name=moving_car_name)

    def collision_occurred(self):
        collision_info = self.airsim_client.simGetCollisionInfo()
        return collision_info.has_collided

    def has_reached_target(self, car_state):
        #  this function gets car state, should it?
        return car_state['x_c1'] > CAR1_DESIRED_POSITION[0]

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
        local_input_car1_perspective = {
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
        return local_input_car1_perspective

    def set_initial_position_and_yaw(self, car_name, start_location_x, start_location_y, car_start_yaw):
        # Convert yaw values from degrees to radians as AirSim uses radians
        car_start_yaw_rad = np.radians(car_start_yaw)
        # Set initial position and yaw of Car1
        initial_position_car = airsim.Vector3r(start_location_x, start_location_y, -1)
        initial_orientation_car = airsim.to_quaternion(0, 0, car_start_yaw_rad)  # Roll, Pitch, Yaw
        initial_pose_car = airsim.Pose(initial_position_car, initial_orientation_car)
        #self.airsim_client.simSetVehiclePose(initial_pose_car, True, car_name)
        self.airsim_client.simSetObjectPose(car_name, initial_pose_car, True)