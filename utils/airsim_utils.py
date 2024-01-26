import airsim
import os
import json

def init_airsim_client():
    """
    Initialize an AirSim client for controlling two cars.

    This function sets up a connection to the AirSim simulation environment,
    enables API control for two cars (Car1 and Car2),
    and sets the throttle to 1 for both cars.

    Returns:
        airsim.CarClient: The initialized AirSim client instance.
    """
    airsim_client = airsim.CarClient()  # Create an AirSim client for car simulation
    airsim_client.confirmConnection()  # Confirm the connection to the AirSim simulator

    airsim_client.enableApiControl(True, "Car1")  # Enable API control for Car1
    airsim_client.enableApiControl(True, "Car2")  # Enable API control for Car2

    # Set car 1 throttle to 1:
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    airsim_client.setCarControls(car_controls, "Car1")

    # Set car 2 throttle to 1:
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    airsim_client.setCarControls(car_controls, "Car2")

    return airsim_client


def move_cars_to_initial_positions(airsim_client, car1_start_location, car2_start_location,
                                   car1_start_yaw, car2_start_yaw):

    # get initial positions according to settings
    # (arbitrary in settings file as long as they are not spawned on top of each other)
    car1_position_from_settings = [airsim_client.simGetObjectPose("Car1").position.x_val,
                                   airsim_client.simGetObjectPose("Car1").position.y_val]
    car2_position_from_settings = [airsim_client.simGetObjectPose("Car2").position.x_val,
                                   airsim_client.simGetObjectPose("Car2").position.y_val]

    car1_start_location[0] -= car1_position_from_settings[0]
    car1_start_location[1] -= car1_position_from_settings[1]
    car2_start_location[0] -= car2_position_from_settings[0]
    car2_start_location[1] -= car2_position_from_settings[1]

    # Set the reference_position for Car1 and Car2 (Do not change this code)
    reference_position = airsim.Pose(airsim.Vector3r(0.0, 0, -1), airsim.Quaternionr(0, 0.0, 0.0, 1.0))
    airsim_client.simSetVehiclePose(reference_position, True, "Car1")
    airsim_client.simSetVehiclePose(reference_position, True, "Car2")

    # Set initial position of Car1
    initial_position_car1 = airsim.Vector3r(car1_start_location[0], car1_start_location[1], -1)
    initial_pose_car1 = airsim.Pose(initial_position_car1, airsim.Quaternionr(0, 0.0, 0.0, 1.0))
    airsim_client.simSetVehiclePose(initial_pose_car1, True, "Car1")

    # Update the position of Car2
    initial_position_car2 = airsim.Vector3r(car2_start_location[0], car2_start_location[1], -1)
    initial_pose_car2 = airsim.Pose(initial_position_car2, airsim.Quaternionr(0, 0.0, 0.0, 1.0))
    airsim_client.simSetVehiclePose(initial_pose_car2, True, "Car2")



def detect_and_handle_collision(airsim_client):
    """
    Detects collision and handles the consequences.

    Args:
        airsim_client: The AirSim client object.

    Returns:
        A tuple indicating if there was a collision and the corresponding reward.
    """
    collision_info = airsim_client.simGetCollisionInfo()
    if collision_info.has_collided:
        # Handling the collision
        # Here, I'm just returning a fixed negative reward. You might want to include more complex logic.
        collision_reward = -1000
        return True, collision_reward
    else:
        # No collision occurred
        return False, 0

