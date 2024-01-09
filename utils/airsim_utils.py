import airsim
import json
import os


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


def update_airsim_settings_file(car1_location, car2_location):
    """
    Update AirSim settings for Car1 and Car2 with new locations and save the modified settings.

    Args:
        car1_location (list[int]): New coordinates [X, Y] for Car1.
        car2_location (list[int]): New coordinates [X, Y] for Car2.

    Returns:
        None

    This function reads the AirSim settings from the 'airsim_settings/settings.json' file,
    updates the positions of Car1 and Car2 with the provided coordinates, and saves the
    modified settings to '~/Documents/AirSim/settings.json'.
    """
    # Define file paths
    settings_directory = 'airsim_settings'
    input_file_name = 'settings.json'
    output_directory = os.path.expanduser('~/Documents/AirSim')
    output_file_name = 'settings.json'

    # Read the original settings
    with open(os.path.join(settings_directory, input_file_name), 'r') as json_file:
        original_settings = json.load(json_file)

    # Modify the data
    original_settings["Vehicles"]["Car1"]['X'] = car1_location[0]
    original_settings["Vehicles"]["Car1"]['Y'] = car1_location[1]
    original_settings["Vehicles"]["Car2"]['X'] = car2_location[0]
    original_settings["Vehicles"]["Car2"]['Y'] = car2_location[1]

    # Write the modified settings to the output file
    with open(os.path.join(output_directory, output_file_name), 'w') as json_file:
        json.dump(original_settings, json_file, indent=4)

    print(f"AirSim settings updated and saved to '{output_file_name}' in '{output_directory}'.")


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

