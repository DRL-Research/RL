import numpy as np


# TODO: consider removing the need to specify car_name
def get_env_state(airsim_client, car_name):
    """
    Retrieve the state of the environment with information about car positions, speeds,
    and velocity components for the specified car.

    Args:
        airsim_client (airsim.CarClient): The AirSim client object.
        car_name (str): The name of the car for which to retrieve the state ("Car1" or "Car2").

    Returns:
        dict: A dictionary containing the environment state information.

    Raises:
        ValueError: If an invalid car_name is provided.
    """

    # Get positions and speeds of both cars
    pose_c1 = airsim_client.simGetObjectPose("Car1").position
    pose_c2 = airsim_client.simGetObjectPose("Car2").position
    speed_c1 = airsim_client.getCarState("Car1").speed
    speed_c2 = airsim_client.getCarState("Car2").speed

    # Create a dictionary to store environment state
    env_state = {
        "x_c1": pose_c1.x_val,
        "y_c1": pose_c1.y_val,
        "v_c1": speed_c1,
        "x_c2": pose_c2.x_val,
        "y_c2": pose_c2.y_val,
        "v_c2": speed_c2,
        "dist_c1_c2": np.sum(np.square(np.array([[pose_c1.x_val, pose_c1.y_val]]) - np.array([[pose_c2.x_val, pose_c2.y_val]])))
    }

    # Determine the velocity based on the car_name parameter
    if car_name == "Car1":
        velocity = airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity
    elif car_name == "Car2":
        velocity = airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity
    else:
        raise ValueError("Invalid car_name")

    # Calculate individual velocity components
    right = max(velocity.x_val, 0)
    left = -min(velocity.x_val, 0)
    forward = max(velocity.y_val, 0)
    backward = -min(velocity.y_val, 0)

    # Store velocity components in the dictionary
    env_state["right"] = right
    env_state["left"] = left
    env_state["forward"] = forward
    env_state["backward"] = backward

    return env_state


def process_car_state_and_control(airsim_client, car_name):
    """
    Process the state and control for a specific car.

    Args:
        airsim_client: The AirSim client object.
        car_name: The name of the car (e.g., "Car1").

    Returns:
        A dictionary containing the state and control information for the car.
    """
    # Fetch the environment state for the specified car
    env_state = get_env_state(airsim_client, car_name)

    # Extract relevant state information for the DNN input
    car_state = np.array([[
        env_state["x_c1"] if car_name == "Car1" else env_state["x_c2"],
        env_state["y_c1"] if car_name == "Car1" else env_state["y_c2"],
        env_state["v_c1"] if car_name == "Car1" else env_state["v_c2"],
        env_state["v_c2"] if car_name == "Car1" else env_state["v_c1"],
        env_state["dist_c1_c2"],
        env_state["right"],
        env_state["left"],
        env_state["forward"],
        env_state["backward"]
    ]])  # Note: Array shaped as [[]] for DNN input

    return {'state': car_state}
