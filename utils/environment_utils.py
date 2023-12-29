import numpy as np


def get_env_state(airsim_client, car_name):

    # Get positions and speeds of both cars
    pose_c1 = airsim_client.simGetObjectPose("Car1").position
    pose_c2 = airsim_client.simGetObjectPose("Car2").position
    speed_c1 = airsim_client.getCarState("Car1").speed
    speed_c2 = airsim_client.getCarState("Car2").speed

    # Create a dictionary to store environment state
    env_state = {"x_c1": pose_c1.x_val,
                 "y_c1": pose_c1.y_val,
                 "v_c1": speed_c1,
                 "x_c2": pose_c2.x_val,
                 "y_c2": pose_c2.y_val,
                 "v_c2": speed_c2,
                 "dist_c1_c2": np.sum(np.square(
                     np.array([[pose_c1.x_val, pose_c1.y_val]]) - np.array([[pose_c2.x_val, pose_c2.y_val]])))}

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
