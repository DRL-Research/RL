import numpy as np


def get_cars_state(airsim_client):

    # Get positions of both cars
    position_c1 = airsim_client.simGetObjectPose("Car1").position
    position_c2 = airsim_client.simGetObjectPose("Car2").position

    # Create a dictionary to store environment state
    cars_state = {
        "x_c1": position_c1.x_val,
        "y_c1": position_c1.y_val,
        "x_c2": position_c2.x_val,
        "y_c2": position_c2.y_val,
        "Vx_c1": airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity.x_val,
        "Vy_c1": airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity.y_val,
        "Vx_c2": airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity.x_val,
        "Vy_c2": airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity.y_val,
        "dist_c1_c2": np.sum(np.square(np.array([[position_c1.x_val, position_c1.y_val]]) - np.array([[position_c2.x_val, position_c2.y_val]])))
    }

    return cars_state

