import numpy as np


def get_cars_state(airsim_client):

    # Get positions of both cars
    position_c1 = airsim_client.simGetObjectPose("Car1").position
    position_c2 = airsim_client.simGetObjectPose("Car2").position

    # Create a dictionary to store environment state
    cars_state = {
        # car1 state:
        "x_c1": position_c1.x_val,
        "y_c1": position_c1.y_val,
        "Vx_c1": airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity.x_val,
        "Vy_c1": airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity.y_val,
        # car2 state:
        "x_c2": position_c2.x_val,
        "y_c2": position_c2.y_val,
        "Vx_c2": airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity.x_val,
        "Vy_c2": airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity.y_val,
        # car1 and car2 distance
        "dist_c1_c2": np.sum(np.square(
            np.array([[position_c1.x_val, position_c1.y_val]]) - np.array([[position_c2.x_val, position_c2.y_val]])))
    }

    return cars_state


def car1_states_to_car2_states_perspective(cars_state_car1_perspective):

    cars_state_car2_perspective = cars_state_car1_perspective.copy()

    # Extract the first 4 elements and the next 4 elements
    first_four = cars_state_car1_perspective[:, :4]
    next_four = cars_state_car1_perspective[:, 4:8]

    # Swap the first 4 elements with the next 4 elements
    cars_state_car2_perspective[:, :4] = next_four
    cars_state_car2_perspective[:, 4:8] = first_four

    return cars_state_car2_perspective

