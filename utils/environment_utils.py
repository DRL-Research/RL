import numpy as np
from config import CAR1_NAME, CAR2_NAME

# TODO: add distance to state (for reward function)
# TODO: change dictionary names to be informative?
# TODO: add function that is empty for now: get_proto_plan()
# TODO: comment get_cars_state and car1_states_to_car2_states_perspective to not interrupt

def get_car_state(airsim_client, car_name):

    car_position = airsim_client.simGetObjectPose(car_name).position
    # Create a dictionary to store car state (pulled from environment)
    car_state = {
        "x_c": car_position.x_val,
        "y_c": car_position.y_val,
        "Vx_c": airsim_client.getCarState(car_name).kinematics_estimated.linear_velocity.x_val,
        "Vy_c": airsim_client.getCarState(car_name).kinematics_estimated.linear_velocity.y_val,
    }
    return car_state


def get_local_input_car1_perspective(airsim_client):
    car1_state = get_car_state(airsim_client, CAR1_NAME)
    car2_state = get_car_state(airsim_client, CAR2_NAME)
    local_input_car1_perspective = np.array([list(car1_state.values()) + list(car2_state.values())])  # + is append
    return local_input_car1_perspective


def get_local_input_car2_perspective(airsim_client):
    car2_state = get_car_state(airsim_client, CAR2_NAME)
    car1_state = get_car_state(airsim_client, CAR1_NAME)
    local_input_car2_perspective = np.array([list(car2_state.values()) + list(car1_state.values())])  # + is append
    return local_input_car2_perspective





def get_cars_state(airsim_client):

    # Get positions of both cars
    position_c1 = airsim_client.simGetObjectPose(CAR1_NAME).position
    position_c2 = airsim_client.simGetObjectPose(CAR2_NAME).position

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


