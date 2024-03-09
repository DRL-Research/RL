import numpy as np
from config import CAR1_NAME, CAR2_NAME

# TODO: add function that is empty for now: get_proto_plan()
# TODO: comment get_cars_state and car1_states_to_car2_states_perspective to not interrupt


def get_car_position_and_speed(airsim_client, car_name):
    car_position = airsim_client.simGetObjectPose(car_name).position
    car_position_and_speed = {
        "x": car_position.x_val,
        "y": car_position.y_val,
        "Vx": airsim_client.getCarState(car_name).kinematics_estimated.linear_velocity.x_val,
        "Vy": airsim_client.getCarState(car_name).kinematics_estimated.linear_velocity.y_val,
    }
    return car_position_and_speed


def get_cars_distance(airsim_client):
    car1_position_and_speed = get_car_position_and_speed(airsim_client, CAR1_NAME)
    car2_position_and_speed = get_car_position_and_speed(airsim_client, CAR2_NAME)
    dist_c1_c2 = np.sum(np.square(
        np.array([[car1_position_and_speed["x"], car1_position_and_speed["y"]]]) -
        np.array([[car2_position_and_speed["x"], car2_position_and_speed["y"]]])))
    return dist_c1_c2


def get_local_input_car1_perspective(airsim_client):
    car1_state = get_car_position_and_speed(airsim_client, CAR1_NAME)
    car2_state = get_car_position_and_speed(airsim_client, CAR2_NAME)
    dist_c1_c2 = get_cars_distance(airsim_client)
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


def get_local_input_car2_perspective(airsim_client):
    car2_state = get_car_position_and_speed(airsim_client, CAR2_NAME)
    car1_state = get_car_position_and_speed(airsim_client, CAR1_NAME)
    dist_c1_c2 = get_cars_distance(airsim_client)
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


#
# def get_global_input____():
#     # Adding 5 values named emb1 to emb5
#     for i in range(1, 6):
#         local_input_car1_perspective[f"emb{i}"] = i  # Assigning example values



