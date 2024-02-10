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

    #
    # # Determine the velocity based on the car_name parameter
    # if car_name == "Car1":
    #     velocity = airsim_client.getCarState("Car2").kinematics_estimated.linear_velocity
    # elif car_name == "Car2":
    #     velocity = airsim_client.getCarState("Car1").kinematics_estimated.linear_velocity
    # else:
    #     raise ValueError("Invalid car_name")

    # # Calculate individual velocity components
    # right = max(velocity.x_val, 0)
    # left = -min(velocity.x_val, 0)
    # forward = max(velocity.y_val, 0)
    # backward = -min(velocity.y_val, 0)
    #
    # # Store velocity components in the dictionary
    # env_state["right"] = right
    # env_state["left"] = left
    # env_state["forward"] = forward
    # env_state["backward"] = backward

    return cars_state

#
# def get_cars_state(airsim_client, car_name):
#     """
#     Process the state and control for a specific car.
#
#     Args:
#         airsim_client: The AirSim client object.
#
#     Returns:
#         A dictionary containing the state and control information for the car.
#     """
#     # Fetch the environment state for the specified car
#     env_state = get_env_state(airsim_client, "Car1")
#
#     # Extract relevant state information for the DNN input
#     car_state = np.array([[
#         env_state["x_c1"] if car_name == "Car1" else env_state["x_c2"],
#         env_state["y_c1"] if car_name == "Car1" else env_state["y_c2"],
#         env_state["v_c1"] if car_name == "Car1" else env_state["v_c2"],
#         env_state["v_c2"] if car_name == "Car1" else env_state["v_c1"],
#         env_state["dist_c1_c2"],
#         env_state["right"],
#         env_state["left"],
#         env_state["forward"],
#         env_state["backward"]
#     ]])  # Note: Array shaped as [[]] for DNN input
#
#     return {'state': car_state}
