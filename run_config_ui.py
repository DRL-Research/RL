import streamlit as st
from src.experiment.scenarios_config import create_full_environment_config, custom_env_configs #CONFIG_EXP1, CONFIG_EXP2, CONFIG_EXP3, CONFIG_EXP4, CONFIG_EXP5
from src.experiment.experiment_config import Experiment
import copy
import os
import random

def get_key_by_value(d, target_value):
    for k, v in d.items():
        if v == target_value:
            return k
    return None

def format_value(v):
    if isinstance(v, str) and v.startswith("Experiment."):
        return v  # write as raw Python expression
    elif isinstance(v, tuple):
        return v  # write as tuple directly
    elif isinstance(v, dict):
        return "{" + ", ".join(f'"{k}": {format_value(v[k])}' for k in v) + "}"
    else:
        return repr(v)

def dump_config_as_python(filepath, config_dict):
    with open(filepath, "w") as f:
        f.write("from src.experiment.experiment_config import Experiment\n\n")
        f.write("CONFIG = {\n")
        for car_group in ["controlled_cars", "static_cars"]:
            f.write(f"    \"{car_group}\": {{\n")
            for car_id, car_data in config_dict.get(car_group, {}).items():
                f.write(f"        \"{car_id}\": {{\n")
                for k, v in car_data.items():
                    formatted = format_value(v)
                    f.write(f'            "{k}": {formatted},\n')
                f.write("        },\n")
            f.write("    },\n")
        f.write("}\n")


st.title("RL Scenario Configurator")

if "page" not in st.session_state:
    st.session_state.page = "Welcome"

st.sidebar.markdown("### Navigation")
if st.sidebar.button("🏠 Welcome"):
    st.session_state.page = "Welcome"
if st.sidebar.button("🛠 Edit Existing Experiment"):
    st.session_state.page = "Edit Existing"
if st.sidebar.button("➕ Add New Experiment"):
    st.session_state.page = "Add New"
page = st.session_state.page

existing_experiments = list(custom_env_configs.keys())

if page == "Welcome":
    welcome_text = (
        "👋 Welcome to the RL Scenario Configurator! Use the sidebar to navigate.\n\n"
        "🛠️ Edit Existing Experiments:\n"
        "- Select from your saved scenarios and customize environment settings.\n"
        "- Adjust initial vehicle locations, speeds, routes, and destinations.\n"
        "- Fine-tune parameters to test various autonomous driving behaviors.\n\n"
        "➕ Add New Experiments:\n"
        "- Create brand new experiment configurations from scratch.\n"
        "- Design custom traffic layouts, vehicle counts, and dynamic settings.\n"
        "- Tailor experiments specifically for your research goals.\n\n"
        "🚗 This tool helps you design, modify, and save experiments to advance your autonomous driving research."
    )

    st.write(welcome_text)

elif page == "Edit Existing":
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_experiment = st.selectbox("Select Experiment", ["--select experiment--"] + existing_experiments, index=0)
    with col2:
        st.write("")

    if selected_experiment != "--select experiment--":
        experiment_name = st.text_input("Experiment Name", selected_experiment)
        config = custom_env_configs[selected_experiment]
        st.session_state.edited_config = copy.deepcopy(config)
        st.session_state.edited_experiment = selected_experiment

        edited_config = st.session_state.edited_config

        st.write("### Controlled Cars")
        for i, (car_id, car_attrs) in enumerate(edited_config.get("controlled_cars", {}).items()):
            with st.expander(f"Controlled Car: {car_id}", expanded=False):
                # updated = edit_car_config(car_id, car_attrs, "controlled")
                # Lane
                possible_lane = Experiment.START_LANES
                lane_keys = list(possible_lane.keys())
                lane_disp = get_key_by_value(possible_lane, car_attrs.get("start_lane", ""))
                lane = st.selectbox(
                    f"Start lane and heading for {car_id}",
                    options=lane_keys,
                    index=lane_keys.index(lane_disp) if lane_disp in lane_keys else 0,
                )

                # Destination
                possible_dest = Experiment.DESTINATION_LANES
                dest_keys = list(possible_dest.keys())
                dest_disp = get_key_by_value(possible_dest, car_attrs.get("destination", ""))
                dest = st.selectbox(
                    f"Destination for {car_id}",
                    options=dest_keys,
                    index=dest_keys.index(dest_disp) if dest_disp in dest_keys else 0,
                )

                # Speed
                possible_speed = Experiment.SPEEDS
                print(type(possible_speed))
                speed_keys = list(possible_speed.keys()) + ["RANDOM FAST/SLOW"]
                speed_disp = get_key_by_value(possible_speed, car_attrs.get("speed", ""))
                speed = st.selectbox(
                    f"Speed for {car_id}",
                    options=speed_keys,
                    index=speed_keys.index(speed_disp) if speed_disp in speed_keys else 0,
                )
                if speed == "RANDOM FAST/SLOW":
                    speed = random.choice(["THROTTLE_SLOW", "THROTTLE_FAST"])

                # Init location split
                init_loc = car_attrs.get("init_location", {})
                longitudinal = st.number_input(
                    f"Longitudinal Position for {car_id}",
                    value=float(init_loc.get("longitudinal", 0)),
                )
                lateral = st.number_input(
                    f"Lateral Position for {car_id}",
                    value=float(init_loc.get("lateral", 0)),
                )

                # Build updated dict with string references to constants
                updated_attrs = {
                    "start_lane": f"Experiment.{lane}",
                    "destination": f"Experiment.{dest}",
                    "speed": f"Experiment.{speed}",
                    "init_location": {
                        "longitudinal": "Experiment.LONGITUDINAL" if longitudinal==Experiment.LONGITUDINAL else longitudinal,
                        "lateral": "Experiment.LATERAL" if lateral==Experiment.LATERAL else lateral,
                    }
                }
                edited_config["controlled_cars"][car_id].update(updated_attrs)

        st.write("### Static Cars")
        for i, (car_id_static, car_attrs_static) in enumerate(edited_config.get("static_cars", {}).items()):
            with st.expander(f"Static Car: {car_id_static}", expanded=False):
                # updated = edit_car_config(car_id, car_attrs, "static")
                # edited_config["static_cars"][car_id].update(updated)
                # Lane
                possible_lane = Experiment.START_LANES
                lane_keys = list(possible_lane.keys())
                lane_disp = get_key_by_value(possible_lane, car_attrs_static.get("start_lane", ""))
                lane = st.selectbox(
                    f"Start lane and heading for {car_id_static}",
                    options=lane_keys,
                    index=lane_keys.index(lane_disp) if lane_disp in lane_keys else 0,
                )

                # Destination
                possible_dest = Experiment.DESTINATION_LANES
                dest_keys = list(possible_dest.keys())
                dest_disp = get_key_by_value(possible_dest, car_attrs_static.get("destination", ""))
                dest = st.selectbox(
                    f"Destination for {car_id_static}",
                    options=dest_keys,
                    index=dest_keys.index(dest_disp) if dest_disp in dest_keys else 0,
                )

                # Speed
                possible_speed = Experiment.SPEEDS
                print(type(possible_speed))
                speed_keys = list(possible_speed.keys()) + ["RANDOM FAST/SLOW"]
                speed_disp = get_key_by_value(possible_speed, car_attrs_static.get("speed", ""))
                speed = st.selectbox(
                    f"Speed for {car_id_static}",
                    options=speed_keys,
                    index=speed_keys.index(speed_disp) if speed_disp in speed_keys else 0,
                )
                if speed == "RANDOM FAST/SLOW":
                    speed = random.choice(["THROTTLE_SLOW", "THROTTLE_FAST"])

                # Init location split
                init_loc = car_attrs_static.get("init_location", {})
                longitudinal = st.number_input(
                    f"Longitudinal Position for {car_id_static}",
                    value=float(init_loc.get("longitudinal", 0)),
                )
                lateral = st.number_input(
                    f"Lateral Position for {car_id_static}",
                    value=float(init_loc.get("lateral", 0)),
                )

                # Build updated dict with string references to constants
                updated_attrs = {
                    "start_lane": f"Experiment.{lane}",
                    "destination": f"Experiment.{dest}",
                    "speed": f"Experiment.{speed}",
                    "init_location": {
                        "longitudinal": "Experiment.LONGITUDINAL" if longitudinal==Experiment.LONGITUDINAL else longitudinal,
                        "lateral": "Experiment.LATERAL" if lateral==Experiment.LATERAL else lateral,
                    }
                }
                edited_config["static_cars"][car_id_static].update(updated_attrs)

        if st.button("Save Changes"):
            # Assume config files are named like "configs/my_experiment.py"
            config_filename = f"src/experiment/configs/{selected_experiment}.py"

            # Dump the edited config to a Python file
            dump_config_as_python(config_filename, edited_config)

            st.session_state.edited_config = edited_config
            st.success(f"Changes saved to `{config_filename}`.")

    else:
        experiment_name = st.text_input("Experiment Name", "Not selected")

elif page == "Add New":
    experiment_name = st.text_input("Experiment Name", "Insert a unique experiment name")
