from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src import project_globals
from src.model.agent_handler import Driver
from src.model.master_model import MasterModel
from src.project_globals import rollout_buffers
from src.training.general_utils import (
    ensure_tensor,
    get_agent_values_from_observation,
    get_scaler_action_and_action_array,
    sanitize_float_vector,
)
from src.training.rollout_buffer_utils import reset_all_buffers


def reshape_drivers_states(all_drivers_states):
    all_drivers_states = (
        all_drivers_states.reshape(-1)
        if isinstance(all_drivers_states, np.ndarray) and len(all_drivers_states.shape) == 2
        else all_drivers_states
    )
    return all_drivers_states


# ── Master 25-D observations (same packing as ``run_proto_action_sweep``) ────


def _pad_vec(vec: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, len(arr))] = arr[: min(size, len(arr))]
    return out


def _normalize_kinematics4(state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.float32).copy()
    out[0:2] /= MasterModel.POSITION_SCALE
    out[2:4] /= MasterModel.SPEED_SCALE
    return out


def _master_slot(vec: np.ndarray, identifier: float, slot_vec_dim: int) -> np.ndarray:
    return np.concatenate([_pad_vec(vec, slot_vec_dim), np.asarray([identifier], dtype=np.float32)])


def _infer_slot_vec_dim(global_emb: np.ndarray, experiment) -> int:
    ge = int(np.asarray(global_emb, dtype=np.float32).reshape(-1).shape[0])
    if experiment is not None:
        emb = int(getattr(experiment, "EMBEDDING_SIZE", ge))
        return max(4, emb)
    return max(4, ge)


def _num_master_slots(experiment) -> int:
    return int(getattr(experiment, "NUM_MASTER_SLOTS", 5)) if experiment is not None else 5


def _normalize_agent_kinematics(car_state: np.ndarray, experiment) -> np.ndarray:
    st = np.asarray(car_state, dtype=np.float32).reshape(-1)
    if experiment is not None and getattr(experiment, "NORMALIZE_AGENT_OBS", False):
        return _normalize_kinematics4(st[:4])
    return st[:4].astype(np.float32)


def _build_local_master_input(
    global_emb: np.ndarray,
    group_states: np.ndarray,
    experiment=None,
) -> np.ndarray:
    slot_vec_dim = _infer_slot_vec_dim(global_emb, experiment)
    num_slots = _num_master_slots(experiment)
    ge = np.asarray(global_emb, dtype=np.float32).reshape(-1)
    norm_master = MasterModel.NORMALIZE_INPUTS or (
        experiment is not None and getattr(experiment, "NORMALIZE_MASTER_INPUTS", False)
    )

    slots = [_master_slot(ge, 1.0, slot_vec_dim)]
    for row in np.asarray(group_states, dtype=np.float32):
        st = _normalize_kinematics4(row[:4]) if norm_master else np.asarray(row[:4], dtype=np.float32)
        slots.append(_master_slot(st, 0.0, slot_vec_dim))
    while len(slots) < num_slots:
        slots.append(np.zeros(slot_vec_dim + 1, dtype=np.float32))
    return np.concatenate(slots[:num_slots]).astype(np.float32)


def _build_global_master_input(
    lm1_emb: np.ndarray,
    lm2_emb=None,
    experiment=None,
    *extra_lm_embs: np.ndarray,
) -> np.ndarray:
    """Pack local-master embeddings into the GM input (one slot per LM, id=1).

    Backward compatible: the classic 2-LM call ``_build_global_master_input(lm1,
    lm2, experiment)`` is unchanged. To support N local masters pass a list/tuple
    of embeddings as the first argument, e.g. ``_build_global_master_input([e0,
    e1, e2, ...], experiment=exp)`` — the hierarchy then scales to more masters.
    """
    # Normalise the call into a flat list of LM embeddings + the experiment.
    if isinstance(lm1_emb, (list, tuple)):
        lm_embs = list(lm1_emb)
        if lm2_emb is not None and experiment is None:
            experiment = lm2_emb
    else:
        lm_embs = [lm1_emb]
        if lm2_emb is not None:
            lm_embs.append(lm2_emb)
        if extra_lm_embs:
            lm_embs.extend(extra_lm_embs)

    slot_vec_dim = max(_infer_slot_vec_dim(e, experiment) for e in lm_embs)
    num_slots = _num_master_slots(experiment)
    slots = [
        _master_slot(np.asarray(e, dtype=np.float32).reshape(-1), 1.0, slot_vec_dim)
        for e in lm_embs[:num_slots]
    ]
    while len(slots) < num_slots:
        slots.append(np.zeros(slot_vec_dim + 1, dtype=np.float32))
    return np.concatenate(slots[:num_slots]).astype(np.float32)


def _unwrap_driver(env):
    """Walk MultiEnvWrapper → DummyVecEnv → … until we reach ``Driver`` (has ``highway_env``)."""
    cur = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "highway_env", None) is not None:
            return cur
        if hasattr(cur, "current") and cur.current is not None:
            cur = cur.current
        elif getattr(cur, "envs", None):
            cur = cur.envs[0]
        elif getattr(cur, "env", None) is not None:
            cur = cur.env
        else:
            break
    raise TypeError(
        f"Could not unwrap to Driver from {type(env).__name__}; "
        "expected a chain ending in an object with highway_env."
    )


def _states_matrix(driver: Driver, prepared) -> np.ndarray:
    arr = np.asarray(prepared, dtype=np.float32).reshape(-1)
    n = arr.shape[0] // 4
    return arr.reshape(n, 4)


def _pad_states_mat_rows(states_mat: np.ndarray, cars_amount: int) -> np.ndarray:
    """Ensure shape (cars_amount, 4); highway vectors can be shorter than experiment.CARS_AMOUNT."""
    sm = np.asarray(states_mat, dtype=np.float32).reshape(-1, 4)
    n = sm.shape[0]
    ca = int(cars_amount)
    if n >= ca:
        return sm[:ca].copy()
    out = np.zeros((ca, 4), dtype=np.float32)
    out[:n] = sm
    return out


def _sync_arrival_flags_length(experiment) -> None:
    """Buffers assume experiment.CARS_AMOUNT; env resets may resize flags to len(controlled_vehicles)."""
    target = int(getattr(experiment, "CARS_AMOUNT", len(project_globals.after_is_arrived_flags)))
    f = project_globals.after_is_arrived_flags
    if len(f) == target:
        return
    if len(f) < target:
        project_globals.after_is_arrived_flags = list(f) + [False] * (target - len(f))
    else:
        project_globals.after_is_arrived_flags = list(f[:target])


def _build_agent_obs_list(states_mat: np.ndarray, lm1_emb, lm2_emb, experiment) -> list[np.ndarray]:
    apm = int(getattr(experiment, "AGENTS_PER_LOCAL_MASTER", 3))
    out = []
    le = np.asarray(lm1_emb, dtype=np.float32).reshape(-1)
    ri = np.asarray(lm2_emb, dtype=np.float32).reshape(-1)
    for i in range(states_mat.shape[0]):
        emb = le if i < apm else ri
        cs = _normalize_agent_kinematics(states_mat[i], experiment)
        out.append(np.concatenate([cs.reshape(-1), emb]).astype(np.float32))
    return out


def run_episode(experiment, total_steps, env, master_model, agent_model, train_both, training_master):
    """Legacy single-vector master rollout (expects Driver wired with ``master_model``)."""
    all_rewards, actions_per_episode = [], []
    steps_counter, episode_sum_of_rewards = 0, 0
    crashed = False

    car_observations, _ = env.reset()
    done, truncated = False, False

    while not done and not truncated:
        steps_counter += 1

        all_drivers_states = env.env.current_state

        embedding, value, log_prob = master_model.get_proto_action(ensure_tensor(all_drivers_states))

        actions = Driver.get_action(
            agent_model,
            car_observations,
            total_steps,
            experiment.EXPLORATION_EXPLOITATION_THRESHOLD,
        )

        cars_scalar_action, cars_action_arrays = [], []
        for action in actions:
            car_scalar_action, car_action_array = get_scaler_action_and_action_array(action)
            cars_scalar_action.append(car_scalar_action)
            cars_action_arrays.append(car_action_array)

        cars_values, cars_log_probas = [], []
        if train_both or not training_master:
            for car_index in range(len(car_observations)):
                car_values, car_log_prob = get_agent_values_from_observation(
                    np.array(car_observations[car_index], dtype=np.float32),
                    cars_action_arrays[car_index],
                    agent_model,
                )
                cars_values.append(car_values)
                cars_log_probas.append(car_log_prob)

        env.render()

        action_tuple = tuple(cars_scalar_action)
        cars_next_obs, reward, done, truncated, info = env.step(action_tuple)
        episode_sum_of_rewards += reward
        all_rewards.append(reward)

        if done and info.get("crashed", False):
            crashed = True

        done_flag = bool(done or truncated)

        all_drivers_states = (
            all_drivers_states.reshape(-1)
            if isinstance(all_drivers_states, np.ndarray) and len(all_drivers_states.shape) == 2
            else all_drivers_states
        )

        if train_both:
            for car_index in range(len(project_globals.after_is_arrived_flags)):
                if not project_globals.after_is_arrived_flags[car_index]:
                    rollout_buffers[car_index].add(
                        np.array(car_observations[car_index], dtype=np.float32),
                        cars_action_arrays[car_index],
                        reward,
                        done_flag,
                        cars_values[car_index],
                        cars_log_probas[car_index],
                    )
            master_model.rollout_buffer.add(all_drivers_states, embedding, reward, done_flag, value, log_prob)
        elif training_master:
            master_model.rollout_buffer.add(all_drivers_states, embedding, reward, done_flag, value, log_prob)
        else:
            for car_index in range(len(project_globals.after_is_arrived_flags)):
                if not project_globals.after_is_arrived_flags[car_index]:
                    rollout_buffers[car_index].add(
                        np.array(car_observations[car_index], dtype=np.float32),
                        cars_action_arrays[car_index],
                        reward,
                        done_flag,
                        cars_values[car_index],
                        cars_log_probas[car_index],
                    )

        car_observations = cars_next_obs

    return episode_sum_of_rewards, actions_per_episode, steps_counter, crashed


def _run_episode_hierarchical(
    episode_idx: int,
    total_steps: int,
    env,
    master_model,
    agent_model,
    experiment,
    *,
    train_both: bool,
    training_local_master: bool,
    training_agent: bool,
    training_global_master: bool,
    collect_bootstrap: bool,
):
    drv = _unwrap_driver(env)
    hw = drv.highway_env

    master_model.rollout_buffer.reset()
    reset_all_buffers()

    raw_obs, info = hw.reset()
    prepared = drv._prepare_state_for_master(raw_obs)
    drv.current_state = prepared
    states_mat = _pad_states_mat_rows(_states_matrix(drv, prepared), experiment.CARS_AMOUNT)
    states_mat = np.nan_to_num(states_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    _sync_arrival_flags_length(experiment)

    global_emb_prev = np.zeros(experiment.EMBEDDING_SIZE, dtype=np.float32)

    episode_sum_of_rewards = 0.0
    actions_per_episode = []
    steps_counter = 0
    crashed = False
    done = truncated = False

    warmup_eps = int(getattr(experiment, "WARMUP_EPISODES", 0))

    fill_lm = train_both or training_local_master
    fill_gm = train_both or training_global_master
    fill_agents = train_both or training_agent

    bootstrap_snap = {}

    reward_mode = getattr(experiment, "AGENT_REWARD_MODE", "global")

    while not done and not truncated:
        steps_counter += 1

        lm1_in = _build_local_master_input(global_emb_prev, states_mat[: experiment.AGENTS_PER_LOCAL_MASTER], experiment)
        lm2_in = _build_local_master_input(
            global_emb_prev,
            states_mat[experiment.AGENTS_PER_LOCAL_MASTER : experiment.CARS_AMOUNT],
            experiment,
        )

        lm1_emb, lv1, lp1 = master_model.get_proto_action(lm1_in)
        lm2_emb, lv2, lp2 = master_model.get_proto_action(lm2_in)
        gm_in = _build_global_master_input(lm1_emb, lm2_emb, experiment)
        gm_emb, gv, gp = master_model.get_proto_action(gm_in)

        lm1_emb = sanitize_float_vector(lm1_emb, dim=int(experiment.EMBEDDING_SIZE))
        lm2_emb = sanitize_float_vector(lm2_emb, dim=int(experiment.EMBEDDING_SIZE))
        gm_emb = sanitize_float_vector(gm_emb, dim=int(experiment.EMBEDDING_SIZE))

        car_observations = _build_agent_obs_list(states_mat, lm1_emb, lm2_emb, experiment)
        _sid = int(experiment.STATE_INPUT_SIZE)
        car_observations = [sanitize_float_vector(o, dim=_sid) for o in car_observations]

        use_random = warmup_eps > 0 and episode_idx <= warmup_eps
        if use_random:
            actions = [np.array([np.random.randint(0, 2)], dtype=np.int64) for _ in car_observations]
            cars_values, cars_log_probas = [], []
            if fill_agents:
                for car_index in range(len(car_observations)):
                    car_values, car_log_prob = get_agent_values_from_observation(
                        np.array(car_observations[car_index], dtype=np.float32),
                        actions[car_index],
                        agent_model,
                    )
                    cars_values.append(car_values)
                    cars_log_probas.append(car_log_prob)
        else:
            actions = Driver.get_action(
                agent_model,
                car_observations,
                total_steps + steps_counter,
                experiment.EXPLORATION_EXPLOITATION_THRESHOLD,
            )
            cars_values, cars_log_probas = [], []
            if fill_agents:
                for car_index in range(len(car_observations)):
                    car_values, car_log_prob = get_agent_values_from_observation(
                        np.array(car_observations[car_index], dtype=np.float32),
                        actions[car_index],
                        agent_model,
                    )
                    cars_values.append(car_values)
                    cars_log_probas.append(car_log_prob)

        cars_scalar_action = []
        cars_action_arrays = []
        for action in actions:
            car_scalar_action, car_action_array = get_scaler_action_and_action_array(action)
            cars_scalar_action.append(car_scalar_action)
            cars_action_arrays.append(car_action_array)

        drv.render()

        action_tuple = tuple(cars_scalar_action)
        raw_next, reward, done, truncated, step_info = hw.step(action_tuple)
        episode_sum_of_rewards += float(reward)
        actions_per_episode.append(tuple(cars_scalar_action))

        if step_info.get("crashed", False) and done:
            crashed = True

        prepared_next = drv._prepare_state_for_master(raw_next)
        drv.current_state = prepared_next
        states_mat = _pad_states_mat_rows(_states_matrix(drv, prepared_next), experiment.CARS_AMOUNT)
        states_mat = np.nan_to_num(states_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        _sync_arrival_flags_length(experiment)

        terminal = bool(done or truncated)
        done_flag = terminal

        per_agent_rewards = step_info.get("agents_rewards")
        if per_agent_rewards is None or len(per_agent_rewards) < experiment.CARS_AMOUNT:
            per_agent_rewards = [float(reward)] * experiment.CARS_AMOUNT

        gm_r = float(reward)
        lm1_r = min(float(x) for x in per_agent_rewards[: experiment.AGENTS_PER_LOCAL_MASTER])
        lm2_r = min(float(x) for x in per_agent_rewards[experiment.AGENTS_PER_LOCAL_MASTER :])
        if reward_mode == "global":
            lm1_r = lm2_r = gm_r

        if fill_lm and len(project_globals.local_master_rollout_buffers) >= 2:
            project_globals.local_master_rollout_buffers[0].add(
                lm1_in,
                lm1_emb,
                lm1_r,
                done_flag,
                lv1,
                lp1,
            )
            project_globals.local_master_rollout_buffers[1].add(
                lm2_in,
                lm2_emb,
                lm2_r,
                done_flag,
                lv2,
                lp2,
            )
        if fill_gm and project_globals.global_master_rollout_buffer is not None:
            project_globals.global_master_rollout_buffer.add(
                gm_in,
                gm_emb,
                gm_r,
                done_flag,
                gv,
                gp,
            )

        if fill_agents:
            for car_index in range(len(project_globals.after_is_arrived_flags)):
                if not project_globals.after_is_arrived_flags[car_index]:
                    ar = gm_r if reward_mode == "global" else (
                        lm1_r if car_index < experiment.AGENTS_PER_LOCAL_MASTER else lm2_r
                    )
                    rollout_buffers[car_index].add(
                        np.array(car_observations[car_index], dtype=np.float32),
                        cars_action_arrays[car_index],
                        ar,
                        done_flag,
                        cars_values[car_index],
                        cars_log_probas[car_index],
                    )

        global_emb_prev = gm_emb.copy()

        if collect_bootstrap:
            lm1_b = _build_local_master_input(global_emb_prev, states_mat[: experiment.AGENTS_PER_LOCAL_MASTER], experiment)
            lm2_b = _build_local_master_input(
                global_emb_prev,
                states_mat[experiment.AGENTS_PER_LOCAL_MASTER : experiment.CARS_AMOUNT],
                experiment,
            )
            e1_b, _, _ = master_model.get_proto_action(lm1_b)
            e2_b, _, _ = master_model.get_proto_action(lm2_b)
            e1_b = sanitize_float_vector(e1_b, dim=int(experiment.EMBEDDING_SIZE))
            e2_b = sanitize_float_vector(e2_b, dim=int(experiment.EMBEDDING_SIZE))
            gm_b = _build_global_master_input(e1_b, e2_b, experiment)
            agent_boot = _build_agent_obs_list(states_mat, e1_b, e2_b, experiment)
            bootstrap_snap = {
                "last_lm1_obs": lm1_b,
                "last_lm2_obs": lm2_b,
                "last_gm_obs": gm_b,
                "last_agent_obs": [
                    sanitize_float_vector(o, dim=int(experiment.STATE_INPUT_SIZE)) for o in agent_boot
                ],
                "last_done": terminal,
            }

    arrived = sum(
        1
        for v in drv._get_unwrapped_env().controlled_vehicles
        if getattr(v, "is_arrived", False)
    )
    arrival_rate = 100.0 * arrived / max(1, experiment.CARS_AMOUNT)

    if collect_bootstrap and not bootstrap_snap:
        lm1_b = _build_local_master_input(global_emb_prev, states_mat[: experiment.AGENTS_PER_LOCAL_MASTER], experiment)
        lm2_b = _build_local_master_input(
            global_emb_prev,
            states_mat[experiment.AGENTS_PER_LOCAL_MASTER : experiment.CARS_AMOUNT],
            experiment,
        )
        e1, _, _ = master_model.get_proto_action(lm1_b)
        e2, _, _ = master_model.get_proto_action(lm2_b)
        e1 = sanitize_float_vector(e1, dim=int(experiment.EMBEDDING_SIZE))
        e2 = sanitize_float_vector(e2, dim=int(experiment.EMBEDDING_SIZE))
        gm_b = _build_global_master_input(e1, e2, experiment)
        agent_boot = _build_agent_obs_list(states_mat, e1, e2, experiment)
        bootstrap_snap = {
            "last_lm1_obs": lm1_b,
            "last_lm2_obs": lm2_b,
            "last_gm_obs": gm_b,
            "last_agent_obs": [
                sanitize_float_vector(o, dim=int(experiment.STATE_INPUT_SIZE)) for o in agent_boot
            ],
            "last_done": True,
        }

    return (
        episode_sum_of_rewards,
        actions_per_episode,
        steps_counter,
        crashed,
        arrival_rate,
        bootstrap_snap if collect_bootstrap else {},
    )


def process_episode(
    episode_idx,
    total_steps,
    env,
    master_model,
    agent_model,
    experiment,
    train_both=False,
    training_master=False,
    *,
    training_local_master=None,
    training_agent=False,
    training_global_master=False,
    collect_bootstrap=False,
    **kwargs,
):
    """
    Unified entry used by ``training_handler`` (hierarchical flags) or legacy callers.

    Returns either a 5-tuple (legacy) or a 6-tuple ending with bootstrap dict when
    ``collect_bootstrap=True``.
    """
    kwargs.pop("training_master", None)

    logger.info("Episode %d", episode_idx)

    hierarchical = training_local_master is not None or training_global_master

    if hierarchical:
        out = _run_episode_hierarchical(
            episode_idx,
            total_steps,
            env,
            master_model,
            agent_model,
            experiment,
            train_both=train_both,
            training_local_master=bool(training_local_master),
            training_agent=training_agent,
            training_global_master=training_global_master,
            collect_bootstrap=collect_bootstrap,
        )
        reward = out[0]
        actions = out[1]
        steps = out[2]
        crashed = out[3]
        arrival_rate = out[4]
        bootstrap = out[5]
        status = "Collision" if crashed else "Success"
        logger.info("Episode %d ended with %s | arrival≈%.1f%%", episode_idx, status, arrival_rate)
        if collect_bootstrap:
            return reward, actions, steps, crashed, arrival_rate, bootstrap
        return reward, actions, steps, crashed, arrival_rate

    master_model.rollout_buffer.reset()
    reset_all_buffers()

    reward, actions, steps, crashed = run_episode(
        experiment,
        total_steps,
        env,
        master_model,
        agent_model,
        train_both=train_both,
        training_master=training_master,
    )
    status = "Collision" if crashed else "Success"
    logger.info("Episode %d ended with %s", episode_idx, status)
    arrival_rate = 0.0
    try:
        inner = _unwrap_driver(env)._get_unwrapped_env()
        arrived = sum(1 for v in inner.controlled_vehicles if getattr(v, "is_arrived", False))
        arrival_rate = 100.0 * arrived / max(1, experiment.CARS_AMOUNT)
    except Exception:
        pass

    if collect_bootstrap:
        drv = _unwrap_driver(env)
        raw_obs, _ = drv.highway_env.reset()
        prepared = drv._prepare_state_for_master(raw_obs)
        sm = _states_matrix(drv, prepared)
        zero_gm = np.zeros(experiment.EMBEDDING_SIZE, dtype=np.float32)
        lm1_in = _build_local_master_input(zero_gm, sm[: experiment.AGENTS_PER_LOCAL_MASTER], experiment)
        lm2_in = _build_local_master_input(
            zero_gm,
            sm[experiment.AGENTS_PER_LOCAL_MASTER : experiment.CARS_AMOUNT],
            experiment,
        )
        e1, _, _ = master_model.get_proto_action(lm1_in)
        e2, _, _ = master_model.get_proto_action(lm2_in)
        gm_in = _build_global_master_input(e1, e2, experiment)
        agent_obs = _build_agent_obs_list(sm, e1, e2, experiment)
        bootstrap = {
            "last_lm1_obs": lm1_in,
            "last_lm2_obs": lm2_in,
            "last_gm_obs": gm_in,
            "last_agent_obs": [np.asarray(o, dtype=np.float32).reshape(-1) for o in agent_obs],
            "last_done": False,
        }
        return reward, actions, steps, crashed, arrival_rate, bootstrap

    return reward, actions, steps, crashed, arrival_rate


def build_local_master_input(global_emb: np.ndarray, group_states: np.ndarray, experiment=None) -> np.ndarray:
    return _build_local_master_input(global_emb, group_states, experiment)


def build_global_master_input(lm1_emb: np.ndarray, lm2_emb: np.ndarray, experiment=None) -> np.ndarray:
    return _build_global_master_input(lm1_emb, lm2_emb, experiment)
