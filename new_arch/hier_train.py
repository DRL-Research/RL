# hier_train.py
"""
Training loop for the 3-level Hierarchical Architecture (Scalability POC)
=========================================================================
Hierarchy:
  GlobalMaster  (1)
    ├─ LocalMaster_0  ->  Agents [0,1,2]
    └─ LocalMaster_1  ->  Agents [3,4,5]

Total vehicles in env: n_agents = n_local_masters * agents_per_master = 2*3 = 6

Each step:
  1. Build GlobalMaster input from LocalMaster aggregate states
  2. GlobalMaster produces global_embedding
  3. For each group g in {0,1}:
       a. Build LocalMaster_g input from its 3 agents' raw states
       b. LocalMaster_g receives global_embedding as guidance
       c. LocalMaster_g produces local_embedding_g
       d. Each agent in group g receives local_embedding_g + its own obs -> action
  4. Step environment
  5. Store transitions for all 3 levels
  6. Every train_every_n_episodes: PPO update for all networks

Metrics collected:
  - episode: reward, collisions, arrivals, mean_speed, min_distance
  - loss: GlobalMaster, LocalMaster_0, LocalMaster_1, AgentGroup_0, AgentGroup_1
"""

import os
import csv
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

from hier_master import (SharedMasterPolicy, make_global_master, make_local_master,
                         build_master_input)
from hier_agent  import HierAgent, HierAgentNet


# ── environment ──────────────────────────────────────────────────────────────

def make_env(config: dict):
    n = config["n_agents"]   # = n_local_masters * agents_per_master
    env = gym.make("intersection-v1", render_mode=None)
    env.unwrapped.configure({
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute":        True,
                "normalize":       False,
                "vehicles_count":  n,
                "see_behind":      True,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type":          "DiscreteMetaAction",
                "target_speeds": config["target_speeds"],
                "longitudinal":  True,
                "lateral":       False,
            },
        },
        "duration":               config["duration"],
        "controlled_vehicles":    n,
        "initial_vehicle_count":  0,
        "spawn_probability":      0,
        "collision_reward":       config["collision_reward"],
        "high_speed_reward":      config["high_speed_reward"],
        "arrived_reward":         config["arrived_reward"],
        "reward_speed_range":     config["reward_speed_range"],
        "policy_frequency":       1,
        "simulation_frequency":   15,
    })
    env.reset()
    return env


# ── safe reset ───────────────────────────────────────────────────────────────

def _min_pairwise_dist(obs_list: list) -> float:
    """Minimum distance between any two agents at episode start."""
    pos = np.array([[o[0], o[1]] for o in obs_list])
    d = float('inf')
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = min(d, float(np.linalg.norm(pos[i] - pos[j])))
    return d if d != float('inf') else 100.0


def safe_reset(env, n_agents: int, min_start_dist: float = 5.0,
               max_retries: int = 20):
    """
    Reset the environment and retry if any two agents start closer than
    min_start_dist metres (avoids un-solvable near-collision scenarios).
    """
    for attempt in range(max_retries):
        obs, info = env.reset()
        obs_list  = parse_obs(obs, n_agents)
        if _min_pairwise_dist(obs_list) >= min_start_dist:
            return obs, info
    # If we never found a clean reset, return the last one and warn once
    return obs, info


# ── observation helpers ───────────────────────────────────────────────────────

def parse_obs(obs, n_agents: int):
    """Return list of n_agents np.ndarray (4,) = [x,y,vx,vy]."""
    if isinstance(obs, tuple):
        return [np.asarray(o[0], dtype=np.float32) for o in obs]
    return [np.asarray(obs[i][0], dtype=np.float32) for i in range(n_agents)]


def aggregate_state(obs_list: list) -> np.ndarray:
    """Mean (x,y,vx,vy) of a group – used as the LocalMaster's 'state' seen by GlobalMaster."""
    return np.mean(np.stack(obs_list), axis=0).astype(np.float32)


def build_global_master_input(lm_states: list, n_local_masters: int) -> np.ndarray:
    """lm_states: list of (4,) arrays, one per LocalMaster."""
    return build_master_input(
        sub_states=[s for s in lm_states],
        type_bits=[1] * n_local_masters,   # 1 = master-type subordinate
        n_active=n_local_masters,
    )


def build_local_master_input(agent_obs_list: list) -> np.ndarray:
    """agent_obs_list: list of (4,) arrays for the group's agents."""
    n = len(agent_obs_list)
    return build_master_input(
        sub_states=agent_obs_list,
        type_bits=[0] * n,                 # 0 = agent-type subordinate
        n_active=n,
    )


# ── coordination reward ───────────────────────────────────────────────────────

def _min_dist(obs_list: list) -> float:
    pos = np.array([[o[0], o[1]] for o in obs_list])
    d   = float('inf')
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = min(d, float(np.linalg.norm(pos[i] - pos[j])))
    return d if d != float('inf') else 100.0


def coordination_reward(all_obs: list, actions: list, config: dict) -> float:
    vels      = np.array([o[2] for o in all_obs])
    vel_std   = float(np.std(vels))
    vel_mean  = float(np.mean(np.abs(vels)))
    min_dist  = _min_dist(all_obs)
    uniq_acts = len(set(actions))
    n         = len(all_obs)

    r = 0.0
    if 2.0 < vel_std < 8.0 and vel_mean > 5.0:
        r += config.get("coord_velocity_bonus", 1.5)
    if min_dist > 15.0:
        r += config.get("coord_safe_bonus", 2.0)
    elif min_dist > 10.0:
        r += config.get("coord_safe_bonus", 1.0)
    elif min_dist < 5.0:
        r -= config.get("coord_danger_penalty", 4.0)
    if 2 <= uniq_acts <= n - 1:
        r += config.get("coord_diversity_bonus", 1.0)
    return r


# ── logger ────────────────────────────────────────────────────────────────────

class HierLogger:
    def __init__(self, save_dir: str, flush_every: int = 50):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir     = save_dir
        self.flush_every  = flush_every
        self.episodes     = []
        self.loss_records = []

    def log_episode(self, ep, total_reward, group_rewards, collisions,
                    arrivals, mean_speed, min_dist, steps):
        self.episodes.append({
            "episode":       ep,
            "total_reward":  round(total_reward, 3),
            "reward_g0":     round(group_rewards[0], 3),
            "reward_g1":     round(group_rewards[1], 3),
            "collisions":    int(collisions),
            "arrivals":      int(arrivals),
            "mean_speed":    round(mean_speed, 3),
            "min_dist":      round(min_dist, 3),
            "steps":         int(steps),
        })
        # Flush to disk periodically so data is visible during training
        if (ep + 1) % self.flush_every == 0:
            self._flush("episodes", self.episodes)

    def log_losses(self, ep: int, loss_dict: dict):
        record = {"episode": ep}
        record.update({k: round(v, 6) for k, v in loss_dict.items()})
        self.loss_records.append(record)
        # Flush losses every flush_every episodes
        if (ep + 1) % self.flush_every == 0:
            self._flush("losses", self.loss_records)

    def _flush(self, name: str, data: list):
        if not data:
            return
        all_keys = []
        seen = set()
        for record in data:
            for k in record.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        fp = os.path.join(self.save_dir, f"{name}.csv")
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(data)

    def save(self):
        self._flush("episodes", self.episodes)
        self._flush("losses",   self.loss_records)
        print(f"  [Logger] saved to {self.save_dir}/")


# ── main training function ────────────────────────────────────────────────────

def train_hier(config: dict, save_dir: str, logger: HierLogger):
    n_lm      = config["n_local_masters"]    # 2
    apm       = config["agents_per_master"]   # 3
    n_agents  = n_lm * apm                    # 6

    env = make_env(config)

    # -- build hierarchy --
    # ONE shared policy for ALL masters (GlobalMaster + LocalMasters)
    master_policy  = SharedMasterPolicy(config)
    global_master  = make_global_master(master_policy)
    local_masters  = [make_local_master(master_policy, mid) for mid in range(n_lm)]

    # ONE shared AgentNet across both agent groups
    shared_agent_net = HierAgentNet(
        obs_dim=4,
        guidance_dim=config["local_embedding_dim"],
        n_actions=len(config["target_speeds"]),
        hidden_dim=config["agent_hidden_dim"],
    )
    agent_groups = [HierAgent(config, gid, shared_net=shared_agent_net) for gid in range(n_lm)]

    total_eps = config["total_episodes"]
    train_every = config["train_every_n_episodes"]
    print_every = config["print_every_n_episodes"]

    print(f"\n{'='*65}")
    print(f"  Hierarchical Scalability POC")
    print(f"  GlobalMaster -> {n_lm} LocalMasters -> {apm} Agents each = {n_agents} total")
    print(f"  Episodes: {total_eps}   |   Train every: {train_every}")
    print(f"{'='*65}\n")

    env_restart_every = 100   # recreate env to prevent C-level segfault accumulation

    for ep in range(total_eps):
        # Periodically recreate the environment to prevent C++ memory buildup
        if ep > 0 and ep % env_restart_every == 0:
            env.close()
            env = make_env(config)

        obs, _ = safe_reset(env, n_agents, min_start_dist=5.0)
        done = truncated = False
        ep_reward   = 0.0
        g_rewards   = [0.0, 0.0]
        collisions  = 0
        arrivals    = 0
        speeds      = []
        min_dists   = []
        step        = 0

        while not done and not truncated:
            # -- parse observations --
            all_obs = parse_obs(obs, n_agents)
            group_obs = [all_obs[:apm], all_obs[apm:]]   # group 0, group 1

            # -- GlobalMaster: aggregate states of LocalMasters --
            lm_states   = [aggregate_state(group_obs[g]) for g in range(n_lm)]
            gm_inp      = build_global_master_input(lm_states, n_lm)
            global_emb  = global_master.get_embedding(gm_inp, guidance=None, deterministic=False)

            # -- LocalMasters: each sees its group + global guidance --
            local_embs  = []
            lm_inp_list = []
            for g in range(n_lm):
                lm_inp = build_local_master_input(group_obs[g])
                lm_inp_list.append(lm_inp)
                local_emb = local_masters[g].get_embedding(lm_inp, guidance=global_emb, deterministic=False)
                local_embs.append(local_emb)

            # -- Agents: each group uses its LocalMaster's embedding --
            all_actions = []
            for g in range(n_lm):
                group_actions = agent_groups[g].get_actions(group_obs[g], local_embs[g])
                all_actions.extend(group_actions)

            # -- environment step --
            next_obs, env_reward, done, truncated, info = env.step(tuple(all_actions))
            next_all_obs  = parse_obs(next_obs, n_agents)
            next_group_obs = [next_all_obs[:apm], next_all_obs[apm:]]

            coord_r      = coordination_reward(all_obs, all_actions, config)
            total_step_r = float(env_reward) + coord_r
            ep_reward   += total_step_r

            # Per-group reward = sum of agents' rewards (each gets total/n_agents)
            agent_r = total_step_r / n_agents
            for g in range(n_lm):
                g_rewards[g] += agent_r * apm

            # Detect collision via info dict (reliable - not affected by reward averaging)
            if info.get("crashed", False):
                collisions += 1

            # Count arrivals: agents_terminated = tuple of per-agent done flags
            agents_terminated = info.get("agents_terminated", ())
            arrivals += sum(1 for t in agents_terminated if t)

            for o in all_obs:
                speeds.append(float(abs(o[2])))
            min_dists.append(_min_dist(all_obs))

            # -- store transitions --
            terminal = done or truncated

            # GlobalMaster: no guidance (root), subordinates are the LocalMasters
            next_lm_states = [aggregate_state(next_group_obs[g]) for g in range(n_lm)]
            next_gm_inp    = build_global_master_input(next_lm_states, n_lm)
            global_master.store(gm_inp,          None,       global_emb,
                                total_step_r, next_gm_inp, None, terminal)

            # LocalMasters: guidance = global_emb (from parent GlobalMaster)
            for g in range(n_lm):
                next_lm_inp_raw = build_local_master_input(next_group_obs[g])
                local_masters[g].store(lm_inp_list[g], global_emb,  local_embs[g],
                                       agent_r * apm,  next_lm_inp_raw, global_emb,
                                       terminal)

            # Agents
            for g in range(n_lm):
                agent_groups[g].store(
                    group_obs[g], local_embs[g],
                    all_actions[g * apm: (g + 1) * apm],
                    [agent_r] * apm,
                    next_group_obs[g],
                    [terminal] * apm,
                )

            obs  = next_obs
            step += 1

        # -- episode summary --
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        mean_mdist = float(np.mean(min_dists)) if min_dists else 0.0
        logger.log_episode(ep, ep_reward, g_rewards, collisions, arrivals,
                           mean_speed, mean_mdist, step)

        # -- PPO update --
        if (ep + 1) % train_every == 0:
            all_losses = {}
            # One update for the SHARED master policy (all master transitions pooled)
            all_losses.update(master_policy.train())
            # One update for the shared agent policy
            for ag in agent_groups:
                all_losses.update(ag.train())
            if all_losses:
                logger.log_losses(ep, all_losses)

        # -- progress print --
        if (ep + 1) % print_every == 0:
            recent   = logger.episodes[-print_every:]
            avg_r    = np.mean([d["total_reward"] for d in recent])
            crash_r  = np.mean([d["collisions"]   for d in recent])
            arr_r    = np.mean([d["arrivals"]      for d in recent])
            print(f"  Ep {ep+1:>4}/{total_eps} | "
                  f"AvgR: {avg_r:>8.1f} | "
                  f"Crashes: {crash_r:.2f} | "
                  f"Arrivals: {arr_r:.2f}")

    # -- save models --
    os.makedirs(save_dir, exist_ok=True)
    # Shared master policy (one file for all master levels)
    master_policy.save(os.path.join(save_dir, "shared_master_policy.pt"))
    # Shared agent policy
    agent_groups[0].save(os.path.join(save_dir, "shared_agent_policy.pt"))

    env.close()
    logger.save()
    print(f"\n  Models saved to {save_dir}/")
    return logger
