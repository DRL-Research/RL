# test_custom_initial_positions.py
"""
Test with CUSTOM initial positions - Full control!
==================================================
Create different test scenarios by manually placing agents
at specific positions around the intersection.
"""

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Normal

# =========================
# Configuration
# =========================
MODELS_DIR = Path(
    r"C:\Users\glebb\Downloads\RL-highway-feature-exp6_2_agents_100_scenarios\F21\RL-highway-feature-exp6_2_agents_100_scenarios\new_arch\MODELS_GOOD"
)

CONFIG = {
    "n_agents": 5,
    "duration": 25,
    "target_speeds": [0, 10, 20],
}

DANGER_THRESHOLD = 5.0
SAVE_DIR = Path(__file__).resolve().parent / "pca_custom_positions"

# =========================
# CUSTOM INITIAL CONFIGURATIONS
# =========================
# Each configuration places 5 agents at specific positions
# Positions are (x, y) coordinates, agents start 75m from center

CONFIGURATIONS = {
    "config_1_spread": [
        # Agents well-spread around intersection
        (-75, 0),  # From west
        (0, -75),  # From south
        (75, 0),  # From east
        (0, 75),  # From north
        (-75, 10),  # From west, slightly offset
    ],

    "config_2_clustered": [
        # Agents clustered (will be close together)
        (-75, 0),
        (-75, 5),
        (-75, 10),
        (0, -75),
        (0, -70),
    ],

    "config_3_opposite": [
        # Agents on opposite sides
        (-75, 0),
        (-75, 5),
        (75, 0),
        (75, 5),
        (0, 75),
    ],

    "config_4_mixed": [
        # Mixed distances
        (-75, 0),
        (-60, 0),
        (0, -75),
        (75, 0),
        (0, 75),
    ],

    "config_5_tight": [
        # Very tight formation
        (-75, 0),
        (-74, 0),
        (-73, 0),
        (-72, 0),
        (-71, 0),
    ],
}


# =========================
# Architecture Detection
# =========================
def detect_architecture(checkpoint_path: str, model_type: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'net' not in ckpt:
        raise ValueError("No 'net' in checkpoint")

    if 'shared.0.weight' in ckpt['net']:
        encoder_name = 'shared'
        hidden_dim = int(ckpt['net']['shared.0.weight'].shape[0])
        input_dim = int(ckpt['net']['shared.0.weight'].shape[1])
    elif 'encoder.0.weight' in ckpt['net']:
        encoder_name = 'encoder'
        hidden_dim = int(ckpt['net']['encoder.0.weight'].shape[0])
        input_dim = int(ckpt['net']['encoder.0.weight'].shape[1])
    elif 'obs_encoder.0.weight' in ckpt['net']:
        encoder_name = 'obs_encoder'
        hidden_dim = int(ckpt['net']['obs_encoder.0.weight'].shape[0])
        input_dim = int(ckpt['net']['obs_encoder.0.weight'].shape[1])
    else:
        raise ValueError("Cannot find encoder")

    if model_type == 'master':
        if 'embedding_dim' in ckpt:
            embedding_dim = int(ckpt['embedding_dim'])
        elif 'log_std' in ckpt['net']:
            embedding_dim = int(ckpt['net']['log_std'].shape[0])
        else:
            raise ValueError("Cannot detect embedding_dim")
        return embedding_dim, hidden_dim, encoder_name
    else:
        return hidden_dim, encoder_name, input_dim


# =========================
# Networks
# =========================
class FlexibleMasterNet(nn.Module):
    def __init__(self, state_dim: int, embedding_dim: int, hidden_dim: int, encoder_name: str):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_name = encoder_name

        encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        setattr(self, encoder_name, encoder)

        self.mean_head = nn.Linear(hidden_dim, embedding_dim)
        self.log_std = nn.Parameter(torch.zeros(embedding_dim) - 1.0)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        encoder = getattr(self, self.encoder_name)
        h = encoder(x)
        emb_mean = self.mean_head(h)
        value = self.value_head(h)
        return emb_mean, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        emb_mean, value = self.forward(x)
        std = self.log_std.exp().clamp(0.05, 0.5)
        dist = Normal(emb_mean, std)
        embedding = emb_mean if deterministic else dist.rsample()
        return embedding, value.squeeze(-1)


class FlexibleAgentNet(nn.Module):
    def __init__(self, obs_dim: int, embedding_dim: int, n_actions: int, hidden_dim: int, encoder_name: str):
        super().__init__()
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.encoder_name = encoder_name

        encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        setattr(self, encoder_name, encoder)

        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, combined_input: torch.Tensor):
        encoder = getattr(self, self.encoder_name)
        h = encoder(combined_input)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


class Master:
    def __init__(self, state_dim: int, embedding_dim: int, hidden_dim: int, encoder_name: str):
        self.device = "cpu"
        self.net = FlexibleMasterNet(state_dim, embedding_dim, hidden_dim, encoder_name).to(self.device)

    def get_action(self, state, deterministic: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embedding, _ = self.net.get_action(state_t, deterministic)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def load(self, path):
        ckpt = torch.load(path + ".pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        print(f"✓ Master loaded")


class Agent:
    def __init__(self, obs_dim: int, embedding_dim: int, n_actions: int, hidden_dim: int, encoder_name: str):
        self.device = "cpu"
        self.net = FlexibleAgentNet(obs_dim, embedding_dim, n_actions, hidden_dim, encoder_name).to(self.device)

    def get_actions(self, local_obs_list, embedding, deterministic=False):
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        embedding_t = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)

        actions = []
        for obs in local_obs_list:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            emb_t = embedding_t.unsqueeze(0)
            combined = torch.cat([obs_t, emb_t], dim=-1)

            with torch.no_grad():
                logits, _ = self.net.forward(combined)
                action = logits.argmax(-1)
            actions.append(int(action.item()))
        return actions

    def load(self, path):
        ckpt = torch.load(path + ".pt", map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        print(f"✓ Agent loaded")


# =========================
# Environment with Custom Positions
# =========================
def make_env():
    env = gym.make("intersection-v1", render_mode=None)

    env.unwrapped.configure({
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": CONFIG["n_agents"],
                "see_behind": True,
            }
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "target_speeds": CONFIG["target_speeds"],
                "longitudinal": True,
                "lateral": False,
            }
        },
        "duration": CONFIG["duration"],
        "controlled_vehicles": CONFIG["n_agents"],
        "initial_vehicle_count": 0,
        "spawn_probability": 0,
        "collision_reward": -300,
        "high_speed_reward": 0.3,
        "arrived_reward": 50,
        "reward_speed_range": [0, 9],
        "policy_frequency": 1,
        "simulation_frequency": 15,
    })

    env.reset()
    return env


def set_custom_positions(env, positions):
    """
    Manually set agent positions after reset.
    positions: list of (x, y) tuples
    """
    controlled_vehicles = env.unwrapped.controlled_vehicles

    if len(controlled_vehicles) != len(positions):
        print(f"Warning: {len(controlled_vehicles)} vehicles but {len(positions)} positions")
        return False

    for i, (vehicle, pos) in enumerate(zip(controlled_vehicles, positions)):
        # Set position
        vehicle.position = np.array(pos, dtype=np.float64)

        # Set heading based on position (point towards center)
        if pos[0] < 0:  # Coming from west
            vehicle.heading = 0
        elif pos[0] > 0:  # Coming from east
            vehicle.heading = np.pi
        elif pos[1] < 0:  # Coming from south
            vehicle.heading = np.pi / 2
        else:  # Coming from north
            vehicle.heading = -np.pi / 2

        # Set initial velocity
        vehicle.velocity = 10.0

    return True


def get_global_state(obs, n_agents):
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs.flatten().astype(np.float32)


def get_local_obs_list(obs, n_agents):
    if isinstance(obs, tuple):
        return [o[0].astype(np.float32) for o in obs]
    return [obs[i][0].astype(np.float32) for i in range(n_agents)]


def compute_min_distance(local_obs_list):
    positions = np.array([[obs[0], obs[1]] for obs in local_obs_list])
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, dist)
    return min_dist if min_dist != float('inf') else 100.0


# =========================
# Data Collection
# =========================
def collect_with_custom_positions(master, agent, configurations, n_runs_per_config):
    """
    Run multiple episodes for each configuration.
    Each configuration = different initial positions.
    """
    all_embeddings = []
    all_labels = []
    all_distances = []
    all_configs = []

    env = make_env()

    print("\n" + "=" * 70)
    print("COLLECTING WITH CUSTOM POSITIONS")
    print("=" * 70)

    for config_name, positions in configurations.items():
        print(f"\nConfiguration: {config_name}")
        print(f"  Positions: {positions}")

        config_embeddings = []
        config_labels = []
        config_distances = []

        for run in range(n_runs_per_config):
            # Reset and set custom positions
            obs, _ = env.reset()
            success = set_custom_positions(env, positions)

            if not success:
                print("  Failed to set positions!")
                continue

            # Get observation after custom positioning
            obs = env.unwrapped._observation()

            done = truncated = False
            step = 0

            while not done and not truncated and step < CONFIG["duration"]:
                global_state = get_global_state(obs, CONFIG["n_agents"])
                embedding = master.get_action(global_state, deterministic=True)

                local_obs_list = get_local_obs_list(obs, CONFIG["n_agents"])
                min_dist = compute_min_distance(local_obs_list)

                actions = agent.get_actions(local_obs_list, embedding, deterministic=True)
                obs, _, done, truncated, _ = env.step(tuple(actions))

                config_embeddings.append(embedding.copy())
                config_distances.append(min_dist)
                config_labels.append("Dangerous" if min_dist < DANGER_THRESHOLD else "Safe")

                step += 1

            if (run + 1) % 10 == 0:
                print(f"  Run {run + 1}/{n_runs_per_config}")

        all_embeddings.extend(config_embeddings)
        all_labels.extend(config_labels)
        all_distances.extend(config_distances)
        all_configs.extend([config_name] * len(config_embeddings))

        n_dangerous = config_labels.count("Dangerous")
        n_safe = config_labels.count("Safe")
        print(f"  ✓ {len(config_embeddings)} steps")
        print(
            f"    Dangerous: {n_dangerous} ({100 * n_dangerous / len(config_embeddings) if config_embeddings else 0:.0f}%)")
        print(f"    Safe: {n_safe} ({100 * n_safe / len(config_embeddings) if config_embeddings else 0:.0f}%)")

    env.close()

    embeddings = np.array(all_embeddings)

    print("\n" + "=" * 70)
    print("COMBINED DATASET")
    print("=" * 70)
    print(f"Total samples: {len(embeddings)}")
    print(f"Dangerous: {all_labels.count('Dangerous')}")
    print(f"Safe: {all_labels.count('Safe')}")
    print(f"Embedding: range=[{embeddings.min():.1f}, {embeddings.max():.1f}], std={embeddings.std():.1f}")

    return embeddings, all_labels, all_distances, all_configs


# =========================
# PCA Visualization
# =========================
def create_pca_plot(embeddings, labels, distances, configs, save_dir):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'label': labels,
        'distance': distances,
        'config': configs
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # LEFT: Color by dangerous/safe
    danger_mask = df['label'] == 'Dangerous'
    ax1.scatter(df[danger_mask]['PC1'], df[danger_mask]['PC2'],
                c='#1f77b4', label='Dangerous (< 5m)', alpha=0.6, s=50, edgecolors='none')

    safe_mask = df['label'] == 'Safe'
    ax1.scatter(df[safe_mask]['PC1'], df[safe_mask]['PC2'],
                c='#ff7f0e', label='Safe (≥ 5m)', alpha=0.6, s=50, edgecolors='none')

    ax1.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax1.set_title('PCA - Colored by Distance\n(Custom Initial Positions)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # RIGHT: Color by configuration
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, config_name in enumerate(CONFIGURATIONS.keys()):
        config_mask = df['config'] == config_name
        ax2.scatter(df[config_mask]['PC1'], df[config_mask]['PC2'],
                    c=colors[i % len(colors)], label=config_name,
                    alpha=0.6, s=50, edgecolors='none')

    ax2.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax2.set_title('PCA - Colored by Configuration\n(Different Starting Positions)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / 'pca_custom_positions.png'
    fig.savefig(output_path, dpi=220)
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    print(f"\nPCA Statistics:")
    print(f"  PC1: [{df['PC1'].min():.1f}, {df['PC1'].max():.1f}]")
    print(f"  PC2: [{df['PC2'].min():.1f}, {df['PC2'].max():.1f}]")
    print(f"  Variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    return df, pca


# =========================
# Main
# =========================
def main():
    print("=" * 70)
    print("PCA ANALYSIS - CUSTOM INITIAL POSITIONS")
    print("=" * 70)
    print("Testing 5 different initial configurations")
    print("All agents start ~75m from intersection center")
    print("=" * 70)

    # Find models
    master_files = list(MODELS_DIR.glob("master*.pt"))
    agent_files = list(MODELS_DIR.glob("agent*.pt"))

    if not master_files or not agent_files:
        raise FileNotFoundError(f"No models in {MODELS_DIR}")

    print(f"\nModels: {master_files[0].name}, {agent_files[0].name}")

    # Detect & load
    embedding_dim, master_hidden, master_encoder = detect_architecture(str(master_files[0]), 'master')
    agent_hidden, agent_encoder, agent_input_dim = detect_architecture(str(agent_files[0]), 'agent')

    print(f"  Master: emb={embedding_dim}, hidden={master_hidden}")
    print(f"  Agent: input={agent_input_dim}, hidden={agent_hidden}")

    master = Master(CONFIG["n_agents"] * 4, embedding_dim, master_hidden, master_encoder)
    agent = Agent(agent_input_dim, embedding_dim, len(CONFIG["target_speeds"]), agent_hidden, agent_encoder)

    master.load(str(master_files[0].with_suffix('')))
    agent.load(str(agent_files[0].with_suffix('')))

    # Collect data
    embeddings, labels, distances, configs = collect_with_custom_positions(
        master, agent, CONFIGURATIONS, n_runs_per_config=20
    )

    # Create PCA
    df, pca = create_pca_plot(embeddings, labels, distances, configs, SAVE_DIR)

    # Save data
    df_full = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    df_full['label'] = labels
    df_full['distance'] = distances
    df_full['config'] = configs
    df_full['PC1'] = df['PC1']
    df_full['PC2'] = df['PC2']
    df_full.to_csv(SAVE_DIR / 'custom_positions_data.csv', index=False)

    print("\n" + "=" * 70)
    print("✓ COMPLETE!")
    print(f"✓ Results: {SAVE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()