"""Entry-point for training the SafeR-ADAIM agent."""
from __future__ import annotations

import argparse
from pathlib import Path

import copy

import torch

from highwayenv.utils import patch_intersection_env, register_intersection_env

from src.experiment import scenarios_config
from src.safe_radaim import (
    SafeIntersectionEnv,
    SafeRADAIMAgent,
    SafeRADAIMConfig,
    SafeRADAIMTrainer,
    SafeRADAIMTrainSettings,
)


def build_environment(config: SafeRADAIMConfig) -> SafeIntersectionEnv:
    patch_intersection_env()
    register_intersection_env()
    base_env = copy.deepcopy(scenarios_config.full_env_config_exp5)
    import gymnasium as gym

    env = gym.make("RELintersection-v0", render_mode=None, config=base_env)
    return SafeIntersectionEnv(
        env,
        min_velocity=config.min_velocity,
        max_velocity=config.max_velocity,
        epsilon_distance_cost=config.epsilon_distance_cost,
        epsilon_collision_cost=config.epsilon_collision_cost,
        epsilon_v=config.epsilon_v,
        epsilon_t=config.epsilon_t,
        epsilon_pass_single=config.epsilon_pass_single,
        epsilon_pass_all=config.epsilon_pass_all,
        epsilon_a=config.epsilon_a,
        epsilon_j=config.epsilon_j,
        safe_distance=config.safe_distance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SafeR-ADAIM agent")
    parser.add_argument("--epochs", type=int, default=32, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1024, help="Environment steps per epoch")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    args = parser.parse_args()

    config = SafeRADAIMConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        device=args.device,
    )

    env = build_environment(config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SafeRADAIMAgent(obs_dim=obs_dim, action_dim=action_dim, config=config)
    settings = SafeRADAIMTrainSettings(total_epochs=config.epochs)
    trainer = SafeRADAIMTrainer(env=env, agent=agent, config=config, settings=settings)
    stats = trainer.train()

    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    torch.save(agent.networks.policy.state_dict(), output_dir / "safe_radaim_policy.pt")
    torch.save(agent.networks.value_fn.state_dict(), output_dir / "safe_radaim_value.pt")
    torch.save(agent.networks.cost_value_fn.state_dict(), output_dir / "safe_radaim_cost_value.pt")

    for entry in stats:
        print(
            f"Epoch {entry.epoch}: reward={entry.total_reward:.2f} cost={entry.total_cost:.2f} "
            f"KL={entry.policy_info.kl_divergence:.4f} case={entry.policy_info.case}"
        )


if __name__ == "__main__":
    main()
