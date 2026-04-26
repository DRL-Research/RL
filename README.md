Base code for scalable masters.

## Architecture (research story)

- **One master policy, many roles:** a single SB3 `PPO` in `MasterModel` maps a fixed **25-D** input (5 slots × (4-D + 1-D **ID bit**)) to a **4-D** embedding. Local masters and the global master **share weights**; they differ only by how `build_master_input` fills slots (`AGENT_BIT` vs `MASTER_BIT`).
- **Second policy for vehicles:** low-level control uses a **separate** agent `PPO` (shared across cars, per-car rollout buffers). Saving produces (at least) master + agent checkpoints — not one combined weight file unless you bundle them yourself.
- **Stable-Baselines3:** the master `PPO` is built with a minimal `_MasterEnv` so SB3 can construct the network. **Training does not use `PPO.learn()` on the intersection.** Rollouts are collected in the custom env loop; updates run via `training_loop_utils.train_model_from_buffer` with PPO-style clipped loss + value + entropy, using experiment `GAMMA`, `GAE_LAMBDA`, `ENT_COEF`, `VF_COEF`, `CLIP_RANGE` (aligned with the master constructor since the fix in `master_model.py`).

## Logging

- **`agent_logs/` / `master_logs/` `progress.csv`:** often empty — SB3 fills these during `learn()`; this project logs manually instead.
- **`episode_metrics.csv`** (experiment folder): one row per episode (`episode`, `reward`, `arrival_pct`, `collision`) written from `training_handler.training_loop`.
- **Collision audit:** `python run_collision_audit.py` (edit constants at top of file) or `python analyze_collisions.py --episodes 80 --out audit.json`; records per-step `vehicle.crashed`, positions, actions, and distances to uncontrolled vehicles.
- **Full step dataset (50 ep):** `python run_collision_dataset.py` → `experiments/collision_steps_dataset.csv` (every agent x,y,vx,vy, crashed each step; intersecting polygon pairs `C0+C3`; `any_new_crash`); plus `collision_episode_summary.json` and `collision_steps_analysis.json`.
- **Why collisions at step ~14:** `python run_collision_step_analysis.py` (after the dataset run) — histogram of first crash step, fraction at step 14, scenario IDs near that time; notes that with `policy_frequency=1` each step ≈ 1 s sim time.

## Execution Flow

```text
main.py / main_final.py / main_sweep*.py
│
└── run_experiment() or training_loop()
    │
    ├── MasterModel
    │   └── Single PPO → embeddings (master_model.py)
    │
    ├── Agent (Driver wrapper)
    │   ├── Wraps environment; builds 8-D obs (state + embedding)
    │   └── gym.make('RELintersection-v0')
    │           └── IntersectionEnv (intersection_class.py)
    │
    ├── Model (Agent PPO)
    │   └── PPO agent (model_handler.py / general_utils.py)
    │
    ├── Training (training_handler.py)
    │   ├── Episodes (episode_utils.process_episode)
    │   └── perform_training_phase → train_model_from_buffer + agent buffers
    │
    ├── Saving / plots / summary.json
    │
    └── Plotting (plotting_utils / main_* save_plots)

