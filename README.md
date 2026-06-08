# Hierarchical Multi-Agent RL — Scalable Coordinated Driving

A 3-level master-agent hierarchy trained with PPO on custom `highway-env` layouts. The core claim: one pair of trained checkpoints (master + agent), deployed at any scale from 3 to 48 agents across independent or connected intersections, consistently reduces crash rates compared to running agents without a master signal.

## Two separate models, two kinds of sharing

The system uses **two distinct PPO models** with different architectures, observation spaces, and action spaces:

| | Master model | Agent model |
|---|---|---|
| Observation | 25-D (5 slots × 5) | 8-D (4-D state + 4-D embedding) |
| Output | 4-D continuous embedding | discrete {slow, fast} |
| Network | ResNet (25→128→128→4) | MLP wide (8→256→256→2) |

What is shared within each role: all Local Masters and the Global Master run the **same single MasterModel instance** (one set of weights for all masters). All agents run the **same single agent model instance** (one set of weights for all agents). There is no parameter sharing between the master model and the agent model.

This is why the system scales: adding more intersections means running the same master model on more inputs, not introducing new parameters. When the system grows beyond 5 Local Masters, intermediate masters group them in sets of ≤5 recursively, forming a tree of depth ⌈log₅(N\_LMs)⌉ — all running the same master weights.

## Architecture

```
                         GM  (Global Master)
                        / | \
                      M1  M2  ...   (intermediate masters when N_LMs > 5)
                     /|   |\
                   LM1  LM2  ...   (Local Masters — one per intersection zone)
                   /|\   /|\
                 a  a  a  a  a  a  (Agents — one per vehicle)
```

**Master observation (25-D):** 5 slots × 5 values each.
Each slot = `[v0, v1, v2, v3, type_bit]`:
- `v0..v3` is either a subordinate agent's 4-D kinematic state or a subordinate master's 4-D embedding.
- `type_bit = 0.0` marks a raw agent state slot; `type_bit = 1.0` marks a master embedding slot.

This identifier bit is what lets the same master model manage either agents or other masters without any architectural change. Empty slots are zero-padded.

**Master output:** a 4-D embedding vector, passed down to all subordinates.

**Agent observation (8-D):** 4-D local kinematic state + 4-D LM embedding received from the master above.

**Agent action:** binary — `{slow, fast}` — mapped to fixed throttle values.

## Topologies evaluated

**Parallel** — each intersection is an independent environment, managed by one Local Master. Agents never cross intersection boundaries. The hierarchy grows by adding more (LM, intersection) pairs.

![Parallel topology — 48 agents, 16 local masters](docs/figures/parallel_M16_N48.png)

**Chain** — intersections are physically connected in a road corridor. Agents route across multiple zones; the LM responsible for a zone receives and hands off agents dynamically as they enter or leave.

![Chain topology — 15 agents, 5 regional local masters](docs/figures/chain_int5_N15.png)

## Results

Evaluated on 30 scenarios per scale, two conditions:
- `normal` — full hierarchy active (GM → LMs → agents)
- `zero_master` — master signal zeroed out at every step; agents navigate on raw state alone

### Parallel scalability (crash rate per intersection, 30 scenarios)

| Agents | Local masters | normal | zero_master |
|---:|---:|---:|---:|
| 3 | 1 | 3% | 30% |
| 6 | 2 | 13% | 70% |
| 12 | 4 | 17% | 73% |
| 18 | 6 | 16% | 73% |
| 24 | 8 | 16% | 72% |
| 36 | 12 | 15% | 72% |
| 48 | 16 | 15% | 71% |

The crash rate gap stays at roughly 55–57 pp across all scales.

### Chain scalability (crash rate, 30 scenarios)

| Intersections | Agents | normal | zero_master |
|---:|---:|---:|---:|
| 2 | 6 | 7% | 80% |
| 3 | 9 | 20% | 67% |
| 4 | 12 | 33% | 97% |
| 5 | 15 | 30% | 77% |

Chain topology is harder — agents must cross zone boundaries, which creates multi-hop coordination conflicts. The master still provides a large benefit, though absolute crash rates are higher than parallel.

## Repository layout

```
.
├── highwayenv/                      # Custom gym environments
│   ├── intersection_class.py        # RELintersection-v0 (single 4-way crossing)
│   ├── double_intersection_class.py # RELdouble-intersection-v0
│   ├── roundabout_class.py          # RELroundabout-v0
│   ├── chain_intersection_class.py  # RELchain-intersection-v0 (connected corridor)
│   ├── CustomControlledVehicle.py
│   ├── custom_action.py
│   └── utils.py
│
├── src/
│   ├── model/
│   │   ├── master_model.py          # MasterModel: ResNet + SB3 PPO wrapper
│   │   ├── agent_handler.py         # Driver: per-vehicle gym.Env wrapper + SB3 PPO
│   │   └── model_handler.py         # save/load utilities
│   ├── training/
│   │   ├── training_handler.py      # training_loop(): outer cycle loop
│   │   ├── training_loop_utils.py   # PPO update, cycle prep, per-step helpers
│   │   ├── episode_utils.py         # process_episode(), master obs packing, identifier bit
│   │   ├── rollout_buffer_utils.py  # buffer reset/fill helpers
│   │   ├── episode_utils.py
│   │   └── general_utils.py         # model init, directory setup, logging
│   ├── experiment/
│   │   ├── experiment_config.py     # Experiment dataclass (all hyperparameters)
│   │   ├── scenarios.py             # Scenario pool generators
│   │   ├── scenarios_config.py      # Per-env scenario presets (exp7)
│   │   ├── scenario_geometry.py     # Intersection geometry helpers
│   │   └── experiment_config.py
│   ├── diagnostics/                 # Collision audit utilities
│   ├── plotting_utils/              # Training curve plotting
│   └── project_globals.py           # Global rollout buffers, step counter
│
├── scripts/
│   ├── training/
│   │   └── main_final.py            # Training entry point (W01 config, 1500 episodes)
│   ├── evaluation/
│   │   ├── run_full_evaluation.py   # Main evaluation script (parallel + chain)
│   │   ├── run_scalability_suite.py # Parallel scaling engine (imported by above)
│   │   ├── run_chain_scalability.py # Chain scaling engine (imported by above)
│   │   ├── run_proto_action_sweep.py# Core inference utilities (shared by all eval scripts)
│   │   ├── run_mixed_layer_experiment.py  # Mixed-layer topology test
│   │   └── run_dynamic_hierarchy.py  # Live architecture change (ramp agents up/down)
│   └── visualization/
│       └── visualize_hierarchy.py   # Generates the two figures in docs/figures/
│
├── tests/
│   └── test_scalability_regression.py  # Fast packing and layout invariant checks
│
├── models/
│   ├── agent/agent.pth              # Trained agent weights (checkpoint 6)
│   └── master/master.pth            # Trained master weights (checkpoint 6)
│
├── logger/                          # Optional Neptune experiment logger
├── docs/figures/                    # Hierarchy diagrams (parallel + chain)
├── setup.py
└── requirements.txt
```

## Training flow

```
main_final.py
  └─ training_loop()                [training_handler.py]
       │  Outer loop: episodes, grouped into cycles.
       │  prepare_models_for_cycle(): sets which heads train this cycle.
       │  FULL_JOINT_TRAINING=True → master + agents both update every cycle.
       │
       ├─ for each episode:
       │    process_episode()        [episode_utils.py]
       │      Runs one full environment episode step by step.
       │      Fills the master rollout buffer and each agent's rollout buffer.
       │
       │      for each env step:
       │        ┌─ MASTER PASS (both LMs and GM in one logical moment) ──────────┐
       │        │                                                                  │
       │        │  For each LM:                                                   │
       │        │    _master_obs_for_lm()  →  pack a 25-D observation:            │
       │        │      slot 0 : [prev_GM_embedding  | id=1]  ← parent feedback    │
       │        │      slot 1 : [agent_0_state (x,y,vx,vy) | id=0]               │
       │        │      slot 2 : [agent_1_state             | id=0]               │
       │        │      slot 3 : [agent_2_state             | id=0]               │
       │        │      slot 4 : [zeros (pad)               | id=0]               │
       │        │    master_model.predict(obs) → 4-D LM embedding                │
       │        │                                                                  │
       │        │  GM (same master_model, different obs):                         │
       │        │    _global_master_obs()  →  pack a 25-D observation:            │
       │        │      slot 0 : [zeros (no parent)  | id=1]  ← GM has no parent  │
       │        │      slot 1 : [LM1_embedding      | id=1]                      │
       │        │      slot 2 : [LM2_embedding      | id=1]                      │
       │        │      slot 3..4: [zeros (pad)       | id=1]                     │
       │        │    master_model.predict(obs) → 4-D GM embedding                │
       │        │      (stored in rollout buffer; broadcast to LMs next step)     │
       │        └──────────────────────────────────────────────────────────────────┘
       │
       │        ┌─ AGENT PASS ───────────────────────────────────────────────────┐
       │        │  For each agent:                                                 │
       │        │    obs = [own_state (4-D) | parent_LM_embedding (4-D)]  = 8-D   │
       │        │    agent_model.predict(obs) → action ∈ {0=slow, 1=fast}         │
       │        └──────────────────────────────────────────────────────────────────┘
       │
       │        env.step(actions) → next states, rewards, done
       │        store transition in rollout buffers
       │
       ├─ every episode (ep_for_train=1):
       │    perform_training_phase()  [training_loop_utils.py]
       │      PPO update on master rollout buffer  (clipped policy + value + entropy loss)
       │      PPO update on agent rollout buffers  (same loss, separate weights)
       │      Both updates use the same SB3 PPO internals — 5 epochs over the buffer.
       │
       └─ save_models()              [model_handler.py]
```

### Identifier bit

Each master observation is 5 slots of 5 values each (25-D total). A slot always has the same shape regardless of what it contains:

```
slot = [v0, v1, v2, v3, id]
```

`v0..v3` is either a 4-D agent kinematic state (x, y, vx, vy, normalized) or a 4-D master embedding. `id` is a single float:

- `id = 0.0` — this slot holds a raw agent state
- `id = 1.0` — this slot holds a master embedding (from a sub-master)

The bit is appended literally to the input vector. The ResNet sees it as one more feature. Because the same weights handle `id=0` and `id=1` slots, a master can manage raw agents, other masters, or a mix of both — without any architectural change. This is also what makes the mixed-layer experiment (`GM → [LM, a3, a4, a5]`) work out of the box: slot 0 gets `id=1` (an LM embedding) and slots 1-3 get `id=0` (raw agent states), and the network was never told they can't coexist.

## Hyperparameters

These are the values used in the training run that produced the saved checkpoint (`experiment_runs/full_26_04_2026-11_40_39`, config `A_base/W_MASTER`, seed 123).

| Parameter | Value | Notes |
|---|---|---|
| Total episodes | 2500 | 5 seeds × 2500 per run |
| Episodes per PPO update | 1 | update after every episode |
| PPO epochs per update | 5 | |
| Agent learning rate | 3e-3 | |
| Master learning rate | 3e-4 | |
| Clip range | 0.2 | |
| Entropy coefficient | 0.005 | |
| Discount (γ) | 0.90 | |
| GAE λ | 0.90 | |
| Value coefficient | 1.0 | |
| Rollout buffer size | 384 | `N_STEPS` |
| Warmup episodes | 200 | random actions for first 200 episodes |
| Peak-lock threshold | 75% | entropy set to 0 once rolling-20 arrival ≥ 75% |
| Agent actions | {5, 10} m/s | slow=5, fast=10 |
| Agent network arch | wide | MLP `[256, 256]` |
| Master network arch | ResNet | 25→128, 2× residual block, 128→128→4 |
| Master obs dim | 25 | 5 slots × (4-D + 1 identifier bit) |
| Master embedding dim | 4 | |
| Agent obs dim | 8 | 4-D kinematic state + 4-D LM embedding |
| Collision reward | -50 | terminal |
| Arrival reward | +50 | terminal |
| High-speed reward | +5/step | per step agent is above speed threshold |
| Starvation reward | 0 | disabled |
| Reward mode | global | one shared reward signal per episode |

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

Requires Python 3.10+.

## Training

> **Note:** `models/agent/agent.pth` and `models/master/master.pth` were produced by a multi-seed run (`experiment_runs/full_26_04_2026-11_40_39`) using the hyperparameters in the table above. The best seed (s123) reached 98.7% arrival on the last 50 episodes.
>
> `scripts/training/main_final.py` is a single-seed reproduction script with the same base config. It will produce a comparable model but with slightly different settings (`ent_coef=0.05`, `ep_for_train=3`). Use it as a starting point if you need to retrain.

```bash
python scripts/training/main_final.py
```

Checkpoints are saved under `experiment_runs/final_<timestamp>/trained_model/`.

## Evaluation

```bash
python scripts/evaluation/run_full_evaluation.py
```

Runs both parallel and chain scalability and writes results to `EVALUATION_RESULTS/<timestamp>/`. Use `--smoke` for a quick end-to-end sanity check. Use `--agent` and `--master` to point at a different checkpoint pair.

Mixed-layer topology experiment (a master managing both sub-masters and raw agents on the same level):

```bash
python scripts/evaluation/run_mixed_layer_experiment.py
```

Dynamic hierarchy (agent count grows and shrinks in one intersection, the master tree is rebuilt live):

```bash
python scripts/evaluation/run_dynamic_hierarchy.py --max-agents 12 --scenarios 40
```

The agent count ramps up from 1 to the maximum and back down to 1. A local master supervises up to 4 agents; once the count crosses 4, 8, 12, ... a new local master is added and a global master coordinates them. The script reuses the same shared weights at every size and reports arrival and crash rate for both the up ramp and the down ramp, so any degradation introduced by adding or removing a master would show up as a gap between the two ramps at the same agent count.

Regenerate the hierarchy diagrams:

```bash
python scripts/visualization/visualize_hierarchy.py
```

## Tests

```bash
python -m pytest tests/
```
