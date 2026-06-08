# Hierarchical Multi-Agent RL for Coordinated Driving

A **3-level hierarchical multi-agent reinforcement-learning** system that coordinates
many self-driving cars through custom [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv)
layouts (intersection, roundabout, double-intersection, and a connected *chain* of
intersections). A single shared **master** network produces low-dimensional *proto*
embeddings that steer a single shared **agent** policy, so the whole hierarchy reuses
just two sets of weights and **scales to more agents and more masters without retraining**.

```
                 ┌─────────────────────┐
                 │   Global Master(s)   │      coordinate the local masters
                 └──────────┬──────────┘
            ┌───────────────┼───────────────┐
     ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
     │ Local Master│  │Local Master│  │Local Master │   one per intersection / group
     └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
        ┌───┼───┐       ┌───┼───┐        ┌───┼───┐
        a   a   a       a   a   a        a   a   a       3 agents (cars) per local master
```

## The identifier bit (key design idea)

Every master reads a fixed **25-D observation = 5 slots × (4-D vector + 1 identifier
bit)**. The identifier bit is what makes the hierarchy uniform and scalable:

| identifier | slot holds | example |
|:---:|---|---|
| `1.0` | a **(sub-)master embedding** | the global feedback in a local-master slot 0; a local-master embedding in a global-master slot |
| `0.0` | a **raw agent state** `(x, y, vx, vy)` | a car supervised directly by this master |

Because the network reads the same vector regardless of whether a child is a master
or a vehicle, **masters and vehicles can be freely mixed on the same layer** — see the
[mixed-layer experiment](#3-mixed-layer-experiment-master--vehicles-on-one-layer).

---

## Repository layout

```
.
├── src/                          # core library (organized by responsibility)
│   ├── model/                    # PPO / RL models
│   │   ├── master_model.py       #   shared MasterModel (SB3 PPO + ResNet) -> proto embedding
│   │   ├── model_handler.py      #   agent PPO wrapper + checkpoint load/save
│   │   └── agent_handler.py      #   Driver gym wrapper: obs = [state ++ master embedding]
│   ├── training/                 # the training pipeline
│   │   ├── training_handler.py   #   training_loop(): rollouts + PPO phases
│   │   ├── episode_utils.py      #   per-step hierarchical rollout + master input packing
│   │   ├── training_loop_utils.py#   manual clipped-PPO updates from rollout buffers
│   │   ├── general_utils.py      #   model/dir/logger initialization
│   │   └── rollout_buffer_utils.py
│   ├── experiment/               # infrastructure: configs + scenario pools
│   │   ├── experiment_config.py  #   Experiment dataclass (all hyperparameters + dims)
│   │   ├── scenarios_config.py   #   env config builders
│   │   └── scenarios.py          #   curated scenario pools / held-out sets
│   ├── diagnostics/              # collision auditing / analysis
│   ├── plotting_utils/           # training-curve plots
│   └── project_globals.py        # shared rollout-buffer state
│
├── highwayenv/                   # custom multi-agent environments (REL* gym ids)
│   ├── intersection_class.py     #   RELintersection-v0      (4-way junction)
│   ├── roundabout_class.py       #   RELroundabout-v0
│   ├── double_intersection_class.py  # RELdouble-intersection-v0
│   ├── chain_intersection_class.py   # RELchain-intersection-v0 (connected corridor)
│   ├── CustomControlledVehicle.py
│   ├── custom_action.py
│   └── utils.py                  #   gym registration of the REL* ids
│
├── logger/                       # optional Neptune logging
├── models_to_check/              # pretrained checkpoints (agent/ + master/)
│
├── scripts/
│   ├── training/                 # ── TRAINING ENTRY POINTS ──
│   │   ├── main_final.py             # production hierarchical training (single intersection)
│   │   ├── train_chain.py            # fine-tune on the connected chain (local-frame normalised)
│   │   └── train_unified_full_scenarios.py
│   ├── evaluation/               # ── EVALUATION / SCALABILITY ──
│   │   ├── run_proto_action_sweep.py     # shared engine: models, packing, episode loop
│   │   ├── run_scalability_suite.py      # PARALLEL scalability (M×K masters/agents)
│   │   ├── run_chain_scalability.py      # CONNECTED-CHAIN scalability (regional masters)
│   │   ├── run_mixed_layer_experiment.py # master + vehicles on the SAME layer
│   │   ├── run_models_evaluation_suite.py
│   │   └── run_crossing_coordination_report.py
│   └── visualization/            # ── FIGURES ──
│       ├── visualize_hierarchy.py        # full "spec" picture (intersections+agents+masters)
│       └── plot_readable_scenarios.py    # per-scenario route diagrams on real road geometry
│
└── tests/
    └── test_scalability_regression.py    # master-input packing + layout backward-compat
```

> Every script in `scripts/**` puts the repo root and all `scripts/` sub-folders on
> `sys.path` automatically, so they can be run directly from anywhere.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pip install -e .                  # makes `src` and `highwayenv` importable
```

Python 3.10+ is required.

---

## Reproducing the experiments

### 1. Parallel scalability (disconnected intersections)
One global master coordinates `M` local masters spread across independent crossings,
each local master owning `K=3` cars. Re-runs the coordination ablations
(`normal / zero_master / const_all_masters / swap_local_masters / zero_global_master`)
at every scale.

```bash
python scripts/evaluation/run_scalability_suite.py
```

### 2. Connected-chain scalability
A single connected corridor of `N` intersections; each intersection is one **regional
local master**, and cars hand off to the next zone's master as they cross. Scales as
`N = 1, 2, 5, 15` intersections → `3, 6, 15, 45` agents. Saves per-scenario layout
plots and JSON for 100 jittered scenarios per scale.

```bash
python scripts/evaluation/run_chain_scalability.py
```

### 3. Mixed-layer experiment (master + vehicles on one layer)
Demonstrates the heterogeneous hierarchy `GM → [ LM → (a0,a1,a2) , a3 , a4 , a5 ]` on
the dense 6-car crossing, where the top master supervises **a sub-master and three raw
cars side-by-side**, separated only by the identifier bit. Compares
`normal / zero_master / zero_top`.

```bash
python scripts/evaluation/run_mixed_layer_experiment.py --scenarios 60
```

### 4. Fine-tuning a more robust chain model
Fine-tunes the pretrained checkpoint on the chain topology, using the **same
local-frame normalization and scenario distribution the evaluation uses** (so the model
is in-distribution at eval time). Saves a rolling-best checkpoint under
`experiment_runs/`.

```bash
python scripts/training/train_chain.py
```

### 5. Full-picture "spec" visualizations
One figure per topology showing the **physical** layout: real road geometry, every
controlled car (position + heading arrow), the local-master groups, and the global
master on top.

```bash
python scripts/visualization/visualize_hierarchy.py --m 6 --k 3 --n-int 5
# -> MODELS_EVALUATION/hierarchy_spec/spec_parallel.png
# -> MODELS_EVALUATION/hierarchy_spec/spec_chain.png
```

---

### Example figures

The full-picture spec images and the mixed-layer result are shipped under
[`docs/figures/`](docs/figures):

| | |
|---|---|
| `spec_parallel.png` | 3 intersections, 6 local masters + 1 global master, 18 agents |
| `spec_chain.png` | 5 connected intersections as 5 regional masters + 1 global master, 15 agents |
| `mixed_layer_results.png` | `GM → [LM → (a0,a1), a2, a3]` ablation: master+vehicles on one layer |

Mixed-layer result on the dense 6-car crossing (50 randomized scenarios),
hierarchy `GM → [ LM → (a0,a1,a2), a3, a4, a5 ]`:

| Condition | Arrival | Crash |
|---|---:|---:|
| `normal` (mixed hierarchy active) | **82.0 %** | **22 %** |
| `zero_top` (mixed master zeroed) | 79.7 % | 30 % |
| `zero_master` (all signals zeroed) | 60.7 % | 68 % |

Zeroing all coordination costs ~21 arrival points and triples the crash rate, so the
heterogeneous layer (a sub-master and raw vehicles side-by-side) is doing real
coordination — not benefiting from an easy scene.

---

## Tests

```bash
python -m pytest tests/ -q
# or:
python tests/test_scalability_regression.py
```

These check that the generalized master-input packing is **backward-compatible** with
the pretrained 2-local-master layout and that larger layouts pack to valid shapes.

---

## Checkpoints

`models_to_check/agent/ckpt_agent6.pth` and `models_to_check/master/ckpt_master6.pth`
are the primary pretrained pair used by the evaluation and visualization scripts.
Training runs write new checkpoints to `experiment_runs/<run>/...` (git-ignored).
