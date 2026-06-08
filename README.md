# Hierarchical Multi-Agent RL for Coordinated Driving

A 3-level hierarchy (Global Master → Local Masters → Agents) trained with shared-weight PPO on custom `highway-env` layouts. The main research claim: the same trained model, without any retraining, coordinates environments with 3 to 48 agents across both parallel and chain topologies.

## Architecture

The hierarchy has three roles:

- **Global Master (GM):** receives embeddings from all Local Masters and emits a global coordination signal.
- **Local Master (LM):** manages up to 5 agents in one intersection zone, aggregates their states, and emits a local embedding upward.
- **Agent:** drives one vehicle; its observation is its local state plus the LM embedding.

All three roles share the same PPO weights. Roles differ only in how inputs are packed, not in parameters. This is what allows the hierarchy to generalize to any number of agents without retraining.

For large scales (more than 5 Local Masters), additional intermediate masters group LMs in sets of 5 recursively, forming a tree of depth log₅(N\_LMs).

## Topologies

**Parallel** — independent intersections, each managed by one LM. Agents do not cross intersection boundaries.

![Parallel topology, 48 agents, 16 local masters](docs/figures/parallel_M16_N48.png)

**Chain** — intersections physically connected in a corridor. Agents route across multiple zones; the LM responsible for a zone hands off agents dynamically as they enter or leave.

![Chain topology, 15 agents, 5 regional local masters](docs/figures/chain_int5_N15.png)

## Results

Evaluated on 100 scenarios per scale. Two conditions:

- `normal`: full hierarchy active (GM → LMs → agents)
- `zero_master`: master signal zeroed out; agents drive on raw state alone

| Scale | Normal crash rate | Zero-master crash rate |
|---|---|---|
| 2 LMs, 6 agents | ~15% | ~75% |
| 4 LMs, 12 agents | ~20% | ~75% |
| 8 LMs, 24 agents | ~22% | ~76% |
| 16 LMs, 48 agents | ~26% | ~77% |

The crash-rate gap stays consistent as the number of agents grows — the hierarchy remains effective without any additional training.

## Repository layout

```
.
├── highwayenv/                 # Custom gym environments
│   ├── intersection_class.py
│   ├── double_intersection_class.py
│   ├── roundabout_class.py
│   └── chain_intersection_class.py
│
├── src/
│   ├── model/                  # MasterModel, AgentHandler, ModelHandler (PPO)
│   ├── training/               # PPO update loop, rollout buffer, episode utils
│   └── experiment/             # Scenario pools, geometry, environment configs
│
├── scripts/
│   ├── training/
│   │   ├── main_final.py       # Main training entry point (W01, 1500 episodes)
│   │   └── train_chain.py      # Fine-tune on connected chain topology
│   ├── evaluation/
│   │   ├── run_full_evaluation.py    # Run the full parallel + chain study
│   │   ├── run_scalability_suite.py  # Parallel scaling engine (imported by above)
│   │   ├── run_chain_scalability.py  # Chain scaling engine (imported by above)
│   │   ├── run_proto_action_sweep.py # Core agent/master inference utilities
│   │   └── run_mixed_layer_experiment.py  # Mixed-layer topology experiment
│   └── visualization/
│       └── visualize_hierarchy.py    # Generate hierarchy + layout diagrams
│
├── tests/
│   └── test_scalability_regression.py  # Fast packing and layout regression checks
│
├── models/
│   ├── agent/agent.pth         # Trained agent weights (checkpoint 6)
│   └── master/master.pth       # Trained master weights (checkpoint 6)
│
├── logger/                     # Optional Neptune experiment logger
├── docs/figures/               # Hierarchy diagrams (parallel + chain)
├── setup.py
└── requirements.txt
```

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

Requires Python 3.10+ and a highway-env installation that includes the custom REL environments.

## Training

Run from the repo root:

```bash
python scripts/training/main_final.py
```

This trains the W01 configuration: 1500 episodes, joint master+agent PPO updates, `agent_lr=3e-3`, `master_lr=3e-4`. Checkpoints are saved under `experiment_runs/`.

To fine-tune on the chain topology (optional):

```bash
python scripts/training/train_chain.py
```

## Evaluation

The main evaluation script runs both parallel and chain scalability and writes results to `EVALUATION_RESULTS/<timestamp>/`:

```bash
python scripts/evaluation/run_full_evaluation.py
```

Options:

```
--smoke           quick sanity run (few scenarios per scale)
--agent PATH      override agent checkpoint (default: models/agent/agent.pth)
--master PATH     override master checkpoint (default: models/master/master.pth)
```

To run the mixed-layer topology experiment separately:

```bash
python scripts/evaluation/run_mixed_layer_experiment.py
```

To regenerate the hierarchy diagrams:

```bash
python scripts/visualization/visualize_hierarchy.py
```

## Tests

```bash
python -m pytest tests/
```

The regression test checks scenario packing and layout invariants without stepping the environment.
