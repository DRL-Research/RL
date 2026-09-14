# HD-MAPS Table 6.4

Chapter 6.2 only: simultaneous arrival, staggered arrival, and hierarchy LM / IM / GM.

Training uses at most two intersections and six agents. Evaluation fleets 12, 18, 24, 36, and 48 are zero-shot.

## Install

Python 3.11.

```
cd HD-MAPS-Reproducibility
pip install -r requirements.txt
pip install -e .
```

## Frozen published simultaneous table

Rebuilds the published simultaneous columns from `artifacts/published/v1/results/simultaneous/`. Staggered values in that printout are the published reference.

```
python -m hdmaps reproduce
```

## Live evaluation of the published Master

```
python -m hdmaps evaluate --protocol both
```

Uses `artifacts/published/v1/checkpoints/agent.pth` and `master_stage2_regional.pth`.

## Retrain (thesis story) then evaluate

The Worker stays frozen. Only the Master is trained.

1. Stage 1: 5000 episodes, 1 intersection, 1-4 agents, seed 42.
2. Stage 2: 2500 episodes, 2 intersections, 4-6 agents, seed 142.
3. Evaluation: 7 fleets, 200 simultaneous episodes, 100 staggered episodes.

```
python -m hdmaps run
```

Writes `outputs/run/`. Checkpoints every 100 episodes.

A recorded run of this protocol is in `artifacts/retrain/v1/`. Load those weights with:

```
python -m hdmaps evaluate --protocol both --init retrain
```

## Recorded retrain vs Table 6.4

Simultaneous n=200. Staggered n=100. Full text: `artifacts/retrain/v1/eval/table.txt`.

### Simultaneous

| Fleet | Hierarchy | Paper crash | Retrain crash | Paper arrival | Retrain arrival |
| ---: | --- | --- | --- | --- | --- |
| 3 | 1 / 0 / 0 | 0.0 [0.0, 1.9] | 7.0 [4.2, 11.4] | 100.0 +/- 0.0 | 93.0 +/- 1.8 |
| 6 | 2 / 0 / 1 | 4.0 [2.0, 7.7] | 0.5 [0.1, 2.8] | 96.9 +/- 1.1 | 99.7 +/- 0.3 |
| 12 | 4 / 0 / 1 | 6.5 [3.8, 10.8] | 0.0 [0.0, 1.9] | 95.0 +/- 1.5 | 100.0 +/- 0.0 |
| 18 | 6 / 2 / 1 | 9.5 [6.2, 14.4] | 0.5 [0.1, 2.8] | 91.9 +/- 1.9 | 99.8 +/- 0.2 |
| 24 | 8 / 2 / 1 | 12.0 [8.2, 17.2] | 0.5 [0.1, 2.8] | 90.0 +/- 2.0 | 99.8 +/- 0.2 |
| 36 | 12 / 3 / 1 | 10.5 [7.0, 15.5] | 0.5 [0.1, 2.8] | 91.0 +/- 2.0 | 99.7 +/- 0.3 |
| 48 | 16 / 4 / 1 | 16.5 [12.0, 22.3] | 0.0 [0.0, 1.9] | 85.5 +/- 2.4 | 100.0 +/- 0.0 |

### Staggered crash

| Fleet | Hierarchy | Paper | Retrain |
| ---: | --- | --- | --- |
| 3 | 1 / 0 / 0 | 2 [0.6, 7.0] | 0.0 [0.0, 3.7] |
| 6 | 2 / 0 / 1 | 2 [0.6, 7.0] | 0.0 [0.0, 3.7] |
| 12 | 4 / 0 / 1 | 3 [1.0, 8.5] | 4.0 [1.6, 9.8] |
| 18 | 6 / 2 / 1 | 7 [3.4, 13.7] | 9.0 [4.8, 16.2] |
| 24 | 8 / 2 / 1 | 3 [1.0, 8.5] | 2.0 [0.6, 7.0] |
| 36 | 12 / 3 / 1 | 2 [0.6, 7.0] | 8.0 [4.1, 15.0] |
| 48 | 16 / 4 / 1 | 7 [3.4, 13.7] | 13.0 [7.8, 21.0] |

A new `python -m hdmaps run` follows the same protocol. It is a new training seed path, not a replay of `artifacts/retrain/v1`.
