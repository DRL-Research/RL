# HD-MAPS Table 6.4

Reproduce **simultaneous arrival**, **staggered arrival**, and **online reconfiguration** (hierarchy `LM / IM / GM`) from Chapter 6.2. This repository does not include comparisons to IPPO, CoPO, or MAPPO.

## Install

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

On Unix, use `source .venv/bin/activate`.

## Reproduce the published table

Rebuild Table 6.4 from frozen per-episode records (no simulator):

```bash
python -m hdmaps reproduce
python -m hdmaps reproduce --smoke
```

Simultaneous crash and arrival are computed from the saved 200 episodes per scale (seeds 2026 and 4242). The hierarchy column is `LM / IM / GM` with fan-out 4. Staggered crash columns are the published reference: no matching frozen episode suite was found for that protocol.

## Re-run the simulator

```bash
python -m hdmaps evaluate --protocol simultaneous --smoke
python -m hdmaps evaluate --protocol staggered --smoke
python -m hdmaps evaluate --protocol both
```

Simultaneous evaluation uses the frozen worker `agent.pth` and the stage-2 regional master that produced Table 6.4. Staggered evaluation attaches vehicles to a Master only inside an 80 m coordination radius, spawning and retiring Masters as zones fill and empty.

## Train

Locked protocol: 11,500 episodes, at most 6 agents, 2 intersections. Validation uses seed 7 and is disjoint from evaluation seeds 2026 and 4242.

```bash
python -m hdmaps train --smoke
python -m hdmaps train
```

New training is statistical. It is not expected to reproduce the published weights bit-for-bit.

## Tests

```bash
pip install pytest
pytest -q
```
