"""
Regression + scalability tests for the variable-master / variable-agent hierarchy.

Fast (no environment stepping): checks that
  1. the generalised global-master packing reproduces the original 2-LM packing
     byte-for-byte (existing 6-agent baseline is unchanged), and
  2. the packing scales to N>2 local masters (more masters) and that the
     scalability harness builds valid M×K layouts (more agents).

Run:
  & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" test_scalability_regression.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the repo root (for ``src`` / ``highwayenv``) and the evaluation scripts
# (for ``run_scalability_suite``) importable when run from tests/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "scripts", "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_global_master_packing_backward_compatible() -> None:
    from src.experiment.experiment_config import Experiment
    from src.training.episode_utils import _build_global_master_input, _master_slot

    exp = Experiment(RENDER_MODE=None)
    e1 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    e2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # Original 2-LM positional call (must be unchanged).
    got = _build_global_master_input(e1, e2, exp)

    svd = max(4, int(exp.EMBEDDING_SIZE))
    slots = [_master_slot(e1, 1.0, svd), _master_slot(e2, 1.0, svd)]
    while len(slots) < int(exp.NUM_MASTER_SLOTS):
        slots.append(np.zeros(svd + 1, dtype=np.float32))
    ref = np.concatenate(slots[: int(exp.NUM_MASTER_SLOTS)]).astype(np.float32)

    assert got.shape == ref.shape == (int(exp.MASTER_OBS_DIM),), (got.shape, ref.shape)
    assert np.allclose(got, ref), "2-LM GM packing changed — baseline regression!"
    print("[ok] global-master packing backward compatible (2 LMs, 25-D)")


def test_global_master_packing_scales_to_many_masters() -> None:
    from src.experiment.experiment_config import Experiment
    from src.training.episode_utils import _build_global_master_input

    exp = Experiment(RENDER_MODE=None)
    embs = [np.full(4, float(i + 1), dtype=np.float32) for i in range(5)]

    # List form: N local masters (here 5, filling all slots).
    got = _build_global_master_input(embs, experiment=exp)
    assert got.shape == (int(exp.MASTER_OBS_DIM),), got.shape

    # First slot must hold the first LM embedding with identifier 1.0.
    assert np.allclose(got[0:4], embs[0]) and got[4] == 1.0
    # Fifth slot holds the fifth LM embedding.
    assert np.allclose(got[20:24], embs[4]) and got[24] == 1.0
    print("[ok] global-master packing scales to 5 local masters")

    # Three-LM call also packs cleanly (extra slots zero-padded).
    got3 = _build_global_master_input(embs[:3], experiment=exp)
    assert np.allclose(got3[10:14], embs[2]) and got3[14] == 1.0
    assert np.allclose(got3[15:25], 0.0), "slots beyond used LMs should be zero"
    print("[ok] global-master packing handles 3 local masters with zero padding")


def test_scalability_harness_layout_construction() -> None:
    import run_scalability_suite as S

    # Curated pool is non-empty and scenario sizing yields exactly N agents.
    assert len(S.ACTIVE_REGULAR_POOL) > 0
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 6, 8):
        sc = S._make_intersection_scenario(n, rng)
        assert len(sc["agents"]) == n, (n, len(sc["agents"]))

    # plan_cells maps M local masters (K cars each) onto physical intersections,
    # and the total controlled-car count is conserved across the layout.
    for m, k in [(1, 3), (2, 3), (4, 3), (5, 5), (15, 3)]:
        cells = S.plan_cells(m, k)
        assert sum(cells) == m * k, (m, k, cells)
        assert all(c % k == 0 for c in cells), (m, k, cells)
    print("[ok] scenario sizing (N agents) + plan_cells layouts valid")


def main() -> None:
    failed = 0
    for fn in (
        test_global_master_packing_backward_compatible,
        test_global_master_packing_scales_to_many_masters,
        test_scalability_harness_layout_construction,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print("\nAll scalability regression tests passed.")


if __name__ == "__main__":
    main()
