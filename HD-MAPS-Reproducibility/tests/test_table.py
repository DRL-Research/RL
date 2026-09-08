from __future__ import annotations

from hdmaps.hierarchy.tree import hierarchy_label
from hdmaps.protocol import PUBLISHED_TABLE, SCALE_CONFIGS
from hdmaps.statistics.metrics import round_wilson, wilson_interval
from hdmaps.statistics.tables import simultaneous_table


def test_hierarchy_matches_table():
    for (n_int, n_agents), row in zip(SCALE_CONFIGS, PUBLISHED_TABLE):
        assert n_agents == row.fleet
        assert hierarchy_label(n_int) == row.hierarchy


def test_simultaneous_crash_and_arrival_match_published():
    computed = simultaneous_table()
    for got, exp in zip(computed, PUBLISHED_TABLE):
        assert got["fleet"] == exp.fleet
        assert round(got["crash_pct"], 1) == exp.sim_crash_pct
        assert got["crash_lo"] == exp.sim_crash_lo
        assert got["crash_hi"] == exp.sim_crash_hi
        assert round(got["arrival_pct"], 1) == exp.sim_arrival_pct
        assert round(got["arrival_sem"], 1) == exp.sim_arrival_sem


def test_wilson_zero_of_200():
    lo, hi = round_wilson(*wilson_interval(0, 200))
    assert lo == 0.0
    assert hi == 1.9


def test_wilson_48_agents():
    lo, hi = round_wilson(*wilson_interval(33, 200))
    assert lo == 12.0
    assert hi == 22.3
