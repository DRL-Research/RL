from __future__ import annotations

import pytest

from hdmaps.reproduce import reproduce


def test_reproduce_cli_table(tmp_path):
    payload = reproduce(smoke=False, out_dir=tmp_path)
    assert "16 / 4 / 1" in payload["table"]
    assert "85.5" in payload["table"]
    assert "Stag crash" in payload["table"]
    assert (tmp_path / "simultaneous_curves.png").is_file()
    assert (tmp_path / "table_6_4.txt").is_file()


@pytest.mark.slow
def test_smoke_evaluate():
    from hdmaps.evaluation.runner import evaluate_simultaneous

    rows = evaluate_simultaneous(smoke=True)
    assert rows[0]["fleet"] == 3
    assert "arrival_pct" in rows[0]
