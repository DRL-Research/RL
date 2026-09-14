from __future__ import annotations

import json
from pathlib import Path

from hdmaps.paths import repo_root
from hdmaps.protocol import PUBLISHED_TABLE
from hdmaps.statistics.figures import plot_simultaneous
from hdmaps.statistics.tables import format_table, published_staggered_table, simultaneous_table


def reproduce(*, smoke: bool = False, out_dir: Path | None = None) -> dict:
    sim_rows = simultaneous_table()
    stag_rows = published_staggered_table()
    if smoke:
        sim_rows = sim_rows[:1]
        stag_rows = stag_rows[:1]
    payload = {
        "simultaneous": sim_rows,
        "published": [
            {
                "fleet": r.fleet,
                "hierarchy": r.hierarchy,
                "sim_crash": r.sim_crash_pct,
                "sim_arrival": r.sim_arrival_pct,
                "stag_crash": r.stag_crash_pct,
            }
            for r in PUBLISHED_TABLE
        ],
        "note": "Simultaneous crash/arrival are rebuilt from frozen suite files. Staggered crash in this table is the published reference; live staggered is python -m hdmaps evaluate --protocol staggered.",
        "table": format_table(sim_rows, stag_rows),
    }
    dest = out_dir or (repo_root() / "outputs" / "reproduce")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "table_6_4.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "table"}, indent=2, default=float),
        encoding="utf-8",
    )
    (dest / "table_6_4.txt").write_text(payload["table"] + "\n", encoding="utf-8")
    plot_simultaneous(sim_rows, dest / "simultaneous_curves.png")
    return payload
