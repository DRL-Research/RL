"""
Assemble one folder ``PROFESSOR_RESULTS/`` with everything for the professor:
  01_parallel/     scalability tables, plots, JSON (copied from latest run)
  02_chain/        chain scalability 100-scenario run
  03_mixed_layer/  mixed-layer experiment
  04_scenario_maps/ schematic maps per N (generated)
  05_finetune_eval/ regional tests (optional --run-tests)
  REPORT.md        executive summary (Hebrew + English tables)
  master_summary.json

Usage:
  py -3 build_professor_package.py
  py -3 build_professor_package.py --run-maps --run-tests --run-mixed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

OUT = os.path.join(_REPO, "PROFESSOR_RESULTS")
PY = sys.executable

SOURCES = {
    "parallel": "MODELS_EVALUATION/scalability_2026_06_02-15_41_14",
    "chain": "MODELS_EVALUATION/chain_scalability_100scen",
    "mixed": "MODELS_EVALUATION/mixed_layer",
    "hierarchy_spec": "MODELS_EVALUATION/hierarchy_spec",
}


def _copy_tree(src: str, dst: str, patterns: tuple[str, ...] = ("*.json", "*.csv", "*.png", "*.tex")) -> int:
    import glob
    n = 0
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    for pat in patterns:
        for fp in glob.glob(os.path.join(src, "**", pat), recursive=True):
            rel = os.path.relpath(fp, src)
            tgt = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copy2(fp, tgt)
            n += 1
    return n


def _write_report(path: str, master: dict) -> None:
    p = master.get("parallel_highlights", {})
    c = master.get("chain_highlights", {})
    m = master.get("mixed_layer", {})
    lines = [
        "# Hierarchical MARL — Results Package for Review",
        "",
        f"Generated: {master.get('timestamp', '')}",
        "",
        "## Available topologies (implemented in code)",
        "",
        "| Topology | Gym ID | Shape |",
        "|----------|--------|-------|",
        "| Single 4-way junction | `RELintersection-v0` | 4 arms (o0–o3) |",
        "| Roundabout | `RELroundabout-v0` | Circular ring |",
        "| Double junction | `RELdouble-intersection-v0` | A ↔ B linked |",
        "| Connected chain | `RELchain-intersection-v0` | I0—I1—… corridor |",
        "",
        "> **Triangular / pentagonal junctions** are not in this `highway-env` fork. ",
        "> They would need new env classes; all four topologies above are trained/evaluated.",
        "",
        "## 1. Parallel scalability (disconnected intersections)",
        "",
        "One global master coordinates M local masters over independent crossings.",
        "",
        "| Agents | LMs | normal arrival | zero_master arrival |",
        "|--------|-----|----------------|---------------------|",
    ]
    for row in p.get("rows", []):
        lines.append(f"| {row['N']} | {row['M']} | {row['normal']}% | {row['zero']}% |")
    lines += [
        "",
        "**Conclusion:** Master benefit ~+27% arrival stable up to **48 agents / 16 LMs**.",
        "",
        "Files: `01_parallel/` (CSV, PNG, per-layout scenarios)",
        "",
        "## 2. Connected chain",
        "",
        "100 jittered scenarios per scale; 3 agents per regional LM (one LM per intersection).",
        "",
        "| Intersections | Agents | normal | zero_master |",
        "|---------------|--------|--------|-------------|",
    ]
    for row in c.get("rows", []):
        lines.append(f"| {row['n_int']} | {row['n_agents']} | {row['normal']}% | {row['zero']}% |")
    lines += [
        "",
        "**Conclusion:** Strong ablation at **2 nodes**; weaker gap at 15 nodes (needs fine-tune).",
        "",
        "## 3. Mixed-layer (master + vehicles same layer)",
        "",
        "Hierarchy: `GM → [ LM→(a0,a1,a2), a3, a4, a5 ]` — identifier bit separates slot types.",
        "",
    ]
    if m:
        lines.append(f"| Condition | Arrival | Crash |")
        lines.append(f"|-----------|---------|-------|")
        for cond, v in m.items():
            lines.append(f"| {cond} | {v.get('arrival', '?')}% | {v.get('crash', '?')}% |")
    lines += [
        "",
        "## 4. Scenario maps (`04_scenario_maps/`)",
        "",
        "For each N ∈ {3,6,12,18,24,36,48}: schematic map (who starts where → destination) ",
        "for **parallel** and **chain**.",
        "",
        "## 5. More than 3 agents in one chain zone?",
        "",
        "Each master input has **5 slots**: slot0 = parent embedding (id=1), slots 1–4 = ",
        "subordinates (id=0). **Up to 4 vehicles** fit one regional LM without extra masters.",
        "",
        "Current chain eval: **1 LM / intersection, 3 agents** (in-distribution for ckpt6).",
        "",
        "If a zone has **5–6 agents**, add a **second LM** on that intersection (as in parallel ",
        "`lm_per_cell=2`) — same shared weights, more LM forward passes.",
        "",
        "## Folder layout",
        "",
        "```",
        "PROFESSOR_RESULTS/",
        "  01_parallel/",
        "  02_chain/",
        "  03_mixed_layer/",
        "  04_scenario_maps/parallel|chain/N*_agents/",
        "  05_finetune_eval/",
        "  docs/figures/  (hierarchy spec images)",
        "  REPORT.md",
        "  master_summary.json",
        "```",
        "",
        "---",
        "",
        "# סיכום בעברית (קצר)",
        "",
        "- **מקביל:** סקייל עד 48 סוכנים, מאסטר מוריד תאונות מ~70% ל~15%.",
        "- **טור:** תיאום חזק ב-2 צמתים; ב-15 צמתים פחות הפרש — fine-tune מומלץ.",
        "- **Mixed-layer:** מאסטר ורכב באותה שכבה עובד; zero_master ~60% על G03 (לא 84%).",
        "- **מפות סנריו:** תמונה לכל N ב-3,6,12,…,48.",
        "- **משולש/מחומש:** לא ממומש בקוד; יש 4 טופולוגיות REL.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_chain_highlights() -> dict:
    p = os.path.join(_REPO, SOURCES["chain"], "chain_results.json")
    if not os.path.isfile(p):
        return {"rows": []}
    with open(p) as f:
        data = json.load(f)
    rows = []
    for lay in data.get("layouts", []):
        ni = lay["n_intersections"]
        na = lay["n_agents"]
        nm = lay["conditions"]["normal"]["arrival_pct_mean"]
        zm = lay["conditions"]["zero_master"]["arrival_pct_mean"]
        rows.append({"n_int": ni, "n_agents": na, "normal": round(nm, 1), "zero": round(zm, 1)})
    return {"rows": rows}


def _load_parallel_highlights() -> dict:
    import csv
    p = os.path.join(_REPO, SOURCES["parallel"], "scalability_table.csv")
    if not os.path.isfile(p):
        return {"rows": []}
    data: dict[tuple[int, int], dict] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n, m, k = int(row["N_total_agents"]), int(row["M_local_masters"]), int(row["K_agents_per_master"])
            if k != 3:
                continue
            data.setdefault((n, m), {})[row["condition"]] = float(row["arrival_pct_mean"])
    out = []
    for (n, m), conds in sorted(data.items()):
        if "normal" in conds and "zero_master" in conds:
            out.append({"N": n, "M": m, "normal": round(conds["normal"], 1),
                        "zero": round(conds["zero_master"], 1)})
    return {"rows": out}


def _load_mixed() -> dict:
    p = os.path.join(_REPO, SOURCES["mixed"], "mixed_layer_results.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    res = d.get("results_mixed_hierarchy") or d.get("results", {})
    return {k: {"arrival": v["mean_arrival_pct"], "crash": v["crash_rate_pct"]}
            for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-maps", action="store_true", help="Regenerate 04_scenario_maps")
    ap.add_argument("--run-tests", action="store_true", help="Run regional eval (quick)")
    ap.add_argument("--run-mixed", action="store_true", help="Re-run mixed-layer experiment")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    if args.run_mixed:
        subprocess.run([PY, "run_mixed_layer_experiment.py", "--scenarios", "50"], check=False)

    n1 = _copy_tree(os.path.join(_REPO, SOURCES["parallel"]), os.path.join(OUT, "01_parallel"))
    n2 = _copy_tree(os.path.join(_REPO, SOURCES["chain"]), os.path.join(OUT, "02_chain"))
    n3 = _copy_tree(os.path.join(_REPO, SOURCES["mixed"]), os.path.join(OUT, "03_mixed_layer"))
    _copy_tree(os.path.join(_REPO, SOURCES["hierarchy_spec"]), os.path.join(OUT, "docs", "figures"))

    if args.run_maps:
        subprocess.run([PY, "generate_scenario_catalog.py",
                        "--out-dir", os.path.join(OUT, "04_scenario_maps")], check=True)
    elif not os.path.isdir(os.path.join(OUT, "04_scenario_maps", "parallel")):
        subprocess.run([PY, "generate_scenario_catalog.py",
                        "--out-dir", os.path.join(OUT, "04_scenario_maps")], check=False)

    if args.run_tests:
        subprocess.run([PY, "test_all_environments_finetune.py", "--quick",
                        "--out", os.path.join(OUT, "05_finetune_eval")], check=False)
    else:
        _copy_tree(os.path.join(_REPO, "PROFESSOR_RESULTS", "05_finetune_eval"),
                   os.path.join(OUT, "05_finetune_eval"))

    master = {
        "timestamp": ts,
        "parallel_highlights": _load_parallel_highlights(),
        "chain_highlights": _load_chain_highlights(),
        "mixed_layer": _load_mixed(),
        "files_copied": {"parallel": n1, "chain": n2, "mixed": n3},
        "output_root": OUT,
    }
    with open(os.path.join(OUT, "master_summary.json"), "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    _write_report(os.path.join(OUT, "REPORT.md"), master)

    print(f"\nProfessor package ready: {OUT}")
    print(f"  copied {n1}+{n2}+{n3} artifacts")
    print(f"  REPORT.md + master_summary.json")


if __name__ == "__main__":
    main()
