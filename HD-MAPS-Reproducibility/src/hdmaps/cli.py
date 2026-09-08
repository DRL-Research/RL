from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hdmaps", description="Reproduce HD-MAPS Table 6.4.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("reproduce", help="Rebuild Table 6.4 from frozen episode records.")
    p_rep.add_argument("--smoke", action="store_true")
    p_rep.add_argument("--out", type=Path, default=None)

    p_eval = sub.add_parser("evaluate", help="Re-run corridor simulation with frozen checkpoints.")
    p_eval.add_argument("--protocol", choices=("simultaneous", "staggered", "both"), default="both")
    p_eval.add_argument("--smoke", action="store_true")
    p_eval.add_argument("--out", type=Path, default=None)

    p_train = sub.add_parser("train", help="Locked HD-MAPS retraining protocol (6 agents / 2 intersections).")
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "reproduce":
        from hdmaps.reproduce import reproduce

        payload = reproduce(smoke=args.smoke, out_dir=args.out)
        print(payload["table"])
        return 0
    if args.cmd == "evaluate":
        from hdmaps.evaluation.runner import evaluate_simultaneous, evaluate_staggered
        from hdmaps.statistics.tables import format_table

        out = args.out or Path("outputs") / "evaluate"
        sim_rows = []
        stag_rows = []
        if args.protocol in ("simultaneous", "both"):
            sim_rows = evaluate_simultaneous(smoke=args.smoke, out_dir=out)
        if args.protocol in ("staggered", "both"):
            raw = evaluate_staggered(smoke=args.smoke, out_dir=out)
            stag_rows = [
                {
                    "fleet": r["fleet"],
                    "hierarchy": r["hierarchy"],
                    "crash_pct": r["crash_pct"],
                    "crash_lo": r["crash_lo"],
                    "crash_hi": r["crash_hi"],
                    "zero_crash_pct": r.get("zero_crash_pct", 0.0),
                }
                for r in raw
            ]
        if sim_rows:
            print(format_table(sim_rows, stag_rows or None))
        elif stag_rows:
            for r in stag_rows:
                print(
                    f"{r['fleet']} {r['hierarchy']} crash={r['crash_pct']:.1f} "
                    f"zero={r['zero_crash_pct']:.1f}"
                )
        return 0
    if args.cmd == "train":
        from hdmaps.training.hdmaps import train

        summary = train(smoke=args.smoke, out_dir=args.out)
        print(summary)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
