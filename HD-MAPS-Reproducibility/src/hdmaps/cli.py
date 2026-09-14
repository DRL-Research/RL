from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hdmaps", description="Reproduce HD-MAPS Table 6.4.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("reproduce", help="Rebuild Table 6.4 from frozen episode records.")
    p_rep.add_argument("--smoke", action="store_true")
    p_rep.add_argument("--out", type=Path, default=None)

    p_eval = sub.add_parser("evaluate", help="Live corridor evaluation.")
    p_eval.add_argument("--protocol", choices=("simultaneous", "staggered", "both"), default="both")
    p_eval.add_argument("--smoke", action="store_true")
    p_eval.add_argument("--out", type=Path, default=None)
    p_eval.add_argument("--init", choices=("published", "retrain"), default="published")
    p_eval.add_argument("--agent", type=Path, default=None)
    p_eval.add_argument("--master", type=Path, default=None)

    p_train = sub.add_parser("train", help="Master-only fine-tune (agent frozen).")
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument("--out", type=Path, default=None)
    p_train.add_argument("--init", choices=("base", "stage1", "published"), default="stage1")
    p_train.add_argument("--episodes", type=int, default=None)
    p_train.add_argument(
        "--pipeline",
        action="store_true",
        help="Run stage1 (1 intersection, 1-4 agents) then stage2 (2 intersections, up to 6 agents).",
    )

    p_run = sub.add_parser("run", help="Full retrain (stage1+stage2) then live evaluation. No frozen-table shortcut.")
    p_run.add_argument("--smoke", action="store_true")
    p_run.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "reproduce":
        from hdmaps.reproduce import reproduce

        payload = reproduce(smoke=args.smoke, out_dir=args.out)
        print(payload["table"])
        return 0
    if args.cmd == "evaluate":
        from hdmaps.evaluation.runner import evaluate_simultaneous, evaluate_staggered
        from hdmaps.paths import retrain_agent_checkpoint, retrain_master_checkpoint
        from hdmaps.statistics.tables import format_table

        out = args.out or Path("outputs") / "evaluate"
        agent = args.agent
        master = args.master
        if args.init == "retrain":
            agent = agent or retrain_agent_checkpoint()
            master = master or retrain_master_checkpoint()
        sim_rows = []
        stag_rows = []
        if args.protocol in ("simultaneous", "both"):
            sim_rows = evaluate_simultaneous(
                smoke=args.smoke, out_dir=out, agent_path=agent, master_path=master
            )
        if args.protocol in ("staggered", "both"):
            stag_rows = evaluate_staggered(
                smoke=args.smoke, out_dir=out, agent_path=agent, master_path=master
            )
        print(format_table(sim_rows, stag_rows or None))
        return 0
    if args.cmd == "train":
        from hdmaps.training.hdmaps import train, train_pipeline

        if args.pipeline:
            summary = train_pipeline(smoke=args.smoke, out_dir=args.out)
        else:
            summary = train(smoke=args.smoke, out_dir=args.out, init=args.init, episodes=args.episodes)
        print(summary)
        return 0
    if args.cmd == "run":
        from hdmaps.evaluation.runner import evaluate_simultaneous, evaluate_staggered
        from hdmaps.statistics.tables import format_table
        from hdmaps.training.hdmaps import train_pipeline

        train_out = (args.out or Path("outputs") / "run") / "train"
        eval_out = (args.out or Path("outputs") / "run") / "eval"
        summary = train_pipeline(smoke=args.smoke, out_dir=train_out)
        print(summary, flush=True)
        agent = train_out / "agent.pth"
        master = train_out / "master.pth"
        sim_rows = evaluate_simultaneous(
            smoke=args.smoke, out_dir=eval_out, agent_path=agent, master_path=master
        )
        stag_rows = evaluate_staggered(
            smoke=args.smoke, out_dir=eval_out, agent_path=agent, master_path=master
        )
        table = format_table(sim_rows, stag_rows)
        print(table, flush=True)
        eval_out.mkdir(parents=True, exist_ok=True)
        (eval_out / "table.txt").write_text(table + "\n", encoding="utf-8")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
