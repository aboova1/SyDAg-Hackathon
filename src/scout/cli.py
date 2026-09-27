"""Commands for the published Field Scout workflow."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build maize scouting evidence.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Check source joins and required files.")
    audit.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))

    features = commands.add_parser("features", help="Extract six-band image features.")
    features.add_argument("--season", type=int, default=2022)
    features.add_argument("--output", type=Path, default=Path("outputs/features_2022.csv"))
    features.add_argument("--force", action="store_true")

    rank_dap = commands.add_parser(
        "evaluate-rank-input-dap-split",
        help="Compare scouting inputs on held-out trial blocks.",
    )
    rank_dap.add_argument("--features", type=Path, default=Path("outputs/features_2022.csv"))
    rank_dap.add_argument("--dap", type=int, required=True)
    rank_dap.add_argument("--seed", type=int, default=42)
    rank_dap.add_argument("--output", type=Path, required=True)

    yield_dap = commands.add_parser(
        "evaluate-yield-dap-split",
        help="Compare yield inputs on held-out trial blocks.",
    )
    yield_dap.add_argument("--features", type=Path, default=Path("outputs/features_2022.csv"))
    yield_dap.add_argument("--dap", type=int, required=True)
    yield_dap.add_argument("--seed", type=int, default=42)
    yield_dap.add_argument("--output", type=Path, required=True)

    rank = commands.add_parser("evaluate-rank", help="Build the calendar-date demo rank.")
    rank.add_argument("--features", type=Path, default=Path("outputs/features_2022.csv"))
    rank.add_argument("--cutoff", default="2022-08-15")
    rank.add_argument("--seed", type=int, default=42)
    rank.add_argument("--output", type=Path, required=True)

    regression = commands.add_parser(
        "evaluate-randomized", help="Build the calendar-date demo yield estimate."
    )
    regression.add_argument("--features", type=Path, default=Path("outputs/features_2022.csv"))
    regression.add_argument("--cutoff", default="2022-08-15")
    regression.add_argument("--seed", type=int, default=42)
    regression.add_argument("--output", type=Path, required=True)

    demo = commands.add_parser("prepare-rank-demo", help="Join saved rank and yield runs.")
    demo.add_argument("--rank-run", type=Path, required=True)
    demo.add_argument("--yield-run", type=Path, required=True)
    demo.add_argument("--output", type=Path, default=Path("outputs/demo_rank_aug15"))

    args = parser.parse_args()
    if args.command == "audit":
        from .audit import audit as run_audit

        result = run_audit(args.root, args.output)
        if not result["structural_pass"]:
            raise SystemExit(1)
    elif args.command == "features":
        from .features import extract

        extract(args.root, args.output, args.season, force=args.force)
    elif args.command == "evaluate-rank-input-dap-split":
        from .rank_input_split_eval import evaluate_rank_input_split

        evaluate_rank_input_split(args.root, args.features, None, args.output,
                                  seed=args.seed, dap=args.dap)
    elif args.command == "evaluate-yield-dap-split":
        from .yield_split_eval import evaluate_yield_split

        evaluate_yield_split(args.root, args.features, None, args.output,
                             seed=args.seed, dap=args.dap)
    elif args.command == "evaluate-rank":
        from .rank_eval import evaluate_rank

        evaluate_rank(args.root, args.features, args.cutoff, args.output, seed=args.seed)
    elif args.command == "evaluate-randomized":
        from .random_eval import evaluate_randomized

        evaluate_randomized(args.root, args.features, args.cutoff, args.output,
                            seed=args.seed)
    elif args.command == "prepare-rank-demo":
        from .demo import prepare_rank_demo

        prepare_rank_demo(args.rank_run, args.yield_run, args.output)


if __name__ == "__main__":
    main()
