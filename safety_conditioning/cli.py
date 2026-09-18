"""Single entry point for active studies and frozen historical reproductions."""
import argparse
import subprocess
import sys
from pathlib import Path
from .artifacts import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    study = commands.add_parser("study", help="Run the locked independent-model replication")
    study.add_argument("--config", type=Path, default=PROJECT / "configs/halfcheetah_replication.json")
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--device", default="cuda")
    train = commands.add_parser("train", help="Train one fixed-recipe model with an explicit seed")
    train.add_argument("--task", choices=["HalfCheetah", "Walker2d"], default="HalfCheetah")
    train.add_argument("--dataset", type=Path)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    collect = commands.add_parser("collect", help="Recreate complete episodes from pinned public experts")
    collect.add_argument("--task", choices=["HalfCheetah", "Walker2d"], default="HalfCheetah")
    collect.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="Verify the refactor against frozen implementations")
    check.add_argument("--device", default="cuda")
    check.add_argument("--output", type=Path, default=PROJECT / "records/consolidation_checks.json")
    for name in ["audit", "report"]:
        command = commands.add_parser(name, help=f"{name.title()} an independent-model study or mismatch probe")
        command.add_argument("--input", type=Path, required=True)
    reproduce = commands.add_parser("reproduce", help="Re-run a frozen historical experiment")
    reproduce.add_argument("task", choices=["pusht", "pendulum", "walker2d", "halfcheetah"])
    reproduce.add_argument("--output", type=Path, required=True)
    commands.add_parser("status", help="Show the canonical and historical result locations")
    mismatch = commands.add_parser("mismatch", help="Probe supplied-safe-set error using frozen outputs")
    mismatch.add_argument("--input", type=Path, required=True)
    mismatch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "study":
        from .study import run_study
        run_study(args.config, args.output, args.device)
    elif args.command == "train":
        from .training import train_policy
        dataset = args.dataset or PROJECT / f"round3/results/{args.task}_training/episodes.npz"
        train_policy(dataset, args.task, args.seed, args.output.resolve(),
                     horizon=8 if args.task == "HalfCheetah" else 12, device=args.device)
    elif args.command == "check":
        from .validation import check_refactor
        check_refactor(args.output, args.device)
    elif args.command == "collect":
        from .collection import collect_episodes
        collect_episodes(args.task, args.output)
    elif args.command == "audit":
        from .artifacts import read_json
        if "relative_errors" in read_json(args.input / "summary.json")["config"]:
            from .mismatch_diagnostics import audit_mismatch
            audit_mismatch(args.input.resolve())
        else:
            from .audit import audit_study
            audit_study(args.input.resolve())
    elif args.command == "report":
        from .artifacts import read_json
        if "relative_errors" in read_json(args.input / "summary.json")["config"]:
            from .mismatch_diagnostics import report_mismatch
            report_mismatch(args.input.resolve())
        else:
            from .reporting import build_report
            build_report(args.input.resolve())
    elif args.command == "reproduce":
        if args.output.exists():
            parser.error("Choose a fresh output directory.")
        if args.task == "pusht":
            command = [sys.executable, str(PROJECT / "src/evaluate.py"), "--output", str(args.output.resolve())]
        else:
            names = {"pendulum": "Pendulum", "walker2d": "Walker2d", "halfcheetah": "HalfCheetah"}
            command = [sys.executable, str(PROJECT / "round3/scripts/replicate.py"), names[args.task],
                       "--output-root", str(args.output.resolve())]
        subprocess.run(command, cwd=PROJECT, check=True)
    elif args.command == "mismatch":
        from .mismatch import analyze_mismatch
        analyze_mismatch(args.input, args.output)
    else:
        print("Active code: safety_conditioning/; environment: round3/.venv")
        print("Current study: results/halfcheetah_replication; protocol: docs/HALFCHEETAH_REPLICATION.md")
        print("Safe-set error probe: results/safe_set_mismatch; protocol: docs/SAFE_SET_MISMATCH.md")
        print("Frozen history: results/heldout; results/pendulum/confirmation; round3/results/*_comparison")
        print("Read RESULTS.md for the current evidence and README.md for commands.")
