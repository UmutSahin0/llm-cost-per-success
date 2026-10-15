"""Command line entry point.

    python -m llm_cost_per_success general {run,analyze}          # cost per successful answer
    python -m llm_cost_per_success tools   {run,analyze,compare}  # eager vs. deferred tools
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

GENERAL_OUTPUT_DIR = "outputs/experiments"


def add_general_commands(commands) -> None:
    parser = commands.add_parser("general", help="Cost per successful task on general questions.")
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="Call the models, record raw runs, then build charts.")
    run.add_argument("--models", required=True, help="Comma-separated OpenRouter model IDs.")
    run.add_argument("--repetitions", type=int, default=5)
    run.add_argument("--name", default="experiment", help="Suffix of the experiment folder.")
    run.add_argument("--description", default="", help="Free-text note stored in experiment.json.")
    run.add_argument("--max-attempts", type=int, default=3, help="LLM calls per task before giving up.")
    run.add_argument("--max-tasks", type=int, default=None, help="Run only the first N tasks (smoke test).")
    run.add_argument("--judge-model", default="openai/gpt-oss-20b")
    run.add_argument("--judge-max-tokens", type=int, default=200)
    run.add_argument("--architecture", default="direct-llm-eval",
                     help="Label such as direct-llm-eval, mcp-agent or rag-agent.")
    run.add_argument("--output-dir", default=GENERAL_OUTPUT_DIR)

    analyze = sub.add_parser("analyze", help="Build summary.csv and charts from an existing runs.csv.")
    analyze.add_argument("--experiment", help="Experiment folder; defaults to the newest one.")
    analyze.add_argument("--output-dir", default=GENERAL_OUTPUT_DIR)


def add_tools_commands(commands) -> None:
    parser = commands.add_parser("tools", help="Eager vs. deferred tool loading (tool routing).")
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="Run the tool-routing benchmark for one or more models.")
    run.add_argument("--models", help="Comma-separated model IDs; defaults to the model set in .env.")
    run.add_argument("--provider", choices=("openai", "openrouter"), default="openrouter")
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--catalog-sizes", default="20,40,80", help="Comma-separated tool-catalog sizes.")
    run.add_argument("--schema-profile", choices=("compact", "rich"), default="rich")
    run.add_argument("--max-tasks", type=int, default=20)
    run.add_argument("--conditions", default="eager,dynamic",
                     help="Comma-separated subset of: eager, dynamic (deferred loading).")

    analyze = sub.add_parser("analyze", help="Build charts for one saved experiment.")
    analyze.add_argument("--experiment", help="Experiment number such as 003; defaults to the latest.")

    compare = sub.add_parser("compare", help="Overlay charts for two or more saved experiments.")
    compare.add_argument("--experiments", nargs="+", required=True, help="IDs such as 025 026 (or 025,026).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_cost_per_success", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="benchmark", required=True)
    add_general_commands(commands)
    add_tools_commands(commands)
    return parser


def run_general(args) -> None:
    from .general.analysis import analyze_experiment, latest_experiment

    if args.action == "run":
        from .general.runner import run_experiment  # imported lazily: pulls in LangChain

        args.model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        directory = run_experiment(args)
    else:
        directory = Path(args.experiment) if args.experiment else latest_experiment(Path(args.output_dir))
    print("Analysis:", analyze_experiment(directory))


def run_tools(args, parser: argparse.ArgumentParser) -> None:
    from .tools import analysis, compare, runner

    if args.action == "run":
        sizes = [int(v) for v in args.catalog_sizes.split(",") if v.strip()]
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        if args.repetitions < 1 or args.max_tasks < 1 or not sizes or min(sizes) < 1:
            parser.error("--repetitions, --max-tasks and --catalog-sizes must be positive")
        if not conditions or set(conditions) - set(runner.CONDITIONS):
            parser.error(f"--conditions must be a subset of {', '.join(runner.CONDITIONS)}")

        models = [m.strip() for m in (args.models or "").split(",") if m.strip()] or [None]
        created = [runner.run_experiment(args.provider, model, args.repetitions, sizes,
                                         args.schema_profile, args.max_tasks, conditions).name
                   for model in models]
        for name in created:
            analysis.analyze_experiment(name)
        if len(created) > 1:
            compare.compare_experiments(created)
    elif args.action == "analyze":
        analysis.analyze_experiment(args.experiment)
    else:
        compare.compare_experiments(args.experiments)


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.benchmark == "general":
        run_general(args)
    else:
        run_tools(args, parser)


if __name__ == "__main__":
    main()
