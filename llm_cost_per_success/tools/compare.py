"""Overlay summary metrics from saved benchmark experiments.

Example:
    python -m llm_cost_per_success tools compare --experiments 025 026 027
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "outputs" / "tool-experiments"
SEGMENT_COLORS = {"cheap": "#22c55e", "mid": "#f59e0b", "premium": "#ef4444"}
_CATALOG = json.loads((ROOT / "data" / "models.json").read_text(encoding="utf-8"))
MODEL_SEGMENTS = {model: info["segment"] for model, info in _CATALOG.items()}
MODEL_COLORS = {model: SEGMENT_COLORS.get(segment, "#64748b") for model, segment in MODEL_SEGMENTS.items()}
CONDITION_MARKERS = {"eager": "o", "dynamic": "^"}
METRICS = [
    ("tool_selection_accuracy", "Success rate: correct tool selection", "ratio", True, "success_rate.png"),
    ("cost_per_correct_selection_usd", "Cost per correct tool selection", "USD", False, "cost_per_success.png"),
    ("correct_tools_per_usd", "Correct tool selections per $1", "correct selections / USD", False, "correct_tools_per_usd.png"),
    ("wrong_routing_spend_rate", "Wasted spend: wrong tool routing", "ratio", True, "waste_rate.png"),
    ("mean_input_tokens", "Mean input tokens", "token", False, "input_tokens.png"),
    ("mean_output_tokens", "Mean output tokens", "token", False, "output_tokens.png"),
    ("mean_reasoning_tokens", "Mean reasoning tokens", "token", False, "reasoning_tokens.png"),
    ("p95_latency_ms", "P95 latency", "milliseconds", False, "p95_latency.png"),
]


def parse_experiment_ids(raw_ids: list[str]) -> list[str]:
    ids = [item.strip() for value in raw_ids for item in value.split(",") if item.strip()]
    if len(ids) < 2:
        raise ValueError("Provide at least two experiments, for example: --experiments 025 026")
    missing = [item for item in ids if not (EXPERIMENTS_DIR / item / "summary.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary.csv for experiment(s): {', '.join(missing)}")
    return ids


def load_summaries(experiment_ids: list[str]) -> pd.DataFrame:
    frames = []
    for experiment_id in experiment_ids:
        frame = pd.read_csv(EXPERIMENTS_DIR / experiment_id / "summary.csv")
        frame["experiment"] = experiment_id
        metadata = json.loads((EXPERIMENTS_DIR / experiment_id / "run_metadata.json").read_text(encoding="utf-8"))
        frame["model"] = metadata.get("model", f"experiment {experiment_id}")
        if "catalog_size" not in frame:
            frame["catalog_size"] = 20
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def format_axis(ax: plt.Axes, ylabel: str, percent: bool) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    elif ylabel == "USD":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:.6f}"))
    elif ylabel == "milliseconds":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    else:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)


def plot_metric(summary: pd.DataFrame, column: str, title: str, ylabel: str, percent: bool, output_path: Path) -> bool:
    frame = summary.dropna(subset=[column]).copy()
    if frame.empty:
        return False
    frame["point_label"] = frame.apply(lambda row: f"{MODEL_SEGMENTS.get(row['model'], row['model'])}\n{int(row['catalog_size'])} tools", axis=1)
    frame = frame.sort_values(["experiment", "catalog_size", "condition"])
    labels = list(dict.fromkeys(frame["point_label"]))
    colors = {model: MODEL_COLORS.get(model, "#64748b") for model in frame["model"].drop_duplicates()}
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.25), 5))
    values = frame.groupby("point_label", sort=False)[column].first().reindex(labels)
    bars = ax.bar(labels, values, color=[colors[frame[frame["point_label"] == label].iloc[0]["model"]] for label in labels])
    ax.bar_label(bars, labels=[f"{v:.1%}" if percent else f"${v:.6f}" if ylabel == "USD" else f"{v:.3f}" for v in values], padding=3, fontsize=8)
    ax.set(title=title, xlabel="model and catalog size")
    format_axis(ax, ylabel, percent)
    experiment_handles = [plt.Line2D([], [], color=color, marker="o", linestyle="", label=str(experiment)) for experiment, color in colors.items()]
    ax.legend(handles=experiment_handles, title="model", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def compare_experiments(raw_ids: list[str]) -> Path:
    """Write overlay charts and ``combined_summary.csv`` for two or more experiments."""
    experiment_ids = parse_experiment_ids(raw_ids)
    summary = load_summaries(experiment_ids)
    output_dir = EXPERIMENTS_DIR / f"compare_{'_'.join(experiment_ids)}" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for column, title, ylabel, percent, filename in METRICS:
        if column in summary.columns:
            written += plot_metric(summary, column, title, ylabel, percent, output_dir / filename)
    combined_path = output_dir.parent / "combined_summary.csv"
    summary.to_csv(combined_path, index=False)
    print(f"Compared experiments: {', '.join(experiment_ids)}")
    print(f"Generated {written} PNG files in: {output_dir}")
    print(f"Combined summary: {combined_path}")
    return output_dir
