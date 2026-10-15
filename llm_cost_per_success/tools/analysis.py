"""Create PNG charts for one saved tool-loading experiment (no API calls)."""

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
OUTPUT_DIR: Path
COLORS = {"eager": "#4C78A8", "dynamic": "#F58518"}


def save_bar_chart(summary: pd.DataFrame, column: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    labels = summary.apply(lambda row: f"{row['condition']}\n{int(row['catalog_size'])} tools", axis=1)
    values = pd.Series(summary[column].values, index=labels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(values.index, values.values, color=[COLORS[name] for name in summary["condition"]])
    ax.set_title(title, pad=12, weight="bold")
    ax.set_ylabel(ylabel)
    if ylabel in {"milliseconds", "USD"}:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if percent:
        ax.set_ylim(0, 1.05)
        labels = [f"{value:.1%}" for value in values]
    else:
        labels = [f"{value:,.0f}" if ylabel == "milliseconds" else f"${value:.6f}" if ylabel == "USD" else f"{value:,.3g}" for value in values]
    ax.bar_label(bars, labels=labels, padding=4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cumulative_progress_chart(runs: pd.DataFrame) -> None:
    """Show what a decision-maker spends and gets as questions accumulate."""
    if runs.empty:
        return
    frame = runs.copy()
    frame["run_at_utc"] = pd.to_datetime(frame["run_at_utc"], utc=True, errors="coerce")
    frame = frame.sort_values("run_at_utc", kind="stable")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    group_columns = ["catalog_size", "condition"]
    for (catalog_size, condition), subset in frame.groupby(group_columns, observed=True, sort=True):
        subset = subset.copy()
        question_number = range(1, len(subset) + 1)
        label = f"{condition} · {int(catalog_size)} tools"
        color = COLORS.get(str(condition), "#666666")
        axes[0].plot(question_number, subset["total_tokens"].cumsum(), marker="o", ms=3, label=label, color=color)
        axes[1].plot(question_number, subset["estimated_cost_usd"].cumsum(), marker="o", ms=3, label=label, color=color)
        axes[2].plot(question_number, subset["accurate"].astype(float).expanding().mean(), marker="o", ms=3, label=label, color=color)
    axes[0].set(title="Benchmark Progress: Cumulative Tokens, Spend, and Accuracy", ylabel="cumulative total tokens")
    axes[1].set(ylabel="cumulative actual cost (USD)")
    axes[2].set(xlabel="question number", ylabel="running accuracy", ylim=(0, 1.05))
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(title="condition", loc="best")
    for axis in axes[:2]:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.yaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:.5f}"))
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "22_benchmark_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_experiment(experiment: str | None = None) -> Path:
    """Write charts to ``<experiment>/plots``; defaults to the newest numbered experiment."""
    global OUTPUT_DIR
    choices = sorted((path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir() and path.name.isdigit()),
                     key=lambda path: int(path.name)) if EXPERIMENTS_DIR.exists() else []
    if experiment:
        DATA_DIR = EXPERIMENTS_DIR / experiment
    elif choices:
        DATA_DIR = choices[-1]
    else:
        raise FileNotFoundError(f"No tool experiments found under {EXPERIMENTS_DIR}")
    if not (DATA_DIR / "raw_runs.csv").exists():
        raise FileNotFoundError(f"No raw_runs.csv found in {DATA_DIR}")
    OUTPUT_DIR = DATA_DIR / "plots"
    OUTPUT_DIR.mkdir(exist_ok=True)
    runs = pd.read_csv(DATA_DIR / "raw_runs.csv")
    summary = pd.read_csv(DATA_DIR / "summary.csv")
    metadata = json.loads((DATA_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    if "catalog_size" not in summary:
        summary["catalog_size"] = 20
    summary["condition"] = pd.Categorical(summary["condition"], ["eager", "dynamic"], ordered=True)
    summary = summary.sort_values(["catalog_size", "condition"])

    save_bar_chart(summary, "mean_input_tokens", "Mean Total Input Tokens", "tokens", "01_input_tokens.png")
    save_bar_chart(summary, "mean_cached_tokens", "Mean Cached Input Tokens", "tokens", "02_cached_tokens.png")
    if "mean_output_tokens" in summary:
        save_bar_chart(summary, "mean_output_tokens", "Mean Output Tokens", "tokens", "03_output_tokens.png")
    if "mean_reasoning_tokens" in summary:
        save_bar_chart(summary, "mean_reasoning_tokens", "Mean Reasoning Tokens", "tokens", "04_reasoning_tokens.png")
    if "mean_total_tokens" in summary:
        save_bar_chart(summary, "mean_total_tokens", "Mean Total Tokens", "tokens", "05_total_tokens.png")
    if "cached_input_rate" in summary:
        save_bar_chart(summary, "cached_input_rate", "Cached Share of Total Input Tokens", "cache hit rate", "06_cached_input_rate.png", percent=True)
    if "mean_input_cost_usd" in summary:
        save_bar_chart(summary, "mean_input_cost_usd", "Estimated Input Cost per Task", "USD", "07_input_cost_usd.png")
    if "mean_output_cost_usd" in summary:
        save_bar_chart(summary, "mean_output_cost_usd", "Estimated Output Cost per Task", "USD", "08_output_cost_usd.png")
    save_bar_chart(summary, "mean_estimated_cost_usd", "Estimated Total Model Cost per Task", "USD", "09_total_cost_usd.png")
    if "mean_cache_savings_usd" in summary:
        save_bar_chart(summary, "mean_cache_savings_usd", "Cache Savings per Task", "USD", "18_cache_savings_usd.png")
    if "cache_cost_savings_rate" in summary:
        save_bar_chart(summary, "cache_cost_savings_rate", "Cache Savings Rate on Total Cost", "savings rate", "19_cache_savings_rate.png", percent=True)
    if "cost_per_correct_selection_usd" in summary:
        success_cost = summary.dropna(subset=["cost_per_correct_selection_usd"])
        if not success_cost.empty:
            save_bar_chart(success_cost, "cost_per_correct_selection_usd", "Actual Spend per Correct Tool Selection", "USD", "20_cost_per_correct_tool.png")
    if "wrong_routing_spend_rate" in summary:
        save_bar_chart(summary, "wrong_routing_spend_rate", "Spend Wasted on Wrong Tool Routing", "wasted spend rate", "21_wrong_routing_spend_rate.png", percent=True)
    save_bar_chart(summary, "mean_latency_ms", "Mean End-to-End Latency", "milliseconds", "10_latency_ms.png")
    save_bar_chart(summary, "tool_selection_accuracy", "Tool Selection Accuracy", "accuracy", "11_tool_accuracy.png", percent=True)
    callable_metric_available = metadata.get("provider") != "openrouter" and metadata.get("architecture") != "v5_openrouter_deferred_ablation"
    if callable_metric_available and "mean_decision_callable_tools" in summary:
        save_bar_chart(
            summary,
            "mean_decision_callable_tools",
            "Mean Callable Tools at Final Selection",
            "tools per question",
            "12_callable_tools_per_question.png",
        )
    if "retrieval_recall_at_k" in summary and summary["retrieval_recall_at_k"].notna().any():
        dynamic_summary = summary[summary["condition"] == "dynamic"].dropna(subset=["retrieval_recall_at_k"])
        save_bar_chart(dynamic_summary, "retrieval_recall_at_k", "Discovery Retrieval Recall@k", "recall", "13_retrieval_recall.png", percent=True)

    valid = runs[runs["error"].fillna("") == ""].copy()
    save_cumulative_progress_chart(valid)
    fig, ax = plt.subplots(figsize=(12, 5))
    for condition in ("eager", "dynamic"):
        subset = valid[valid["condition"] == condition]
        ax.scatter(subset["task_id"], subset["latency_ms"], label=condition, alpha=0.65, s=34, color=COLORS[condition])
    ax.set(title="Latency Distribution by Task", xlabel="task", ylabel="milliseconds")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="condition")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_latency_by_task.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    failures = int((runs["error"].fillna("") != "").sum())
    print(f"Model: {metadata['model']}")
    print(f"Experiment: {DATA_DIR.name}")
    print(f"Generated PNG files in: {OUTPUT_DIR}")
    print(f"Rows: {len(runs)} | API error rows excluded from charts: {failures}")
    return OUTPUT_DIR
