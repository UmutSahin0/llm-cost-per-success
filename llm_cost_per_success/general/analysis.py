"""Turn a saved ``runs.csv`` into summary metrics and PNG charts (no API calls)."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .tasks import DATA_DIR  # noqa: E402

SEGMENT_COLORS = {"cheap": "#22c55e", "mid": "#f59e0b", "premium": "#ef4444"}
DEFAULT_COLOR = "#64748b"

# metric key -> (title, y-axis label, caption explaining how to read it)
BAR_CHARTS = {
    "cost_per_success": ("Cost per successful task", "USD",
                         "Total API cost divided by successful tasks; lower is better."),
    "inflate": ("Cost inflation factor", "x",
                "Cost per success / cost of a first-attempt success. 1.0 is ideal; above it is retry and failure overhead."),
    "success_rate": ("Success rate", "ratio",
                     "Share of tasks that ended with a correct answer; higher is better."),
    "first_attempt_success_rate": ("First-attempt success rate", "ratio",
                                   "Share of tasks solved by the first LLM call, with no retry."),
    "p95_latency": ("P95 latency", "s",
                    "95% of tasks finished faster than this; lower is better."),
    "waste_rate": ("Wasted cost rate", "ratio",
                   "Share of total cost spent on tasks that never succeeded; lower is better."),
    "retry_rate": ("Retries per task", "calls",
                   "Average extra LLM calls per task; lower is better."),
    "input_tokens_per_success": ("Input tokens per success", "tokens",
                                 "All input tokens spent divided by successful tasks; lower is more efficient."),
    "output_tokens_per_success": ("Output tokens per success", "tokens",
                                  "All output tokens spent divided by successful tasks; lower is more efficient."),
    "reasoning_tokens_per_success": ("Reasoning tokens per success", "tokens",
                                     "Reported reasoning tokens divided by successful tasks; lower is more efficient."),
}


def _num(row: dict, key: str) -> float:
    return float(row[key])


def _is_true(row: dict, key: str) -> bool:
    return row[key] == "True"


def load_catalog(path: Path = DATA_DIR / "models.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_cost(row: dict, price: dict) -> float:
    """Billed cost of a run; falls back to catalog prices when OpenRouter reported none."""
    billed = _num(row, "cost_usd")
    if billed:
        return billed
    input_price = price.get("input_per_million_usd", 0)
    output_price = price.get("output_per_million_usd", 0)
    reasoning_price = price.get("reasoning_per_million_usd", output_price)
    return (_num(row, "input_tokens") * input_price
            + _num(row, "output_tokens") * output_price
            + _num(row, "reasoning_tokens") * reasoning_price) / 1_000_000


def _ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else None


def summarize_model(model: str, rows: list[dict], price: dict) -> dict:
    """Compute every reported metric for the runs of a single model."""
    good = [r for r in rows if _is_true(r, "success")]
    cost = sum(run_cost(r, price) for r in rows)
    first_try_costs = [run_cost(r, price) for r in rows if _is_true(r, "first_attempt_success")]
    ideal_unit = statistics.mean(first_try_costs) if first_try_costs else None
    cost_per_success = _ratio(cost, len(good))

    all_input = sum(_num(r, "input_tokens") for r in rows)
    all_output = sum(_num(r, "output_tokens") for r in rows)
    good_input = sum(_num(r, "input_tokens") for r in good)
    good_output = sum(_num(r, "output_tokens") for r in good)
    input_price = price.get("input_per_million_usd", 0)
    output_price = price.get("output_per_million_usd", 0)
    latencies = sorted(_num(r, "latency_seconds") for r in rows)

    return {
        "model": model,
        "segment": price.get("segment", "unknown"),
        "input_price_1m": input_price,
        "output_price_1m": output_price,
        "effective_input_price_1m": _ratio(input_price * all_input, good_input),
        "effective_output_price_1m": _ratio(output_price * all_output, good_output),
        "avg_input_tokens": statistics.mean(_num(r, "input_tokens") for r in rows),
        "avg_output_tokens": statistics.mean(_num(r, "output_tokens") for r in rows),
        "success_rate": len(good) / len(rows),
        "first_attempt_success_rate": sum(_is_true(r, "first_attempt_success") for r in rows) / len(rows),
        "actual_total_cost_usd": cost,
        "ideal_one_shot_total_cost_usd": ideal_unit * len(rows) if ideal_unit else None,
        "cost_per_success": cost_per_success,
        "inflate": _ratio(cost_per_success, ideal_unit) if cost_per_success else None,
        "waste_rate": _ratio(sum(run_cost(r, price) for r in rows if not _is_true(r, "success")), cost) or 0,
        "retry_rate": sum(_num(r, "retry_count") for r in rows) / len(rows),
        "avg_calls": statistics.mean(_num(r, "llm_calls") for r in rows),
        "input_tokens_per_success": _ratio(good_input, len(good)),
        "output_tokens_per_success": _ratio(good_output, len(good)),
        "reasoning_tokens_per_success": _ratio(sum(_num(r, "reasoning_tokens") for r in rows), len(good)),
        "p95_latency": latencies[max(0, int(0.95 * len(latencies)) - 1)],
    }


def summarize(rows: list[dict], catalog: dict) -> list[dict]:
    models = sorted({r["model"] for r in rows})
    return [summarize_model(m, [r for r in rows if r["model"] == m], catalog.get(m, {})) for m in models]


def _save(fig, caption: str, path: Path, bottom: float) -> None:
    fig.text(0.5, 0.01, caption, ha="center", wrap=True, fontsize=8)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bar(summary: list[dict], key: str, out: Path) -> None:
    title, ylabel, caption = BAR_CHARTS[key]
    data = [row for row in summary if row[key] is not None]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [SEGMENT_COLORS.get(row["segment"], DEFAULT_COLOR) for row in data]
    bars = ax.bar([row["model"] for row in data], [row[key] for row in data], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for bar, row in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row[key]:.6g}", ha="center")
    segments = sorted({row["segment"] for row in data if row["segment"] in SEGMENT_COLORS})
    if segments:
        ax.legend(handles=[Patch(color=SEGMENT_COLORS[s], label=s.capitalize()) for s in segments],
                  title="Model segment")
    _save(fig, caption, out / f"{key}.png", bottom=0.07)


def plot_heatmap(rows: list[dict], out: Path) -> None:
    tasks = sorted({r["task_id"] for r in rows})
    models = sorted({r["model"] for r in rows})

    def rate(model: str, task: str) -> float:
        cell = [r for r in rows if r["model"] == model and r["task_id"] == task]
        return sum(_is_true(r, "success") for r in cell) / max(1, len(cell))

    fig, ax = plt.subplots(figsize=(14, max(3, len(models))))
    image = ax.imshow([[rate(m, t) for t in tasks] for m in models], vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(tasks)), tasks, rotation=90)
    ax.set_yticks(range(len(models)), models)
    fig.colorbar(image, label="Success rate")
    ax.set_title("Success rate by task and model")
    _save(fig, "Red task/model cells are the first candidates for failure analysis and retry strategy.",
          out / "task_success_heatmap.png", bottom=0.06)


def plot_ideal_vs_actual(summary: list[dict], out: Path) -> None:
    data = [row for row in summary if row["ideal_one_shot_total_cost_usd"] is not None]
    if not data:
        return
    positions = range(len(data))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.2 for i in positions], [r["ideal_one_shot_total_cost_usd"] for r in data], 0.4,
           label="Ideal: every task solved first try", color="#60a5fa")
    ax.bar([i + 0.2 for i in positions], [r["actual_total_cost_usd"] for r in data], 0.4,
           label="Actual: retries and failures included", color="#ef4444")
    ax.set_xticks(list(positions), [r["model"] for r in data], rotation=20)
    ax.set_ylabel("Total USD")
    ax.set_title("Ideal vs. actual total spend")
    ax.legend()
    _save(fig, "Ideal = mean cost of first-attempt successes x number of tasks. "
               "Actual = cost of every call in this experiment.",
          out / "ideal_vs_actual_total_cost.png", bottom=0.08)


def plot_effective_price(summary: list[dict], out: Path) -> None:
    data = [row for row in summary if row["effective_input_price_1m"] is not None]
    if not data:
        return
    positions = range(len(data))
    series = [("input_price_1m", "List input / 1M", "#60a5fa", -0.3),
              ("effective_input_price_1m", "Effective input / 1M", "#1d4ed8", 0.0),
              ("output_price_1m", "List output / 1M", "#fca5a5", 0.3),
              ("effective_output_price_1m", "Effective output / 1M", "#dc2626", 0.6)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color, offset in series:
        ax.bar([i + offset for i in positions], [r[key] for r in data], 0.3, label=label, color=color)
    ax.set_xticks([i + 0.15 for i in positions], [r["model"] for r in data], rotation=20)
    ax.set_ylabel("USD / 1M tokens")
    ax.set_title("List price vs. effective price after failures")
    ax.legend()
    _save(fig, "Effective price = cost of all attempts / tokens of successful tasks only. "
               "It charges the cost of failures and retries to the work that did get done.",
          out / "effective_token_price.png", bottom=0.07)


def analyze_experiment(directory: Path, catalog_path: Path = DATA_DIR / "models.json") -> Path:
    """Write ``summary.csv`` and all charts to ``<directory>/analysis``."""
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    with (directory / "runs.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(rows, load_catalog(catalog_path))

    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)

    for key in BAR_CHARTS:
        plot_bar(summary, key, out)
    plot_heatmap(rows, out)
    plot_ideal_vs_actual(summary, out)
    plot_effective_price(summary, out)
    return out


def latest_experiment(root: Path) -> Path:
    candidates = [d for d in Path(root).glob("*") if d.is_dir() and (d / "runs.csv").exists()]
    if not candidates:
        raise SystemExit(f"No experiment with a runs.csv found under {root}.")
    return max(candidates, key=lambda d: d.stat().st_mtime)
