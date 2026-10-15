"""Run reproducible eager vs. deferred tool-loading benchmarks.

Supported providers: OpenAI Responses ``tool_search`` and OpenRouter Responses
``openrouter:tool_search``. Both use the same stateless controlled ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .catalog import build_catalog, public_schema


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "outputs" / "tool-experiments"
CONDITIONS = ("eager", "dynamic")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
SYSTEM_PROMPT = (
    "You are a tool-routing agent. Call exactly one function for every user request. "
    "Use request_clarification only when essential information is missing, and "
    "respond_without_tool only for writing or reasoning requests that need no external action. "
    "Do not answer in prose."
)


@dataclass
class Metrics:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    reported_cost_usd: float = 0.0


def response_metrics(response: Any, latency_ms: float) -> Metrics:
    usage = response.usage
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return Metrics(
        input_tokens=input_tokens,
        cached_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
        output_tokens=output_tokens,
        reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)),
        latency_ms=latency_ms,
        reported_cost_usd=float(getattr(usage, "cost", 0) or 0),
    )


def invoke(client: OpenAI, model: str, tools: list[dict[str, Any]], prompt: str) -> tuple[Any, Metrics]:
    started_at = time.perf_counter()
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        tools=tools,
        parallel_tool_calls=False,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return response, response_metrics(response, (time.perf_counter() - started_at) * 1000)


def function_call(response: Any) -> Any | None:
    return next((item for item in response.output if item.type == "function_call"), None)


def loaded_tool_names(items: list[Any]) -> list[str]:
    """Flatten OpenAI ``tool_search_output.tools`` items, including namespaces."""
    names: list[str] = []
    for item in items:
        nested = getattr(item, "tools", None)
        if nested:
            names.extend(loaded_tool_names(list(nested)))
        elif getattr(item, "name", None):
            names.append(item.name)
    return names


def pricing(model: str) -> dict[str, float]:
    """USD per million tokens: ``data/models.json`` first, then ``.env`` overrides as fallback."""
    catalog = json.loads((DATA_DIR / "models.json").read_text(encoding="utf-8"))
    listed = catalog.get(model, {})
    input_price = listed.get("input_per_million_usd", float(os.getenv("INPUT_PRICE_PER_MILLION", "0")))
    return {
        "input_price_per_million_usd": input_price,
        "cached_input_price_per_million_usd": listed.get(
            "cached_input_per_million_usd", float(os.getenv("CACHED_INPUT_PRICE_PER_MILLION", input_price))),
        "output_price_per_million_usd": listed.get(
            "output_per_million_usd", float(os.getenv("OUTPUT_PRICE_PER_MILLION", "0"))),
    }


def cost_breakdown(metrics: Metrics, rates: dict[str, float]) -> dict[str, float]:
    uncached_input = max(metrics.input_tokens - metrics.cached_tokens, 0)
    uncached_cost = uncached_input * rates["input_price_per_million_usd"] / 1_000_000
    cached_cost = metrics.cached_tokens * rates["cached_input_price_per_million_usd"] / 1_000_000
    output_cost = metrics.output_tokens * rates["output_price_per_million_usd"] / 1_000_000
    estimated_cost = uncached_cost + cached_cost + output_cost
    without_cache = metrics.input_tokens * rates["input_price_per_million_usd"] / 1_000_000 + output_cost
    return {
        "uncached_input_cost_usd": uncached_cost,
        "cached_input_cost_usd": cached_cost,
        "input_cost_usd": uncached_cost + cached_cost,
        "output_cost_usd": output_cost,
        "estimated_cost_usd": estimated_cost,
        "estimated_cost_without_cache_usd": without_cache,
        "cache_savings_usd": without_cache - estimated_cost,
    }


def benchmark_tools(catalog: list[dict[str, Any]], deferred: bool, provider: str) -> list[dict[str, Any]]:
    """Build a controlled ablation where only business-tool deferral changes."""
    search_type = "tool_search" if provider == "openai" else "openrouter:tool_search"
    # OpenAI requires one deferred tool when tool_search is present. Keeping this
    # inert sentinel in both arms prevents it from becoming a confounder.
    sentinel = {
        "type": "function",
        "name": "benchmark_tool_search_sentinel",
        "description": "Internal benchmark sentinel. Never call this tool for a user request.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "defer_loading": True,
    }
    business_tools = [{**public_schema(tool), "defer_loading": deferred} for tool in catalog]
    return [{"type": search_type}, sentinel, *business_tools]


def run_task(
    client: OpenAI,
    model: str,
    task: dict[str, str],
    condition: str,
    catalog_size: int,
    repetition: int,
    schema_profile: str,
    provider: str,
    rates: dict[str, float],
) -> dict[str, Any]:
    catalog = build_catalog(catalog_size, schema_profile)
    selected_tool = ""
    loaded_names: list[str] = []
    output_types = ""
    error = ""
    used_tool_search = False
    try:
        response, metrics = invoke(
            client,
            model,
            benchmark_tools(catalog, deferred=condition == "dynamic", provider=provider),
            task["prompt"],
        )
        selected = function_call(response)
        selected_tool = selected.name if selected else ""
        output_types = "|".join(item.type for item in response.output)
        used_tool_search = any(item.type in {"tool_search_call", "openrouter:tool_search"} for item in response.output)
        loaded_names = loaded_tool_names([item for item in response.output if item.type == "tool_search_output"])
    except Exception as exc:  # Failed API calls remain visible in raw_runs.csv.
        metrics = Metrics()
        error = repr(exc)

    # OpenRouter does not expose the hosted search result, so never guess it.
    loaded_count: int | None = None if provider == "openrouter" and condition == "dynamic" else len(set(loaded_names))
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "architecture": f"{provider}_hosted_tool_search",
        "workload": "stateless_controlled_ablation",
        "condition": condition,
        "schema_deferred": condition == "dynamic",
        "catalog_size": catalog_size,
        "task_id": task["id"],
        "task_group": task["group"],
        "prompt": task["prompt"],
        "expected_tool": task["expected_tool"],  # Evaluation-only: never sent to the model.
        "selected_tool": selected_tool,
        # For tasks that need no tool, making no function call is the correct behaviour.
        "accurate": (not selected_tool) if task["expected_tool"] == "respond_without_tool" else selected_tool == task["expected_tool"],
        "api_calls": 1,
        "initial_callable_tool_count": len(catalog) if condition == "eager" else 0,
        "decision_callable_tool_count": len(catalog) if condition == "eager" else loaded_count,
        "native_tool_search_used": used_tool_search,
        "native_loaded_tools": "|".join(loaded_names),
        "native_loaded_tool_count": loaded_count,
        "native_output_types": output_types,
        **asdict(metrics),
        "cache_hit_rate": metrics.cached_tokens / metrics.input_tokens if metrics.input_tokens else 0.0,
        **cost_breakdown(metrics, rates),
        "error": error,
        "repetition": repetition,
    }


def next_experiment_dir() -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in EXPERIMENTS_DIR.iterdir() if path.is_dir() and path.name.isdigit()]
    destination = EXPERIMENTS_DIR / f"{max(existing, default=0) + 1:03d}"
    destination.mkdir()
    return destination


def mean(values: list[float | int]) -> float:
    return statistics.fmean(values) if values else 0.0


def catalog_stats(sizes: list[int], schema_profile: str) -> list[dict[str, Any]]:
    rows = []
    for size in sizes:
        for tool in build_catalog(size, schema_profile):
            schema = public_schema(tool)
            rows.append({
                "catalog_size": size,
                "tool_name": tool["name"],
                "description_chars": len(tool["description"]),
                "description_words": len(tool["description"].split()),
                "schema_json_chars": len(json.dumps(schema, ensure_ascii=False, separators=(",", ":"))),
                "parameter_count": len(tool["parameters"]["properties"]),
            })
    return rows


def write_experiment(rows: list[dict[str, Any]], metadata: dict[str, Any], destination: Path, sizes: list[int], schema_profile: str) -> None:
    with (destination / "raw_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for size in sorted(set(sizes)):
        catalog = build_catalog(size, schema_profile)
        schema_sizes = [len(json.dumps(public_schema(tool), ensure_ascii=False, separators=(",", ":"))) for tool in catalog]
        for condition in [c for c in CONDITIONS if any(row["condition"] == c for row in rows)]:
            group = [row for row in rows if row["catalog_size"] == size and row["condition"] == condition]
            successful = [row for row in group if not row["error"]]
            total_input = sum(row["input_tokens"] for row in successful)
            total_cached = sum(row["cached_tokens"] for row in successful)
            total_cost = sum(row["estimated_cost_usd"] for row in group)
            total_without_cache = sum(row["estimated_cost_without_cache_usd"] for row in successful)
            correct = sum(row["accurate"] for row in successful)
            decision_counts = [row["decision_callable_tool_count"] for row in successful if row["decision_callable_tool_count"] is not None]
            loaded_counts = [row["native_loaded_tool_count"] for row in successful if row["native_loaded_tool_count"] is not None]
            wrong_cost = sum(row["estimated_cost_usd"] for row in group if not row["accurate"])
            summary_rows.append({
                "catalog_size": size,
                "condition": condition,
                "runs": len(group),
                "catalog_tool_count": len(catalog),
                "mean_tool_description_chars": mean([len(tool["description"]) for tool in catalog]),
                "mean_tool_description_words": mean([len(tool["description"].split()) for tool in catalog]),
                "mean_tool_schema_json_chars": mean(schema_sizes),
                "total_catalog_schema_json_chars": sum(schema_sizes),
                "failure_rate": 1 - len(successful) / len(group) if group else 0.0,
                "tool_selection_accuracy": sum(row["accurate"] for row in group) / len(group) if group else 0.0,
                "wrong_routing_spend_rate": wrong_cost / total_cost if total_cost else "",
                "correct_tools_per_usd": correct / total_cost if total_cost else 0.0,
                "mean_initial_callable_tools": mean([row["initial_callable_tool_count"] for row in successful]),
                "mean_decision_callable_tools": mean(decision_counts) if decision_counts else "",
                "mean_native_loaded_tools": mean(loaded_counts) if loaded_counts else "",
                "mean_input_tokens": mean([row["input_tokens"] for row in successful]),
                "mean_cached_tokens": mean([row["cached_tokens"] for row in successful]),
                "cached_input_rate": total_cached / total_input if total_input else 0.0,
                "mean_output_tokens": mean([row["output_tokens"] for row in successful]),
                "mean_reasoning_tokens": mean([row["reasoning_tokens"] for row in successful]),
                "mean_total_tokens": mean([row["total_tokens"] for row in successful]),
                "mean_input_cost_usd": mean([row["input_cost_usd"] for row in successful]),
                "mean_output_cost_usd": mean([row["output_cost_usd"] for row in successful]),
                "mean_cache_savings_usd": mean([row["cache_savings_usd"] for row in successful]),
                "cache_cost_savings_rate": (total_without_cache - total_cost) / total_without_cache if total_without_cache else 0.0,
                "cost_per_correct_selection_usd": total_cost / correct if correct else "",
                "mean_latency_ms": mean([row["latency_ms"] for row in successful]),
                "p95_latency_ms": sorted(row["latency_ms"] for row in successful)[max(0, round(0.95 * len(successful)) - 1)] if successful else 0.0,
                "mean_reported_cost_usd": mean([row["reported_cost_usd"] for row in successful]),
                "mean_estimated_cost_usd": mean([row["estimated_cost_usd"] for row in successful]),
            })

    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    stats = catalog_stats(sizes, schema_profile)
    with (destination / "catalog_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)
    metadata["catalog_profiles"] = {
        str(size): {
            "tool_count": size,
            "mean_description_chars": mean([row["description_chars"] for row in stats if row["catalog_size"] == size]),
            "mean_description_words": mean([row["description_words"] for row in stats if row["catalog_size"] == size]),
            "mean_schema_json_chars": mean([row["schema_json_chars"] for row in stats if row["catalog_size"] == size]),
            "total_schema_json_chars": sum(row["schema_json_chars"] for row in stats if row["catalog_size"] == size),
        }
        for size in sizes
    }
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(DATA_DIR / "tool_tasks.json", destination / "tasks_snapshot.json")


def make_client(provider: str) -> tuple[OpenAI, str]:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required. Add it to .env or your shell environment.")
        return OpenAI(api_key=key), os.getenv("OPENAI_MODEL", "gpt-6-luna")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required. Add it to .env or your shell environment.")
    return (
        OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", default_headers={"X-OpenRouter-Title": "llm-cost-per-success"}),
        os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-luna"),
    )


def run_experiment(
    provider: str,
    model: str | None,
    repetitions: int,
    catalog_sizes: list[int],
    schema_profile: str,
    max_tasks: int,
    conditions: list[str],
) -> Path:
    """Run one model through every catalog size and condition; return the experiment folder."""
    client, default_model = make_client(provider)
    model = model or default_model
    rates = pricing(model)
    tasks = json.loads((DATA_DIR / "tool_tasks.json").read_text(encoding="utf-8"))[:max_tasks]
    rows = []
    for size in catalog_sizes:
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                for condition in conditions:
                    row = run_task(client, model, task, condition, size, repetition,
                                   schema_profile, provider, rates)
                    rows.append(row)
                    print(model, size, condition, task["id"], row["selected_tool"], row["accurate"])

    destination = next_experiment_dir()
    write_experiment(
        rows,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "architecture": f"{provider}_hosted_tool_search",
            "workload": "stateless_controlled_ablation",
            "history_messages": 0,
            "catalog_sizes": catalog_sizes,
            "conditions": conditions,
            "schema_profile": schema_profile,
            "task_count": len(tasks),
            "repetitions": repetitions,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "pricing_usd_per_million_tokens": rates,
            "method": "Both conditions use hosted tool search and the same inert deferred sentinel; only business-tool defer_loading differs. OpenRouter does not expose the loaded subset.",
        },
        destination,
        catalog_sizes,
        schema_profile,
    )
    print(f"Experiment saved to: {destination}")
    return destination
