"""Run tasks against models through OpenRouter and record one CSV row per task.

A task is retried (up to ``max_attempts``) until an LLM judge says PASS. Cost,
tokens and latency are accumulated across all attempts, so the resulting rows
reflect the *real* price of getting one correct answer.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI

from .tasks import DATA_DIR, Task, load_tasks, tasks_fingerprint

OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_ANSWER_TOKENS = 256
# The task set is Turkish, so the retry hint and judge prompt stay Turkish too.
RETRY_NOTE = (
    "\nÖnceki yanıt doğrulayıcıdan geçmedi. Hesabı ve istenen çıktıyı yeniden "
    "kontrol et; yalnızca doğru nihai cevabı ver."
)
JUDGE_TEMPLATE = (
    "Görev: {prompt}\nReferans cevap: {expected}\nAday cevap: {answer}\n"
    "Aday cevap görevi doğru yerine getiriyorsa yalnız PASS, aksi halde yalnız FAIL yaz."
)


@dataclass
class RunRecord:
    run_id: str
    model: str
    task_id: str
    repeat: int
    success: bool
    first_attempt_success: bool
    retry_count: int
    llm_calls: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    answer: str
    judge_model: str
    judge_verdict: str
    error: str
    timestamp: str


FIELDS = [f.name for f in fields(RunRecord)]


def extract_usage(message) -> tuple[int, int, int, float]:
    """Return ``(input, output, reasoning, cost_usd)`` from a LangChain message."""
    meta = message.response_metadata
    usage = meta.get("token_usage") or message.usage_metadata or {}
    return (
        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0,
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0,
        usage.get("reasoning_tokens", 0) or 0,
        float(usage.get("cost", meta.get("cost", 0)) or 0),
    )


def make_chat(model: str, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_URL,
        temperature=0,
        max_tokens=max_tokens,
        max_retries=0,
    )


def run_task(llm, judge, task: Task, *, model: str, judge_model: str,
             repeat: int, max_attempts: int) -> RunRecord:
    """Attempt one task until the judge passes it or attempts run out."""
    start = time.perf_counter()
    answer = error = ""
    verdict = "FAIL"
    first_attempt = False
    calls = 0
    tokens = [0, 0, 0]
    cost = 0.0

    for attempt in range(1, max_attempts + 1):
        calls += 1
        try:
            reply = llm.invoke(task.prompt + ("" if attempt == 1 else RETRY_NOTE))
            answer = str(reply.content).strip()
            *used, spent = extract_usage(reply)
            tokens = [total + part for total, part in zip(tokens, used)]
            cost += spent
            verdict = str(judge.invoke(JUDGE_TEMPLATE.format(
                prompt=task.prompt, expected=task.expected, answer=answer)).content).strip().upper()
        except Exception as exc:  # an API failure ends this task, not the whole run
            error = f"{type(exc).__name__}: {exc}"
            break
        if verdict.startswith("PASS"):
            first_attempt = attempt == 1
            break

    return RunRecord(
        run_id=str(uuid.uuid4()), model=model, task_id=task.id, repeat=repeat,
        success=verdict.startswith("PASS") and not error,
        first_attempt_success=first_attempt, retry_count=calls - 1, llm_calls=calls,
        latency_seconds=round(time.perf_counter() - start, 3),
        input_tokens=tokens[0], output_tokens=tokens[1], reasoning_tokens=tokens[2],
        cost_usd=cost, answer=answer, judge_model=judge_model, judge_verdict=verdict,
        error=error, timestamp=datetime.now().isoformat(),
    )


def write_manifest(directory: Path, args, tasks: list[Task], categories: dict[str, str],
                   catalog: dict) -> None:
    manifest = {
        "experiment_id": directory.name,
        "started_at": datetime.now().astimezone().isoformat(),
        "description": args.description,
        "architecture": args.architecture,
        "models": args.model_list,
        "parameters": {
            "temperature": 0, "max_tokens": MAX_ANSWER_TOKENS,
            "max_attempts": args.max_attempts, "max_tasks": args.max_tasks,
            "repetitions": args.repetitions,
        },
        "judge": {
            "model": args.judge_model, "max_tokens": args.judge_max_tokens,
            "cost_included_in_target_metrics": False,
            "rubric": "Return PASS only if candidate correctly fulfills the task; otherwise FAIL.",
        },
        "eval_set": {
            "task_count": len(tasks), "categories": categories,
            "tasks": [{"id": t.id, "prompt": t.prompt, "expected_output": t.expected} for t in tasks],
            "sha256": tasks_fingerprint(tasks),
        },
        "pricing_catalog": catalog,
        "output": "runs.csv",
        "metrics": FIELDS,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (directory / "experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(args) -> Path:
    """Run every model x repetition x task and return the experiment directory."""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (see .env.example).")

    tasks, categories = load_tasks()
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise SystemExit("--max-tasks must be at least 1.")

    catalog = json.loads((DATA_DIR / "models.json").read_text(encoding="utf-8"))
    directory = Path(args.output_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{args.name}"
    directory.mkdir(parents=True)
    write_manifest(directory, args, tasks, categories, catalog)

    judge = make_chat(args.judge_model, args.judge_max_tokens)
    with (directory / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for model in args.model_list:
            llm = make_chat(model, MAX_ANSWER_TOKENS)
            for repeat in range(1, args.repetitions + 1):
                for task in tasks:
                    record = run_task(llm, judge, task, model=model, judge_model=args.judge_model,
                                      repeat=repeat, max_attempts=args.max_attempts)
                    writer.writerow(asdict(record))
                    handle.flush()
                    print(model, task.id, "PASS" if record.success else "FAIL")
    print("Experiment:", directory)
    return directory
