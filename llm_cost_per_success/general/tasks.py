"""Evaluation task set loaded from ``data/tasks.json``."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    expected: str


def load_tasks(path: Path = DATA_DIR / "tasks.json") -> tuple[list[Task], dict[str, str]]:
    """Return ``(tasks, category_descriptions)`` from a tasks JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw["tasks"]], raw["categories"]


def tasks_fingerprint(tasks: list[Task]) -> str:
    """SHA-256 of the task set, stored in the manifest to detect eval drift."""
    payload = json.dumps([(t.id, t.prompt, t.expected) for t in tasks], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
