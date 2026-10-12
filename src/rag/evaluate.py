"""Retrieval quality over a set of questions with known answers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.ingest import Services, sha256_of
from rag.search import search


@dataclass(frozen=True)
class Question:
    question: str
    document: str
    page: int | None


@dataclass
class Report:
    recall: dict[int, float]
    mrr: float
    misses: list[str]


def load_questions(path: Path) -> list[Question]:
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            questions.append(Question(raw["question"], raw["document"], raw.get("page")))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: expected {{question, document, page}}: {exc}") from exc
    return questions


def _document_id(document: str) -> str:
    path = Path(document).expanduser()
    return sha256_of(path) if path.exists() else document


def evaluate(services: Services, questions: list[Question], ks: tuple[int, ...] = (1, 3, 8)) -> Report:
    found = {k: 0 for k in ks}
    reciprocal = 0.0
    misses = []
    for q in questions:
        target = _document_id(q.document)
        hits = search(services, q.question, limit=max(ks))
        rank = next((i for i, h in enumerate(hits, start=1) if h.document_id.startswith(target)
                     and (q.page is None or h.page_start <= q.page <= h.page_end)), None)
        if rank is None:
            misses.append(q.question)
            continue
        reciprocal += 1 / rank
        for k in ks:
            found[k] += rank <= k
    n = len(questions) or 1
    return Report(recall={k: found[k] / n for k in ks}, mrr=reciprocal / n, misses=misses)
