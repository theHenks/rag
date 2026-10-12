import json
from pathlib import Path

from rag.evaluate import evaluate, load_questions
from rag.ingest import Services, ingest_file
from rag.metadata import Overrides
from rag.parse import load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


def test_recall_and_mrr(config, fake_embedder, tmp_path):
    services = Services(config, embedder=fake_embedder, parser=lambda pdf, d: load_parsed(FIX / "paper"))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF x")
    ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    questions = tmp_path / "eval.jsonl"
    questions.write_text("\n".join(json.dumps(q) for q in [
        {"question": "Majorana Demonstrator half-life limit", "document": str(pdf), "page": None},
        {"question": "nothing relevant at all", "document": "f" * 12, "page": None},
    ]))
    report = evaluate(services, load_questions(questions))
    assert report.recall[8] == 0.5 and 0 < report.mrr <= 0.5
    assert report.misses == ["nothing relevant at all"]
