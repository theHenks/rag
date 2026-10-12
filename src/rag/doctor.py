"""Checks that the services rag depends on are present and consistent."""
from __future__ import annotations

import shutil
from pathlib import Path

from rag.config import Config


def run_checks(config: Config) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            results.append((name, True, fn() or "ok"))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    def qdrant():
        from rag.embed import ModelEmbedder
        from rag.index import Index
        Index(config, ModelEmbedder(config).dense_size).ensure_collection()
        return f"{config.qdrant_url}, collection {config.collection!r} matches the configuration"

    def mineru():
        from rag.parse import mineru_version
        models = Path.home() / ".mineru" / "models"
        if not models.exists():
            raise FileNotFoundError(f"{models} is missing; run `uv run mineru-kit models download --tier "
                                    f"{config.mineru_tier} --source huggingface`")
        return f"mineru {mineru_version()}, models in {models}"

    def ollama():
        from rag.enrich import check_vision
        check_vision(config)
        return f"{config.vision_model} at {config.ollama_url}"

    def disk():
        config.data_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(config.data_dir).free / 1e9
        if free < 20:
            raise RuntimeError(f"only {free:.0f} GB free under {config.data_dir}")
        return f"{free:.0f} GB free"

    check("mineru", mineru)
    check("qdrant", qdrant)
    check("ollama (only needed for --enrich)", ollama)
    check("disk", disk)
    return results
