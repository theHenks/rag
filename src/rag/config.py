"""Settings, read from a TOML file whose keys are the fields of `Config`."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import platformdirs


def _default_data_dir() -> Path:
    return Path(platformdirs.user_data_dir("rag"))


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=_default_data_dir)
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rag"
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    sparse_model: str = "Qdrant/bm25"
    device: str = "mps"
    chunk_tokens: int = 450
    mineru_tier: str = "standard"
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "gemma4:31b-mlx"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.sqlite3"

    @property
    def cache_dir(self) -> Path:
        """MinerU output, one directory per document checksum."""
        return self.data_dir / "mineru"

    @property
    def models_dir(self) -> Path:
        """fastembed's model cache; its default lives under $TMPDIR, which macOS clears."""
        return self.data_dir / "models"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "worker.log"


def config_path() -> Path:
    """`$RAG_CONFIG` if set, else `config.toml` in the platform config directory."""
    override = os.environ.get("RAG_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_config_dir("rag")) / "config.toml"


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"unknown keys in {path}: {', '.join(unknown)} (known keys: {', '.join(sorted(known))})"
        )
    if "data_dir" in raw:
        raw["data_dir"] = Path(raw["data_dir"]).expanduser()
    return Config(**raw)
