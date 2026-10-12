from pathlib import Path

import pytest

from rag.config import Config, load_config


def test_missing_file_gives_defaults(tmp_path):
    config = load_config(tmp_path / "absent.toml")
    assert config.collection == "rag"
    assert config.dense_model == "Qwen/Qwen3-Embedding-0.6B"


def test_values_are_read_and_paths_expanded(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('collection = "test"\nchunk_tokens = 300\ndata_dir = "~/somewhere"\n')
    config = load_config(path)
    assert config.collection == "test"
    assert config.chunk_tokens == 300
    assert config.data_dir == Path.home() / "somewhere"
    assert config.catalog_path == Path.home() / "somewhere" / "catalog.sqlite3"


def test_unknown_keys_are_rejected_by_name(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('colection = "typo"\n')
    with pytest.raises(ValueError, match="unknown keys in .*config.toml: colection"):
        load_config(path)


def test_rag_config_environment_variable_selects_the_file(tmp_path, monkeypatch):
    path = tmp_path / "other.toml"
    path.write_text('collection = "from-env"\n')
    monkeypatch.setenv("RAG_CONFIG", str(path))
    assert load_config().collection == "from-env"
