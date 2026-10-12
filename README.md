# rag

Hybrid search over a personal PDF library: conference talks downloaded with
[indico-rag](../indico-timetable-tool/indico-rag), papers, theses and notes. Claude Code
queries it through MCP; the same functions are available as the `rag` command.

PDFs are parsed once with [MinerU](https://github.com/opendatalab/MinerU) into text,
tables, equations and figures, split into chunks (one per slide for talks), and stored
in [Qdrant](https://qdrant.tech) with a dense vector (Qwen3-Embedding-0.6B) and a BM25
vector. A search runs both and merges the rankings, so it finds paraphrases as well as
exact terms such as isotopes, run numbers and names.

## Setup

Needs macOS on Apple Silicon, Python 3.12, [uv](https://docs.astral.sh/uv/), and, for
figure descriptions, [Ollama](https://ollama.com) with a vision model.

```bash
mkdir .venv.nosync && ln -s .venv.nosync .venv   # see below
uv sync
scripts/qdrant start                       # Qdrant on 127.0.0.1:6333
uv run mineru-kit models download --tier standard --source huggingface
uv run rag doctor
claude mcp add --scope user rag -- uv --directory "$PWD" run rag mcp
```

`scripts/qdrant` downloads a pinned Qdrant release once and runs it in the
background; its storage and log are under `~/Library/Application Support/rag/qdrant`.
It does not start at login: run `scripts/qdrant start` after a reboot.

If the repository is inside an iCloud-synced folder such as `~/Desktop`, the virtual
environment must not be synced: iCloud evicts its files when the disk is full and
reading an evicted file blocks, which hangs `uv run` and Python imports. iCloud skips
names ending in `.nosync`, hence the symlink.

MinerU's models take about 2 GB under `~/.mineru`, the embedding model about 1.2 GB
under `~/.cache/huggingface`, and MinerU's output roughly 1–5 MB per document under
the data directory.

## Use

From Claude Code: "add ~/Desktop/indico-talks/2357_legend-collaboration-meeting-in-naples
with the tag legend", then ask questions. Adding runs in the background; ask for its
status.

```bash
rag add <path> [--type paper] [--tag legend] [--enrich] [--background]
rag search "Tl-208 background in PPC detectors" [--type talk] [--event 2357] [--since 2025-01-01]
rag status [job-id]
rag enrich <document-id>…        describe figures with the vision model (~10–30 s per figure)
rag rm <document-id>
rag rebuild                      re-index from cached MinerU output after a model change
rag eval [questions.jsonl]       recall@k and MRR over your own questions
```

A talk is recognized from the `_manifest.json` that indico-rag writes into an event
folder; its title, speakers, session, start time and Indico link come from there. The
manifest must contain an `event` block, which indico-rag writes since it records the
event host and title.

## Configuration

`~/Library/Application Support/rag/config.toml` (or the file named by `$RAG_CONFIG`).
Keys: `qdrant_url`, `collection`, `dense_model`, `sparse_model`, `device`,
`chunk_tokens`, `mineru_tier`, `ollama_url`, `vision_model`, `data_dir`. An unknown
key is an error.

The collection records the embedding model and chunker it was built with. If the
configuration changes either, every command stops with an error until `rag rebuild`
has re-indexed the library.

## Evaluating retrieval

Write questions you would actually ask to `eval.jsonl` in the data directory, one per
line: `{"question": "...", "document": "<path to the PDF>", "page": 12}` (`page` may
be `null`). `rag eval` reports how often the right page is among the top 1, 3 and 8
results. To compare embedding models, point a second config file at another
collection and model, run `RAG_CONFIG=other.toml rag rebuild`, then `rag eval` with
each.

## Tests

`uv run pytest` needs Qdrant running and fails if it is not. `uv run pytest -m slow`
additionally runs MinerU, the embedding model and Ollama.

## License

MinerU is distributed under the MinerU Open Source License (Apache 2.0 with
additional conditions); offering a service built on it to others requires stating
that MinerU is used.
