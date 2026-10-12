# rag — design

Date: 2026-10-12

## Purpose

A local retrieval system over a personal library of PDFs: conference talks
downloaded by `indico-rag`, papers, theses, and notes. Its main consumer is an
agent (Claude Code) that queries it through MCP and writes answers with
citations; asking a question means asking the agent. The same functions are
available from a command line.

It is used by one person on one machine (Apple M3 Max, 96 GB). Hosting it for
others is not a goal; moving the vector store to a hosted Qdrant later should
need configuration, not code changes.

### Success criteria

- An agent can find the right document and page for concept questions, exact
  terms (isotopes, run numbers, names), and filtered questions ("LEGEND
  meetings since 2025", "papers only").
- Every result carries enough to cite it: title, page range, local path, and
  the Indico URL for talks.
- Adding a folder from the agent does not block the agent, and every file that
  could not be indexed is reported with its error.
- On the evaluation set (see Testing), the correct page is in the top 8 results
  for most questions; the exact threshold is set once the set exists.

### Out of scope

Generating answers with a local LLM; watched folders; non-PDF inputs; page-image
embeddings (ColPali-style multivectors); re-ranking (revisited only if the
evaluation shows a need); multi-user access and authentication.

## Architecture

```
            ┌──────────── rag (Python, uv) ────────────┐
 Claude ──► │ mcp_server ─┐                            │
 Code       │ cli ────────┼─► search ─► embed ─► index ─┼──► Qdrant (Docker)
            │             └─► jobs ──► worker process   │
            │                   parse ─► metadata ─►    │
            │                   chunk ─► enrich ─►      │──► Ollama (gemma4)
            │                   embed ─► index          │
            │ catalog (SQLite)   MinerU cache (files)   │
            └───────────────────────────────────────────┘
```

- **Qdrant** runs as a local server in Docker with a named volume
  (`docker-compose.yml` in the repository), so the MCP server, the CLI, and the
  ingest worker can use it at the same time.
- **MinerU 4** converts PDFs into typed blocks (text, title, table, equation,
  figure) with page numbers and bounding boxes, including OCR for scanned pages.
- **Dense embeddings** come from a small local model on MPS, loaded in Python.
  Candidates are Qwen3-Embedding-0.6B and bge-m3; the choice is made with the
  evaluation set during implementation.
- **Sparse embeddings** are BM25, with the IDF computed by Qdrant
  (`modifier: idf`).
- **Figure descriptions** come from a vision model served by Ollama (gemma4 by
  default), only for documents with enrichment switched on.

### Stores

- **Catalog (SQLite)** is the authoritative record: documents, their paths,
  metadata, enrichment flag, status, and the MinerU, chunker, and embedding
  settings each was indexed with; jobs and per-file progress.
- **MinerU cache** holds MinerU's output (block JSON, cropped figure images,
  markdown) per file checksum and MinerU version.
- **Qdrant** is a derived index. `rag rebuild` recreates it from the catalog
  and the cache without running MinerU.

The catalog and the cache live under `platformdirs.user_data_dir("rag")`;
`config.toml` under `platformdirs.user_config_dir("rag")` holds the Qdrant URL,
collection name, embedding model, Ollama URL and model, and chunk size.

## Data model

### Documents

A document is identified by the SHA-256 of its file contents. A moved or
renamed file adds a path to the existing document. A changed file at a known
path is a new document; the old one is removed from the index.

Metadata:

| field | indexed in Qdrant | source |
| --- | --- | --- |
| `document_id` (checksum) | yes | computed |
| `source_type`: `talk`, `paper`, `thesis`, `note`, `other` | yes | Indico manifest → `talk`; otherwise given at `add`, default `other` |
| `title` | no (text) | manifest, then MinerU title block, then PDF metadata, then filename |
| `authors` | yes | manifest speakers, PDF metadata, or given at `add` |
| `date` | yes (datetime) | manifest start time, PDF metadata, or given at `add` |
| `tags` | yes | given at `add` |
| `event_id`, `event_title`, `session` | yes | Indico manifest |
| `indico_url` | no | Indico manifest |
| `paths` | no | catalog |

A PDF belongs to an Indico event when a `_manifest.json` written by
`indico-rag` in it or one of its ancestor directories (up to the folder passed
to `add`) lists that file's checksum. Values given at `add` override automatic
ones.

### Chunks

Chunks are built from MinerU blocks:

- **Slide decks** (documents from an Indico manifest, or whose pages are all
  landscape): one chunk per page.
- **Other documents:** consecutive blocks under the same heading are grouped up
  to the configured size (default 450 tokens of the embedding model's
  tokenizer). A table or equation block is never split. A table larger than
  one chunk is split by rows, repeating the header row in each part.
- **Figures** (enrichment on): one `figure` chunk per figure block, holding the
  caption and the vision model's description.

The text that is embedded starts with a header line
`<document title> › <section heading>`; the stored chunk text does not include
it.

Each chunk's payload holds: `document_id`, `chunk_index`, `kind`
(`slide`, `text`, `table`, `figure`), `page_start`, `page_end`, block bounding
boxes, the stored text, and the document's indexed metadata fields.

The point ID is a UUIDv5 of the checksum, the chunk index, and the chunker
version, so writing a document again replaces its points rather than adding
duplicates.

### Collection settings

The Qdrant collection records, in a reserved point or collection metadata, the
embedding model name, its vector size, the chunker version, and the sparse
model. On startup, every command compares them with the configuration and
stops with an error naming both values when they differ; the remedy is
`rag rebuild`.

## Ingestion

### Pipeline per file

```
discover → hash → catalog → parse → metadata → chunk → [enrich] → embed → write → indexed
```

- **discover:** PDFs under the given path. Other files are listed in the job
  report as skipped.
- **hash / catalog:** a checksum already indexed with the current settings is
  skipped (a new path is recorded).
- **parse:** MinerU runs only if the cache has no entry for this checksum and
  MinerU version.
- **write:** delete the document's points by `document_id`, then upsert the
  new ones. The status becomes `indexed` only after the upsert succeeds.

### Jobs and the worker

- `add_documents` (MCP) and `rag add --background` record a job in the catalog
  and start a detached `rag worker` process if none is running, then return the
  job ID. `rag add` without `--background` runs the same code in the
  foreground with a progress display.
- At most one worker runs at a time, enforced by a lock in the catalog. It
  processes queued jobs in order and exits when the queue is empty. Jobs are
  serial so that MinerU, the embedding model, and the vision model do not
  compete for GPU memory.
- The worker records per-file stage and outcome in the catalog;
  `ingest_status` reads only the catalog and works across MCP server restarts.
- A job marked `running` whose worker is no longer alive is reported as
  `interrupted` and resumed; completed files are skipped by checksum.

### Failures

- **Per file:** an exception marks that file `failed` with the stage and the
  exception text; the job continues. A job with failures ends as
  `completed_with_failures` and lists them. Running `add` on the same path
  again retries only the failed and new files. Nothing is retried
  automatically.
- **Whole job:** Qdrant unreachable, collection settings not matching the
  configuration, MinerU not importable, or Ollama unreachable while enrichment
  is on stop the job immediately with status `failed` and the error.

### Enrichment and removal

- `enrich(document_ids)` sets the flag and runs enrich → embed → write on the
  cached MinerU output, adding `figure` chunks.
- `remove_document` deletes the catalog entry and its points. A document whose
  files no longer exist on disk is not removed automatically; `rag status`
  lists it.

## Search

One Qdrant query with two prefetches, a dense search and a BM25 search of 50
candidates each, both under the same payload filter, fused with reciprocal rank
fusion. At most `max_per_document` results (default 3) come from one document.

A result contains: chunk text, score, `chunk_id`, `document_id`, `title`,
`source_type`, `authors`, `date`, `page_start`, `page_end`, the first existing
local path, and a link: `indico_url` for talks, otherwise
`file://<path>#page=<page_start>`.

## Interfaces

### MCP tools (`rag mcp`, stdio)

| tool | purpose |
| --- | --- |
| `search(query, limit=8, source_type?, tags?, event?, authors?, date_from?, date_to?, document_id?, max_per_document=3)` | hybrid search with filters |
| `read_context(chunk_id, before=1, after=1)` | neighboring chunks of a result |
| `get_document(document_id)` | metadata, heading outline, page count, status |
| `list_documents(filters…)` | browse the catalog |
| `add_documents(path, source_type?, tags?, enrich=false)` | queue an ingest job; returns its ID |
| `ingest_status(job_id?)` | progress and failures of one job, or recent jobs |
| `enrich(document_ids)` | add figure descriptions to indexed documents |
| `remove_document(document_id)` | remove from catalog and index |

### CLI

```
rag add <path> [--type T] [--tag X]… [--enrich] [--background]
rag search "<query>" [--type T] [--event E] [--tag X] [--limit N]
rag status [job-id]
rag enrich <document-id>…
rag rm <document-id>
rag rebuild
rag doctor        checks Qdrant, MinerU, Ollama, collection settings
rag eval          runs the evaluation set
rag mcp           MCP server (started by Claude Code)
rag worker        ingest worker (started by add)
```

## Testing

- **Chunker and metadata** run on real MinerU output, captured once from a
  public NOW 2026 slide deck, a CC-licensed arXiv paper with tables and
  equations, and a scanned page, and committed as fixtures. No collaboration
  material is committed.
- **Index and search** run against a real Qdrant in Docker using a temporary
  collection, marked `slow`.
- **Worker:** queue order, the single-worker lock, resuming after a killed
  worker, per-file failure isolation, and whole-job errors.
- **MCP:** through the MCP SDK's in-process client, checking tool schemas and
  result shapes.
- **Fail-fast errors** (settings mismatch, Qdrant down, Ollama missing) are
  asserted by message.
- **Evaluation:** about 30 questions with the expected document and page,
  stored in the data directory (not in the repository). `rag eval` reports
  recall@k and the rank of the first correct result. It decides the embedding
  model and whether re-ranking is needed.

## Repository layout

```
rag/
├── src/rag/
│   ├── catalog.py      SQLite: documents, jobs, settings
│   ├── parse.py        MinerU call and cache
│   ├── metadata.py     Indico manifest, PDF metadata, overrides
│   ├── chunk.py        blocks → chunks
│   ├── enrich.py       figure descriptions via Ollama
│   ├── embed.py        dense model and BM25
│   ├── index.py        collection, write, delete, settings check
│   ├── search.py       hybrid query, filters, per-document cap
│   ├── jobs.py         queue and worker
│   ├── mcp_server.py
│   └── cli.py
├── tests/
├── docker-compose.yml
└── README.md
```

## Licensing

MinerU's code is under the MinerU Open Source License: Apache 2.0 plus a
commercial license requirement above 100 million monthly active users or USD 20
million monthly revenue, and a requirement to state prominently that MinerU is
used when offering an online service based on it to third parties. Neither
affects personal use; a hosted deployment would need the attribution. `rag`
calls MinerU as a dependency and can carry any permissive license.

The license of the model weights MinerU downloads is separate from the code
license and is checked when the MinerU backend is pinned, as the first
implementation step.
