# Working on rag

## Shape

```
parse.py     MinerU → cached blocks          the only module that knows MinerU's formats
metadata.py  indico-rag manifest, overrides  the only module that knows the manifest
chunk.py     blocks → chunks
embed.py     dense + BM25
index.py     Qdrant                          the only module that talks to Qdrant
ingest.py    one document end to end
jobs.py      queue + single worker
search.py, mcp_server.py, cli.py on top
```

## Facts the code depends on

- MinerU 4's in-process API is `mineru.parse(path, tier=...)`. Its content list is
  rendered from the saved middle JSON via `ParseResult.from_json(...)`; rendering
  from the live result embeds images as base64. Bounding boxes are 0–1000 integers,
  `page_idx` is 0-based. Slide titles carry `text_level`; paper section headings
  often do not. Logos can carry their institute name as an `image_footnote`.
- MinerU starts PDF render workers with `spawn`; anything that may parse must be
  run under `if __name__ == "__main__":` or the workers die with "PDF render worker
  disconnected".
- The NOW 2026 deck has 30 images on 16 pages, most of them logos; that is why
  `enrich.figures_worth_describing` drops small and repeated images.
- MinerU's table HTML garbles values with super- and subscripts. The table image is
  kept in the cache.
- The `mcp` package 2.x has no `FastMCP`; the server class is
  `mcp.server.mcpserver.MCPServer`, and `mcp.Client(server)` tests it in process.
  Synchronous tools run in worker threads. A tool returning a bare `dict` gives no
  structured result; `dict[str, Any]` does. Only `ToolError` messages reach the client.
- Ollama answers gemma4 about twice as fast with `"think": false`.
- Files under `~/Desktop` may be iCloud placeholders; reading one downloads it, and
  on a full disk iCloud also evicts files of a virtual environment kept there.
- Docker Desktop on this network inherits a broken WPAD proxy and cannot pull
  images, so Qdrant runs as the native binary (`scripts/qdrant`).

## Tests

Test fixtures are MinerU output for a public NOW 2026 deck and arXiv:2207.07638
(CC BY 4.0). Never commit collaboration-internal documents or their parses.
