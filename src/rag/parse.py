"""MinerU parsing, its on-disk cache, and the block model built from it.

A parse directory holds MinerU's own output (`middle_json.json`, `images/`, …)
plus two files written here: `content_list.json` (MinerU's flat block list)
and `meta.json` (title, authors, creation date, page sizes). Everything after
parsing reads only those two files and the images.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from html.parser import HTMLParser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class MinerUMissing(RuntimeError):
    pass


class UnknownBlockType(ValueError):
    pass


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    page: int
    bbox: tuple[int, int, int, int]
    level: int | None = None
    image: str | None = None
    caption: str = ""
    rows: tuple[str, ...] = ()


@dataclass(frozen=True)
class Parsed:
    directory: Path
    title: str | None
    authors: list[str]
    created_at: str | None
    pages: list[tuple[float, float]]
    blocks: list[Block] = field(default_factory=list)

    def image_path(self, block: Block) -> Path:
        if block.image is None:
            raise ValueError(f"block on page {block.page} has no image")
        return self.directory / block.image

    @property
    def landscape(self) -> bool:
        return bool(self.pages) and all(w > h for w, h in self.pages)


_DROPPED = {"header", "footer", "page_number", "aside_text", "index"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


class _TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(_clean("".join(self._cell)))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def table_rows(html: str) -> list[str]:
    parser = _TableRows()
    parser.feed(html)
    return [" | ".join(cells) for cells in parser.rows if any(cells)]


def blocks_from_content_list(items: list[dict]) -> list[Block]:
    """Turn MinerU's content list (v1) into blocks, dropping page furniture."""
    blocks: list[Block] = []
    for item in items:
        kind = item["type"]
        if kind in _DROPPED:
            continue
        page = item["page_idx"]
        bbox = tuple(item.get("bbox") or (0, 0, 0, 0))
        if kind in ("text", "page_footnote"):
            text = _clean(item.get("text", ""))
            if text:
                level = item.get("text_level")
                blocks.append(Block("title" if level else "text", text, page, bbox, level=level))
        elif kind in ("image", "chart"):
            caption = _clean(" ".join(item.get(f"{kind}_caption", []) + item.get(f"{kind}_footnote", [])))
            blocks.append(Block("figure", caption, page, bbox, image=item.get("img_path"), caption=caption))
        elif kind == "table":
            caption = _clean(" ".join(item.get("table_caption", [])))
            rows = table_rows(item.get("table_body", "")) + [
                f"Note: {_clean(note)}" for note in item.get("table_footnote", [])]
            text = "\n".join(([caption] if caption else []) + rows)
            blocks.append(Block("table", text, page, bbox, image=item.get("img_path"),
                                caption=caption, rows=tuple(rows)))
        elif kind == "equation":
            blocks.append(Block("equation", f"$$ {item['text']} $$", page, bbox, image=item.get("img_path")))
        elif kind == "list":
            text = "\n".join(_clean(entry) for entry in item.get("list_items", []) if _clean(entry))
            if text:
                blocks.append(Block("text", text, page, bbox))
        elif kind == "code":
            caption = _clean(" ".join(item.get("code_caption", [])))
            text = "\n".join(([caption] if caption else []) + [item.get("code_body", "")])
            blocks.append(Block("text", text.strip(), page, bbox))
        else:
            raise UnknownBlockType(f"MinerU produced a block of unknown type {kind!r} on page {page}")
    return blocks


def mineru_version() -> str:
    try:
        return version("mineru")
    except PackageNotFoundError as exc:
        raise MinerUMissing("MinerU is not installed in this environment; run `uv sync`") from exc


def parse_directory(cache_dir: Path, document_id: str, version: str, tier: str) -> Path:
    return cache_dir / document_id / f"{version}-{tier}"


def load_parsed(directory: Path) -> Parsed:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    items = json.loads((directory / "content_list.json").read_text(encoding="utf-8"))
    return Parsed(
        directory=directory, title=meta.get("title"), authors=list(meta.get("authors") or []),
        created_at=meta.get("created_at"), pages=[tuple(p) for p in meta["pages"]],
        blocks=blocks_from_content_list(items),
    )


def parse_pdf(pdf: Path, document_id: str, *, cache_dir: Path, tier: str) -> Parsed:
    """Parse `pdf` with MinerU unless its parse directory already exists.

    MinerU writes into a temporary sibling directory that is renamed into
    place only when complete, so an interrupted parse is never mistaken for
    a cached one. The caller must run under `if __name__ == "__main__":`
    because MinerU starts its PDF render workers with `spawn`.
    """
    directory = parse_directory(cache_dir, document_id, mineru_version(), tier)
    if directory.exists():
        return load_parsed(directory)

    import mineru
    from mineru.parser import ParseResult
    from mineru.parser.writer import FileBasedDataWriter
    from mineru.render import render_content_list

    staging = directory.with_name(f"{directory.name}.tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        result = mineru.parse(str(pdf), tier=tier)
        result.save(FileBasedDataWriter(str(staging)))
        text = (staging / "middle_json.json").read_text(encoding="utf-8")
        middle = json.loads(text)
        content = render_content_list(ParseResult.from_json(text).middle_json, asset_base_url="")
        document = middle["metadata"]["document"]
        meta = {
            "title": document.get("title"),
            "authors": document.get("authors", []),
            "created_at": document.get("created_at"),
            "pages": [[p["width_pt"], p["height_pt"]] for p in middle["extensions"]["docvortex_layout"]["pages"]],
        }
        (staging / "content_list.json").write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_parsed(directory)
