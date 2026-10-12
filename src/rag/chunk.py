"""Blocks → chunks.

Slide decks give one chunk per page. Other documents are split into sections
at title blocks, and each section's blocks are packed greedily up to the token
budget. A block is never split, except a table larger than the budget, which
is split by rows with its header row repeated.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby

from rag.parse import Block, Parsed

CHUNKER_VERSION = 1


@dataclass(frozen=True)
class Chunk:
    index: int
    kind: str
    text: str
    heading: str
    page_start: int
    page_end: int
    bboxes: list[list[int]]


@dataclass(frozen=True)
class _Part:
    kind: str
    text: str
    heading: str
    blocks: tuple[Block, ...]


def _block_text(block: Block) -> str:
    return block.caption if block.kind == "figure" else block.text


def _split_table(block: Block, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    header, body = block.rows[0], block.rows[1:]
    pieces: list[list[str]] = [[]]
    prefix_first = ([block.caption] if block.caption else []) + [header]
    for row in body:
        prefix = prefix_first if len(pieces) == 1 else [header]
        candidate = prefix + pieces[-1] + [row]
        if pieces[-1] and count("\n".join(candidate)) > max_tokens:
            pieces.append([])
        pieces[-1].append(row)
    return ["\n".join((prefix_first if i == 0 else [header]) + rows) for i, rows in enumerate(pieces)]


def _pack(blocks: list[Block], heading: str, kind: str, max_tokens: int,
          count: Callable[[str], int]) -> list[_Part]:
    parts: list[_Part] = []
    current: list[Block] = []
    tokens = 0

    def flush():
        nonlocal current, tokens
        if current:
            only_tables = all(b.kind == "table" for b in current)
            parts.append(_Part("table" if kind == "text" and only_tables else kind,
                               "\n".join(_block_text(b) for b in current), heading, tuple(current)))
        current, tokens = [], 0

    for block in blocks:
        text = _block_text(block)
        if not text:
            continue
        n = count(text)
        if block.kind == "table" and n > max_tokens and len(block.rows) > 1:
            flush()
            for piece in _split_table(block, max_tokens, count):
                parts.append(_Part("table", piece, heading, (block,)))
            continue
        if current and tokens + n > max_tokens:
            flush()
        current.append(block)
        tokens += n
    flush()
    return parts


def _to_chunk(index: int, part: _Part) -> Chunk:
    pages = [b.page + 1 for b in part.blocks]
    return Chunk(index=index, kind=part.kind, text=part.text, heading=part.heading,
                 page_start=min(pages), page_end=max(pages),
                 bboxes=[[b.page + 1, *b.bbox] for b in part.blocks])


def chunk_document(parsed: Parsed, *, slides: bool, max_tokens: int,
                   count_tokens: Callable[[str], int]) -> list[Chunk]:
    parts: list[_Part] = []
    if slides:
        for _, page_blocks in groupby(parsed.blocks, key=lambda b: b.page):
            page_blocks = list(page_blocks)
            heading = next((b.text for b in page_blocks if b.kind == "title"), "")
            parts.extend(_pack(page_blocks, heading, "slide", max_tokens, count_tokens))
    else:
        heading, section = "", []
        for block in parsed.blocks:
            if block.kind == "title":
                parts.extend(_pack(section, heading, "text", max_tokens, count_tokens))
                heading, section = block.text, []
            else:
                section.append(block)
        parts.extend(_pack(section, heading, "text", max_tokens, count_tokens))
    return [_to_chunk(i, part) for i, part in enumerate(parts)]


def figure_chunks(parsed: Parsed, descriptions: dict[Block, str], *, start: int) -> list[Chunk]:
    chunks = []
    for offset, (block, description) in enumerate(descriptions.items()):
        page = block.page + 1
        text = f"{block.caption}\n\n{description}" if block.caption else description
        chunks.append(Chunk(index=start + offset, kind="figure", text=text,
                            heading=f"Figure on page {page}", page_start=page, page_end=page,
                            bboxes=[[page, *block.bbox]]))
    return chunks


def embedding_text(title: str, chunk: Chunk) -> str:
    header = f"{title} › {chunk.heading}" if chunk.heading else title
    return f"{header}\n{chunk.text}"
