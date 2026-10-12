"""Figure descriptions from a vision model served by Ollama."""
from __future__ import annotations

import base64
import hashlib
from collections import defaultdict
from pathlib import Path

import httpx

from rag.config import Config
from rag.parse import Block, Parsed

_MIN_AREA = 20_000  # 2 % of a page, in MinerU's 0–1000 × 0–1000 bbox units
_REPEATS = 3
_PROMPT = (
    "This image is a figure from a physics document. Describe what it shows in two to four "
    "sentences: the kind of plot or diagram, the quantities on its axes, and the main result "
    "it conveys. Do not speculate beyond what is visible.\n\nCaption: "
)


class VisionUnavailable(RuntimeError):
    pass


def check_vision(config: Config) -> None:
    try:
        response = httpx.get(f"{config.ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise VisionUnavailable(f"Ollama is not reachable at {config.ollama_url} ({exc}); start Ollama") from exc
    names = {m["name"] for m in response.json().get("models", [])}
    if config.vision_model not in names:
        raise VisionUnavailable(
            f"the vision model {config.vision_model} is not available in Ollama; "
            f"run `ollama pull {config.vision_model}` or set vision_model in the configuration")


def figures_worth_describing(parsed: Parsed) -> list[Block]:
    candidates = [b for b in parsed.blocks if b.kind == "figure" and b.image is not None]
    pages_by_digest: dict[str, set[int]] = defaultdict(set)
    digests: dict[Block, str] = {}
    for block in candidates:
        digest = hashlib.sha256(parsed.image_path(block).read_bytes()).hexdigest()
        digests[block] = digest
        pages_by_digest[digest].add(block.page)
    chosen = []
    for block in candidates:
        x0, y0, x1, y1 = block.bbox
        if (x1 - x0) * (y1 - y0) < _MIN_AREA:
            continue
        if len(pages_by_digest[digests[block]]) >= _REPEATS:
            continue
        chosen.append(block)
    return chosen


def describe_figure(image: Path, caption: str, *, config: Config, client: httpx.Client) -> str:
    payload = {
        "model": config.vision_model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": _PROMPT + (caption or "(none)"),
                      "images": [base64.b64encode(image.read_bytes()).decode()]}],
    }
    try:
        response = client.post(f"{config.ollama_url}/api/chat", json=payload, timeout=300)
    except httpx.TransportError as exc:
        raise VisionUnavailable(f"Ollama is not reachable at {config.ollama_url} ({exc})") from exc
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def describe_figures(parsed: Parsed, config: Config) -> dict[Block, str]:
    with httpx.Client() as client:
        return {block: describe_figure(parsed.image_path(block), block.caption, config=config, client=client)
                for block in figures_worth_describing(parsed)}
