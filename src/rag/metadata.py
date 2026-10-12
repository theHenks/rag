"""Document metadata from indico-rag manifests, MinerU output, and user overrides.

Precedence for every field: override, then manifest, then the document itself,
then a fallback (the filename for the title, "other" for the source type).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.parse import Parsed

SOURCE_TYPES = ("talk", "paper", "thesis", "note", "other")
MANIFEST_NAME = "_manifest.json"

_GENERIC_TITLE = re.compile(
    r"^(powerpoint presentation|presentation|untitled.*|slide \d+|microsoft (word|powerpoint) - .*"
    r"|.*\.(pptx?|docx?|key|tex|pdf|odp))$",
    re.IGNORECASE,
)


class OldManifest(ValueError):
    pass


@dataclass(frozen=True)
class Overrides:
    source_type: str | None = None
    tags: tuple[str, ...] = ()
    title: str | None = None
    authors: tuple[str, ...] = ()
    date: str | None = None

    def __post_init__(self):
        if self.source_type is not None and self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {', '.join(SOURCE_TYPES)}, not {self.source_type!r}")

    @classmethod
    def from_args(cls, args: dict) -> "Overrides":
        return cls(source_type=args.get("source_type"), tags=tuple(args.get("tags") or ()),
                   title=args.get("title"), authors=tuple(args.get("authors") or ()), date=args.get("date"))

    def to_args(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class ManifestTalk:
    title: str
    speakers: list[str]
    session: str
    start: str | None
    indico_url: str
    event_id: str
    event_title: str


@dataclass(frozen=True)
class Metadata:
    title: str
    source_type: str
    authors: list[str]
    date: str | None
    tags: list[str]
    event_id: str | None
    event_title: str | None
    session: str | None
    indico_url: str | None
    slides: bool


def find_manifest_talk(pdf: Path, sha256: str, root: Path) -> ManifestTalk | None:
    """The manifest entry for this file's checksum, looking in the PDF's directory
    and its ancestors up to and including `root`."""
    root = root.resolve()
    directory = pdf.resolve().parent
    while True:
        manifest = directory / MANIFEST_NAME
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for record in payload.get("talks", {}).values():
                if record.get("sha256") != sha256:
                    continue
                event = payload.get("event")
                if event is None:
                    raise OldManifest(
                        f"{manifest} has no event block; re-run `indico-rag fetch` on that event "
                        "with a current indico-rag to add it (downloaded talks are skipped)")
                return ManifestTalk(
                    title=record["title"], speakers=list(record.get("speakers") or []),
                    session=record.get("session") or "", start=record.get("start"),
                    indico_url=event["host"] + record["indico_url"],
                    event_id=str(event["id"]), event_title=event["title"],
                )
        if directory == root or directory == directory.parent:
            return None
        directory = directory.parent


def _first_heading(parsed: Parsed) -> str | None:
    first_page = [b for b in parsed.blocks if b.kind == "title" and b.page == 0]
    if not first_page:
        return None
    return min(first_page, key=lambda b: b.level or 99).text


def _usable(title: str | None) -> str | None:
    if title and title.strip() and not _GENERIC_TITLE.match(title.strip()):
        return title.strip()
    return None


def resolve_metadata(pdf: Path, parsed: Parsed, talk: ManifestTalk | None,
                     overrides: Overrides) -> Metadata:
    title = (overrides.title or (talk.title if talk else None) or _usable(_first_heading(parsed))
             or _usable(parsed.title) or pdf.stem)
    source_type = overrides.source_type or ("talk" if talk else "other")
    authors = list(overrides.authors) or (talk.speakers if talk else []) or parsed.authors
    date = overrides.date or (talk.start if talk else None) or parsed.created_at
    return Metadata(
        title=title, source_type=source_type, authors=authors, date=date, tags=list(overrides.tags),
        event_id=talk.event_id if talk else None, event_title=talk.event_title if talk else None,
        session=talk.session if talk else None, indico_url=talk.indico_url if talk else None,
        slides=talk is not None or parsed.landscape,
    )
