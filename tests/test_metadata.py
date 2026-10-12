import json
from pathlib import Path

import pytest

from rag.metadata import ManifestTalk, OldManifest, Overrides, find_manifest_talk, resolve_metadata
from rag.parse import Block, Parsed

EVENT = {"host": "https://agenda.infn.it", "id": "50058", "title": "NOW 2026",
         "url": "https://agenda.infn.it/event/50058/"}


def _manifest(directory: Path, sha: str, event=EVENT):
    payload = {"counts": {}, "talks": {"1": {
        "title": "JUNO results", "speakers": ["A. Caccianiga"], "session": "Session I",
        "start": "2026-09-01T09:00:00", "indico_url": "/event/50058/contributions/1/", "sha256": sha}}}
    if event:
        payload["event"] = event
    (directory / "_manifest.json").write_text(json.dumps(payload))


def _parsed(title=None, blocks=(), pages=((612.0, 792.0),), authors=()):
    return Parsed(directory=Path("/x"), title=title, authors=list(authors), created_at="2023-02-13T02:16:23Z",
                  pages=list(pages), blocks=list(blocks))


def test_manifest_in_an_ancestor_directory_is_found(tmp_path):
    day = tmp_path / "50058_now" / "2026-09-01" / "Session-I"
    day.mkdir(parents=True)
    _manifest(tmp_path / "50058_now", "abc")
    talk = find_manifest_talk(day / "talk.pdf", "abc", root=tmp_path)
    assert talk.indico_url == "https://agenda.infn.it/event/50058/contributions/1/"
    assert (talk.event_id, talk.event_title, talk.speakers) == ("50058", "NOW 2026", ["A. Caccianiga"])


def test_search_stops_at_the_root(tmp_path):
    _manifest(tmp_path, "abc")
    inner = tmp_path / "inner"
    inner.mkdir()
    assert find_manifest_talk(inner / "talk.pdf", "abc", root=inner) is None


def test_a_checksum_not_in_the_manifest_is_not_a_talk(tmp_path):
    _manifest(tmp_path, "abc")
    assert find_manifest_talk(tmp_path / "talk.pdf", "zzz", root=tmp_path) is None


def test_old_manifest_is_reported_not_ignored(tmp_path):
    _manifest(tmp_path, "abc", event=None)
    with pytest.raises(OldManifest, match="re-run `indico-rag fetch`"):
        find_manifest_talk(tmp_path / "talk.pdf", "abc", root=tmp_path)


def test_talk_metadata_comes_from_the_manifest():
    talk = ManifestTalk(title="JUNO results", speakers=["A. Caccianiga"], session="Session I",
                        start="2026-09-01T09:00:00", indico_url="https://h/event/1/contributions/1/",
                        event_id="1", event_title="NOW 2026")
    meta = resolve_metadata(Path("/x/talk.pdf"), _parsed(pages=((720.0, 540.0),)), talk, Overrides(tags=("now",)))
    assert (meta.title, meta.source_type, meta.authors, meta.date) == (
        "JUNO results", "talk", ["A. Caccianiga"], "2026-09-01T09:00:00")
    assert meta.slides and meta.tags == ["now"] and meta.event_title == "NOW 2026"


def test_generic_pdf_title_is_not_used():
    heading = Block("title", "Majorana Demonstrator final result", 0, (0, 0, 1, 1), level=1)
    meta = resolve_metadata(Path("/x/paper.pdf"), _parsed(title="PowerPoint Presentation", blocks=[heading]),
                            None, Overrides())
    assert meta.title == "Majorana Demonstrator final result"
    meta = resolve_metadata(Path("/x/my-thesis.pdf"), _parsed(title="Microsoft Word - thesis.docx"), None, Overrides())
    assert meta.title == "my-thesis"


def test_overrides_win_and_defaults_apply():
    meta = resolve_metadata(Path("/x/p.pdf"), _parsed(title="Real Title", authors=["PDF Author"]), None,
                            Overrides(source_type="paper", authors=("Given Author",)))
    assert (meta.title, meta.source_type, meta.authors) == ("Real Title", "paper", ["Given Author"])
    assert meta.date == "2023-02-13T02:16:23Z" and not meta.slides
    assert resolve_metadata(Path("/x/p.pdf"), _parsed(), None, Overrides()).source_type == "other"


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValueError, match="source_type must be one of"):
        Overrides(source_type="slides")
