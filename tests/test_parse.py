import json
from pathlib import Path

import pytest

from rag.parse import Block, UnknownBlockType, blocks_from_content_list, load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


def test_paper_blocks():
    parsed = load_parsed(FIX / "paper")
    kinds = {b.kind for b in parsed.blocks}
    assert kinds == {"title", "text", "table", "equation", "figure"}
    title = next(b for b in parsed.blocks if b.kind == "title")
    assert title.text.startswith("Final Result of the Majorana Demonstrator")
    table = next(b for b in parsed.blocks if b.kind == "table")
    assert table.rows[0] == " | PPCs | ICPCs"  # the header row's first cell is empty
    assert any(r.startswith("Total Exposure | 67.94 kg yr") for r in table.rows)
    assert table.rows[-1] == "Note: ∗A/E in the case of ICPCs"
    equation = next(b for b in parsed.blocks if b.kind == "equation")
    assert equation.text.startswith("$$ T _ {1 / 2} > \\ln (2)")
    chart = next(b for b in parsed.blocks if b.kind == "figure")
    assert chart.caption.startswith("FIG. 1.")
    assert parsed.image_path(chart).exists()
    assert not parsed.landscape


def test_page_numbers_headers_and_asides_are_dropped():
    parsed = load_parsed(FIX / "slides")
    first_page = [b.text for b in parsed.blocks if b.page == 0]
    assert "Dip. Interateneo di Fisica Università di Bari" not in first_page
    assert parsed.landscape
    assert parsed.authors == ["Paolo Bernardini"]


def test_bold_markers_and_whitespace_are_cleaned():
    [block] = blocks_from_content_list(
        [{"type": "text", "text": "**Antonio  Surdo**\n on behalf", "bbox": [0, 0, 1, 1], "page_idx": 0}])
    assert block.text == "Antonio Surdo on behalf"


def test_unknown_block_types_fail_loudly():
    with pytest.raises(UnknownBlockType, match="'hologram'"):
        blocks_from_content_list([{"type": "hologram", "page_idx": 0}])


def test_code_blocks_become_text():
    [block] = blocks_from_content_list([{"type": "code", "sub_type": "code", "code_body": "x = 1",
                                         "code_caption": ["Listing 1"], "code_footnote": [],
                                         "bbox": [0, 0, 1, 1], "page_idx": 2}])
    assert (block.kind, block.text, block.page) == ("text", "Listing 1\nx = 1", 2)


import subprocess
import sys


@pytest.mark.slow
def test_parse_pdf_runs_mineru_and_caches(tmp_path):
    pdf = Path.home() / "Desktop/indico-talks/50058_now-2026/2026-09-01" \
        / "Session-I-(plenary)-Standard-three-neutrino-oscillations/0845_Surdo_Welcome-to-NOW-2026.pdf"
    script = (
        "import sys, pathlib\n"
        "from rag.parse import parse_pdf\n"
        "if __name__ == '__main__':\n"
        "    p = parse_pdf(pathlib.Path(sys.argv[1]), 'x' * 64, cache_dir=pathlib.Path(sys.argv[2]), tier='standard')\n"
        "    print(len(p.pages), p.landscape)\n"
    )
    runner = tmp_path / "run.py"
    runner.write_text(script)
    out = subprocess.run([sys.executable, str(runner), str(pdf), str(tmp_path / "cache")],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["16", "True"]
    [directory] = (tmp_path / "cache" / ("x" * 64)).iterdir()
    assert (directory / "content_list.json").exists()
