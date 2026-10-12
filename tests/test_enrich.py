from pathlib import Path

import pytest

from rag.config import Config
from rag.enrich import VisionUnavailable, check_vision, describe_figures, figures_worth_describing
from rag.parse import Block, Parsed, load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


def test_logos_and_tiny_images_are_not_described(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "logo.jpg").write_bytes(b"same logo")
    for page in range(3):
        (tmp_path / "images" / f"logo{page}.jpg").write_bytes(b"same logo")
    (tmp_path / "images" / "plot.jpg").write_bytes(b"a plot")
    (tmp_path / "images" / "tiny.jpg").write_bytes(b"tiny")
    blocks = [Block("figure", "", p, (0, 0, 300, 300), image=f"images/logo{p}.jpg") for p in range(3)]
    blocks += [Block("figure", "", 0, (0, 0, 500, 500), image="images/plot.jpg"),
               Block("figure", "", 1, (0, 0, 100, 100), image="images/tiny.jpg"),
               Block("figure", "", 1, (0, 0, 500, 500), image=None)]
    parsed = Parsed(directory=tmp_path, title=None, authors=[], created_at=None,
                    pages=[(720.0, 540.0)] * 3, blocks=blocks)
    assert [b.image for b in figures_worth_describing(parsed)] == ["images/plot.jpg"]


def test_unreachable_ollama_names_the_url(tmp_path):
    with pytest.raises(VisionUnavailable, match="http://localhost:1"):
        check_vision(Config(data_dir=tmp_path, ollama_url="http://localhost:1"))


def test_missing_model_names_the_pull_command(tmp_path):
    with pytest.raises(VisionUnavailable, match="ollama pull no-such-model:1b"):
        check_vision(Config(data_dir=tmp_path, vision_model="no-such-model:1b"))


@pytest.mark.slow
def test_real_description_of_the_paper_chart(tmp_path):
    parsed = load_parsed(FIX / "paper")
    # The fixture keeps only this chart's image, so the document is narrowed to it first.
    chart = [b for b in parsed.blocks if b.image == "images/page_3_chart_1.jpg"]
    only_chart = Parsed(parsed.directory, parsed.title, parsed.authors, parsed.created_at, parsed.pages, chart)
    [description] = describe_figures(only_chart, Config(data_dir=tmp_path)).values()
    assert "spectrum" in description.lower() or "energy" in description.lower()


def test_a_slow_figure_fails_that_figure_not_the_job(tmp_path):
    import httpx
    from rag.enrich import describe_figure

    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)
    image = tmp_path / "f.jpg"
    image.write_bytes(b"jpg")
    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(httpx.ReadTimeout):
            describe_figure(image, "", config=Config(data_dir=tmp_path), client=client)
