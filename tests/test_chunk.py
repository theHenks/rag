from pathlib import Path

from rag.chunk import Chunk, chunk_document, embedding_text, figure_chunks
from rag.parse import Block, Parsed, load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


def words(text: str) -> int:
    return len(text.split())


def _parsed(blocks, pages=((612.0, 792.0),) * 3):
    return Parsed(directory=Path("/nonexistent"), title=None, authors=[], created_at=None,
                  pages=list(pages), blocks=blocks)


def test_slides_give_one_chunk_per_page_with_text():
    parsed = load_parsed(FIX / "slides")
    chunks = chunk_document(parsed, slides=True, max_tokens=450, count_tokens=words)
    assert all(c.kind == "slide" for c in chunks)
    assert all(c.page_start == c.page_end for c in chunks)
    pages = [c.page_start for c in chunks]
    assert pages == sorted(set(pages)) and pages[0] == 1
    assert chunks[0].heading == "NOW 2026 Neutrino Oscillation Workshop"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_prose_is_packed_under_headings_up_to_the_budget():
    blocks = [
        Block("title", "Intro", 0, (0, 0, 1, 1), level=1),
        *[Block("text", "w " * 100, 0, (0, i, 1, 1)) for i in range(5)],
        Block("title", "Method", 1, (0, 0, 1, 1), level=2),
        Block("text", "m " * 50, 1, (0, 0, 1, 1)),
    ]
    chunks = chunk_document(_parsed(blocks), slides=False, max_tokens=250, count_tokens=words)
    assert [(c.heading, words(c.text)) for c in chunks] == [("Intro", 200), ("Intro", 200), ("Intro", 100), ("Method", 50)]
    assert all(c.kind == "text" for c in chunks)


def test_tables_and_equations_are_never_split_across_chunks():
    blocks = [Block("text", "a " * 200, 0, (0, 0, 1, 1)),
              Block("equation", "$$ E = mc^2 $$", 0, (0, 0, 1, 1)),
              Block("table", "h1 | h2\n1 | 2", 1, (0, 0, 1, 1), rows=("h1 | h2", "1 | 2"))]
    chunks = chunk_document(_parsed(blocks), slides=False, max_tokens=202, count_tokens=words)
    assert [c.text for c in chunks][1:] == ["$$ E = mc^2 $$\nh1 | h2\n1 | 2"]
    assert chunks[1].page_start == 1 and chunks[1].page_end == 2


def test_a_large_table_is_split_by_rows_repeating_the_header():
    rows = ("name | value",) + tuple(f"row{i} | {i}" for i in range(10))
    table = Block("table", "Table 1\n" + "\n".join(rows), 0, (0, 0, 1, 1), caption="Table 1", rows=rows)
    chunks = chunk_document(_parsed([table]), slides=False, max_tokens=12, count_tokens=words)
    assert all(c.kind == "table" for c in chunks)
    assert chunks[0].text.startswith("Table 1\nname | value\nrow0 | 0")
    assert all(c.text.splitlines()[0 if i else 1] == "name | value" for i, c in enumerate(chunks))
    assert sum(len(c.text.splitlines()) for c in chunks) == 1 + 10 + len(chunks)


def test_paper_fixture_produces_a_table_chunk_and_pages():
    parsed = load_parsed(FIX / "paper")
    chunks = chunk_document(parsed, slides=False, max_tokens=450, count_tokens=words)
    assert any("Total Exposure | 67.94 kg yr" in c.text for c in chunks)
    assert chunks[-1].page_end == 8
    # The paper's only title block is its title; any text MinerU places before it has no heading.
    assert sum(c.heading.startswith("Final Result") for c in chunks) >= len(chunks) - 1


def test_embedding_text_prefixes_title_and_heading():
    chunk = Chunk(0, "text", "body", "Section", 1, 1, [])
    assert embedding_text("Doc", chunk) == "Doc › Section\nbody"
    assert embedding_text("Doc", Chunk(0, "text", "body", "", 1, 1, [])) == "Doc\nbody"


def test_figure_chunks_hold_caption_and_description():
    figure = Block("figure", "FIG. 1.", 3, (1, 2, 3, 4), image="images/a.jpg", caption="FIG. 1.")
    [chunk] = figure_chunks(_parsed([figure]), {figure: "An energy spectrum."}, start=7)
    assert (chunk.index, chunk.kind, chunk.page_start) == (7, "figure", 4)
    assert chunk.text == "FIG. 1.\n\nAn energy spectrum."
    assert chunk.heading == "Figure on page 4"
