import pytest

from rag_platform.services.chunking import ChunkingConfig, chunk_text


def test_recursive_chunking_preserves_offsets_and_overlap() -> None:
    text = "zero one two three four five\n\nsix seven eight nine ten eleven"
    drafts = chunk_text(
        text,
        "recursive",
        ChunkingConfig(target_words=5, overlap_words=1, minimum_words=1),
    )

    assert drafts[0].content.split()[-1] == drafts[1].content.split()[0]
    for draft in drafts:
        source_words = text[draft.start_offset : draft.end_offset].split()
        assert source_words == draft.content.split()


def test_paragraph_chunking_retains_section_title_and_merges_small_tail() -> None:
    text = "# Experience\n\nBuilt APIs and search systems.\n\nShort tail."
    drafts = chunk_text(
        text,
        "paragraph",
        ChunkingConfig(target_words=7, overlap_words=0, minimum_words=3),
    )

    assert len(drafts) == 1
    assert drafts[0].section_title == "Experience"
    assert drafts[0].start_offset == 0
    assert drafts[0].end_offset == len(text)


def test_chunking_configuration_and_strategy_are_validated() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(target_words=10, overlap_words=10)
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_text("content", "unknown", ChunkingConfig())
