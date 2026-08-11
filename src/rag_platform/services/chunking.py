import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChunkingConfig:
    target_words: int = 330
    overlap_words: int = 45
    minimum_words: int = 20

    def __post_init__(self) -> None:
        if self.target_words < 1:
            raise ValueError("chunk target must be positive")
        if self.overlap_words < 0 or self.overlap_words >= self.target_words:
            raise ValueError("chunk overlap must be non-negative and smaller than the target")
        if self.minimum_words < 1 or self.minimum_words > self.target_words:
            raise ValueError("minimum chunk size must be between one and the target")


@dataclass(frozen=True)
class ChunkDraft:
    content: str
    chunk_index: int
    start_offset: int
    end_offset: int
    section_title: str | None = None


class ChunkingStrategy(Protocol):
    def split(self, text: str, config: ChunkingConfig) -> list[ChunkDraft]: ...


class RecursiveTextChunker:
    def split(self, text: str, config: ChunkingConfig) -> list[ChunkDraft]:
        words = list(re.finditer(r"\S+", text))
        if not words:
            return []
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + config.target_words)
            if end < len(words):
                paragraph_end = _paragraph_boundary(text, words, start, end)
                if paragraph_end is not None:
                    end = paragraph_end
            ranges.append((start, end))
            if end == len(words):
                break
            start = max(start + 1, end - config.overlap_words)
        if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < config.minimum_words:
            ranges[-2] = (ranges[-2][0], ranges[-1][1])
            ranges.pop()
        return [
            _draft(text, words, start_index, end_index, index)
            for index, (start_index, end_index) in enumerate(ranges)
        ]


class ParagraphChunker:
    def split(self, text: str, config: ChunkingConfig) -> list[ChunkDraft]:
        paragraph_matches = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL))
        if not paragraph_matches:
            return []
        drafts: list[ChunkDraft] = []
        group: list[re.Match[str]] = []
        group_words = 0
        for paragraph in paragraph_matches:
            paragraph_words = len(re.findall(r"\S+", paragraph.group()))
            if paragraph_words > config.target_words:
                if group:
                    drafts.append(_paragraph_draft(group, len(drafts)))
                    group = []
                    group_words = 0
                nested = RecursiveTextChunker().split(paragraph.group(), config)
                drafts.extend(
                    ChunkDraft(
                        item.content,
                        len(drafts),
                        paragraph.start() + item.start_offset,
                        paragraph.start() + item.end_offset,
                        item.section_title,
                    )
                    for item in nested
                )
                continue
            if group and group_words + paragraph_words > config.target_words:
                drafts.append(_paragraph_draft(group, len(drafts)))
                group = []
                group_words = 0
            group.append(paragraph)
            group_words += paragraph_words
        if group:
            drafts.append(_paragraph_draft(group, len(drafts)))
        if len(drafts) > 1 and len(drafts[-1].content.split()) < config.minimum_words:
            previous = drafts[-2]
            tail = drafts[-1]
            drafts[-2] = ChunkDraft(
                _normalized_slice(text, previous.start_offset, tail.end_offset),
                previous.chunk_index,
                previous.start_offset,
                tail.end_offset,
                previous.section_title,
            )
            drafts.pop()
        return drafts


def chunk_text(text: str, strategy: str, config: ChunkingConfig) -> list[ChunkDraft]:
    strategies: dict[str, ChunkingStrategy] = {
        "recursive": RecursiveTextChunker(),
        "paragraph": ParagraphChunker(),
    }
    try:
        selected = strategies[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown chunking strategy: {strategy}") from exc
    return selected.split(text, config)


def _paragraph_boundary(
    text: str,
    words: list[re.Match[str]],
    start: int,
    end: int,
) -> int | None:
    minimum_end = start + max(1, int((end - start) * 0.7))
    for index in range(end - 1, minimum_end - 1, -1):
        between = text[words[index - 1].end() : words[index].start()]
        if "\n\n" in between:
            return index
    return None


def _draft(
    text: str,
    words: list[re.Match[str]],
    start: int,
    end: int,
    index: int,
) -> ChunkDraft:
    start_offset = words[start].start()
    end_offset = words[end - 1].end()
    content = " ".join(match.group() for match in words[start:end])
    return ChunkDraft(content, index, start_offset, end_offset, _section_title(content))


def _paragraph_draft(paragraphs: list[re.Match[str]], index: int) -> ChunkDraft:
    start_offset = paragraphs[0].start()
    end_offset = paragraphs[-1].end()
    content = "\n\n".join(paragraph.group().strip() for paragraph in paragraphs)
    return ChunkDraft(content, index, start_offset, end_offset, _section_title(content))


def _section_title(content: str) -> str | None:
    first_line = content.splitlines()[0].strip()
    if first_line.startswith("#"):
        title = first_line.lstrip("#").strip()
        return title[:500] or None
    if first_line.endswith(":") and len(first_line) <= 120:
        return first_line[:-1].strip() or None
    return None


def _normalized_slice(text: str, start: int, end: int) -> str:
    return re.sub(r"[ \t]+", " ", text[start:end].strip())
