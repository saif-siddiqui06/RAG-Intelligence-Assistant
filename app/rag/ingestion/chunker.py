"""Intelligent, configurable chunking.

Recursively splits on a prioritized list of separators (paragraph,
line, sentence, word, character) so chunks break on natural boundaries
instead of mid-sentence, then greedily merges the resulting pieces up
to `chunk_size` with `chunk_overlap` carried between consecutive
chunks. This is a from-scratch reimplementation of the well-known
recursive-character splitting strategy — no LangChain dependency.
"""
from dataclasses import dataclass, field

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: list[str] = field(default_factory=lambda: list(DEFAULT_SEPARATORS))

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if not self.separators:
            raise ValueError("separators must not be empty")


class RecursiveCharacterChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split_text(self, text: str) -> list[str]:
        chunks = self._split(text, self.config.separators)
        return [c.strip() for c in chunks if c.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        separator = separators[-1]
        next_separators: list[str] = []
        for i, candidate in enumerate(separators):
            if candidate == "" or candidate in text:
                separator = candidate
                next_separators = separators[i + 1 :]
                break

        pieces = list(text) if separator == "" else text.split(separator)

        chunks: list[str] = []
        buffer: list[str] = []
        for piece in pieces:
            if len(piece) < self.config.chunk_size:
                buffer.append(piece)
                continue
            if buffer:
                chunks.extend(self._merge(buffer, separator))
                buffer = []
            if next_separators:
                chunks.extend(self._split(piece, next_separators))
            else:
                chunks.extend(self._hard_slice(piece))
        if buffer:
            chunks.extend(self._merge(buffer, separator))
        return chunks

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []

        for piece in pieces:
            candidate = current + [piece]
            if current and len(separator.join(candidate)) > self.config.chunk_size:
                chunks.append(separator.join(current))
                current = self._carry_overlap(current, separator)
            current.append(piece)
        if current:
            chunks.append(separator.join(current))
        return chunks

    def _carry_overlap(self, pieces: list[str], separator: str) -> list[str]:
        overlap: list[str] = []
        for piece in reversed(pieces):
            candidate = [piece] + overlap
            if len(separator.join(candidate)) > self.config.chunk_overlap:
                break
            overlap = candidate
        return overlap

    def _hard_slice(self, text: str) -> list[str]:
        step = max(self.config.chunk_size - self.config.chunk_overlap, 1)
        return [text[i : i + self.config.chunk_size] for i in range(0, len(text), step)]
