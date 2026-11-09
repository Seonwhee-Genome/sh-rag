"""Recursive, separator-aware text chunking with overlap."""

from __future__ import annotations

from collections import deque

# Coarse to fine. Sentence enders include CJK punctuation.
SEPARATORS = ["\n\n", "\n", ". ", "。", "? ", "! ", "; ", ", ", " ", ""]


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most ``chunk_size`` characters.

    Text is broken at the coarsest separator that works (paragraph, line,
    sentence, word, then character), and consecutive chunks share roughly
    ``overlap`` characters so ideas that straddle a boundary stay retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))
    text = text.strip()
    if not text:
        return []
    return _merge(_split(text, SEPARATORS, chunk_size), chunk_size, overlap)


def _split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sep = next((s for s in separators if s and s in text), "")
    if not sep:
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]  # keep separators attached
    finer = separators[separators.index(sep) + 1 :]
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > size:
            out.extend(_split(piece, finer, size))
        else:
            out.append(piece)
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: deque[str] = deque()
    length = 0
    for piece in pieces:
        if window and length + len(piece) > size:
            chunks.append("".join(window).strip())
            # Keep a tail of the previous chunk as overlap, as long as it still
            # leaves room for the incoming piece.
            while window and (length > overlap or length + len(piece) > size):
                length -= len(window.popleft())
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [c for c in chunks if c]
