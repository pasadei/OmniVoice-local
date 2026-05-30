"""Sentence splitting utilities for low-latency streaming TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

_TERMINAL_PUNCT = {".", "!", "?", "。", "！", "？", ";", ":"}
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.",
}
_TAG_RE = re.compile(r"\[[^\[\]\n]+\]")


@dataclass
class SentenceSplitter:
    """Split text into TTS-friendly chunks with min/max length controls."""

    max_chars: int = 150
    min_chars: int = 20
    abbreviations: set[str] = field(default_factory=lambda: set(_ABBREVIATIONS))

    def split_text(self, text: str) -> list[str]:
        """Split complete text into synthesize-ready sentence chunks."""
        stripped = text.strip()
        if not stripped:
            return []

        raw_sentences, remainder = self.extract_complete_sentences(stripped)
        if remainder:
            raw_sentences.append(remainder.strip())
        return self._post_process(raw_sentences)

    def extract_complete_sentences(self, buffer: str) -> tuple[list[str], str]:
        """Return complete sentences detected in buffer and leftover tail."""
        if not buffer:
            return [], ""

        boundaries: list[int] = []
        in_tag = False

        for idx, ch in enumerate(buffer):
            if ch == "[":
                in_tag = True
            elif ch == "]":
                in_tag = False

            if in_tag or ch not in _TERMINAL_PUNCT:
                continue

            token = self._last_token(buffer[: idx + 1]).lower()
            if token in self.abbreviations:
                continue
            boundaries.append(idx + 1)

        if not boundaries:
            return [], buffer

        sentences: list[str] = []
        prev = 0
        for boundary in boundaries:
            chunk = buffer[prev:boundary].strip()
            if chunk:
                sentences.append(chunk)
            prev = boundary

        remainder = buffer[prev:]
        return sentences, remainder

    def _post_process(self, sentences: Iterable[str]) -> list[str]:
        result: list[str] = []
        pending = ""

        for sentence in sentences:
            for piece in self._split_long_sentence(sentence):
                merged = f"{pending} {piece}".strip() if pending else piece
                if len(merged) < self.min_chars:
                    pending = merged
                    continue
                result.append(merged)
                pending = ""

        if pending:
            if result:
                result[-1] = f"{result[-1]} {pending}".strip()
            else:
                result.append(pending)

        return result

    def _split_long_sentence(self, sentence: str) -> list[str]:
        if len(sentence) <= self.max_chars:
            return [sentence.strip()]

        chunks: list[str] = []
        remaining = sentence.strip()
        while len(remaining) > self.max_chars:
            split_idx = self._find_split_index(remaining)
            chunks.append(remaining[:split_idx].strip())
            remaining = remaining[split_idx:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    def _find_split_index(self, text: str) -> int:
        limit = min(len(text), self.max_chars)
        window = text[:limit]

        comma = window.rfind(",")
        if comma >= self.min_chars:
            return comma + 1

        space = window.rfind(" ")
        if space >= self.min_chars:
            return space + 1

        return limit

    @staticmethod
    def _last_token(text: str) -> str:
        text = text.rstrip()
        if not text:
            return ""
        last_space = text.rfind(" ")
        return text[last_space + 1 :]
