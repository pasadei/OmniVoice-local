"""Utilities for cleaning assistant text before sentence splitting and TTS."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_LINE_RE = re.compile(r"(?m)^\s*```[^\n]*\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_BULLET_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_EMPHASIS_MARK_RE = re.compile(r"[*_~]+")
_ROLE_LABEL_RE = re.compile(
    r"(?i)^\s*(?:assistant|agent|caller|user|system|context)\s*:\s*"
)
_INLINE_CONTEXT_LABEL_RE = re.compile(r"(?i)\b(?:caller|user|system|context)\s*:")
_INLINE_ASSISTANT_LABEL_RE = re.compile(r"(?i)\b(?:assistant|agent)\s*:")
_REPEATED_MARK_RE = re.compile(r"([.,:;])\1+")
_REPEATED_BANG_OR_QUESTION_RE = re.compile(r"[!?]{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
_JSON_TEXT_KEYS = ("reply", "response", "text", "message", "content", "assistant")
_NON_VERBAL_TAG_RE = re.compile(r"\[([^\[\]\n]+)\]")
_NO_JSON_TEXT = object()
_ALLOWED_NON_VERBAL_TAGS = {
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
}


def sanitize_assistant_text(text: Any) -> str:
    """Remove formatting noise while keeping spoken meaning intact."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = _strip_code_fence_lines(cleaned)
    json_text = _extract_json_text(cleaned)
    if json_text is _NO_JSON_TEXT:
        cleaned = ""
    elif json_text is not None:
        cleaned = json_text

    cleaned = _strip_prompt_scaffolding(cleaned)
    cleaned = _strip_inline_formatting(cleaned)
    cleaned = _strip_markdown_line_prefixes(cleaned)
    cleaned = _filter_non_verbal_tags(cleaned)
    cleaned = _collapse_punctuation_noise(cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip()


def _strip_code_fence_lines(text: str) -> str:
    return _FENCE_LINE_RE.sub("", text).strip()


def _extract_json_text(text: str) -> object | str | None:
    candidate = text.strip()
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, (dict, list)):
        return _NO_JSON_TEXT

    unwrapped = _unwrap_json_text(parsed)
    if unwrapped is None:
        return _NO_JSON_TEXT
    return unwrapped


def _unwrap_json_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    if isinstance(value, dict):
        for key in _JSON_TEXT_KEYS:
            if key not in value:
                continue
            unwrapped = _unwrap_json_text(value[key])
            if unwrapped:
                return unwrapped
        return None

    if isinstance(value, list):
        parts = [part for item in value if (part := _unwrap_json_text(item))]
        if parts:
            return " ".join(parts)

    return None


def _strip_markdown_line_prefixes(text: str) -> str:
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _ROLE_LABEL_RE.sub("", line)
        line = _HEADING_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def _strip_prompt_scaffolding(text: str) -> str:
    if _INLINE_CONTEXT_LABEL_RE.search(text):
        matches = list(_INLINE_ASSISTANT_LABEL_RE.finditer(text))
        if matches:
            trailing_text = text[matches[-1].end() :].strip()
            if trailing_text:
                return trailing_text

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = _ROLE_LABEL_RE.sub("", raw_line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines) if cleaned_lines else text


def _strip_inline_formatting(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    return _EMPHASIS_MARK_RE.sub("", text)


def _collapse_punctuation_noise(text: str) -> str:
    text = _REPEATED_MARK_RE.sub(r"\1", text)

    def _replace_bang_or_question(match: re.Match[str]) -> str:
        punctuation = match.group(0)
        return "?" if "?" in punctuation else "!"

    return _REPEATED_BANG_OR_QUESTION_RE.sub(_replace_bang_or_question, text)


def _filter_non_verbal_tags(text: str) -> str:
    def _replace_tag(match: re.Match[str]) -> str:
        tag = match.group(1)
        if tag in _ALLOWED_NON_VERBAL_TAGS:
            return match.group(0)
        return ""

    return _NON_VERBAL_TAG_RE.sub(_replace_tag, text)
