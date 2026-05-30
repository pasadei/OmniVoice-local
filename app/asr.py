"""Thin ASR adapter around faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _resolve_device(device: str) -> dict[str, Any]:
    if device.startswith("cuda:"):
        _, _, index = device.partition(":")
        if index.isdigit():
            return {"device": "cuda", "device_index": int(index)}
    return {"device": device}


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    language_probability: float | None = None


class FasterWhisperASR:
    """Transcribe mono 16 kHz PCM audio with faster-whisper."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        from faster_whisper import WhisperModel

        model_kwargs = {
            **_resolve_device(device),
            "compute_type": compute_type,
        }
        self._model = WhisperModel(model_name, **model_kwargs)

    def transcribe(
        self,
        pcm_bytes: bytes,
        *,
        language_hint: str | None = None,
        session_id: str | None = None,
    ) -> TranscriptionResult:
        del session_id
        if not pcm_bytes:
            return TranscriptionResult(text="", language=language_hint)
        if len(pcm_bytes) % 2 != 0:
            raise ValueError("PCM audio must contain 16-bit samples.")

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0

        kwargs: dict[str, Any] = {"beam_size": 1}
        if language_hint:
            kwargs["language"] = language_hint

        segments, info = self._model.transcribe(audio, **kwargs)
        detected_language = getattr(info, "language", None)
        detected_language_probability = getattr(info, "language_probability", None)
        if not isinstance(detected_language, str) or not detected_language.strip():
            detected_language = language_hint
        if not isinstance(detected_language_probability, (float, int)):
            detected_language_probability = None

        return TranscriptionResult(
            text="".join(segment.text for segment in segments).strip(),
            language=detected_language,
            language_probability=(
                float(detected_language_probability)
                if detected_language_probability is not None
                else None
            ),
        )
