"""Audio encoding helpers for WebSocket chunk streaming."""

from __future__ import annotations

import io
import wave
from typing import Literal

import numpy as np
import torch

AudioFormat = Literal["pcm", "wav"]


def tensor_to_pcm16(audio: torch.Tensor) -> bytes:
    """Convert `(1, T)` float tensor to signed 16-bit little-endian PCM."""
    samples = audio[0].detach().cpu().float().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


def encode_audio_chunk(audio: torch.Tensor, output_format: AudioFormat, sample_rate: int = 24000) -> bytes:
    """Encode an OmniVoice output tensor as `pcm` or self-contained `wav`."""
    pcm = tensor_to_pcm16(audio)
    if output_format == "pcm":
        return pcm
    if output_format == "wav":
        return _pcm_to_wav(pcm, sample_rate)
    raise ValueError(f"Unsupported output_format '{output_format}'. Use pcm or wav.")


def _pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()
