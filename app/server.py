#!/usr/bin/env python3
"""
OmniVoice TTS HTTP Server
=========================

Purpose:
    FastAPI-based HTTP server wrapping the OmniVoice TTS model.
    Provides voice cloning via a pre-configured voices directory,
    voice design via speaker attribute instructions, and auto-voice mode.
    All OmniVoice generation parameters are exposed through the API.

    Optionally serves the built-in OmniVoice Gradio demo UI on a separate
    port, sharing the same loaded model instance (no extra VRAM).

Usage:
    # Start with default settings:
    python server.py

    # Custom GPU and voices directory:
    OMNIVOICE_DEVICE=cuda:1 OMNIVOICE_VOICES_DIR=/voices python server.py

    # Inside container (typical):
    # Model and voices are mounted as bind volumes, see compose.yaml

Environment variables:
    OMNIVOICE_MODEL          - HuggingFace model ID or local path
                               (default: k2-fsa/OmniVoice)
    OMNIVOICE_DEVICE         - PyTorch device string
                               (default: cuda:0)
    OMNIVOICE_DTYPE          - Model dtype: float16, bfloat16, float32
                               (default: float16)
    OMNIVOICE_VOICES_DIR     - Path to voice assets directory
                               (default: /voices)
    OMNIVOICE_SAMPLES_DIR    - Compatibility alias for OMNIVOICE_VOICES_DIR
    OMNIVOICE_HOST           - Server bind address (default: 0.0.0.0)
    OMNIVOICE_PORT           - Server bind port (default: 8000)
    OMNIVOICE_OUTPUT_FORMAT  - Default output audio format: wav, mp3, flac, ogg
                               (default: wav)
    OMNIVOICE_GRADIO_ENABLED - Enable Gradio UI: true/false (default: true)
    OMNIVOICE_GRADIO_PORT    - Gradio UI port (default: 8001)
    OMNIVOICE_WYOMING_ENABLED - Enable Wyoming TCP API for Home Assistant
                               (default: false)
    OMNIVOICE_WYOMING_HOST   - Wyoming bind address (default: 0.0.0.0)
    OMNIVOICE_WYOMING_PORT   - Wyoming TCP port (default: 10200)
    OMNIVOICE_API_KEY        - API key for bearer-token authentication.
                               When set, all endpoints except /health require
                               the header: Authorization: Bearer <key>
                               Leave unset or empty to disable auth entirely.
    OMNIVOICE_CORS_ORIGINS   - Comma-separated allowed CORS origins
                               (default: empty = CORS disabled)
    OMNIVOICE_MAX_UPLOAD_BYTES - Max text file upload size in bytes
                               (default: 10485760 = 10 MB)
    OMNIVOICE_WS_ENABLED      - Enable /ws/tts endpoint (default: true)
    OMNIVOICE_WS_DEFAULT_NUM_STEP - Default WS diffusion steps (default: 16)
    OMNIVOICE_WS_MAX_SENTENCE_CHARS - WS max sentence length (default: 150)
    OMNIVOICE_WS_MIN_SENTENCE_CHARS - WS min sentence length (default: 20)
    OMNIVOICE_WS_BUFFER_TIMEOUT_MS - Flush text buffer timeout in streaming mode (default: 2000)
    OMNIVOICE_CONVERSATION_WS_ENABLED - Enable /ws/conversation endpoint (default: true)
    OMNIVOICE_ASR_MODEL      - faster-whisper model name or path (default: small)
    OMNIVOICE_ASR_DEVICE     - faster-whisper device: auto, cpu, cuda, cuda:0, etc. (default: auto)
    OMNIVOICE_ASR_COMPUTE_TYPE - faster-whisper compute type (default: default)
    OMNIVOICE_OLLAMA_HOST    - Ollama API base URL (default: http://localhost:11434)
    OMNIVOICE_OLLAMA_MODEL   - Ollama model for assistant replies (default: gemma4)
    OMNIVOICE_LLM_PROVIDER   - Assistant backend: "ollama" or "openai" (default: ollama)
    OMNIVOICE_OPENAI_BASE_URL - OpenAI-compatible API base URL (default: http://localhost:8010/v1)
    OMNIVOICE_OPENAI_MODEL   - OpenAI-compatible model name (default: gpt-4o-mini)
    OMNIVOICE_OPENAI_API_KEY - API key for OpenAI-compatible endpoint (default: unset)
"""

import hmac
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torchaudio
import uvicorn
from audio_encoder import encode_audio_chunk
from fastapi import Depends, FastAPI, Form, HTTPException, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from conversation_ws import create_conversation_router
from sentence_splitter import SentenceSplitter
from text_sanitizer import sanitize_assistant_text
from ws_handler import WSConfig, _generate_one, create_ws_router

if TYPE_CHECKING:
    from assistant_backends import AssistantBackend

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omnivoice-server")


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
def _resolve_voices_dir() -> Path:
    voices_dir = os.environ.get("OMNIVOICE_VOICES_DIR")
    if voices_dir:
        return Path(voices_dir)

    samples_dir = os.environ.get("OMNIVOICE_SAMPLES_DIR")
    if samples_dir:
        return Path(samples_dir)

    return Path("/voices")


def _resolve_samples_dir() -> Path:
    return Path(os.environ.get("OMNIVOICE_SAMPLES_DIR", "/samples"))


MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
DTYPE_STR = os.environ.get("OMNIVOICE_DTYPE", "float16")
VOICES_DIR = _resolve_voices_dir()
SAMPLES_DIR = _resolve_samples_dir()
HOST = os.environ.get("OMNIVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("OMNIVOICE_PORT", "8000"))
DEFAULT_OUTPUT_FORMAT = os.environ.get("OMNIVOICE_OUTPUT_FORMAT", "wav")
GRADIO_ENABLED = os.environ.get("OMNIVOICE_GRADIO_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
GRADIO_PORT = int(os.environ.get("OMNIVOICE_GRADIO_PORT", "8001"))
WYOMING_ENABLED = os.environ.get("OMNIVOICE_WYOMING_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)
WYOMING_HOST = os.environ.get("OMNIVOICE_WYOMING_HOST", "0.0.0.0")
WYOMING_PORT = int(os.environ.get("OMNIVOICE_WYOMING_PORT", "10200"))
_OMNIVOICE_DEFAULT_LANGUAGES = (
    "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,"
    "fi,fo,fr,ga,gl,gu,ha,he,hi,hr,hu,hy,id,is,it,ja,jv,ka,kk,km,kn,ko,"
    "ku,ky,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,"
    "or,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,"
    "th,tk,tl,tr,tt,uk,ur,uz,vi,yo,zh,zu"
)
WYOMING_LANGUAGES = os.environ.get(
    "OMNIVOICE_WYOMING_LANGUAGES", _OMNIVOICE_DEFAULT_LANGUAGES
)
API_KEY = os.environ.get("OMNIVOICE_API_KEY", "").strip()
CORS_ORIGINS = os.environ.get("OMNIVOICE_CORS_ORIGINS", "").strip()
MAX_UPLOAD_BYTES = int(
    os.environ.get("OMNIVOICE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
)  # 10 MB
WS_ENABLED = os.environ.get("OMNIVOICE_WS_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
WS_DEFAULT_NUM_STEP = int(os.environ.get("OMNIVOICE_WS_DEFAULT_NUM_STEP", "16"))
WS_MAX_SENTENCE_CHARS = int(os.environ.get("OMNIVOICE_WS_MAX_SENTENCE_CHARS", "150"))
WS_MIN_SENTENCE_CHARS = int(os.environ.get("OMNIVOICE_WS_MIN_SENTENCE_CHARS", "20"))
WS_BUFFER_TIMEOUT_MS = int(os.environ.get("OMNIVOICE_WS_BUFFER_TIMEOUT_MS", "2000"))
CONVERSATION_WS_ENABLED = os.environ.get(
    "OMNIVOICE_CONVERSATION_WS_ENABLED", "true"
).lower() in ("true", "1", "yes")
ASR_MODEL = os.environ.get("OMNIVOICE_ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("OMNIVOICE_ASR_DEVICE", "auto")
ASR_COMPUTE_TYPE = os.environ.get("OMNIVOICE_ASR_COMPUTE_TYPE", "default")
ASR_LANGUAGE_STICKY_MIN_PROB = float(
    os.environ.get("OMNIVOICE_ASR_LANGUAGE_STICKY_MIN_PROB", "0.80")
)
OLLAMA_HOST = os.environ.get("OMNIVOICE_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OMNIVOICE_OLLAMA_MODEL", "gemma4")
LLM_BACKEND = os.environ.get("OMNIVOICE_LLM_PROVIDER", "ollama").lower()
OPENAI_BASE_URL = os.environ.get("OMNIVOICE_OPENAI_BASE_URL", "http://localhost:8010/v1")
OPENAI_MODEL = os.environ.get("OMNIVOICE_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OMNIVOICE_OPENAI_API_KEY", None)
CONVERSATION_WARMUP_ENABLED = os.environ.get(
    "OMNIVOICE_CONVERSATION_WARMUP_ENABLED", "false"
).lower() in ("true", "1", "yes")

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

SAMPLE_RATE = 24000

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}

# OpenAI response_format → internal format + MIME type
# "opus" uses an Ogg container; "aac" falls back to mp3; "pcm" is raw s16le
OPENAI_FORMAT_MAP = {
    "mp3": ("mp3", "audio/mpeg"),
    "opus": ("ogg", "audio/ogg"),
    "aac": ("mp3", "audio/mpeg"),  # best-effort fallback
    "flac": ("flac", "audio/flac"),
    "wav": ("wav", "audio/wav"),
    "pcm": ("pcm", "audio/pcm"),  # raw s16le, handled specially
}

# OpenAI model → diffusion steps
OPENAI_MODEL_STEPS = {
    "tts-1": 16,
    "tts-1-hd": 32,
}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> None:
    """Dependency that enforces API key auth when OMNIVOICE_API_KEY is set."""
    if not API_KEY:
        return  # Auth disabled — all requests allowed
    if credentials is None or not hmac.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide: Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Voice directory scanner
# ---------------------------------------------------------------------------


def scan_voices(directory: Path) -> dict:
    """
    Scan the voices directory for audio + transcript pairs.

    Expected structure:
        some-name.wav   + some-name.txt
        another.mp3     + another.txt

    Audio and transcript files are matched by stem (filename without extension).
    If a .txt transcript is missing, the sample is still loaded but ref_text
    will be None (OmniVoice will use Whisper to auto-transcribe).

    Returns:
        dict[str, dict] mapping voice name -> {audio_path, ref_text}
    """
    samples = {}
    if not directory.is_dir():
        log.warning("Voices directory does not exist: %s", directory)
        return samples

    audio_files = {}
    txt_files = {}

    for f in directory.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files[f.stem] = f
        elif f.suffix.lower() == ".txt":
            txt_files[f.stem] = f

    for stem, audio_path in audio_files.items():
        ref_text = None
        if stem in txt_files:
            try:
                ref_text = txt_files[stem].read_text(encoding="utf-8").strip()
            except Exception as exc:
                log.warning("Failed to read transcript %s: %s", txt_files[stem], exc)
        samples[stem] = {
            "audio_path": str(audio_path),
            "ref_text": ref_text,
        }
        log.info(
            "  sample: %-30s  audio: %s  transcript: %s",
            stem,
            audio_path.name,
            "yes" if ref_text else "no (will auto-transcribe)",
        )

    log.info("Loaded %d voice sample(s) from %s", len(samples), directory)
    return samples


scan_samples = scan_voices


def _load_voice_samples() -> dict:
    samples = scan_voices(VOICES_DIR)
    if samples or SAMPLES_DIR == VOICES_DIR:
        return samples

    legacy_samples = scan_voices(SAMPLES_DIR)
    if legacy_samples:
        log.warning(
            "No voices found in %s; falling back to legacy samples directory %s",
            VOICES_DIR,
            SAMPLES_DIR,
        )
    return legacy_samples


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    """Request body for the /tts endpoint."""

    # Required
    text: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize.",
    )

    # Voice selection (mutually exclusive approaches)
    sample: Optional[str] = Field(
        None,
        description=(
            "Name of a voice asset (stem without extension) from the "
            "voices directory. Enables voice cloning mode."
        ),
    )
    instruct: Optional[str] = Field(
        None,
        description=(
            "Speaker attribute instruction for voice design mode, e.g. "
            "'female, low pitch, british accent'. Ignored if 'sample' is set."
        ),
    )

    # Override ref_text for a sample (optional)
    ref_text: Optional[str] = Field(
        None,
        description=(
            "Override the transcript for the reference audio. "
            "If omitted, the .txt file next to the sample is used. "
            "If that is also missing, Whisper auto-transcribes."
        ),
    )

    # Decoding parameters
    num_step: Optional[int] = Field(
        None,
        ge=1,
        le=256,
        description="Number of diffusion unmasking steps (default: 32). Use 16 for faster inference.",
    )
    guidance_scale: Optional[float] = Field(
        None,
        ge=0.0,
        le=50.0,
        description="Classifier-free guidance scale (default: 2.0).",
    )
    t_shift: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Time-step shift for noise schedule (default: 0.1).",
    )
    denoise: Optional[bool] = Field(
        None,
        description="Prepend denoising token (default: True).",
    )

    # Sampling parameters
    position_temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Temperature for mask-position selection (default: 5.0). 0 = greedy.",
    )
    class_temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Temperature for token sampling (default: 0.0). 0 = greedy.",
    )
    layer_penalty_factor: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Penalty for deeper codebook layers (default: 5.0).",
    )

    # Duration / speed
    duration: Optional[float] = Field(
        None,
        gt=0.0,
        le=600.0,
        description="Fixed output duration in seconds. Overrides 'speed'.",
    )
    speed: Optional[float] = Field(
        None,
        gt=0.0,
        le=10.0,
        description="Speed factor (>1.0 faster, <1.0 slower). Ignored when 'duration' is set.",
    )

    # Pre/post processing
    preprocess_prompt: Optional[bool] = Field(
        None,
        description="Preprocess reference audio (remove silences, add punctuation). Default: True.",
    )
    postprocess_output: Optional[bool] = Field(
        None,
        description="Post-process output audio (remove long silences). Default: True.",
    )

    # Long-form generation
    audio_chunk_duration: Optional[float] = Field(
        None,
        gt=0.0,
        le=120.0,
        description="Target chunk duration in seconds for long text (default: 15.0).",
    )
    audio_chunk_threshold: Optional[float] = Field(
        None,
        gt=0.0,
        le=600.0,
        description="Estimated duration threshold to activate chunking (default: 30.0).",
    )

    # Output
    output_format: Optional[str] = Field(
        None,
        description="Output audio format: wav, mp3, flac, ogg. Default from OMNIVOICE_OUTPUT_FORMAT env.",
    )


class SampleInfo(BaseModel):
    """Info about a single voice asset."""

    name: str
    audio_file: str
    has_transcript: bool
    transcript_preview: Optional[str] = None


class OpenAISpeechRequest(BaseModel):
    """
    OpenAI-compatible /v1/audio/speech request body.
    See: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    model: str = Field(
        ...,
        description="TTS model to use. 'tts-1' = fast (16 steps), 'tts-1-hd' = quality (32 steps).",
    )
    input: str = Field(
        ...,
        description="Text to synthesize (max 4096 characters).",
        max_length=4096,
    )
    voice: str = Field(
        ...,
        description=(
            "Voice to use. Standard OpenAI voices (alloy, ash, coral, echo, fable, nova, "
            "onyx, sage, shimmer) map to a voice asset of the same name when available, "
            "or fall back to auto-voice mode. Any sample name loaded on the server can also "
            "be used directly."
        ),
    )
    response_format: str = Field(
        "mp3",
        description="Output format: mp3, opus, aac, flac, wav, pcm. Note: aac is served as mp3.",
    )
    speed: Optional[float] = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Speech speed factor (0.25–4.0, default 1.0).",
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

# Globals populated at startup
model = None
voice_samples: dict = {}
wyoming_server = None
model_lock: asyncio.Lock = asyncio.Lock()
active_ws_connections = 0
conversation_service = None


class ConversationService:
    """Server-side ASR + assistant TTS for `/ws/conversation`."""

    def __init__(
        self,
        asr_adapter: Any,
        *,
        assistant_backend: "AssistantBackend",
    ):
        self._asr_adapter = asr_adapter
        self._assistant_backend = assistant_backend
        self._splitter = SentenceSplitter(
            max_chars=WS_MAX_SENTENCE_CHARS,
            min_chars=max(1, WS_MIN_SENTENCE_CHARS),
        )
        self._voice_prompt_cache: dict[tuple[str, str | None], Any] = {}
        self._pending_turn_metrics: dict[str, dict[str, Any]] = {}

    async def warmup(self, *, sample: str | None = None) -> None:
        warmup_audio = b"\x00\x00" * (16000 // 4)
        with contextlib.suppress(Exception):
            await self.transcribe(
                warmup_audio, 16000, session_id=None, language_hint=None
            )
        with contextlib.suppress(Exception):
            await self._generate_assistant_text(
                "Reply with ok.",
                session_id="warmup",
                history=[],
                language_hint=None,
            )
        voice_clone_prompt = None
        if sample:
            with contextlib.suppress(Exception):
                voice_clone_prompt = await self._get_voice_clone_prompt(sample)
        with contextlib.suppress(Exception):
            gen_kwargs: dict[str, Any] = {
                "text": "Ok.",
                "num_step": WS_DEFAULT_NUM_STEP,
            }
            if voice_clone_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = voice_clone_prompt
            await self._generate_sentence_audio(gen_kwargs)

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        *,
        session_id: str | None = None,
        language_hint: str | None = None,
    ) -> Any:
        if sample_rate != 16000:
            raise RuntimeError(
                "Conversation ASR currently expects mono 16 kHz PCM audio."
            )
        started = time.monotonic()
        transcript = await asyncio.to_thread(
            _call_with_supported_kwargs,
            self._asr_adapter.transcribe,
            audio_bytes,
            session_id=session_id,
            language_hint=language_hint,
        )
        if session_id:
            self._pending_turn_metrics[session_id] = {
                "turn_started_at": started,
                "asr_ms": int((time.monotonic() - started) * 1000),
                "detected_language": self._extract_detected_language(transcript),
                "detected_language_probability": self._extract_detected_language_probability(
                    transcript
                ),
                "language_hint": language_hint,
            }
        return transcript

    async def respond(
        self,
        transcript: str,
        response_id: int,
        *,
        sample: str | None = None,
        instruct: str | None = None,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ):
        text = str(transcript).strip()
        if not text:
            self._discard_pending_turn_metrics(session_id)
            yield {"type": "response_done", "response_id": response_id}
            return

        turn_metrics = self._begin_turn_metrics(
            session_id=session_id,
            language_hint=language_hint,
        )
        llm_started = time.monotonic()
        response_text = sanitize_assistant_text(
            await self._generate_assistant_text(
                text,
                session_id=session_id,
                history=history or [],
                language_hint=language_hint,
            )
        )
        llm_ms = int((time.monotonic() - llm_started) * 1000)
        if not response_text:
            raise RuntimeError("Assistant backend returned an empty response.")

        sentences = self._splitter.split_text(response_text)
        if not sentences:
            yield {"type": "response_done", "response_id": response_id}
            return

        tts_started = time.monotonic()
        tts_first_chunk_ms = None
        voice_clone_prompt = None
        if sample:
            voice_clone_prompt = await self._get_voice_clone_prompt(sample)

        last_sentence_index = len(sentences) - 1
        for sentence_index, sentence in enumerate(sentences):
            yield {
                "type": "assistant_text",
                "text": sentence,
                "response_id": response_id,
            }

            gen_kwargs: dict[str, Any] = {
                "text": sentence,
                "num_step": WS_DEFAULT_NUM_STEP,
            }
            if voice_clone_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = voice_clone_prompt
            elif instruct is not None:
                gen_kwargs["instruct"] = instruct

            started = time.monotonic()
            audio = await self._generate_sentence_audio(gen_kwargs)
            generation_ms = int((time.monotonic() - started) * 1000)
            duration_ms = int(audio.shape[-1] * 1000 / SAMPLE_RATE)
            if tts_first_chunk_ms is None:
                tts_first_chunk_ms = int((time.monotonic() - tts_started) * 1000)

            yield {
                "type": "audio_chunk",
                "response_id": response_id,
                "chunk_index": sentence_index,
                "sentence": sentence,
                "duration_ms": duration_ms,
                "generation_time_ms": generation_ms,
                "is_last": sentence_index == last_sentence_index,
                "payload": encode_audio_chunk(
                    audio, output_format="pcm", sample_rate=SAMPLE_RATE
                ),
            }

        self._log_completed_turn(
            session_id=session_id,
            response_id=response_id,
            turn_metrics=turn_metrics,
            llm_ms=llm_ms,
            tts_first_chunk_ms=tts_first_chunk_ms,
            tts_total_ms=int((time.monotonic() - tts_started) * 1000),
        )
        yield {"type": "response_done", "response_id": response_id}

    def validate_sample(self, sample: str | None) -> None:
        if sample and sample not in voice_samples:
            raise RuntimeError(f"Sample '{sample}' not found.")

    def clear_voice_prompt_cache(self) -> None:
        self._voice_prompt_cache.clear()

    async def _generate_sentence_audio(self, gen_kwargs: dict[str, Any]):
        generation_task = asyncio.create_task(
            _generate_one(lambda: model, model_lock, gen_kwargs)
        )
        try:
            return await asyncio.shield(generation_task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await generation_task
            raise

    async def _get_voice_clone_prompt(self, sample: str) -> Any:
        if sample not in voice_samples:
            raise RuntimeError(f"Sample '{sample}' not found.")

        sample_info = voice_samples[sample]
        ref_text = sample_info.get("ref_text")
        cache_key = (sample, ref_text)
        cached_prompt = self._voice_prompt_cache.get(cache_key)
        if cached_prompt is not None:
            return cached_prompt

        if model is None:
            raise RuntimeError("Model not loaded yet.")

        kwargs = {"ref_audio": sample_info["audio_path"]}
        if ref_text is not None:
            kwargs["ref_text"] = ref_text

        async with model_lock:
            prompt = await asyncio.to_thread(model.create_voice_clone_prompt, **kwargs)

        self._voice_prompt_cache[cache_key] = prompt
        return prompt

    async def _generate_assistant_text(
        self,
        transcript: str,
        *,
        session_id: str | None,
        history: list[dict[str, str]],
        language_hint: str | None,
    ) -> str:
        generate_response = self._assistant_backend.generate_response
        supports_context = _callable_supports_any_keyword(
            generate_response,
            {"session_id", "history", "language_hint"},
        )
        if supports_context:
            return await _call_with_supported_kwargs(
                generate_response,
                transcript,
                session_id=session_id,
                history=history,
                language_hint=language_hint,
            )

        prompt = _build_assistant_prompt(
            transcript,
            session_id=session_id,
            history=history,
            language_hint=language_hint,
        )
        return await generate_response(prompt)

    def _begin_turn_metrics(
        self,
        *,
        session_id: str | None,
        language_hint: str | None,
    ) -> dict[str, Any]:
        turn_metrics = {
            "turn_started_at": time.monotonic(),
            "asr_ms": None,
            "detected_language": None,
            "language_hint": language_hint,
        }
        if not session_id:
            return turn_metrics

        stored_metrics = self._pending_turn_metrics.pop(session_id, None)
        if stored_metrics is None:
            return turn_metrics

        if language_hint is not None:
            stored_metrics["language_hint"] = language_hint
        return stored_metrics

    def _discard_pending_turn_metrics(self, session_id: str | None) -> None:
        if session_id:
            self._pending_turn_metrics.pop(session_id, None)

    def _extract_detected_language(self, transcript: Any) -> str | None:
        if isinstance(transcript, dict):
            language = transcript.get("language") or transcript.get("detected_language")
        else:
            language = getattr(transcript, "language", None)
        if not isinstance(language, str) or not language.strip():
            return None
        return language.strip()

    def _extract_detected_language_probability(self, transcript: Any) -> float | None:
        if isinstance(transcript, dict):
            probability = transcript.get("language_probability") or transcript.get(
                "detected_language_probability"
            )
        else:
            probability = getattr(transcript, "language_probability", None)
        if not isinstance(probability, (float, int)):
            return None
        return float(probability)

    def _assistant_backend_name(self) -> str:
        return type(self._assistant_backend).__name__

    def _assistant_model_name(self) -> str | None:
        for attribute_name in ("_model", "model"):
            value = getattr(self._assistant_backend, attribute_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _log_completed_turn(
        self,
        *,
        session_id: str | None,
        response_id: int,
        turn_metrics: dict[str, Any],
        llm_ms: int,
        tts_first_chunk_ms: int | None,
        tts_total_ms: int,
    ) -> None:
        payload = {
            "event": "conversation_turn_completed",
            "session_id": session_id,
            "response_id": response_id,
            "detected_language": turn_metrics.get("detected_language"),
            "detected_language_probability": turn_metrics.get(
                "detected_language_probability"
            ),
            "language_hint": turn_metrics.get("language_hint"),
            "assistant_backend": self._assistant_backend_name(),
            "assistant_model": self._assistant_model_name(),
            "asr_ms": turn_metrics.get("asr_ms"),
            "llm_ms": llm_ms,
            "tts_first_chunk_ms": tts_first_chunk_ms,
            "tts_total_ms": tts_total_ms,
            "turn_total_ms": int(
                (time.monotonic() - turn_metrics["turn_started_at"]) * 1000
            ),
        }
        log.info(json.dumps(payload, sort_keys=True))


class _ConversationServiceProxy:
    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        *,
        session_id: str | None = None,
        language_hint: str | None = None,
    ) -> Any:
        return await _require_conversation_service().transcribe(
            audio_bytes,
            sample_rate,
            session_id=session_id,
            language_hint=language_hint,
        )

    def validate_sample(self, sample: str | None) -> None:
        _require_conversation_service().validate_sample(sample)

    async def respond(
        self,
        transcript: str,
        response_id: int,
        *,
        sample: str | None = None,
        instruct: str | None = None,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ):
        async for event in _require_conversation_service().respond(
            transcript,
            response_id,
            sample=sample,
            instruct=instruct,
            session_id=session_id,
            history=history,
            language_hint=language_hint,
        ):
            yield event


def _require_conversation_service() -> ConversationService:
    if conversation_service is None:
        raise RuntimeError("Conversation service not loaded yet.")
    return conversation_service


def _create_conversation_service() -> ConversationService:
    from asr import FasterWhisperASR

    if LLM_BACKEND == "openai":
        from assistant_backends import OpenAIAssistantBackend
        from openai import AsyncOpenAI

        api_key = OPENAI_API_KEY or "not-used"
        assistant_backend: Any = OpenAIAssistantBackend(
            AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=api_key),
            OPENAI_MODEL,
        )
    else:
        from assistant_backends import OllamaAssistantBackend
        from ollama import AsyncClient

        assistant_backend = OllamaAssistantBackend(
            AsyncClient(host=OLLAMA_HOST), OLLAMA_MODEL
        )

    return ConversationService(
        FasterWhisperASR(
            model_name=ASR_MODEL,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        ),
        assistant_backend=assistant_backend,
    )


conversation_service_proxy = _ConversationServiceProxy()


def _callable_supports_any_keyword(func: Any, names: set[str]) -> bool:
    parameters = inspect.signature(func).parameters.values()
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name in names
        for parameter in parameters
    )


def _call_with_supported_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    parameters = inspect.signature(func).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return func(*args, **kwargs)

    supported_names = {parameter.name for parameter in parameters}
    supported_kwargs = {
        name: value for name, value in kwargs.items() if name in supported_names
    }
    return func(*args, **supported_kwargs)


def _build_assistant_prompt(
    transcript: str,
    *,
    session_id: str | None,
    history: list[dict[str, str]],
    language_hint: str | None,
) -> str:
    if not history and not language_hint and not session_id:
        return transcript

    lines = ["Continue the phone conversation naturally."]
    if session_id:
        lines.append(f"Session ID: {session_id}")
    if language_hint:
        lines.append(f"Caller language hint: {language_hint}")
    if history:
        lines.append("Recent conversation:")
        for turn in history[-3:]:
            user_text = str(turn.get("user") or "").strip()
            assistant_text = str(turn.get("assistant") or "").strip()
            if user_text:
                lines.append(f"Caller: {user_text}")
            if assistant_text:
                lines.append(f"Assistant: {assistant_text}")
    lines.append(f"Caller: {transcript}")
    return "\n".join(lines)


def _inc_active_ws(delta: int) -> None:
    """Increment/decrement active WebSocket connection count."""
    global active_ws_connections
    active_ws_connections = max(0, active_ws_connections + delta)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    global model, voice_samples, wyoming_server, conversation_service

    dtype = DTYPE_MAP.get(DTYPE_STR)
    if dtype is None:
        log.error(
            "Invalid OMNIVOICE_DTYPE=%s. Use: float16, bfloat16, float32", DTYPE_STR
        )
        sys.exit(1)

    if API_KEY:
        log.info("API key authentication: ENABLED")
    else:
        log.warning(
            "API key authentication: DISABLED (set OMNIVOICE_API_KEY to enable)"
        )

    log.info("Loading OmniVoice model: %s", MODEL_ID)
    log.info("  device=%s  dtype=%s", DEVICE, DTYPE_STR)

    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=dtype,
    )
    log.info("Model loaded successfully.")

    log.info("Scanning voices directory: %s", VOICES_DIR)
    voice_samples = _load_voice_samples()

    if CONVERSATION_WS_ENABLED:
        log.info("Loading ASR model: %s", ASR_MODEL)
        conversation_service = _create_conversation_service()
        if CONVERSATION_WARMUP_ENABLED:
            warmup_sample = next(iter(sorted(voice_samples.keys())), None)
            log.info("Warming up conversation stack")
            await conversation_service.warmup(sample=warmup_sample)

    # Launch Gradio UI in a background thread, sharing the same model
    if GRADIO_ENABLED:
        _launch_gradio()

    if WYOMING_ENABLED:
        wyoming_server = _launch_wyoming_server()

    yield  # Application runs here

    if wyoming_server is not None:
        wyoming_server.stop()
        wyoming_server = None
    conversation_service = None

    log.info("Shutting down OmniVoice server.")


app = FastAPI(
    title="OmniVoice TTS Server",
    description=(
        "HTTP API for OmniVoice zero-shot multilingual TTS. "
        "Supports voice cloning (via pre-loaded voices), "
        "voice design (via speaker attribute instructions), "
        "and auto-voice mode."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

ws_config = WSConfig(
    enabled=WS_ENABLED,
    default_num_step=WS_DEFAULT_NUM_STEP,
    max_sentence_chars=WS_MAX_SENTENCE_CHARS,
    min_sentence_chars=WS_MIN_SENTENCE_CHARS,
    buffer_timeout_ms=WS_BUFFER_TIMEOUT_MS,
    sample_rate=SAMPLE_RATE,
)
app.include_router(
    create_ws_router(
        get_model=lambda: model,
        get_voice_samples=lambda: voice_samples,
        model_lock=model_lock,
        config=ws_config,
        active_counter=_inc_active_ws,
    )
)
if CONVERSATION_WS_ENABLED:
    app.include_router(
        create_conversation_router(
            service=conversation_service_proxy,
            sticky_language_min_prob=ASR_LANGUAGE_STICKY_MIN_PROB,
            active_counter=_inc_active_ws,
        )
    )

# CORS middleware — allow Gradio pages to fetch from the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(",") if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Gradio integration
# ---------------------------------------------------------------------------


def _launch_gradio():
    """
    Start the OmniVoice Gradio demo in a background thread.
    Uses the already loaded 'model' global — no extra VRAM.
    Also starts the conversation test page on a separate port.
    """
    try:
        from omnivoice.cli.demo import build_demo
    except ImportError:
        log.warning(
            "Could not import omnivoice.cli.demo.build_demo. "
            "Gradio UI will not be available."
        )
        return

    log.info("Starting Gradio UI on port %d ...", GRADIO_PORT)
    demo = build_demo(model=model, checkpoint=MODEL_ID)

    def _run():
        demo.launch(
            server_name=HOST,
            server_port=GRADIO_PORT,
            share=False,
            prevent_thread_lock=False,
            css="footer {display:none !important}",
        )

    thread = threading.Thread(target=_run, name="gradio-ui", daemon=True)
    thread.start()
    log.info("Gradio UI started on port %d", GRADIO_PORT)


# ---------------------------------------------------------------------------
# Wyoming protocol integration
# ---------------------------------------------------------------------------


class _WyomingServer:
    """Minimal Wyoming TCP server for Home Assistant integration."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._thread = threading.Thread(
            target=self._run, name="wyoming-server", daemon=True
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[asyncio.base_events.Server] = None
        self._ready = threading.Event()

    def start(self):
        self._thread.start()
        self._ready.wait(timeout=10)

    def stop(self):
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._shutdown)
        self._thread.join(timeout=10)

    def _shutdown(self):
        if self._server is not None:
            self._server.close()
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _serve(self):
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        log.info("Wyoming TCP server listening on %s:%d", self.host, self.port)
        self._ready.set()
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        synth_state = {"voice": None, "chunks": []}
        synthesized = False  # track whether we already synthesized in this connection
        peer = writer.get_extra_info("peername")
        log.info("Wyoming client connected: %s", peer)
        try:
            while True:
                event = await _wyoming_read_event(reader)
                if event is None:
                    break
                event_type = event.get("type")
                data = event.get("data") or {}
                log.debug("Wyoming event from %s: %s", peer, event_type)

                if event_type == "describe":
                    await _wyoming_send_event(writer, _wyoming_info_event())
                elif event_type == "synthesize":
                    if synthesized:
                        log.debug("Skipping duplicate synthesize event from %s", peer)
                        continue
                    text = str(data.get("text", "")).strip()
                    if text:
                        voice = data.get("voice")
                        await _wyoming_send_tts(writer, text, voice)
                        synthesized = True
                elif event_type == "synthesize-start":
                    synth_state["voice"] = data.get("voice")
                    synth_state["chunks"] = []
                elif event_type == "synthesize-chunk":
                    chunk = str(data.get("text", ""))
                    if chunk:
                        synth_state["chunks"].append(chunk)
                elif event_type == "synthesize-stop":
                    if synthesized:
                        log.debug(
                            "Skipping duplicate streaming synthesize from %s", peer
                        )
                    else:
                        text = "".join(synth_state["chunks"]).strip()
                        if text:
                            await _wyoming_send_tts(writer, text, synth_state["voice"])
                            synthesized = True
                    await _wyoming_send_event(writer, {"type": "synthesize-stopped"})
                    synth_state["voice"] = None
                    synth_state["chunks"] = []
                else:
                    log.debug("Ignoring Wyoming event type: %s", event_type)
        except Exception as exc:
            log.warning("Wyoming connection error from %s: %s", peer, exc)
        finally:
            writer.close()
            await writer.wait_closed()
            log.info("Wyoming client disconnected: %s", peer)


def _launch_wyoming_server() -> _WyomingServer:
    server = _WyomingServer(WYOMING_HOST, WYOMING_PORT)
    server.start()
    return server


def _wyoming_info_event() -> dict:
    speakers = [{"name": s} for s in sorted(voice_samples.keys())]
    languages = _resolve_wyoming_languages()
    voice_info = {
        "name": "omnivoice",
        "languages": languages,
        "attribution": {
            "name": "k2-fsa OmniVoice",
            "url": "https://github.com/k2-fsa/OmniVoice",
        },
        "installed": True,
        "description": "OmniVoice zero-shot multilingual TTS",
    }
    if speakers:
        voice_info["speakers"] = speakers

    tts_program = {
        "name": "omnivoice",
        "attribution": {
            "name": "OmniVoice Local Server",
            "url": "https://github.com/k2-fsa/OmniVoice",
        },
        "installed": True,
        "description": "OmniVoice Wyoming TTS service",
        "voices": [voice_info],
        "supports_synthesize_streaming": True,
    }

    return {
        "type": "info",
        "data": {"tts": [tts_program]},
    }


def _resolve_wyoming_languages() -> list[str]:
    """
    Return supported language codes for Wyoming `describe`.

    Tries to auto-detect from the loaded model/tokenizer first.
    If unavailable, falls back to OMNIVOICE_WYOMING_LANGUAGES (comma-separated).
    """
    discovered: list[str] = []
    tokenizer = getattr(model, "tokenizer", None)

    # Common patterns from Whisper-like tokenizers
    if tokenizer is not None:
        for attr in (
            "_LANGUAGE_CODES",
            "LANGUAGE_CODES",
            "language_codes",
            "languages",
        ):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, (list, tuple, set)):
                discovered = [str(code).strip() for code in value if str(code).strip()]
                if discovered:
                    break

    # Model-level optional method
    if not discovered:
        get_languages = getattr(model, "get_supported_languages", None)
        if callable(get_languages):
            try:
                value = get_languages()
                if isinstance(value, (list, tuple, set)):
                    discovered = [
                        str(code).strip() for code in value if str(code).strip()
                    ]
            except Exception as exc:
                log.debug(
                    "Failed to get languages from model.get_supported_languages: %s",
                    exc,
                )

    if discovered:
        return sorted(set(discovered))

    fallback = [code.strip() for code in WYOMING_LANGUAGES.split(",") if code.strip()]
    if not fallback:
        fallback = ["en"]
    return sorted(set(fallback))


async def _wyoming_send_tts(
    writer: asyncio.StreamWriter, text: str, voice: Optional[dict]
):
    sample = None
    if isinstance(voice, dict):
        candidate = voice.get("name") or voice.get("speaker")
        if isinstance(candidate, str):
            sample = candidate

    try:
        gen_kwargs = _build_gen_kwargs(
            text=text, sample=sample if sample in voice_samples else None
        )
        audio, elapsed = await _run_model_async(gen_kwargs)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        log.warning("[wyoming] TTS generation refused: %s", detail)
        await _wyoming_send_event(writer, {"type": "error", "data": {"text": detail}})
        return
    duration_s = audio.shape[-1] / SAMPLE_RATE
    log.info("[wyoming] Generated %.2fs audio in %.2fs", duration_s, elapsed)

    samples = audio[0].cpu().float().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16).tobytes()

    await _wyoming_send_event(
        writer,
        {
            "type": "audio-start",
            "data": {"rate": SAMPLE_RATE, "width": 2, "channels": 1},
        },
    )
    chunk_size = 8192
    for i in range(0, len(pcm), chunk_size):
        await _wyoming_send_event(
            writer,
            {
                "type": "audio-chunk",
                "data": {"rate": SAMPLE_RATE, "width": 2, "channels": 1},
                "payload": pcm[i : i + chunk_size],
            },
        )
    await _wyoming_send_event(writer, {"type": "audio-stop"})


async def _wyoming_send_event(writer: asyncio.StreamWriter, event: dict):
    payload = event.get("payload")
    header = {
        "type": event["type"],
    }
    data = event.get("data")
    if data:
        header["data"] = data
    if payload:
        header["payload_length"] = len(payload)

    writer.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
    if payload:
        writer.write(payload)
    await writer.drain()


async def _wyoming_read_event(reader: asyncio.StreamReader) -> Optional[dict]:
    line = await reader.readline()
    if not line:
        return None
    header = json.loads(line.decode("utf-8").strip())

    data = header.get("data", {})
    data_length = int(header.get("data_length") or 0)
    if data_length > 0:
        extra_data = await reader.readexactly(data_length)
        extra_json = json.loads(extra_data.decode("utf-8"))
        if isinstance(extra_json, dict):
            data = {**data, **extra_json}

    payload_length = int(header.get("payload_length") or 0)
    if payload_length > 0:
        await reader.readexactly(payload_length)  # payload not needed for TTS requests

    return {"type": header.get("type"), "data": data}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/duplex", tags=["UI"])
async def duplex_ui():
    """Serve the full-duplex conversation UI."""
    return FileResponse(
        path=Path(__file__).parent / "duplex.html",
        media_type="text/html",
    )


@app.get("/health", tags=["System"])
async def health():
    """Health check. Returns 503 while the model is still loading."""
    ready = model is not None
    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if ready else "loading",
            "model_loaded": ready,
            "model": MODEL_ID,
            "device": DEVICE,
            "gradio_enabled": GRADIO_ENABLED,
            "gradio_port": GRADIO_PORT if GRADIO_ENABLED else None,
            "websocket_enabled": WS_ENABLED,
            "conversation_websocket_enabled": CONVERSATION_WS_ENABLED,
            "active_ws_connections": active_ws_connections,
        },
    )


@app.get("/voices", response_model=list[SampleInfo], tags=["Voices"])
@app.get("/samples", response_model=list[SampleInfo], tags=["Voices"])
async def list_voices(_: None = Depends(require_api_key)):
    """List all available voice assets."""
    result = []
    for name, info in sorted(voice_samples.items()):
        preview = None
        if info["ref_text"]:
            preview = info["ref_text"][:120] + (
                "..." if len(info["ref_text"]) > 120 else ""
            )
        result.append(
            SampleInfo(
                name=name,
                audio_file=Path(info["audio_path"]).name,
                has_transcript=info["ref_text"] is not None,
                transcript_preview=preview,
            )
        )
    return result


list_samples = list_voices


@app.post("/voices/reload", tags=["Voices"])
@app.post("/samples/reload", tags=["Voices"])
async def reload_voices(_: None = Depends(require_api_key)):
    """Re-scan the voices directory (e.g. after adding new files)."""
    global voice_samples
    voice_samples = _load_voice_samples()
    if conversation_service is not None:
        conversation_service.clear_voice_prompt_cache()
    return {"status": "ok", "count": len(voice_samples)}


reload_samples = reload_voices


# ---------------------------------------------------------------------------
# Shared generation logic
# ---------------------------------------------------------------------------


def _build_gen_kwargs(
    text: str,
    sample: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_text: Optional[str] = None,
    num_step: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    t_shift: Optional[float] = None,
    denoise: Optional[bool] = None,
    position_temperature: Optional[float] = None,
    class_temperature: Optional[float] = None,
    layer_penalty_factor: Optional[float] = None,
    duration: Optional[float] = None,
    speed: Optional[float] = None,
    preprocess_prompt: Optional[bool] = None,
    postprocess_output: Optional[bool] = None,
    audio_chunk_duration: Optional[float] = None,
    audio_chunk_threshold: Optional[float] = None,
) -> dict:
    """Build kwargs dict for model.generate() from request parameters."""
    gen_kwargs: dict = {"text": text}

    # Voice cloning mode
    if sample is not None:
        if sample not in voice_samples:
            available = sorted(voice_samples.keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Sample '{sample}' not found.",
                    "available_samples": available,
                },
            )
        sample_info = voice_samples[sample]
        gen_kwargs["ref_audio"] = sample_info["audio_path"]

        # ref_text priority: request body > .txt file > None (auto-transcribe)
        resolved_ref_text = (
            ref_text if ref_text is not None else sample_info["ref_text"]
        )
        if resolved_ref_text is not None:
            gen_kwargs["ref_text"] = resolved_ref_text

    elif instruct is not None:
        gen_kwargs["instruct"] = instruct

    # Generation parameters — only pass if explicitly set
    params = {
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "t_shift": t_shift,
        "denoise": denoise,
        "position_temperature": position_temperature,
        "class_temperature": class_temperature,
        "layer_penalty_factor": layer_penalty_factor,
        "duration": duration,
        "speed": speed,
        "preprocess_prompt": preprocess_prompt,
        "postprocess_output": postprocess_output,
        "audio_chunk_duration": audio_chunk_duration,
        "audio_chunk_threshold": audio_chunk_threshold,
    }
    for key, value in params.items():
        if value is not None:
            gen_kwargs[key] = value

    return gen_kwargs


def _run_model(gen_kwargs: dict) -> tuple[torch.Tensor, float]:
    """
    Run model.generate() and return (audio_tensor, elapsed_seconds).
    Raises HTTPException on failure.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.monotonic()
    try:
        audio_tensors = model.generate(**gen_kwargs)
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")
    elapsed = time.monotonic() - t0

    if not audio_tensors:
        raise HTTPException(status_code=500, detail="Model returned empty audio.")

    return audio_tensors[0], elapsed  # (1, T), seconds


async def _run_model_async(gen_kwargs: dict) -> tuple[torch.Tensor, float]:
    """
    Async wrapper around model.generate() serialized by a global asyncio lock.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.monotonic()
    try:
        async with model_lock:
            audio_tensors = await asyncio.to_thread(model.generate, **gen_kwargs)
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}")
    elapsed = time.monotonic() - t0

    if not audio_tensors:
        raise HTTPException(status_code=500, detail="Model returned empty audio.")

    return audio_tensors[0], elapsed


def _encode_audio(audio: torch.Tensor | np.ndarray, fmt: str, mime_type: str) -> tuple[bytes, str]:
    """
    Encode audio tensor to the requested format.
    Returns (bytes, mime_type).
    For PCM: raw signed 16-bit little-endian mono.
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    elif not isinstance(audio, torch.Tensor):
        raise TypeError(f"audio must be Tensor or ndarray, got {type(audio).__name__}")

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio, got shape {tuple(audio.shape)}")

    audio = audio.detach().cpu().float().clamp_(-1.0, 1.0)

    if fmt == "pcm":
        pcm_data = (audio[0].numpy() * 32767).astype(np.int16).tobytes()
        return pcm_data, "audio/pcm"

    buf = io.BytesIO()
    torchaudio.save(buf, audio, SAMPLE_RATE, format=fmt)
    buf.seek(0)
    return buf.read(), mime_type


def _audio_response(
    audio: torch.Tensor, elapsed: float, fmt: str, mime_type: str, log_tag: str = ""
) -> Response:
    """Build a Response from an audio tensor with timing headers."""
    content, actual_mime = _encode_audio(audio, fmt, mime_type)
    duration_s = audio.shape[-1] / SAMPLE_RATE
    rtf = elapsed / max(duration_s, 0.001)

    log.info(
        "%sGenerated %.2fs audio in %.2fs (RTF=%.3f) format=%s",
        f"[{log_tag}] " if log_tag else "",
        duration_s,
        elapsed,
        rtf,
        fmt,
    )

    return Response(
        content=content,
        media_type=actual_mime,
        headers={
            "X-Audio-Duration": f"{duration_s:.3f}",
            "X-Generation-Time": f"{elapsed:.3f}",
            "X-RTF": f"{rtf:.4f}",
        },
    )


async def _generate_audio(
    gen_kwargs: dict, output_format: Optional[str], sample_name: Optional[str]
) -> Response:
    """Run model.generate() and return audio Response."""
    fmt = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
    if fmt not in MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format '{fmt}'. Use: {list(MIME_TYPES.keys())}",
        )

    audio, elapsed = await _run_model_async(gen_kwargs)
    return _audio_response(
        audio, elapsed, fmt, MIME_TYPES[fmt], log_tag=sample_name or ""
    )


# ---------------------------------------------------------------------------
# TTS endpoints
# ---------------------------------------------------------------------------


@app.post("/tts", tags=["TTS"])
async def synthesize(req: TTSRequest, _: None = Depends(require_api_key)):
    """
    Synthesize speech from text (JSON body).

    Modes (determined by which fields are set):
      1. Voice cloning:  set 'sample' (and optionally 'ref_text')
      2. Voice design:   set 'instruct' (no 'sample')
      3. Auto voice:     neither 'sample' nor 'instruct'
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    gen_kwargs = _build_gen_kwargs(
        text=req.text,
        sample=req.sample,
        instruct=req.instruct,
        ref_text=req.ref_text,
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        t_shift=req.t_shift,
        denoise=req.denoise,
        position_temperature=req.position_temperature,
        class_temperature=req.class_temperature,
        layer_penalty_factor=req.layer_penalty_factor,
        duration=req.duration,
        speed=req.speed,
        preprocess_prompt=req.preprocess_prompt,
        postprocess_output=req.postprocess_output,
        audio_chunk_duration=req.audio_chunk_duration,
        audio_chunk_threshold=req.audio_chunk_threshold,
    )
    return await _generate_audio(gen_kwargs, req.output_format, req.sample)


@app.post("/tts/file", tags=["TTS"])
async def synthesize_from_file(
    _: None = Depends(require_api_key),
    text_file: UploadFile = File(
        ..., description="Text file (.txt) with content to synthesize."
    ),
    sample: Optional[str] = Form(None, description="Voice sample name for cloning."),
    instruct: Optional[str] = Form(
        None, description="Speaker attributes for voice design."
    ),
    ref_text: Optional[str] = Form(
        None, description="Override transcript for the sample."
    ),
    num_step: Optional[int] = Form(None, description="Diffusion steps (default: 32)."),
    guidance_scale: Optional[float] = Form(
        None, description="CFG scale (default: 2.0)."
    ),
    t_shift: Optional[float] = Form(
        None, description="Time-step shift (default: 0.1)."
    ),
    denoise: Optional[bool] = Form(
        None, description="Prepend denoising token (default: true)."
    ),
    position_temperature: Optional[float] = Form(
        None, description="Mask-position temperature (default: 5.0)."
    ),
    class_temperature: Optional[float] = Form(
        None, description="Token sampling temperature (default: 0.0)."
    ),
    layer_penalty_factor: Optional[float] = Form(
        None, description="Codebook layer penalty (default: 5.0)."
    ),
    duration: Optional[float] = Form(
        None, description="Fixed output duration in seconds."
    ),
    speed: Optional[float] = Form(
        None, description="Speed factor (>1 faster, <1 slower)."
    ),
    preprocess_prompt: Optional[bool] = Form(
        None, description="Preprocess reference audio (default: true)."
    ),
    postprocess_output: Optional[bool] = Form(
        None, description="Post-process output (default: true)."
    ),
    audio_chunk_duration: Optional[float] = Form(
        None, description="Chunk duration for long text (default: 15.0)."
    ),
    audio_chunk_threshold: Optional[float] = Form(
        None, description="Chunking activation threshold (default: 30.0)."
    ),
    output_format: Optional[str] = Form(
        None, description="Output format: wav, mp3, flac, ogg."
    ),
):
    """
    Synthesize speech from a text file (multipart/form-data).

    Usage:
        curl -X POST http://localhost:8000/tts/file \\
          -F "text_file=@input.txt" \\
          -F "sample=my-voice" \\
          -F "num_step=32" \\
          -o output.wav
    """
    raw = await text_file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES} bytes.",
        )
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="Text file must be UTF-8 encoded."
        ) from None

    if not text:
        raise HTTPException(status_code=400, detail="Text file is empty.")

    gen_kwargs = _build_gen_kwargs(
        text=text,
        sample=sample,
        instruct=instruct,
        ref_text=ref_text,
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        denoise=denoise,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        layer_penalty_factor=layer_penalty_factor,
        duration=duration,
        speed=speed,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
    )
    return await _generate_audio(gen_kwargs, output_format, sample)


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints  (/v1/audio/speech)
# ---------------------------------------------------------------------------


def _openai_error(
    status_code: int, message: str, error_type: str = "invalid_request_error"
) -> JSONResponse:
    """Return an error response in the OpenAI error envelope format."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.post("/v1/audio/speech", tags=["OpenAI-compatible"])
async def openai_speech(req: OpenAISpeechRequest, _: None = Depends(require_api_key)):
    """
    OpenAI-compatible TTS endpoint.

    Drop-in replacement for https://api.openai.com/v1/audio/speech.

    Voice resolution order:
      1. If a voice sample named exactly ``voice`` is loaded → voice cloning mode.
      2. Otherwise → auto-voice mode (model picks a voice).

    Model → quality mapping:
      ``tts-1``     → 16 diffusion steps  (faster)
      ``tts-1-hd``  → 32 diffusion steps  (higher quality)
      any other     → server default

    Example::

        curl http://localhost:8000/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{"model":"tts-1","input":"Hello world","voice":"alloy"}' \\
          --output speech.mp3
    """
    if not req.input.strip():
        return _openai_error(400, "Field 'input' must not be empty.")

    fmt_key = req.response_format.lower()
    if fmt_key not in OPENAI_FORMAT_MAP:
        return _openai_error(
            400,
            f"Unsupported response_format '{req.response_format}'. "
            f"Supported: {list(OPENAI_FORMAT_MAP.keys())}",
        )
    internal_fmt, mime_type = OPENAI_FORMAT_MAP[fmt_key]

    # Resolve voice → sample or auto
    voice_key = req.voice.lower()
    matched_sample: Optional[str] = None
    if voice_key in voice_samples:
        matched_sample = voice_key
    elif req.voice in voice_samples:
        matched_sample = req.voice

    # Map model name to diffusion steps
    num_step = OPENAI_MODEL_STEPS.get(req.model)

    try:
        gen_kwargs = _build_gen_kwargs(
            text=req.input,
            sample=matched_sample,
            speed=req.speed,
            num_step=num_step,
        )
        audio, elapsed = await _run_model_async(gen_kwargs)
    except HTTPException as exc:
        return _openai_error(
            exc.status_code,
            exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            error_type="server_error",
        )
    except Exception as exc:
        log.exception("OpenAI endpoint: unexpected generation error")
        return _openai_error(500, f"Generation error: {exc}", error_type="server_error")

    return _audio_response(
        audio,
        elapsed,
        internal_fmt,
        mime_type,
        log_tag=f"openai voice={req.voice} model={req.model}",
    )


@app.get("/v1/models", tags=["OpenAI-compatible"])
async def openai_list_models(_: None = Depends(require_api_key)):
    """
    OpenAI-compatible model list endpoint.

    Returns the two TTS model IDs recognized by /v1/audio/speech.
    """
    now = int(time.time())
    models = [
        {"id": "tts-1", "object": "model", "created": now, "owned_by": "omnivoice"},
        {"id": "tts-1-hd", "object": "model", "created": now, "owned_by": "omnivoice"},
    ]
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
