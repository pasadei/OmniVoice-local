#!/usr/bin/env python3
"""
OmniVoice TTS HTTP Server
=========================

Purpose:
    FastAPI-based HTTP server wrapping the OmniVoice TTS model.
    Provides voice cloning via a pre-configured sample directory,
    voice design via speaker attribute instructions, and auto-voice mode.
    All OmniVoice generation parameters are exposed through the API.

    Optionally serves the built-in OmniVoice Gradio demo UI on a separate
    port, sharing the same loaded model instance (no extra VRAM).

Usage:
    # Start with default settings:
    python server.py

    # Custom GPU and sample directory:
    OMNIVOICE_DEVICE=cuda:1 OMNIVOICE_SAMPLES_DIR=/samples python server.py

    # Inside container (typical):
    # Model and samples are mounted as bind volumes, see compose.yaml

Environment variables:
    OMNIVOICE_MODEL          - HuggingFace model ID or local path
                               (default: k2-fsa/OmniVoice)
    OMNIVOICE_DEVICE         - PyTorch device string
                               (default: cuda:0)
    OMNIVOICE_DTYPE          - Model dtype: float16, bfloat16, float32
                               (default: float16)
    OMNIVOICE_SAMPLES_DIR    - Path to voice samples directory
                               (default: /samples)
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
"""

import hmac
import io
import json
import logging
import os
import sys
import threading
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

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
MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
DTYPE_STR = os.environ.get("OMNIVOICE_DTYPE", "float16")
SAMPLES_DIR = Path(os.environ.get("OMNIVOICE_SAMPLES_DIR", "/samples"))
HOST = os.environ.get("OMNIVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("OMNIVOICE_PORT", "8000"))
DEFAULT_OUTPUT_FORMAT = os.environ.get("OMNIVOICE_OUTPUT_FORMAT", "wav")
GRADIO_ENABLED = os.environ.get("OMNIVOICE_GRADIO_ENABLED", "true").lower() in ("true", "1", "yes")
GRADIO_PORT = int(os.environ.get("OMNIVOICE_GRADIO_PORT", "8001"))
WYOMING_ENABLED = os.environ.get("OMNIVOICE_WYOMING_ENABLED", "false").lower() in ("true", "1", "yes")
WYOMING_HOST = os.environ.get("OMNIVOICE_WYOMING_HOST", "0.0.0.0")
WYOMING_PORT = int(os.environ.get("OMNIVOICE_WYOMING_PORT", "10200"))
WYOMING_LANGUAGES = os.environ.get("OMNIVOICE_WYOMING_LANGUAGES", "en")
API_KEY = os.environ.get("OMNIVOICE_API_KEY", "").strip()
CORS_ORIGINS = os.environ.get("OMNIVOICE_CORS_ORIGINS", "").strip()
MAX_UPLOAD_BYTES = int(os.environ.get("OMNIVOICE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB

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
    "mp3":  ("mp3",  "audio/mpeg"),
    "opus": ("ogg",  "audio/ogg"),
    "aac":  ("mp3",  "audio/mpeg"),   # best-effort fallback
    "flac": ("flac", "audio/flac"),
    "wav":  ("wav",  "audio/wav"),
    "pcm":  ("pcm",  "audio/pcm"),    # raw s16le, handled specially
}

# OpenAI model → diffusion steps
OPENAI_MODEL_STEPS = {
    "tts-1":    16,
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
# Sample directory scanner
# ---------------------------------------------------------------------------

def scan_samples(directory: Path) -> dict:
    """
    Scan the samples directory for audio + transcript pairs.

    Expected structure:
        some-name.wav   + some-name.txt
        another.mp3     + another.txt

    Audio and transcript files are matched by stem (filename without extension).
    If a .txt transcript is missing, the sample is still loaded but ref_text
    will be None (OmniVoice will use Whisper to auto-transcribe).

    Returns:
        dict[str, dict] mapping sample name -> {audio_path, ref_text}
    """
    samples = {}
    if not directory.is_dir():
        log.warning("Samples directory does not exist: %s", directory)
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
            "Name of a voice sample (stem without extension) from the "
            "samples directory. Enables voice cloning mode."
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
    """Info about a single voice sample."""
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
            "onyx, sage, shimmer) map to a voice sample of the same name when available, "
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


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    global model, voice_samples, wyoming_server

    dtype = DTYPE_MAP.get(DTYPE_STR)
    if dtype is None:
        log.error("Invalid OMNIVOICE_DTYPE=%s. Use: float16, bfloat16, float32", DTYPE_STR)
        sys.exit(1)

    if API_KEY:
        log.info("API key authentication: ENABLED")
    else:
        log.warning("API key authentication: DISABLED (set OMNIVOICE_API_KEY to enable)")

    log.info("Loading OmniVoice model: %s", MODEL_ID)
    log.info("  device=%s  dtype=%s", DEVICE, DTYPE_STR)

    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=dtype,
    )
    log.info("Model loaded successfully.")

    log.info("Scanning samples directory: %s", SAMPLES_DIR)
    voice_samples = scan_samples(SAMPLES_DIR)

    # Launch Gradio UI in a background thread, sharing the same model
    if GRADIO_ENABLED:
        _launch_gradio()

    if WYOMING_ENABLED:
        wyoming_server = _launch_wyoming_server()

    yield  # Application runs here

    if wyoming_server is not None:
        wyoming_server.stop()
        wyoming_server = None

    log.info("Shutting down OmniVoice server.")


app = FastAPI(
    title="OmniVoice TTS Server",
    description=(
        "HTTP API for OmniVoice zero-shot multilingual TTS. "
        "Supports voice cloning (via pre-loaded samples), "
        "voice design (via speaker attribute instructions), "
        "and auto-voice mode."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (enabled when OMNIVOICE_CORS_ORIGINS is set)
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")],
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
        )

    thread = threading.Thread(target=_run, name="gradio-ui", daemon=True)
    thread.start()
    log.info("Gradio UI thread started.")


# ---------------------------------------------------------------------------
# Wyoming protocol integration
# ---------------------------------------------------------------------------

class _WyomingServer:
    """Minimal Wyoming TCP server for Home Assistant integration."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._thread = threading.Thread(target=self._run, name="wyoming-server", daemon=True)
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
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        log.info("Wyoming TCP server listening on %s:%d", self.host, self.port)
        self._ready.set()
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        synth_state = {"voice": None, "chunks": []}
        peer = writer.get_extra_info("peername")
        log.info("Wyoming client connected: %s", peer)
        try:
            while True:
                event = await _wyoming_read_event(reader)
                if event is None:
                    break
                event_type = event.get("type")
                data = event.get("data") or {}

                if event_type == "describe":
                    await _wyoming_send_event(writer, _wyoming_info_event())
                elif event_type == "synthesize":
                    text = str(data.get("text", "")).strip()
                    if text:
                        voice = data.get("voice")
                        await _wyoming_send_tts(writer, text, voice)
                elif event_type == "synthesize-start":
                    synth_state["voice"] = data.get("voice")
                    synth_state["chunks"] = []
                elif event_type == "synthesize-chunk":
                    chunk = str(data.get("text", ""))
                    if chunk:
                        synth_state["chunks"].append(chunk)
                elif event_type == "synthesize-stop":
                    text = "".join(synth_state["chunks"]).strip()
                    if text:
                        await _wyoming_send_tts(writer, text, synth_state["voice"])
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
        "data": {
            "tts": [tts_program]
        },
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
        for attr in ("_LANGUAGE_CODES", "LANGUAGE_CODES", "language_codes", "languages"):
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
                    discovered = [str(code).strip() for code in value if str(code).strip()]
            except Exception as exc:
                log.debug("Failed to get languages from model.get_supported_languages: %s", exc)

    if discovered:
        return sorted(set(discovered))

    fallback = [code.strip() for code in WYOMING_LANGUAGES.split(",") if code.strip()]
    if not fallback:
        fallback = ["en"]
    return sorted(set(fallback))


async def _wyoming_send_tts(writer: asyncio.StreamWriter, text: str, voice: Optional[dict]):
    sample = None
    if isinstance(voice, dict):
        candidate = voice.get("name") or voice.get("speaker")
        if isinstance(candidate, str):
            sample = candidate

    try:
        gen_kwargs = _build_gen_kwargs(text=text, sample=sample if sample in voice_samples else None)
        audio, elapsed = _run_model(gen_kwargs)
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
        {"type": "audio-start", "data": {"rate": SAMPLE_RATE, "width": 2, "channels": 1}},
    )
    chunk_size = 8192
    for i in range(0, len(pcm), chunk_size):
        await _wyoming_send_event(
            writer,
            {
                "type": "audio-chunk",
                "data": {"rate": SAMPLE_RATE, "width": 2, "channels": 1},
                "payload": pcm[i:i + chunk_size],
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

@app.get("/health", tags=["System"])
async def health():
    """Health check. Returns 503 while the model is still loading."""
    ready = model is not None
    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if ready else "loading",
            "model": MODEL_ID,
            "device": DEVICE,
            "gradio_enabled": GRADIO_ENABLED,
            "gradio_port": GRADIO_PORT if GRADIO_ENABLED else None,
        },
    )


@app.get("/samples", response_model=list[SampleInfo], tags=["Samples"])
async def list_samples(_: None = Depends(require_api_key)):
    """List all available voice samples."""
    result = []
    for name, info in sorted(voice_samples.items()):
        preview = None
        if info["ref_text"]:
            preview = info["ref_text"][:120] + ("..." if len(info["ref_text"]) > 120 else "")
        result.append(SampleInfo(
            name=name,
            audio_file=Path(info["audio_path"]).name,
            has_transcript=info["ref_text"] is not None,
            transcript_preview=preview,
        ))
    return result


@app.post("/samples/reload", tags=["Samples"])
async def reload_samples(_: None = Depends(require_api_key)):
    """Re-scan the samples directory (e.g. after adding new files)."""
    global voice_samples
    voice_samples = scan_samples(SAMPLES_DIR)
    return {"status": "ok", "count": len(voice_samples)}


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
        resolved_ref_text = ref_text if ref_text is not None else sample_info["ref_text"]
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


def _encode_audio(audio: torch.Tensor, fmt: str, mime_type: str) -> tuple[bytes, str]:
    """
    Encode audio tensor to the requested format.
    Returns (bytes, mime_type).
    For PCM: raw signed 16-bit little-endian mono.
    """
    if fmt == "pcm":
        samples = audio[0].cpu().float().numpy()
        samples = np.clip(samples, -1.0, 1.0)
        pcm_data = (samples * 32767).astype(np.int16).tobytes()
        return pcm_data, "audio/pcm"

    buf = io.BytesIO()
    torchaudio.save(buf, audio.cpu(), SAMPLE_RATE, format=fmt)
    buf.seek(0)
    return buf.read(), mime_type


def _audio_response(audio: torch.Tensor, elapsed: float, fmt: str, mime_type: str, log_tag: str = "") -> Response:
    """Build a Response from an audio tensor with timing headers."""
    content, actual_mime = _encode_audio(audio, fmt, mime_type)
    duration_s = audio.shape[-1] / SAMPLE_RATE
    rtf = elapsed / max(duration_s, 0.001)

    log.info(
        "%sGenerated %.2fs audio in %.2fs (RTF=%.3f) format=%s",
        f"[{log_tag}] " if log_tag else "",
        duration_s, elapsed, rtf, fmt,
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


def _generate_audio(gen_kwargs: dict, output_format: Optional[str], sample_name: Optional[str]) -> Response:
    """Run model.generate() and return audio Response."""
    fmt = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
    if fmt not in MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format '{fmt}'. Use: {list(MIME_TYPES.keys())}",
        )

    audio, elapsed = _run_model(gen_kwargs)
    return _audio_response(audio, elapsed, fmt, MIME_TYPES[fmt], log_tag=sample_name or "")


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
    return _generate_audio(gen_kwargs, req.output_format, req.sample)


@app.post("/tts/file", tags=["TTS"])
async def synthesize_from_file(
    _: None = Depends(require_api_key),
    text_file: UploadFile = File(..., description="Text file (.txt) with content to synthesize."),
    sample: Optional[str] = Form(None, description="Voice sample name for cloning."),
    instruct: Optional[str] = Form(None, description="Speaker attributes for voice design."),
    ref_text: Optional[str] = Form(None, description="Override transcript for the sample."),
    num_step: Optional[int] = Form(None, description="Diffusion steps (default: 32)."),
    guidance_scale: Optional[float] = Form(None, description="CFG scale (default: 2.0)."),
    t_shift: Optional[float] = Form(None, description="Time-step shift (default: 0.1)."),
    denoise: Optional[bool] = Form(None, description="Prepend denoising token (default: true)."),
    position_temperature: Optional[float] = Form(None, description="Mask-position temperature (default: 5.0)."),
    class_temperature: Optional[float] = Form(None, description="Token sampling temperature (default: 0.0)."),
    layer_penalty_factor: Optional[float] = Form(None, description="Codebook layer penalty (default: 5.0)."),
    duration: Optional[float] = Form(None, description="Fixed output duration in seconds."),
    speed: Optional[float] = Form(None, description="Speed factor (>1 faster, <1 slower)."),
    preprocess_prompt: Optional[bool] = Form(None, description="Preprocess reference audio (default: true)."),
    postprocess_output: Optional[bool] = Form(None, description="Post-process output (default: true)."),
    audio_chunk_duration: Optional[float] = Form(None, description="Chunk duration for long text (default: 15.0)."),
    audio_chunk_threshold: Optional[float] = Form(None, description="Chunking activation threshold (default: 30.0)."),
    output_format: Optional[str] = Form(None, description="Output format: wav, mp3, flac, ogg."),
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
        raise HTTPException(status_code=400, detail="Text file must be UTF-8 encoded.") from None

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
    return _generate_audio(gen_kwargs, output_format, sample)


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints  (/v1/audio/speech)
# ---------------------------------------------------------------------------

def _openai_error(status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    """Return an error response in the OpenAI error envelope format."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
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
        audio, elapsed = _run_model(gen_kwargs)
    except HTTPException as exc:
        return _openai_error(exc.status_code, exc.detail if isinstance(exc.detail, str) else str(exc.detail), error_type="server_error")
    except Exception as exc:
        log.exception("OpenAI endpoint: unexpected generation error")
        return _openai_error(500, f"Generation error: {exc}", error_type="server_error")

    return _audio_response(audio, elapsed, internal_fmt, mime_type, log_tag=f"openai voice={req.voice} model={req.model}")


@app.get("/v1/models", tags=["OpenAI-compatible"])
async def openai_list_models(_: None = Depends(require_api_key)):
    """
    OpenAI-compatible model list endpoint.

    Returns the two TTS model IDs recognized by /v1/audio/speech.
    """
    now = int(time.time())
    models = [
        {"id": "tts-1",    "object": "model", "created": now, "owned_by": "omnivoice"},
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
