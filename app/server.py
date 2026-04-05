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
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

# Globals populated at startup
model = None
voice_samples: dict = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    global model, voice_samples

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

    yield  # Application runs here

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


def _generate_audio(gen_kwargs: dict, output_format: Optional[str], sample_name: Optional[str]) -> Response:
    """Run model.generate() and return audio Response."""
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

    audio = audio_tensors[0]  # (1, T)

    fmt = (output_format or DEFAULT_OUTPUT_FORMAT).lower()
    if fmt not in MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format '{fmt}'. Use: {list(MIME_TYPES.keys())}",
        )

    buf = io.BytesIO()
    torchaudio.save(buf, audio.cpu(), SAMPLE_RATE, format=fmt)
    buf.seek(0)

    duration_s = audio.shape[-1] / SAMPLE_RATE
    log.info(
        "Generated %.2fs audio in %.2fs (RTF=%.3f) format=%s sample=%s",
        duration_s, elapsed, elapsed / max(duration_s, 0.001), fmt,
        sample_name or "(none)",
    )

    return Response(
        content=buf.read(),
        media_type=MIME_TYPES[fmt],
        headers={
            "X-Audio-Duration": f"{duration_s:.3f}",
            "X-Generation-Time": f"{elapsed:.3f}",
            "X-RTF": f"{elapsed / max(duration_s, 0.001):.4f}",
        },
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
        raise HTTPException(status_code=400, detail="Text file must be UTF-8 encoded.")

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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
