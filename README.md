# OmniVoice TTS Server

OmniVoice TTS server with Gradio web UI, REST API, OpenAI-compatible endpoint, and Wyoming (Home Assistant) endpoint sharing a single model instance. Voice cloning via voices directory, voice design, full generation parameter control. NVIDIA GPU support (Ampere, Ada Lovelace, Blackwell). Docker/Podman-compose ready.

Based on [OmniVoice](https://github.com/k2-fsa/OmniVoice) — zero-shot multilingual TTS for 600+ languages. Thanks to the [k2-fsa](https://github.com/k2-fsa) team for open-sourcing the model.

## Overview

The server loads OmniVoice once and exposes it through multiple interfaces simultaneously:

- **REST API** (port `8000`) — FastAPI with JSON and multipart/form-data endpoints, full parameter control
- **OpenAI-Compatible API** (port `8000`) — drop-in replacement for the OpenAI TTS API (`/v1/audio/speech`, `/v1/models`), works with the official OpenAI SDK
- **WebSocket Streaming API** (port `8000`) — sentence-level progressive audio streaming (`/ws/tts`)
- **Full-Duplex Conversation** (port `8000`) — real-time bidirectional voice conversation via `/ws/conversation` with browser-based test UI at `/duplex`
- **Wyoming TCP API** (port `10200`) — Home Assistant compatible TTS service for Assist pipelines

All interfaces share the same model instance — no extra VRAM.

**Three generation modes:**

- **Voice cloning** — pick a named voice from the canonical voices directory
- **Voice design** — describe the voice with attributes (`female, low pitch, british accent`)
- **Auto voice** — let the model choose a voice automatically

## Directory Structure

```
omnivoice-server/
├── compose.yaml         # Docker/Podman compose
├── Dockerfile
├── README.md
├── app/
│   └── server.py        # FastAPI server + Gradio launcher
└── voices/              # Voice sample pairs (bind-mounted)
    ├── john-doe.wav
    ├── john-doe.txt
    ├── narrator.mp3
    └── narrator.txt
```

## Voices

Place audio files and their transcripts in `./voices/`. Files are paired by stem (filename without extension):

```
my-voice.wav    ←  audio file (wav, mp3, flac, ogg, opus, m4a, aac)
my-voice.txt    ←  plain text transcript of the audio
```

The transcript file is optional — if missing, OmniVoice will auto-transcribe the audio using Whisper at request time (slower, slightly less accurate).

`voices` is the canonical public terminology and `./voices/` is the canonical asset directory. Legacy `/samples` endpoints and wire fields such as `sample` and `available_samples` intentionally remain compatibility names for now.

Legacy deployments that still keep assets only in `./samples/` continue to work. The server scans `./voices/` first and automatically falls back to the legacy samples mount when the canonical directory is empty.

You can add or remove voices at runtime and call `POST /voices/reload` to re-scan. Compatibility aliases remain available at `GET /samples` and `POST /samples/reload`.

## Quick Start

```bash
# 1. Create the canonical voice directory and place assets there
mkdir -p ./voices
cp my-voice.wav my-voice.txt ./voices/

# 2. Configure local secrets (optional but recommended)
cp .env.example .env
# Edit .env for local-only secrets:
# - set HF_TOKEN if you need Hugging Face access for gated/rate-limited assets
# - set OMNIVOICE_API_KEY if you want bearer-token auth and enable its compose.yaml mapping
# .env is local and ignored by git; compose.yaml stays safe to commit because it only references env vars

# 3. Build and start
podman-compose up -d --build
# or: docker compose up -d --build

# 4. Check health (model loading takes ~60-120s)
curl http://localhost:8000/health

# 5. Open Gradio UI
# http://localhost:8001

# 6. Generate speech via API
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "sample": "my-voice"}' \
  -o output.wav
```

If you enable API key auth, keep the secret in `.env` and uncomment the `OMNIVOICE_API_KEY=${OMNIVOICE_API_KEY}` reference in `compose.yaml`. In this compose setup, setting `OMNIVOICE_API_KEY` in `.env` alone does not reach the container until that mapping is enabled.


### Wyoming / Home Assistant

Enable Wyoming by setting:

- `OMNIVOICE_WYOMING_ENABLED=true`
- `OMNIVOICE_WYOMING_HOST=0.0.0.0`
- `OMNIVOICE_WYOMING_PORT=10200`

In Home Assistant add a Wyoming provider pointing to this container IP and port `10200`.

Supported events:
- `describe` → returns `info` with TTS model metadata and available sample speakers
- `synthesize` → returns PCM audio stream (`audio-start`/`audio-chunk`/`audio-stop`)
- `synthesize-start`/`synthesize-chunk`/`synthesize-stop` streaming text mode

## API Reference

## WebSocket Streaming API

Endpoint: `ws://localhost:8000/ws/tts`

Current auth behavior: both `/ws/tts` and `/ws/conversation` currently bypass `OMNIVOICE_API_KEY`. REST and OpenAI-style HTTP endpoints can be protected with bearer-token auth when `OMNIVOICE_API_KEY` is set, but the current WebSocket implementation accepts direct connections without WebSocket authentication headers or tokens.

Supported inbound messages:

- `{"type":"synthesize","text":"...","sample":"optional","instruct":"optional","output_format":"pcm|wav","num_step":16,"speed":1.0}`
- `{"type":"text_chunk","text":"..."}` (incremental LLM token stream mode)
- `{"type":"text_flush"}` (force synthesis of buffered text)
- `{"type":"set_voice","sample":"my-voice","ref_text":"optional override"}`. If the sample is unknown, the server currently sends an `error` event and closes the connection.

Server outbound messages:

- `audio_chunk` JSON metadata message followed by a binary audio message (`pcm` or `wav`) for each synthesized sentence.
- `voice_set` acknowledgment after a successful `set_voice` request, for example `{"type":"voice_set","sample":"my-voice"}`.
- `done` once the current operation completes. Its numeric totals (`total_chunks`, `total_duration_ms`, `total_generation_time_ms`) are cumulative for the current WebSocket session, not just the most recent operation.
- `error` for invalid messages / unknown samples.

WebSocket voice selection rules:

- `sample` is the compatibility field name for selecting a loaded voice from `GET /voices`.
- When `sample` resolves to a loaded voice, cloning wins and `instruct` is ignored.
- When `sample` is omitted, `/ws/tts` uses `instruct` for voice design.
- When `sample` is present but unknown, `/ws/tts` falls back to `instruct` only if `instruct` was also provided; `set_voice` still treats an unknown sample as a session error.
- For incremental `/ws/tts` streaming, `instruct` supplied on `text_chunk` is reused by later chunks and `text_flush` until the buffered utterance completes.

Voice clone prompts are cached per WebSocket session and reused for each chunk until `set_voice` is called again.

### Remote playback client

The real playback workflow lives in `examples/ws_playback_client.py`. It is meant to run on the machine that has the speakers or headphones attached, while the OmniVoice server can run on another host.

Because `/ws/tts` is currently unauthenticated, the example connects directly to the WebSocket URL and does not send an API key.

The playback client depends on `websockets` and `pyaudio`:

```bash
python -m pip install websockets pyaudio
```

Debian/Ubuntu: if `pyaudio` wheels are unavailable, install PortAudio headers first with `sudo apt install portaudio19-dev python3-pyaudio` and retry the pip install in your environment.

macOS: install PortAudio first with `brew install portaudio`, then run `python -m pip install pyaudio websockets`.

Live playback only supports `pcm`. The example client rejects any other `--output-format` because it writes raw 24 kHz mono PCM frames directly to the local audio device.

```bash
python examples/ws_playback_client.py \
  --url ws://server-host:8000/ws/tts \
  --text "Hello from the remote playback client." \
  --sample your-sample-name \
  --output-format pcm
```

The client prints each `audio_chunk` / `done` event as JSON and plays the matching binary PCM frames through the local speakers.

### Experimental full duplex conversation client

`examples/ws_duplex_client.py` is an experimental workstation-side client for `/ws/conversation`. It captures 16 kHz microphone audio, uses local Silero VAD to bracket turns, and plays the assistant's streamed 24 kHz PCM response through the local speakers.

**Browser-based test UI:** A standalone HTML test page is available at `http://localhost:8000/duplex`. This page provides a web interface for testing full-duplex conversations directly in your browser without requiring the Python client. The page automatically detects the WebSocket URL and loads available voices.

**HTTPS requirement for microphone access:** Modern browsers require HTTPS (or WSS) for microphone access via `getUserMedia()`, except on `localhost`. For remote testing over HTTPS, use a tunneling service like ngrok:

```bash
# Install ngrok from https://ngrok.com/download
ngrok http 8000
```

ngrok will provide an HTTPS URL (e.g., `https://abc123.ngrok.io`). Open `https://abc123.ngrok.io/duplex` in your browser to access the full-duplex test UI with microphone support.

Caveats:

- The client is experimental and documents the current reference flow rather than a stable public interface.
- It assumes a 16 kHz mono microphone uplink and 24 kHz mono PCM speaker playback.
- `/ws/conversation` currently bypasses `OMNIVOICE_API_KEY`, so the example connects directly without sending an API key.
- The smoke flow below is an operator runbook for a real workstation/server setup and was not manually validated in this environment.

Workstation dependencies:

- `websockets`
- `pyaudio`
- `silero-vad`
- `torch`

```bash
python -m pip install websockets pyaudio silero-vad torch
```

Server-side conversation dependencies:

- `faster-whisper`
- `ollama` Python client
- Reachable Ollama server

```bash
python -m pip install faster-whisper ollama
```

The server-side conversation path uses `faster-whisper` for ASR and Ollama for assistant text generation. By default it connects to `OMNIVOICE_OLLAMA_HOST=http://localhost:11434` and requests `OMNIVOICE_OLLAMA_MODEL=gemma4`.

Assistant governance behavior for `/ws/conversation`:

- Each WebSocket session gets an internal `session_id` that is passed to the assistant backend for turn correlation.
- The server keeps a short rolling history of the latest completed turns and reuses it on later assistant requests to keep replies grounded in the ongoing call.
- Language selection is sticky per session: an explicit `session_start.language` override wins, otherwise the latest ASR-detected language is reused as the next `language_hint`.
- Assistant text is sanitized before TTS to strip common formatting noise and obvious wrapper content so spoken output stays concise and TTS-friendly for the existing duplex protocol.
- After each completed turn, the server emits one JSON info log for operators with `session_id`, `response_id`, detected language, language hint, assistant backend/model, and per-stage latency fields (`asr_ms`, `llm_ms`, `tts_first_chunk_ms`, `tts_total_ms`, `turn_total_ms`). This is server-side diagnostics only; no new WebSocket messages are exposed to clients.
- `session_start` also accepts optional `instruct` voice-design text. If `sample` is absent, later TTS turns use that `instruct`; if both `sample` and `instruct` are provided, the selected voice still wins and `instruct` is ignored for TTS.

### `/ws/conversation` protocol

Endpoint: `ws://localhost:8000/ws/conversation`

Inbound JSON messages:

- `{"type":"session_start","sample":"optional","instruct":"optional","sample_rate":16000,"language":"optional"}` starts or restarts the session. `sample` and `instruct` are supplied here, not on later turn messages. If both are present, `sample` wins for TTS. Sending a new `session_start` resets session state, clears buffered audio/history, assigns a new internal session id, and interrupts any active response.
- `{"type":"speech_start"}` begins caller audio capture for the next turn.
- `{"type":"speech_end"}` ends caller audio capture and triggers ASR plus assistant/TTS processing.
- `{"type":"input_audio_chunk","audio":"<base64 pcm16le>"}` appends audio as JSON when binary frames are inconvenient.
- Binary WebSocket frames can also be sent directly during active capture and are treated as raw 16 kHz mono PCM audio.

Outbound JSON messages:

- `session_started` acknowledges `session_start` with the active `sample`, `sample_rate`, and optional `language` override.
- `listening` confirms that the server is collecting caller audio.
- `transcript_final` delivers the final ASR text for the captured turn.
- `assistant_text` streams assistant text chunks for the current `response_id`.
- `audio_chunk` sends metadata for a synthesized assistant audio chunk; the matching binary audio payload is sent in the next WebSocket frame.
- `response_done` marks the end of the current assistant response.
- `interrupted` reports that an in-flight response was cancelled, usually because the caller barged in or a new `session_start` replaced the session.
- `error` reports recoverable protocol or validation problems.

Basic turn lifecycle:

1. Client sends `session_start`.
2. Server replies with `session_started`.
3. Client sends `speech_start`, then one or more binary frames or `input_audio_chunk` messages.
4. Client sends `speech_end`.
5. Server replies with `transcript_final`, then zero or more `assistant_text` and `audio_chunk` events, each `audio_chunk` immediately followed by its binary audio frame.
6. Server finishes the turn with `response_done`.

Example invocation:

```bash
python examples/ws_duplex_client.py \
  --url ws://server-host:8000/ws/conversation \
  --sample your-sample-name \
  --language it
```

Use `--language <code>` when you want to force the ASR language for the whole session, for example `--language it` for Italian. If omitted, the server falls back to per-session language detection and sticky language hints.

You can also use inferred voice design directly from the duplex client:

```bash
python examples/ws_duplex_client.py \
  --url ws://server-host:8000/ws/conversation \
  --instruct "female, calm, warm voice" \
  --language it
```

If you provide both `--sample` and `--instruct`, the client forwards both and the server keeps the canonical precedence rule: the selected `sample` wins and `instruct` is ignored for TTS.

```bash
python examples/ws_duplex_client.py \
  --url ws://server-host:8000/ws/conversation \
  --sample your-sample-name \
  --instruct "male, deep, robotic voice" \
  --language it
```

Manual duplex smoke flow for a real workstation + server setup:

```bash
# Prerequisite: start or point to a reachable Ollama server and ensure the model exists.
# Example:
#   export OMNIVOICE_OLLAMA_HOST=http://localhost:11434
#   ollama pull gemma4
docker compose up -d --build
python examples/ws_duplex_client.py \
  --url ws://localhost:8000/ws/conversation \
  --sample your-sample-name \
  --language it
```

Or test inferred voice design instead of cloning:

```bash
python examples/ws_duplex_client.py \
  --url ws://localhost:8000/ws/conversation \
  --instruct "female, calm, warm voice" \
  --language it
```

Expected behavior:

- The operator speaks first to start the turn.
- The assistant responds after that utterance is transcribed.
- The user barges in while playback is active.
- Local playback stops immediately.
- A new assistant response starts after the interruption turn is transcribed.

### Manual real-stack smoke test

This is a manual integration check that complements the pytest coverage.

Prerequisites:

- Docker Compose installed
- Enough time for the initial model download and startup warmup
- NVIDIA GPU access with a working CUDA container runtime if you use the stock `compose.yaml`; edit the compose file first if you need a CPU-only or non-NVIDIA deployment
- At least one valid voice available in `./voices`
- `websocat` installed on the machine running the test

Start the stack:

```bash
docker compose up -d --build
```

Then poll readiness until the model finishes loading. A single early `curl` may fail or report that startup is still in progress while weights download and the model initializes:

```bash
curl http://localhost:8000/health
```

Repeat that check until `/health` reports ready, then open a WebSocket session:

```bash
websocat ws://localhost:8000/ws/tts
```

Then send a synthesize request that references a real voice from `./voices`:

```json
{"type":"synthesize","text":"Ciao! Come posso aiutarti oggi?","sample":"your-sample-name","output_format":"pcm","num_step":16}
```

Expected result:

- One or more `audio_chunk` JSON events appear first
- Each `audio_chunk` event is followed by a binary payload in the terminal output
- A final `done` event arrives after the last chunk

If you want local speaker playback instead of inspecting the raw binary frames in the terminal, run `examples/ws_playback_client.py` from the playback machine against the same `ws://.../ws/tts` endpoint.

### `GET /health`

Health check. Returns model status, device info, and Gradio state. It returns HTTP `503` while the model is still loading and HTTP `200` once the server is ready to accept synthesis requests.

### `GET /voices`

List all loaded voices from the canonical `./voices/` directory. `GET /samples` remains available as a compatibility alias.

**Response example:**

```json
[
  {
    "name": "john-doe",
    "audio_file": "john-doe.wav",
    "has_transcript": true,
    "transcript_preview": "This is a sample recording of my voice..."
  }
]
```

### `POST /voices/reload`

Re-scan the canonical voices directory. Use after adding/removing voice files without restarting the container. `POST /samples/reload` remains available as a compatibility alias.

### `POST /tts`

Synthesize speech from JSON body. Returns raw audio bytes with `Content-Type` matching the requested format.

**Response headers:**

| Header | Description |
|---|---|
| `X-Audio-Duration` | Generated audio duration in seconds |
| `X-Generation-Time` | Wall-clock generation time in seconds |
| `X-RTF` | Real-time factor (generation time / audio duration) |

### `POST /tts/file`

Synthesize speech from a text file via `multipart/form-data`. Same parameters as `/tts`, but text is uploaded as a file instead of passed in JSON.

```bash
curl -X POST http://localhost:8000/tts/file \
  -F "text_file=@input.txt" \
  -F "sample=my-voice" \
  -F "num_step=32" \
  -F "speed=0.9" \
  -o output.wav
```

### Request Parameters

**Voice selection** (pick one approach or neither for auto-voice):

| Field | Type | Description |
|---|---|---|
| `text` | string | **Required.** Text to synthesize. Supports approved expressive tags like `[laughter]` plus pronunciation control. |
| `sample` | string | Compatibility alias for a voice name (stem) used for voice cloning. Must exist in the voices directory. |
| `instruct` | string | Speaker attributes for voice design, e.g. `"female, low pitch, british accent"`. Ignored if `sample` is set. |
| `ref_text` | string | Override transcript for the sample audio. Falls back to .txt file, then Whisper. |

**Decoding parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `num_step` | int | 32 | Diffusion unmasking steps. 16 = faster, 32 = higher quality. |
| `guidance_scale` | float | 2.0 | Classifier-free guidance scale. |
| `t_shift` | float | 0.1 | Time-step shift for noise schedule. |
| `denoise` | bool | true | Prepend denoising token. |

**Sampling parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `position_temperature` | float | 5.0 | Temperature for mask-position selection. 0 = greedy. |
| `class_temperature` | float | 0.0 | Temperature for token sampling. 0 = greedy. |
| `layer_penalty_factor` | float | 5.0 | Penalty for deeper codebook layers. |

**Duration and speed:**

| Field | Type | Default | Description |
|---|---|---|---|
| `duration` | float | null | Fixed output duration in seconds. Overrides `speed`. |
| `speed` | float | null | Speed factor. >1.0 = faster speech, <1.0 = slower. Ignored when `duration` is set. |

**Pre/post processing:**

| Field | Type | Default | Description |
|---|---|---|---|
| `preprocess_prompt` | bool | true | Remove long silences from reference audio, add punctuation to ref text. |
| `postprocess_output` | bool | true | Remove long silences from generated audio. |

**Long-form generation:**

| Field | Type | Default | Description |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | Target chunk duration in seconds when splitting long text. |
| `audio_chunk_threshold` | float | 30.0 | Estimated audio duration above which chunking activates. |

**Output:**

| Field | Type | Default | Description |
|---|---|---|---|
| `output_format` | string | wav | Output audio format: `wav`, `mp3`, `flac`, `ogg`. |

### Examples

**Voice cloning (JSON):**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Today we are launching a new product line.",
    "sample": "john-doe",
    "num_step": 32,
    "speed": 0.9
  }' -o cloned.wav
```

**Voice cloning from text file:**

```bash
curl -X POST http://localhost:8000/tts/file \
  -F "text_file=@chapter1.txt" \
  -F "sample=narrator" \
  -F "num_step=32" \
  -o chapter1.wav
```

**Voice design:**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning everyone.",
    "instruct": "female, young adult, high pitch, american accent",
    "num_step": 16,
    "guidance_scale": 2.5
  }' -o designed.wav
```

**Auto voice with approved expressive tags:**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[laughter] That was really funny!",
    "output_format": "mp3"
  }' -o auto.mp3
```

**Precise duration control + silence removal off:**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This sentence will be exactly ten seconds long.",
    "sample": "narrator",
    "duration": 10.0,
    "postprocess_output": false
  }' -o precise.wav
```

**Long text with custom chunking:**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Very long article text goes here...",
    "sample": "narrator",
    "audio_chunk_duration": 20.0,
    "audio_chunk_threshold": 25.0
  }' -o longform.wav
```

## OpenAI-Compatible API

The server exposes a drop-in replacement for the OpenAI TTS API, so any client that works with `https://api.openai.com` can be pointed at this server instead.

### Limitations compared to OpenAI API

- **No voice listing endpoint.** The OpenAI API standard does not define an endpoint for listing available voices (there is no `GET /v1/voices`). To discover which voices are loaded, use OmniVoice's native `GET /voices` endpoint.
- **Voice names are not restricted to the standard set.** OpenAI's API documents a fixed set of voices (`alloy`, `ash`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`), but OmniVoice accepts **any** voice name that matches a loaded voice file. For example, if you place `my-custom-voice.wav` in the voices directory, `"voice": "my-custom-voice"` will work.
- **Silent fallback on unknown voice.** If the `voice` value does not match any loaded voice, the request does **not** fail — it silently falls back to auto-voice mode (model picks a random voice). This differs from the `/tts` endpoint, which returns a 404 error with a list of available voices.

### `POST /v1/audio/speech`

Identical contract to the [OpenAI TTS endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Request body (JSON):**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | `tts-1` (fast, 16 steps) or `tts-1-hd` (quality, 32 steps) |
| `input` | string | yes | Text to synthesize. Max 4096 characters. |
| `voice` | string | yes | Voice name. If a voice with this name is loaded, voice cloning is used. Otherwise auto-voice mode. Standard OpenAI voices (`alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`, etc.) **work when matching voice files exist.** |
| `response_format` | string | no | `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`. Note: `aac` is served as mp3. |
| `speed` | float | no | Speech speed factor (0.25–4.0). |

**curl example:**

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello world!", "voice": "alloy"}' \
  -o speech.mp3
```

**Python openai SDK example:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-key",   # or any string when auth is disabled
)

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="narrator",    # must match a loaded voice name
    input="Good evening, and welcome to the show.",
    response_format="mp3",
)
response.stream_to_file("output.mp3")
```

**Voice resolution:**

1. If a loaded voice named exactly `voice` (case-insensitive) exists → voice cloning mode using that voice.
2. Otherwise → auto-voice mode (model picks a voice).

To use OpenAI's standard voice names (`alloy`, `nova`, etc.) with real cloning, place matching voice files in `./voices/`:

```
voices/alloy.wav + voices/alloy.txt
voices/nova.wav  + voices/nova.txt
```

**Model → quality mapping:**

| `model` value | Diffusion steps | Notes |
|---|---|---|
| `tts-1` | 16 | Faster inference |
| `tts-1-hd` | 32 | Higher quality |
| any other | server default (32) | Accepted without error |

**Supported `response_format` values:**

| Value | Container | MIME type |
|---|---|---|
| `mp3` | MP3 | `audio/mpeg` |
| `opus` | Ogg/Opus | `audio/ogg` |
| `aac` | MP3 (fallback) | `audio/mpeg` |
| `flac` | FLAC | `audio/flac` |
| `wav` | WAV | `audio/wav` |
| `pcm` | Raw s16le mono | `audio/pcm` |

### `GET /v1/models`

Returns the list of available TTS model IDs in the OpenAI format.

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "tts-1",    "object": "model", "created": 1234567890, "owned_by": "omnivoice"},
    {"id": "tts-1-hd", "object": "model", "created": 1234567890, "owned_by": "omnivoice"}
  ]
}
```

### Errors

Errors from `/v1/audio/speech` are returned in the OpenAI error envelope format:

```json
HTTP 400
{
  "error": {
    "message": "Unsupported response_format 'xyz'. Supported: ['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

## Error Handling

The server returns structured JSON errors:

These response field names still use legacy compatibility terminology on the wire. For example, unknown-voice errors return `available_samples` today even though `voices` / `/voices` are the canonical docs and endpoint names.

**Unknown sample:**

```json
HTTP 404
{
  "detail": {
    "error": "Sample 'nonexistent' not found.",
    "available_samples": ["john-doe", "narrator"]
  }
}
```

**Invalid output format:**

```json
HTTP 400
{
  "detail": "Unsupported output format 'aiff'. Use: ['wav', 'mp3', 'flac', 'ogg']"
}
```

**Model not ready:**

```json
HTTP 503
{
  "detail": "Model not loaded yet."
}
```

**Generation failure:**

```json
HTTP 500
{
  "detail": "Generation error: <exception details>"
}
```

## Authentication

API key authentication is **optional**. When disabled (default), all endpoints are open.

To enable it, set the `OMNIVOICE_API_KEY` environment variable to any non-empty string. Once set, authenticated HTTP API routes require:

```
Authorization: Bearer <your-key>
```

**curl example:**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "sample": "my-voice"}' \
  -o output.wav
```

`GET /health` is always open so Docker/Podman healthchecks continue to work without credentials. The current `/ws/tts` and `/ws/conversation` WebSocket endpoints are also unauthenticated and do not enforce `OMNIVOICE_API_KEY`.

**Setup:**

```bash
cp .env.example .env
# Set OMNIVOICE_API_KEY in .env
# Uncomment OMNIVOICE_API_KEY=${OMNIVOICE_API_KEY} in compose.yaml so it is passed into the container
```

`.env` is local and ignored by git. Keep secrets there; `compose.yaml` is safe to commit because it contains only environment-variable references.

## Configuration

All configuration is via environment variables. Use `.env` for local secret values such as `HF_TOKEN` and `OMNIVOICE_API_KEY`, and let `compose.yaml` reference those variables:

| Variable | Default | Description |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | HuggingFace model ID or local path |
| `OMNIVOICE_DEVICE` | `cuda:0` | PyTorch device (`cuda:0`, `cuda:1`, etc.) |
| `OMNIVOICE_DTYPE` | `float16` | Model dtype: `float16`, `bfloat16`, `float32` |
| `OMNIVOICE_VOICES_DIR` | `/voices` | Canonical path to voice assets inside container |
| `OMNIVOICE_SAMPLES_DIR` | `/samples` | Compatibility alias and legacy fallback path used when the canonical voices directory is empty |
| `OMNIVOICE_HOST` | `0.0.0.0` | Server bind address |
| `OMNIVOICE_PORT` | `8000` | API server port |
| `OMNIVOICE_OUTPUT_FORMAT` | `wav` | Default output format when not specified in request |
| `OMNIVOICE_GRADIO_ENABLED` | `true` | Enable/disable Gradio web UI |
| `OMNIVOICE_GRADIO_PORT` | `8001` | Gradio web UI port |
| `OMNIVOICE_API_KEY` | _(unset)_ | API key for bearer-token auth. Unset = auth disabled |
| `OMNIVOICE_CORS_ORIGINS` | _(empty)_ | Comma-separated list of allowed CORS origins. Empty = CORS disabled |
| `OMNIVOICE_MAX_UPLOAD_BYTES` | `10485760` | Max text file upload size in bytes (default 10 MB) |
| `OMNIVOICE_CONVERSATION_WS_ENABLED` | `true` | Enable/disable the experimental `/ws/conversation` endpoint and its server-side assistant governance flow |
| `OMNIVOICE_ASR_MODEL` | `small` | `faster-whisper` model name or local path used for conversation ASR, language detection, and turn latency logging |
| `OMNIVOICE_ASR_DEVICE` | `auto` | `faster-whisper` device selection: `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `OMNIVOICE_ASR_COMPUTE_TYPE` | `default` | `faster-whisper` compute type passed to the ASR model |
| `OMNIVOICE_OLLAMA_HOST` | `http://localhost:11434` | Ollama server base URL used for governed conversation assistant requests |
| `OMNIVOICE_OLLAMA_MODEL` | `gemma4` | Ollama model used to generate assistant text; its name is also included in completed-turn diagnostic logs |

### Using a Pre-Downloaded Model

To avoid downloading the model on every container rebuild, download it once and bind-mount:

```bash
# Download model (one time)
mkdir -p /srv/models/OmniVoice
# Use huggingface-cli, git lfs, or manual download

# Update compose.yaml volumes:
#   - /srv/models/OmniVoice:/models/OmniVoice
# Update compose.yaml environment:
#   - OMNIVOICE_MODEL=/models/OmniVoice
```

### Multi-GPU

To use a specific GPU, set `OMNIVOICE_DEVICE=cuda:1` (or whichever index).

For multiple instances on different GPUs, duplicate the service in `compose.yaml` with different device settings and ports.

## Supported Voice Design Attributes

Combine freely with commas:

- **Gender:** male, female
- **Age:** child, teenager, young adult, middle-aged, elderly
- **Pitch:** very low pitch, low pitch, moderate pitch, high pitch, very high pitch
- **Style:** whisper
- **English accent:** american accent, british accent, australian accent, indian accent, chinese accent, canadian accent, korean accent, portuguese accent, russian accent, japanese accent
- **Chinese dialect:** 四川话, 陕西话, 河南话, 贵州话, 云南话, 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话

## Approved Expressive Tags

Direct `/tts` and `/ws/tts` requests pass your text through as provided, so approved tags can be sent inline there. `/ws/conversation` is different: assistant-generated text is sanitized before TTS, and only the exact allowlist below is preserved. Matching is exact, so case or spacing variants such as `[Laughter]` or `[ laughter ]`, plus unapproved tags, are removed from assistant output.

`[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`

## Pronunciation Control

**Chinese:** Use pinyin with tone numbers inline: `这批货物打ZHE2出售`

**English:** Use CMU dictionary format in brackets: `You could probably still make [IH1 T] look good.`

## Notes

- First request after startup may be slow due to model warm-up.
- Gradio UI and API share a single model and single GIL — concurrent requests are executed sequentially, which is expected for single-GPU setups.
- Whisper auto-transcription (when no .txt is provided) adds latency. Providing transcripts is recommended.
- The output sample rate is always 24 kHz (set by OmniVoice).
- The `start_period` in healthcheck is 120s to account for model loading time. Adjust if your storage is slow.
- `num_step` values above 32 may cause artifacts. The model was trained with 32 steps; use 16 for speed or 32 for quality.
