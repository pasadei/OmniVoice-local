# OmniVoice TTS Server

OmniVoice TTS server with Gradio web UI and REST API sharing a single model instance. Voice cloning via sample directory, voice design, full generation parameter control. NVIDIA GPU support (Ampere, Ada Lovelace, Blackwell). Docker/Podman-compose ready.

Based on [OmniVoice](https://github.com/k2-fsa/OmniVoice) — zero-shot multilingual TTS for 600+ languages. Thanks to the [k2-fsa](https://github.com/k2-fsa) team for open-sourcing the model.

## Overview

The server loads OmniVoice once and exposes it through two interfaces simultaneously:

- **Gradio Web UI** (port `8001`) — the built-in OmniVoice demo with voice cloning and voice design tabs
- **REST API** (port `8000`) — FastAPI with JSON and multipart/form-data endpoints, full parameter control

Both share the same model instance — no extra VRAM.

**Three generation modes:**

- **Voice cloning** — pick a named sample from the samples directory
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
└── samples/             # Voice sample pairs (bind-mounted)
    ├── john-doe.wav
    ├── john-doe.txt
    ├── narrator.mp3
    └── narrator.txt
```

## Voice Samples

Place audio files and their transcripts in `./samples/`. Files are paired by stem (filename without extension):

```
my-voice.wav    ←  audio file (wav, mp3, flac, ogg, opus, m4a, aac)
my-voice.txt    ←  plain text transcript of the audio
```

The transcript file is optional — if missing, OmniVoice will auto-transcribe the audio using Whisper at request time (slower, slightly less accurate).

You can add or remove samples at runtime and call `POST /samples/reload` to re-scan.

## Quick Start

```bash
# 1. Place voice samples
cp my-voice.wav my-voice.txt ./samples/

# 2. Build and start
podman-compose up -d --build
# or: docker compose up -d --build

# 3. Check health (model loading takes ~60-120s)
curl http://localhost:8000/health

# 4. Open Gradio UI
# http://localhost:8001

# 5. Generate speech via API
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "sample": "my-voice"}' \
  -o output.wav
```

## API Reference

### `GET /health`

Health check. Returns model status, device info, and Gradio state.

### `GET /samples`

List all loaded voice samples.

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

### `POST /samples/reload`

Re-scan the samples directory. Use after adding/removing sample files without restarting the container.

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
| `text` | string | **Required.** Text to synthesize. Supports inline tags like `[laughter]` and pronunciation control. |
| `sample` | string | Sample name (stem) for voice cloning. Must exist in samples directory. |
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

**Auto voice with non-verbal tags:**

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

## Error Handling

The server returns structured JSON errors:

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

## Configuration

All configuration is via environment variables (set in `compose.yaml`):

| Variable | Default | Description |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | HuggingFace model ID or local path |
| `OMNIVOICE_DEVICE` | `cuda:0` | PyTorch device (`cuda:0`, `cuda:1`, etc.) |
| `OMNIVOICE_DTYPE` | `float16` | Model dtype: `float16`, `bfloat16`, `float32` |
| `OMNIVOICE_SAMPLES_DIR` | `/samples` | Path to voice samples inside container |
| `OMNIVOICE_HOST` | `0.0.0.0` | Server bind address |
| `OMNIVOICE_PORT` | `8000` | API server port |
| `OMNIVOICE_OUTPUT_FORMAT` | `wav` | Default output format when not specified in request |
| `OMNIVOICE_GRADIO_ENABLED` | `true` | Enable/disable Gradio web UI |
| `OMNIVOICE_GRADIO_PORT` | `8001` | Gradio web UI port |

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

## Supported Non-Verbal Tags

Insert directly in the text field:

`[laughter]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`, `[sniff]`, `[sigh]`

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
