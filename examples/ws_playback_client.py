from __future__ import annotations

import argparse
import asyncio
import json
import sys


def build_synthesize_request(
    *,
    text: str,
    output_format: str = "pcm",
    sample: str | None = None,
    num_step: int | None = None,
) -> dict[str, object]:
    if output_format.lower() != "pcm":
        raise ValueError("Only pcm output_format is supported for live playback")

    request: dict[str, object] = {
        "type": "synthesize",
        "text": text,
        "output_format": "pcm",
    }
    if sample is not None:
        request["sample"] = sample
    if num_step is not None:
        request["num_step"] = num_step
    return request


async def consume_stream(websocket, audio_stream, *, event_sink=None) -> None:
    if event_sink is None:
        event_sink = lambda event: print(json.dumps(event), flush=True)

    pending_chunk = None

    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            if pending_chunk is None:
                raise RuntimeError("Received binary frame without audio_chunk metadata")
            audio_stream.write(message)
            pending_chunk = None
            continue

        event = json.loads(message)
        event_type = event.get("type")
        if event_type == "error":
            event_sink(event)
            raise RuntimeError(event.get("message") or "Server returned an error event")
        if event_type == "done":
            if pending_chunk is not None:
                raise RuntimeError("Received done before audio bytes")
            event_sink(event)
            return
        if event_type != "audio_chunk":
            raise RuntimeError(f"Unexpected event type: {event_type}")
        if pending_chunk is not None:
            raise RuntimeError("Received audio_chunk metadata before audio bytes")
        pending_chunk = event
        event_sink(event)


async def play(
    url: str,
    request: dict[str, object],
    websockets_module,
    pyaudio_module,
    *,
    event_sink=None,
) -> None:
    async with websockets_module.connect(url) as websocket:
        await websocket.send(json.dumps(request))

        audio_api = pyaudio_module.PyAudio()
        audio_stream = None
        try:
            audio_stream = audio_api.open(
                format=pyaudio_module.paInt16,
                channels=1,
                rate=24000,
                output=True,
            )
            await consume_stream(websocket, audio_stream, event_sink=event_sink)
        finally:
            if audio_stream is not None:
                audio_stream.stop_stream()
                audio_stream.close()
            audio_api.terminate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream OmniVoice WebSocket TTS to the local speakers."
    )
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts")
    parser.add_argument("--text", required=True)
    parser.add_argument("--sample")
    parser.add_argument("--num-step", type=int)
    parser.add_argument("--output-format", default="pcm")
    args = parser.parse_args(argv)

    try:
        request = build_synthesize_request(
            text=args.text,
            output_format=args.output_format,
            sample=args.sample,
            num_step=args.num_step,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        import pyaudio
    except ModuleNotFoundError:
        print(
            "Missing dependency: install pyaudio to enable playback.", file=sys.stderr
        )
        return 1

    try:
        import websockets
    except ModuleNotFoundError:
        print(
            "Missing dependency: install websockets to connect to the server.",
            file=sys.stderr,
        )
        return 1

    try:
        asyncio.run(play(args.url, request, websockets, pyaudio))
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
