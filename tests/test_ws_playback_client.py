from __future__ import annotations

import asyncio
import importlib.util
import json
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "ws_playback_client.py"


def _load_module():
    if not MODULE_PATH.exists():
        return types.SimpleNamespace(
            build_synthesize_request=lambda **kwargs: (_ for _ in ()).throw(
                NotImplementedError
            ),
        )

    spec = importlib.util.spec_from_file_location("ws_playback_client", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_synthesize_request_rejects_non_pcm_output_format():
    client = _load_module()

    with pytest.raises(ValueError, match="Only pcm output_format is supported"):
        client.build_synthesize_request(text="hello", output_format="wav")


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = iter(messages)

    async def recv(self):
        return next(self._messages)


class FakeAudioStream:
    def __init__(self):
        self.writes = []
        self.stopped = False
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_consume_stream_writes_binary_after_audio_chunk_metadata():
    client = _load_module()
    consume_stream = getattr(client, "consume_stream", None)

    assert callable(consume_stream)

    events = []
    stream = FakeAudioStream()
    websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "audio_chunk",
                    "chunk_index": 0,
                    "sentence": "Hello",
                    "is_last": True,
                }
            ),
            b"\x00\x01\x02\x03",
            json.dumps(
                {
                    "type": "done",
                    "total_chunks": 1,
                    "total_duration_ms": 1,
                    "total_generation_time_ms": 2,
                }
            ),
        ]
    )

    asyncio.run(consume_stream(websocket, stream, event_sink=events.append))

    assert stream.writes == [b"\x00\x01\x02\x03"]
    assert events == [
        {"type": "audio_chunk", "chunk_index": 0, "sentence": "Hello", "is_last": True},
        {
            "type": "done",
            "total_chunks": 1,
            "total_duration_ms": 1,
            "total_generation_time_ms": 2,
        },
    ]


def test_consume_stream_raises_on_server_error_event():
    client = _load_module()
    events = []
    stream = FakeAudioStream()
    websocket = FakeWebSocket(
        [
            json.dumps(
                {"type": "error", "message": "Sample missing", "code": "UNKNOWN_SAMPLE"}
            )
        ]
    )

    with pytest.raises(RuntimeError, match="Sample missing"):
        asyncio.run(client.consume_stream(websocket, stream, event_sink=events.append))

    assert stream.writes == []
    assert events == [
        {"type": "error", "message": "Sample missing", "code": "UNKNOWN_SAMPLE"}
    ]


def test_consume_stream_raises_on_binary_frame_without_metadata():
    client = _load_module()
    stream = FakeAudioStream()
    websocket = FakeWebSocket([b"\x00\x01"])

    with pytest.raises(
        RuntimeError, match="Received binary frame without audio_chunk metadata"
    ):
        asyncio.run(
            client.consume_stream(websocket, stream, event_sink=lambda event: None)
        )

    assert stream.writes == []


def test_consume_stream_raises_when_done_arrives_before_pending_audio_bytes():
    client = _load_module()
    events = []
    stream = FakeAudioStream()
    websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "audio_chunk",
                    "chunk_index": 0,
                    "sentence": "Hello",
                    "is_last": True,
                }
            ),
            json.dumps(
                {
                    "type": "done",
                    "total_chunks": 1,
                    "total_duration_ms": 1,
                    "total_generation_time_ms": 2,
                }
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="Received done before audio bytes"):
        asyncio.run(client.consume_stream(websocket, stream, event_sink=events.append))

    assert stream.writes == []
    assert events == [
        {"type": "audio_chunk", "chunk_index": 0, "sentence": "Hello", "is_last": True}
    ]


class FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeClientWebSocket(FakeWebSocket):
    def __init__(self, messages):
        super().__init__(messages)
        self.sent = []

    async def send(self, data: str) -> None:
        self.sent.append(data)


class FakePyAudioInstance:
    def __init__(self):
        self.open_calls = []
        self.stream = FakeAudioStream()
        self.terminated = False

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        return self.stream

    def terminate(self):
        self.terminated = True


class FakePyAudioOpenErrorInstance:
    def __init__(self):
        self.terminated = False

    def open(self, **kwargs):
        raise RuntimeError("open failed")

    def terminate(self):
        self.terminated = True


def test_play_sends_request_and_streams_audio_to_pyaudio():
    client = _load_module()
    play = getattr(client, "play", None)

    assert callable(play)

    websocket = FakeClientWebSocket(
        [
            json.dumps(
                {
                    "type": "audio_chunk",
                    "chunk_index": 0,
                    "sentence": "Hi",
                    "is_last": True,
                }
            ),
            b"\x00\x01",
            json.dumps(
                {
                    "type": "done",
                    "total_chunks": 1,
                    "total_duration_ms": 1,
                    "total_generation_time_ms": 1,
                }
            ),
        ]
    )
    pyaudio_instance = FakePyAudioInstance()
    pyaudio_module = types.SimpleNamespace(
        paInt16="fake-format",
        PyAudio=lambda: pyaudio_instance,
    )
    websockets_module = types.SimpleNamespace(
        connect=lambda url: FakeConnection(websocket)
    )
    events = []

    asyncio.run(
        play(
            "ws://example/ws/tts",
            {"type": "synthesize", "text": "Hi", "output_format": "pcm"},
            websockets_module,
            pyaudio_module,
            event_sink=events.append,
        )
    )

    assert websocket.sent == [
        json.dumps({"type": "synthesize", "text": "Hi", "output_format": "pcm"})
    ]
    assert pyaudio_instance.open_calls == [
        {"format": "fake-format", "channels": 1, "rate": 24000, "output": True}
    ]
    assert pyaudio_instance.stream.writes == [b"\x00\x01"]
    assert pyaudio_instance.stream.stopped is True
    assert pyaudio_instance.stream.closed is True
    assert pyaudio_instance.terminated is True
    assert events == [
        {"type": "audio_chunk", "chunk_index": 0, "sentence": "Hi", "is_last": True},
        {
            "type": "done",
            "total_chunks": 1,
            "total_duration_ms": 1,
            "total_generation_time_ms": 1,
        },
    ]


def test_play_terminates_pyaudio_when_stream_open_fails():
    client = _load_module()
    websocket = FakeClientWebSocket([])
    pyaudio_instance = FakePyAudioOpenErrorInstance()
    pyaudio_module = types.SimpleNamespace(
        paInt16="fake-format",
        PyAudio=lambda: pyaudio_instance,
    )
    websockets_module = types.SimpleNamespace(
        connect=lambda url: FakeConnection(websocket)
    )

    with pytest.raises(RuntimeError, match="open failed"):
        asyncio.run(
            client.play(
                "ws://example/ws/tts",
                {"type": "synthesize", "text": "Hi", "output_format": "pcm"},
                websockets_module,
                pyaudio_module,
            )
        )

    assert websocket.sent == [
        json.dumps({"type": "synthesize", "text": "Hi", "output_format": "pcm"})
    ]
    assert pyaudio_instance.terminated is True


def test_main_returns_non_zero_for_invalid_output_format(capsys):
    client = _load_module()
    main = getattr(client, "main", None)

    assert callable(main)

    exit_code = main(["--text", "hello", "--output-format", "wav"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Only pcm output_format is supported for live playback" in captured.err
