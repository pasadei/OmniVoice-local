from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
import threading
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "ws_duplex_client.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ws_duplex_client", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePlayback:
    def __init__(self):
        self.interrupt_calls = 0
        self.chunks = []

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def enqueue(self, chunk: bytes) -> None:
        self.chunks.append(chunk)


class FakeStream:
    def __init__(self):
        self.stop_calls = 0
        self.close_calls = 0
        self.start_calls = 0
        self.writes = []
        self.write_started = threading.Event()
        self.write_release = threading.Event()
        self.block_writes = False

    def stop_stream(self) -> None:
        self.stop_calls += 1

    def start_stream(self) -> None:
        self.start_calls += 1

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        self.write_started.set()
        if self.block_writes:
            self.write_release.wait(timeout=1)

    def close(self) -> None:
        self.close_calls += 1


class FakePyAudioInstance:
    def __init__(self):
        self.open_calls = []
        self.terminated = False
        self.streams = []

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        self.terminated = True


class FakeTTYStream(io.StringIO):
    def __init__(self, *, is_tty: bool):
        super().__init__()
        self._is_tty = is_tty
        self.flush_calls = 0

    def isatty(self) -> bool:
        return self._is_tty

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()


class FakeVisualizer:
    def __init__(self):
        self.renders = []
        self.finish_calls = 0
        self.state_label = None

    def render(self, level: float, *, state_label: str | None = None) -> None:
        if state_label is not None:
            self.state_label = state_label
        self.renders.append((level, state_label))

    def finish(self) -> None:
        self.finish_calls += 1


def test_duplex_state_interrupts_local_playback_on_speech_start():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()

    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert playback.interrupt_calls == 1

    assert state.start_local_speech(playback) is None
    assert playback.interrupt_calls == 1


def test_calculate_pcm16_level_returns_zero_for_empty_audio():
    client = _load_module()

    assert client.calculate_pcm16_level(b"") == 0.0


def test_calculate_pcm16_level_uses_peak_absolute_sample():
    client = _load_module()
    quiet = int(0.25 * 32768).to_bytes(2, byteorder="little", signed=True)
    loud = int(-0.5 * 32768).to_bytes(2, byteorder="little", signed=True)

    assert client.calculate_pcm16_level(quiet + loud) == pytest.approx(0.5)


def test_calculate_pcm16_level_ignores_odd_trailing_byte():
    client = _load_module()
    sample = int(0.25 * 32768).to_bytes(2, byteorder="little", signed=True)

    assert client.calculate_pcm16_level(sample + b"\xff") == pytest.approx(0.25)


def test_terminal_visualizer_renders_single_line_bar_and_tracks_state_label():
    client = _load_module()
    stream = FakeTTYStream(is_tty=True)
    visualizer = client.TerminalVisualizer(stream=stream, width=8)

    visualizer.render(0.5, state_label="LISTEN")

    assert visualizer.enabled is True
    assert visualizer.state_label == "LISTEN"
    assert stream.getvalue() == "\rLISTEN [####----]"
    assert stream.flush_calls == 1


def test_terminal_visualizer_clears_stale_characters_on_shorter_render():
    client = _load_module()
    stream = FakeTTYStream(is_tty=True)
    visualizer = client.TerminalVisualizer(stream=stream, width=4)

    visualizer.render(1.0, state_label="LONG")
    visualizer.render(0.0, state_label="S")

    assert stream.getvalue() == "\rLONG [####]\rS [----]   "


def test_terminal_visualizer_finish_appends_newline_once():
    client = _load_module()
    stream = FakeTTYStream(is_tty=True)
    visualizer = client.TerminalVisualizer(stream=stream, width=4)

    visualizer.render(1.0, state_label="PLAY")
    visualizer.finish()
    visualizer.finish()

    assert stream.getvalue() == "\rPLAY [####]\n"


def test_terminal_visualizer_disabled_skips_rendering_and_cleanup():
    client = _load_module()
    stream = FakeTTYStream(is_tty=False)
    visualizer = client.TerminalVisualizer(stream=stream, width=6)

    visualizer.render(0.5, state_label="IDLE")
    visualizer.finish()

    assert visualizer.enabled is False
    assert visualizer.state_label == "IDLE"
    assert stream.getvalue() == ""


def test_terminal_visualizer_uses_current_stderr_when_stream_is_omitted(monkeypatch):
    client = _load_module()
    stdout_stream = FakeTTYStream(is_tty=True)
    stderr_stream = FakeTTYStream(is_tty=True)

    monkeypatch.setattr(sys, "stdout", stdout_stream)
    monkeypatch.setattr(sys, "stderr", stderr_stream)

    visualizer = client.TerminalVisualizer(width=4)
    visualizer.render(0.5, state_label="NOW")

    assert stderr_stream.getvalue() == "\rNOW [##--]"


def test_terminal_visualizer_default_constructor_stays_enabled_when_stderr_is_tty(
    monkeypatch,
):
    client = _load_module()
    stdout_stream = FakeTTYStream(is_tty=False)
    stderr_stream = FakeTTYStream(is_tty=True)

    monkeypatch.setattr(sys, "stdout", stdout_stream)
    monkeypatch.setattr(sys, "stderr", stderr_stream)

    visualizer = client.TerminalVisualizer(width=4)
    visualizer.render(0.5, state_label="NOW")
    visualizer.finish()

    assert visualizer.enabled is True
    assert stderr_stream.getvalue() == "\rNOW [##--]\n"


def test_terminal_visualizer_default_constructor_disables_when_stderr_is_not_a_tty(
    monkeypatch,
):
    client = _load_module()
    stdout_stream = FakeTTYStream(is_tty=True)
    stderr_stream = FakeTTYStream(is_tty=False)

    monkeypatch.setattr(sys, "stdout", stdout_stream)
    monkeypatch.setattr(sys, "stderr", stderr_stream)

    visualizer = client.TerminalVisualizer(width=4)
    visualizer.render(0.5, state_label="NOW")
    visualizer.finish()

    assert visualizer.enabled is False
    assert stderr_stream.getvalue() == ""


def test_build_session_start_includes_explicit_language_override_and_instruct():
    client = _load_module()

    assert client.build_session_start(
        sample="agent", language="it", instruct="Warm and patient narrator."
    ) == {
        "type": "session_start",
        "sample_rate": 16000,
        "sample": "agent",
        "language": "it",
        "instruct": "Warm and patient narrator.",
    }


def test_duplex_state_drops_stale_audio_after_local_interrupt():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()

    assert state.handle_event(
        {"type": "assistant_text", "response_id": 1}, playback
    ) == {
        "type": "assistant_text",
        "response_id": 1,
    }
    assert state.handle_event({"type": "audio_chunk", "response_id": 1}, playback) == {
        "type": "audio_chunk",
        "response_id": 1,
    }

    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert state.handle_audio_bytes(b"old-response", playback) is False
    assert playback.chunks == []
    assert state.end_local_speech() == {"type": "speech_end"}

    assert (
        state.handle_event({"type": "audio_chunk", "response_id": 1}, playback) is None
    )
    assert state.handle_event({"type": "audio_chunk", "response_id": 2}, playback) == {
        "type": "audio_chunk",
        "response_id": 2,
    }
    assert state.handle_audio_bytes(b"new-response", playback) is True
    assert playback.chunks == [b"new-response"]


def test_duplex_state_tracks_session_and_response_lifecycle():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()

    assert state.handle_event(
        {"type": "session_started", "sample": "agent", "sample_rate": 16000},
        playback,
    ) == {"type": "session_started", "sample": "agent", "sample_rate": 16000}
    assert state.session_started is True
    assert state.sample == "agent"
    assert state.sample_rate == 16000

    assert state.handle_event({"type": "listening"}, playback) == {"type": "listening"}
    assert state.server_listening is True

    assert state.handle_event(
        {"type": "transcript_final", "text": "hello"}, playback
    ) == {
        "type": "transcript_final",
        "text": "hello",
    }
    assert state.server_listening is False

    assert state.handle_event(
        {"type": "assistant_text", "text": "hi", "response_id": 3}, playback
    ) == {"type": "assistant_text", "text": "hi", "response_id": 3}
    assert state.active_response_id == 3

    assert state.handle_event({"type": "interrupted", "response_id": 3}, playback) == {
        "type": "interrupted",
        "response_id": 3,
    }
    assert state.active_response_id is None

    assert (
        state.handle_event({"type": "response_done", "response_id": 3}, playback)
        is None
    )


def test_duplex_state_renders_listening_when_local_speech_starts_without_barge_in():
    client = _load_module()
    visualizer = FakeVisualizer()
    state = client.DuplexState(visualizer=visualizer)
    playback = FakePlayback()

    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert state.end_local_speech() == {"type": "speech_end"}

    assert visualizer.renders == [
        (0.0, "LISTENING"),
        (0.0, "IDLE"),
    ]


def test_duplex_state_renders_interrupted_only_for_barge_in():
    client = _load_module()
    visualizer = FakeVisualizer()
    state = client.DuplexState(visualizer=visualizer)
    playback = FakePlayback()

    assert state.handle_event({"type": "listening"}, playback) == {"type": "listening"}
    assert state.handle_event(
        {"type": "assistant_text", "text": "hi", "response_id": 1}, playback
    ) == {"type": "assistant_text", "text": "hi", "response_id": 1}
    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert state.end_local_speech() == {"type": "speech_end"}

    assert visualizer.renders == [
        (0.0, "LISTENING"),
        (0.0, "ASSISTANT"),
        (0.0, "INTERRUPTED"),
        (0.0, "IDLE"),
    ]


def test_open_audio_streams_and_main_use_expected_runtime_configuration(monkeypatch):
    client = _load_module()
    pyaudio_instance = FakePyAudioInstance()
    pyaudio_module = types.SimpleNamespace(
        paInt16="pcm16",
        PyAudio=lambda: pyaudio_instance,
    )

    audio_api, input_stream, output_stream = client.open_audio_streams(pyaudio_module)

    assert pyaudio_instance.open_calls == [
        {
            "format": "pcm16",
            "channels": 1,
            "rate": 16000,
            "input": True,
            "frames_per_buffer": 512,
        },
        {"format": "pcm16", "channels": 1, "rate": 24000, "output": True},
    ]

    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    audio_api.terminate()
    assert pyaudio_instance.terminated is True

    captured = {}

    async def fake_run_client(
        url,
        sample,
        language,
        instruct,
        websockets_module,
        pyaudio_module,
        torch_module,
        silero_vad_module,
    ):
        captured.update(
            {
                "url": url,
                "sample": sample,
                "language": language,
                "instruct": instruct,
                "websockets": websockets_module,
                "pyaudio": pyaudio_module,
                "torch": torch_module,
                "silero_vad": silero_vad_module,
            }
        )

    monkeypatch.setattr(client, "run_client", fake_run_client)
    monkeypatch.setitem(sys.modules, "websockets", types.ModuleType("websockets"))
    monkeypatch.setitem(sys.modules, "pyaudio", types.ModuleType("pyaudio"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "silero_vad", types.ModuleType("silero_vad"))

    assert (
        client.main(
            [
                "--url",
                "ws://example/ws/conversation",
                "--sample",
                "agent",
                "--language",
                "it",
                "--instruct",
                "Warm and patient narrator.",
            ]
        )
        == 0
    )
    assert captured == {
        "url": "ws://example/ws/conversation",
        "sample": "agent",
        "language": "it",
        "instruct": "Warm and patient narrator.",
        "websockets": sys.modules["websockets"],
        "pyaudio": sys.modules["pyaudio"],
        "torch": sys.modules["torch"],
        "silero_vad": sys.modules["silero_vad"],
    }


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = iter(messages)

    async def recv(self):
        return next(self._messages)


class FakeConnectContext:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_playback_controller_interrupt_drops_buffered_audio():
    client = _load_module()
    stream = FakeStream()
    stream.block_writes = True
    playback = client.PlaybackController(stream)

    task = asyncio.create_task(playback.run())
    playback.enqueue(b"first")
    await asyncio.to_thread(stream.write_started.wait, 1)
    playback.enqueue(b"second")

    playback.interrupt()
    playback.enqueue(b"third")
    stream.write_release.set()

    for _ in range(20):
        if stream.writes == [b"first", b"third"]:
            break
        await asyncio.sleep(0.01)
    await playback.close()
    await task

    assert stream.writes == [b"first", b"third"]
    assert stream.stop_calls == 0
    assert stream.start_calls == 0


@pytest.mark.asyncio
async def test_playback_controller_updates_visualizer_from_playback_frames():
    client = _load_module()
    stream = FakeStream()
    visualizer = FakeVisualizer()
    playback = client.PlaybackController(stream, visualizer=visualizer)
    sample = int(0.5 * 32768).to_bytes(2, byteorder="little", signed=True)

    task = asyncio.create_task(playback.run())
    playback.enqueue(sample)

    for _ in range(20):
        if stream.writes == [sample]:
            break
        await asyncio.sleep(0.01)
    await playback.close()
    await task

    assert stream.writes == [sample]
    assert visualizer.renders == [(pytest.approx(0.5), None)]
    assert visualizer.finish_calls == 1


@pytest.mark.asyncio
async def test_response_done_keeps_assistant_state_until_playback_drains():
    client = _load_module()
    stream = FakeStream()
    stream.block_writes = True
    visualizer = FakeVisualizer()
    playback = client.PlaybackController(stream, visualizer=visualizer)
    state = client.DuplexState(visualizer=visualizer)
    sample = int(0.5 * 32768).to_bytes(2, byteorder="little", signed=True)

    playback.set_on_drain(state.playback_drained)

    task = asyncio.create_task(playback.run())

    assert state.handle_event(
        {"type": "assistant_text", "text": "hi", "response_id": 1}, playback
    ) == {"type": "assistant_text", "text": "hi", "response_id": 1}
    assert state.handle_event({"type": "audio_chunk", "response_id": 1}, playback) == {
        "type": "audio_chunk",
        "response_id": 1,
    }
    assert state.handle_audio_bytes(sample, playback) is True
    await asyncio.to_thread(stream.write_started.wait, 1)

    renders_before_done = list(visualizer.renders)
    assert state.handle_event(
        {"type": "response_done", "response_id": 1}, playback
    ) == {
        "type": "response_done",
        "response_id": 1,
    }

    assert visualizer.renders == renders_before_done
    assert visualizer.state_label == "ASSISTANT"

    stream.write_release.set()
    for _ in range(20):
        if visualizer.state_label == "IDLE":
            break
        await asyncio.sleep(0.01)

    await playback.close()
    await task

    assert visualizer.state_label == "IDLE"


@pytest.mark.asyncio
async def test_local_barge_in_during_playback_drain_renders_interrupted():
    client = _load_module()
    stream = FakeStream()
    stream.block_writes = True
    visualizer = FakeVisualizer()
    playback = client.PlaybackController(stream, visualizer=visualizer)
    state = client.DuplexState(visualizer=visualizer)
    sample = int(0.5 * 32768).to_bytes(2, byteorder="little", signed=True)

    playback.set_on_drain(state.playback_drained)

    task = asyncio.create_task(playback.run())

    assert state.handle_event(
        {"type": "assistant_text", "text": "hi", "response_id": 1}, playback
    ) == {"type": "assistant_text", "text": "hi", "response_id": 1}
    assert state.handle_event({"type": "audio_chunk", "response_id": 1}, playback) == {
        "type": "audio_chunk",
        "response_id": 1,
    }
    assert state.handle_audio_bytes(sample, playback) is True
    await asyncio.to_thread(stream.write_started.wait, 1)

    assert state.handle_event(
        {"type": "response_done", "response_id": 1}, playback
    ) == {
        "type": "response_done",
        "response_id": 1,
    }

    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert visualizer.state_label == "INTERRUPTED"

    stream.write_release.set()
    await playback.close()
    await task


@pytest.mark.asyncio
async def test_run_client_builds_visualizer_and_attaches_it_to_playback_and_state(
    monkeypatch,
):
    client = _load_module()
    input_stream = FakeStream()
    output_stream = FakeStream()
    audio_api = FakePyAudioInstance()
    captured = {}
    created_visualizers = []

    class RecordingVisualizer(FakeVisualizer):
        def __init__(self):
            super().__init__()
            created_visualizers.append(self)

    class RecordingPlayback:
        def __init__(self, stream, *, visualizer=None):
            captured["playback_stream"] = stream
            captured["playback_visualizer"] = visualizer

        def set_on_drain(self, callback):
            captured["playback_on_drain"] = callback

        async def run(self):
            return None

        async def close(self):
            return None

    class RecordingState:
        def __init__(self, sample=None, *, visualizer=None):
            captured["state_sample"] = sample
            captured["state_visualizer"] = visualizer

        def playback_drained(self):
            captured["state_playback_drained"] = True

    websocket = types.SimpleNamespace()

    async def fake_send(message):
        captured.setdefault("messages", []).append(message)

    websocket.send = fake_send

    async def fake_receive_loop(ws, state, playback, *, event_sink=None):
        captured["receive_loop"] = (ws, state, playback, event_sink)

    async def fake_microphone_loop(
        ws,
        state,
        input_stream_arg,
        playback,
        torch_module,
        silero_vad_module,
    ):
        captured["microphone_loop"] = (
            ws,
            state,
            input_stream_arg,
            playback,
            torch_module,
            silero_vad_module,
        )

    monkeypatch.setattr(
        client,
        "open_audio_streams",
        lambda pyaudio_module: (audio_api, input_stream, output_stream),
    )
    monkeypatch.setattr(client, "TerminalVisualizer", RecordingVisualizer)
    monkeypatch.setattr(client, "PlaybackController", RecordingPlayback)
    monkeypatch.setattr(client, "DuplexState", RecordingState)
    monkeypatch.setattr(client, "receive_loop", fake_receive_loop)
    monkeypatch.setattr(client, "microphone_loop", fake_microphone_loop)

    websockets_module = types.SimpleNamespace(
        connect=lambda url: FakeConnectContext(websocket)
    )
    pyaudio_module = types.SimpleNamespace()
    torch_module = types.SimpleNamespace()
    silero_vad_module = types.SimpleNamespace()

    await client.run_client(
        "ws://example/ws/conversation",
        "agent",
        None,
        "Warm and patient narrator.",
        websockets_module,
        pyaudio_module,
        torch_module,
        silero_vad_module,
    )

    assert len(created_visualizers) == 1
    assert captured["playback_stream"] is output_stream
    assert captured["playback_visualizer"] is created_visualizers[0]
    assert captured["state_sample"] == "agent"
    assert captured["state_visualizer"] is created_visualizers[0]
    assert captured["playback_on_drain"] == captured["receive_loop"][1].playback_drained
    assert json.loads(captured["messages"][0]) == {
        "type": "session_start",
        "sample_rate": 16000,
        "sample": "agent",
        "instruct": "Warm and patient narrator.",
    }


@pytest.mark.asyncio
async def test_playback_controller_close_drops_buffered_audio():
    client = _load_module()
    stream = FakeStream()
    stream.block_writes = True
    playback = client.PlaybackController(stream)

    task = asyncio.create_task(playback.run())
    playback.enqueue(b"first")
    await asyncio.to_thread(stream.write_started.wait, 1)
    playback.enqueue(b"second")

    await playback.close()
    stream.write_release.set()
    await task

    assert stream.writes == [b"first"]


@pytest.mark.asyncio
async def test_receive_loop_raises_on_binary_frame_without_metadata():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()
    websocket = FakeWebSocket([b"orphan-audio"])

    with pytest.raises(
        RuntimeError, match="Received binary frame without audio_chunk metadata"
    ):
        await client.receive_loop(
            websocket, state, playback, event_sink=lambda event: None
        )


@pytest.mark.asyncio
async def test_receive_loop_raises_when_metadata_is_not_followed_by_binary():
    client = _load_module()
    state = client.DuplexState(active_response_id=1, last_response_id=1)
    playback = FakePlayback()
    websocket = FakeWebSocket(
        [
            '{"type": "audio_chunk", "response_id": 1}',
            '{"type": "response_done", "response_id": 1}',
        ]
    )

    with pytest.raises(RuntimeError, match="Received response_done before audio bytes"):
        await client.receive_loop(
            websocket, state, playback, event_sink=lambda event: None
        )


@pytest.mark.asyncio
async def test_receive_loop_drops_stale_binary_after_local_barge_in():
    client = _load_module()
    state = client.DuplexState(active_response_id=1, last_response_id=1)
    playback = FakePlayback()
    websocket = FakeWebSocket(
        [
            '{"type": "audio_chunk", "response_id": 1}',
            b"stale-audio",
            '{"type": "error", "message": "stop", "code": "STOP"}',
        ]
    )

    def event_sink(event):
        if event["type"] == "audio_chunk":
            assert state.start_local_speech(playback) == {"type": "speech_start"}

    with pytest.raises(RuntimeError, match="stop"):
        await client.receive_loop(websocket, state, playback, event_sink=event_sink)

    assert playback.chunks == []


@pytest.mark.asyncio
async def test_receive_loop_allows_interrupted_after_audio_metadata_without_binary():
    client = _load_module()
    state = client.DuplexState(active_response_id=1, last_response_id=1)
    playback = FakePlayback()
    events = []
    websocket = FakeWebSocket(
        [
            '{"type": "audio_chunk", "response_id": 1}',
            '{"type": "interrupted", "response_id": 1}',
            '{"type": "error", "message": "stop", "code": "STOP"}',
        ]
    )

    with pytest.raises(RuntimeError, match="stop"):
        await client.receive_loop(websocket, state, playback, event_sink=events.append)

    assert events == [
        {"type": "audio_chunk", "response_id": 1},
        {"type": "interrupted", "response_id": 1},
        {"type": "error", "message": "stop", "code": "STOP"},
    ]
