import importlib.util
from pathlib import Path
import asyncio
import base64
import json
import sys
import threading
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from starlette.websockets import WebSocketState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from conversation_ws import (
    ConversationSessionState,
    _forward_service_response,
    _interrupt_active_response,
    create_conversation_router,
)


APP_DIR = Path(__file__).resolve().parents[1] / "app"

AUDIO_CHUNK_A = b"a" * 12000
AUDIO_CHUNK_B = b"b" * 12000
AUDIO_CHUNK_C = b"c" * 12000
AUDIO_CHUNK_D = b"d" * 12000


def _load_app_module(module_name: str, filename: str):
    module_path = APP_DIR / filename
    assert module_path.is_file(), f"Missing module file: {module_path}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeConversationService:
    def __init__(self):
        self.transcribe_calls = []
        self.transcribe_contexts = []
        self.transcript_results = []
        self.response_started = []
        self.response_contexts = []
        self.response_released = {}

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        *,
        session_id: str | None = None,
        language_hint: str | None = None,
    ):
        self.transcribe_calls.append((audio_bytes, sample_rate))
        self.transcribe_contexts.append(
            {"session_id": session_id, "language_hint": language_hint}
        )
        if self.transcript_results:
            return self.transcript_results.pop(0)
        return "hello there"

    async def respond(
        self,
        transcript: str,
        response_id: int,
        *,
        sample: str | None = None,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ):
        self.response_started.append((transcript, response_id, sample))
        self.response_contexts.append(
            {
                "session_id": session_id,
                "history": history,
                "language_hint": language_hint,
            }
        )
        yield {"type": "assistant_text", "text": f"answering {transcript}"}

        gate = self.response_released.setdefault(
            response_id, __import__("asyncio").Event()
        )
        await gate.wait()

        yield {
            "type": "audio_chunk",
            "response_id": response_id,
            "payload": b"audio-bytes",
        }
        yield {"type": "response_done", "response_id": response_id}


class FakeAssistantBackend:
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.calls = []

    async def generate_response(self, transcript: str) -> str:
        self.calls.append(transcript)
        return self.response_text


class FakeTranscriptionResult:
    def __init__(
        self,
        text: str,
        language: str | None = None,
        language_probability: float | None = None,
    ):
        self.text = text
        self.language = language
        self.language_probability = language_probability


def _make_client(*, max_input_bytes: int = 1024 * 1024):
    service = FakeConversationService()
    app = FastAPI()
    app.include_router(
        create_conversation_router(service=service, max_input_bytes=max_input_bytes)
    )
    return TestClient(app), service


def _make_client_with_service(service, *, max_input_bytes: int = 1024 * 1024):
    app = FastAPI()
    app.include_router(
        create_conversation_router(service=service, max_input_bytes=max_input_bytes)
    )
    return TestClient(app)


def test_faster_whisper_asr_adapter_transcribes_pcm_bytes(monkeypatch):
    model_calls = {}

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            model_calls["init"] = (model_name, kwargs)

        def transcribe(self, audio, **kwargs):
            model_calls["audio"] = audio
            model_calls["transcribe_kwargs"] = kwargs

            class Segment:
                def __init__(self, text):
                    self.text = text

            return iter([Segment(" hello"), Segment(" world")]), types.SimpleNamespace(
                language="fr"
            )

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    asr = _load_app_module("asr_under_test", "asr.py")
    adapter = asr.FasterWhisperASR(
        model_name="small",
        device="cuda:1",
        compute_type="int8_float16",
    )

    transcript = adapter.transcribe(
        np.array([0, 32767, -32768], dtype=np.int16).tobytes(),
        language_hint="es",
    )

    assert transcript.text == "hello world"
    assert transcript.language == "fr"
    assert model_calls["init"] == (
        "small",
        {"device": "cuda", "device_index": 1, "compute_type": "int8_float16"},
    )
    assert model_calls["transcribe_kwargs"] == {"beam_size": 1, "language": "es"}
    assert model_calls["audio"].tolist() == pytest.approx([0.0, 32767 / 32768, -1.0])


@pytest.mark.asyncio
async def test_server_conversation_service_requires_16khz_audio(monkeypatch):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            return f"heard {len(pcm_bytes)} bytes"

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_backend=FakeAssistantBackend()
    )

    with pytest.raises(RuntimeError, match="16 kHz PCM"):
        await service.transcribe(b"abc", 8000)


@pytest.mark.asyncio
async def test_server_conversation_service_passes_language_hint_to_asr(monkeypatch):
    asr_calls = []

    class FakeASR:
        def transcribe(
            self,
            pcm_bytes: bytes,
            *,
            language_hint: str | None = None,
            session_id: str | None = None,
        ):
            asr_calls.append(
                {
                    "pcm_bytes": pcm_bytes,
                    "language_hint": language_hint,
                    "session_id": session_id,
                }
            )
            return FakeTranscriptionResult("heard caller", language="es")

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_backend=FakeAssistantBackend()
    )

    result = await service.transcribe(
        b"abc",
        16000,
        session_id="session-123",
        language_hint="fr",
    )

    assert result.text == "heard caller"
    assert result.language == "es"
    assert asr_calls == [
        {
            "pcm_bytes": b"abc",
            "language_hint": "fr",
            "session_id": "session-123",
        }
    ]


@pytest.mark.asyncio
async def test_server_conversation_service_passes_session_context_to_backend(
    monkeypatch,
):
    backend_calls = []

    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError(f"transcribe should not be called: {pcm_bytes!r}")

    class ContextAwareBackend:
        async def generate_response(
            self,
            transcript: str,
            *,
            session_id: str | None = None,
            history: list[dict[str, str]] | None = None,
            language_hint: str | None = None,
        ) -> str:
            backend_calls.append(
                {
                    "transcript": transcript,
                    "session_id": session_id,
                    "history": history,
                    "language_hint": language_hint,
                }
            )
            return "Assistant sentence."

    class FakeAudio:
        shape = (1, 2400)

    async def fake_generate_sentence_audio(gen_kwargs):
        del gen_kwargs
        return FakeAudio()

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_backend=ContextAwareBackend()
    )
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")

    events = [
        event
        async for event in service.respond(
            "Current caller message",
            5,
            session_id="session-123",
            history=[
                {"user": "First caller turn", "assistant": "First assistant reply"},
                {"user": "Second caller turn", "assistant": "Second assistant reply"},
            ],
            language_hint="de",
        )
    ]

    assert backend_calls == [
        {
            "transcript": "Current caller message",
            "session_id": "session-123",
            "history": [
                {"user": "First caller turn", "assistant": "First assistant reply"},
                {"user": "Second caller turn", "assistant": "Second assistant reply"},
            ],
            "language_hint": "de",
        }
    ]
    assert events == [
        {"type": "assistant_text", "text": "Assistant sentence.", "response_id": 5},
        {
            "type": "audio_chunk",
            "response_id": 5,
            "chunk_index": 0,
            "sentence": "Assistant sentence.",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm",
        },
        {"type": "response_done", "response_id": 5},
    ]


@pytest.mark.asyncio
async def test_server_conversation_service_logs_completed_turn_metrics(
    monkeypatch, caplog
):
    class FakeASR:
        def transcribe(
            self,
            pcm_bytes: bytes,
            *,
            language_hint: str | None = None,
            session_id: str | None = None,
        ):
            assert pcm_bytes == b"abc"
            assert language_hint == "fr"
            assert session_id == "session-123"
            return FakeTranscriptionResult("heard caller", language="es")

    class FakeAudio:
        shape = (1, 2400)

    async def fake_generate_sentence_audio(gen_kwargs):
        assert gen_kwargs["text"] == "Assistant sentence."
        return FakeAudio()

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    assistant_backend = FakeAssistantBackend("Assistant sentence.")
    assistant_backend._model = "test-model"
    service = server.ConversationService(FakeASR(), assistant_backend=assistant_backend)
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")
    caplog.set_level("INFO", logger="omnivoice-server")

    transcript = await service.transcribe(
        b"abc",
        16000,
        session_id="session-123",
        language_hint="fr",
    )
    events = [
        event
        async for event in service.respond(
            transcript.text,
            6,
            session_id="session-123",
            history=[],
            language_hint="fr",
        )
    ]

    assert events == [
        {"type": "assistant_text", "text": "Assistant sentence.", "response_id": 6},
        {
            "type": "audio_chunk",
            "response_id": 6,
            "chunk_index": 0,
            "sentence": "Assistant sentence.",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm",
        },
        {"type": "response_done", "response_id": 6},
    ]

    log_payloads = []
    for record in caplog.records:
        try:
            log_payloads.append(json.loads(record.message))
        except json.JSONDecodeError:
            continue

    completed_turn_logs = [
        payload
        for payload in log_payloads
        if payload.get("event") == "conversation_turn_completed"
    ]

    assert len(completed_turn_logs) == 1
    assert completed_turn_logs[0] == {
        "event": "conversation_turn_completed",
        "session_id": "session-123",
        "response_id": 6,
        "detected_language": "es",
        "detected_language_probability": None,
        "language_hint": "fr",
        "assistant_backend": "FakeAssistantBackend",
        "assistant_model": "test-model",
        "asr_ms": completed_turn_logs[0]["asr_ms"],
        "llm_ms": completed_turn_logs[0]["llm_ms"],
        "tts_first_chunk_ms": completed_turn_logs[0]["tts_first_chunk_ms"],
        "tts_total_ms": completed_turn_logs[0]["tts_total_ms"],
        "turn_total_ms": completed_turn_logs[0]["turn_total_ms"],
    }
    assert completed_turn_logs[0]["asr_ms"] >= 0
    assert completed_turn_logs[0]["llm_ms"] >= 0
    assert completed_turn_logs[0]["tts_first_chunk_ms"] >= 0
    assert (
        completed_turn_logs[0]["tts_total_ms"]
        >= completed_turn_logs[0]["tts_first_chunk_ms"]
    )
    assert (
        completed_turn_logs[0]["turn_total_ms"]
        >= completed_turn_logs[0]["tts_total_ms"]
    )


@pytest.mark.asyncio
async def test_server_conversation_service_clears_empty_turn_metrics_before_next_log(
    monkeypatch, caplog
):
    class FakeASR:
        def transcribe(
            self,
            pcm_bytes: bytes,
            *,
            language_hint: str | None = None,
            session_id: str | None = None,
        ):
            assert pcm_bytes == b"abc"
            assert language_hint == "it"
            assert session_id == "session-123"
            return FakeTranscriptionResult("   ", language="it")

    class FakeAudio:
        shape = (1, 2400)

    async def fake_generate_sentence_audio(gen_kwargs):
        assert gen_kwargs["text"] == "Assistant sentence."
        return FakeAudio()

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    assistant_backend = FakeAssistantBackend("Assistant sentence.")
    assistant_backend._model = "test-model"
    service = server.ConversationService(FakeASR(), assistant_backend=assistant_backend)
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")
    caplog.set_level("INFO", logger="omnivoice-server")

    transcript = await service.transcribe(
        b"abc",
        16000,
        session_id="session-123",
        language_hint="it",
    )
    empty_turn_events = [
        event
        async for event in service.respond(
            transcript.text,
            1,
            session_id="session-123",
            history=[],
            language_hint="it",
        )
    ]
    non_empty_turn_events = [
        event
        async for event in service.respond(
            "Actual caller transcript",
            2,
            session_id="session-123",
            history=[],
            language_hint="fr",
        )
    ]

    assert empty_turn_events == [{"type": "response_done", "response_id": 1}]
    assert non_empty_turn_events == [
        {"type": "assistant_text", "text": "Assistant sentence.", "response_id": 2},
        {
            "type": "audio_chunk",
            "response_id": 2,
            "chunk_index": 0,
            "sentence": "Assistant sentence.",
            "duration_ms": 100,
            "generation_time_ms": non_empty_turn_events[1]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm",
        },
        {"type": "response_done", "response_id": 2},
    ]

    log_payloads = []
    for record in caplog.records:
        try:
            log_payloads.append(json.loads(record.message))
        except json.JSONDecodeError:
            continue

    completed_turn_logs = [
        payload
        for payload in log_payloads
        if payload.get("event") == "conversation_turn_completed"
    ]

    assert completed_turn_logs == [
        {
            "event": "conversation_turn_completed",
            "session_id": "session-123",
            "response_id": 2,
            "detected_language": None,
            "detected_language_probability": None,
            "language_hint": "fr",
            "assistant_backend": "FakeAssistantBackend",
            "assistant_model": "test-model",
            "asr_ms": None,
            "llm_ms": completed_turn_logs[0]["llm_ms"],
            "tts_first_chunk_ms": completed_turn_logs[0]["tts_first_chunk_ms"],
            "tts_total_ms": completed_turn_logs[0]["tts_total_ms"],
            "turn_total_ms": completed_turn_logs[0]["turn_total_ms"],
        }
    ]


@pytest.mark.asyncio
async def test_server_conversation_service_streams_response_scoped_text_and_audio(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAudio:
        shape = (1, 2400)

    class FakeModel:
        def __init__(self):
            self.prompt_calls = []

        def create_voice_clone_prompt(self, **kwargs):
            self.prompt_calls.append(kwargs)
            return {"cached": True, **kwargs}

    generated_kwargs = []
    assistant_backend = FakeAssistantBackend(
        "Assistant first sentence. Assistant second sentence?"
    )
    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = FakeModel()
    server.voice_samples = {
        "agent-voice": {"audio_path": "/tmp/agent.wav", "ref_text": "hello"}
    }

    async def fake_generate_one(get_model, model_lock, gen_kwargs):
        generated_kwargs.append(gen_kwargs)
        assert get_model() is server.model
        assert model_lock is server.model_lock
        return FakeAudio()

    monkeypatch.setattr(server, "_generate_one", fake_generate_one)
    monkeypatch.setattr(
        server,
        "encode_audio_chunk",
        lambda audio, output_format, sample_rate=24000: (
            f"{output_format}:{sample_rate}:{audio.shape[-1]}".encode("ascii")
        ),
    )

    service = server.ConversationService(
        FakeASR(),
        assistant_backend=assistant_backend,
    )

    events = [
        event
        async for event in service.respond(
            "Caller transcript that should not be echoed.",
            7,
            sample="agent-voice",
            instruct="Ignore this prompt.",
        )
    ]

    assert events == [
        {
            "type": "assistant_text",
            "text": "Assistant first sentence.",
            "response_id": 7,
        },
        {
            "type": "audio_chunk",
            "response_id": 7,
            "chunk_index": 0,
            "sentence": "Assistant first sentence.",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": False,
            "payload": b"pcm:24000:2400",
        },
        {
            "type": "assistant_text",
            "text": "Assistant second sentence?",
            "response_id": 7,
        },
        {
            "type": "audio_chunk",
            "response_id": 7,
            "chunk_index": 1,
            "sentence": "Assistant second sentence?",
            "duration_ms": 100,
            "generation_time_ms": events[3]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm:24000:2400",
        },
        {"type": "response_done", "response_id": 7},
    ]
    assert generated_kwargs == [
        {
            "text": "Assistant first sentence.",
            "num_step": server.WS_DEFAULT_NUM_STEP,
            "voice_clone_prompt": {
                "cached": True,
                "ref_audio": "/tmp/agent.wav",
                "ref_text": "hello",
            },
        },
        {
            "text": "Assistant second sentence?",
            "num_step": server.WS_DEFAULT_NUM_STEP,
            "voice_clone_prompt": {
                "cached": True,
                "ref_audio": "/tmp/agent.wav",
                "ref_text": "hello",
            },
        },
    ]
    assert assistant_backend.calls == ["Caller transcript that should not be echoed."]
    assert server.model.prompt_calls == [
        {"ref_audio": "/tmp/agent.wav", "ref_text": "hello"}
    ]


@pytest.mark.asyncio
async def test_server_conversation_service_passes_instruct_to_tts_without_sample(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAudio:
        shape = (1, 2400)

    generated_kwargs = []
    assistant_backend = FakeAssistantBackend("Assistant sentence.")
    server = _load_server_module(monkeypatch, conversation_enabled=True)

    async def fake_generate_sentence_audio(gen_kwargs):
        generated_kwargs.append(gen_kwargs)
        return FakeAudio()

    service = server.ConversationService(
        FakeASR(),
        assistant_backend=assistant_backend,
    )
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")

    events = [
        event
        async for event in service.respond(
            "Caller transcript that should not be echoed.",
            8,
            instruct="Calm and reassuring support agent.",
        )
    ]

    assert events == [
        {
            "type": "assistant_text",
            "text": "Assistant sentence.",
            "response_id": 8,
        },
        {
            "type": "audio_chunk",
            "response_id": 8,
            "chunk_index": 0,
            "sentence": "Assistant sentence.",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm",
        },
        {"type": "response_done", "response_id": 8},
    ]
    assert generated_kwargs == [
        {
            "text": "Assistant sentence.",
            "num_step": server.WS_DEFAULT_NUM_STEP,
            "instruct": "Calm and reassuring support agent.",
        }
    ]
    assert assistant_backend.calls == ["Caller transcript that should not be echoed."]


@pytest.mark.asyncio
async def test_server_conversation_service_sanitizes_assistant_output_before_tts(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAudio:
        shape = (1, 2400)

    generated_texts = []
    assistant_backend = FakeAssistantBackend(
        '```json\n{"reply":"# Update\\n- Your order ships tomorrow!!!\\n- Tracking number follows."}\n```'
    )
    server = _load_server_module(monkeypatch, conversation_enabled=True)

    async def fake_generate_sentence_audio(gen_kwargs):
        generated_texts.append(gen_kwargs["text"])
        return FakeAudio()

    service = server.ConversationService(
        FakeASR(),
        assistant_backend=assistant_backend,
    )
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")

    events = [
        event
        async for event in service.respond(
            "Caller transcript that should not be echoed.",
            8,
        )
    ]

    assert events == [
        {
            "type": "assistant_text",
            "text": "Update Your order ships tomorrow!",
            "response_id": 8,
        },
        {
            "type": "audio_chunk",
            "response_id": 8,
            "chunk_index": 0,
            "sentence": "Update Your order ships tomorrow!",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": False,
            "payload": b"pcm",
        },
        {
            "type": "assistant_text",
            "text": "Tracking number follows.",
            "response_id": 8,
        },
        {
            "type": "audio_chunk",
            "response_id": 8,
            "chunk_index": 1,
            "sentence": "Tracking number follows.",
            "duration_ms": 100,
            "generation_time_ms": events[3]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm",
        },
        {"type": "response_done", "response_id": 8},
    ]
    assert generated_texts == [
        "Update Your order ships tomorrow!",
        "Tracking number follows.",
    ]


def test_server_registers_conversation_router_when_enabled(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=True)

    route_paths = {route.path for route in server.app.routes}

    assert "/ws/conversation" in route_paths
    assert server.CONVERSATION_WS_ENABLED is True
    assert server.ASR_MODEL == "tiny.en"
    assert server.ASR_DEVICE == "cpu"
    assert server.ASR_COMPUTE_TYPE == "int8"


def test_server_skips_conversation_router_when_disabled(monkeypatch):
    server = _load_server_module(
        monkeypatch, conversation_enabled=False, broken_asr_import=True
    )

    route_paths = {route.path for route in server.app.routes}

    assert "/ws/conversation" not in route_paths
    assert server.CONVERSATION_WS_ENABLED is False


def test_server_conversation_defaults_ollama_model_to_gemma4(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=True)

    assert server.OLLAMA_MODEL == "gemma4"


def test_server_conversation_service_uses_configured_ollama_host_and_model(
    monkeypatch,
):
    client_init = []
    backend_init = []
    asr_init = []

    class FakeAsyncClient:
        def __init__(self, host):
            client_init.append(host)

    class FakeAssistantBackendImpl:
        def __init__(self, client, model):
            backend_init.append((client, model))

        async def generate_response(self, transcript: str) -> str:
            raise AssertionError(
                f"generate_response should not be called: {transcript}"
            )

    class FakeASR:
        def __init__(self, model_name, device, compute_type):
            asr_init.append((model_name, device, compute_type))

    monkeypatch.setenv("OMNIVOICE_OLLAMA_HOST", "http://ollama.internal:11434")
    monkeypatch.setenv("OMNIVOICE_OLLAMA_MODEL", "custom-gemma")
    monkeypatch.setitem(
        sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    )
    monkeypatch.setitem(
        sys.modules,
        "assistant_backends",
        types.SimpleNamespace(OllamaAssistantBackend=FakeAssistantBackendImpl),
    )

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    monkeypatch.setitem(
        sys.modules, "asr", types.SimpleNamespace(FasterWhisperASR=FakeASR)
    )

    service = server._create_conversation_service()

    assert client_init == ["http://ollama.internal:11434"]
    assert asr_init == [("tiny.en", "cpu", "int8")]
    assert len(backend_init) == 1
    assert backend_init[0][0].__class__ is FakeAsyncClient
    assert backend_init[0][1] == "custom-gemma"
    assert isinstance(service._assistant_backend, FakeAssistantBackendImpl)


@pytest.mark.asyncio
async def test_ollama_assistant_backend_enforces_phone_agent_plain_text_policy():
    chat_calls = []

    class FakeAsyncClient:
        async def chat(self, **kwargs):
            chat_calls.append(kwargs)
            return {"message": {"content": "I can help with that now."}}

    assistant_backends = _load_app_module(
        "assistant_backends_under_test", "assistant_backends.py"
    )
    backend = assistant_backends.OllamaAssistantBackend(
        FakeAsyncClient(), "phone-agent-model"
    )

    reply = await backend.generate_response("I need to reschedule my appointment.")

    assert reply == "I can help with that now."
    assert len(chat_calls) == 1
    assert chat_calls[0]["model"] == "phone-agent-model"
    assert chat_calls[0]["stream"] is False
    assert chat_calls[0]["messages"][1] == {
        "role": "user",
        "content": "I need to reschedule my appointment.",
    }
    system_prompt = chat_calls[0]["messages"][0]["content"]
    assert chat_calls[0]["messages"][0]["role"] == "system"
    assert "phone" in system_prompt.lower()
    assert "calm" in system_prompt.lower()
    assert "direct" in system_prompt.lower()
    assert "reassuring" in system_prompt.lower()
    assert "plain text" in system_prompt.lower()
    assert "markdown" in system_prompt.lower()
    assert "only these non-verbal tags" in system_prompt.lower()
    assert "[laughter]" in system_prompt
    assert "[question-en]" in system_prompt
    assert "overuse" in system_prompt.lower()


@pytest.mark.asyncio
async def test_ollama_assistant_backend_uses_structured_context_messages():
    chat_calls = []

    class FakeAsyncClient:
        async def chat(self, **kwargs):
            chat_calls.append(kwargs)
            return {"message": {"content": "I can help with that now."}}

    assistant_backends = _load_app_module(
        "assistant_backends_context_under_test", "assistant_backends.py"
    )
    backend = assistant_backends.OllamaAssistantBackend(
        FakeAsyncClient(), "phone-agent-model"
    )

    reply = await backend.generate_response(
        "Current caller message",
        session_id="session-123",
        history=[
            {"user": "First caller turn", "assistant": "First assistant reply"},
            {"user": "Second caller turn", "assistant": "Second assistant reply"},
        ],
        language_hint="de",
    )

    assert reply == "I can help with that now."
    assert chat_calls == [
        {
            "model": "phone-agent-model",
            "messages": [
                {
                    "role": "system",
                    "content": assistant_backends.PHONE_AGENT_SYSTEM_PROMPT,
                },
                {
                    "role": "system",
                    "content": "Conversation context: session_id=session-123; caller_language_hint=de.",
                },
                {
                    "role": "system",
                    "content": "Reply only in the caller's language for this turn: German (de).",
                },
                {"role": "user", "content": "First caller turn"},
                {"role": "assistant", "content": "First assistant reply"},
                {"role": "user", "content": "Second caller turn"},
                {"role": "assistant", "content": "Second assistant reply"},
                {"role": "user", "content": "Current caller message"},
            ],
            "stream": False,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {},
        {"message": {}},
        {"message": {"content": None}},
        {"message": {"content": ["not", "text"]}},
    ],
)
async def test_ollama_assistant_backend_raises_for_malformed_response(response):
    class FakeAsyncClient:
        async def chat(self, **kwargs):
            del kwargs
            return response

    assistant_backends = _load_app_module(
        "assistant_backends_malformed_under_test", "assistant_backends.py"
    )
    backend = assistant_backends.OllamaAssistantBackend(
        FakeAsyncClient(), "phone-agent-model"
    )

    with pytest.raises(
        RuntimeError, match="Assistant backend returned no usable message content"
    ):
        await backend.generate_response("Please help me with my order.")


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        (
            '```json\n{"reply":"# Update\\n- Your order ships tomorrow!!!\\n- Tracking number follows."}\n```',
            "Update Your order ships tomorrow! Tracking number follows.",
        ),
        (
            "  ## Status\n\n* We found your reservation??\n\n* It is confirmed.  ",
            "Status We found your reservation? It is confirmed.",
        ),
        (
            "Please review [order details](https://example.com/orders/42) next.",
            "Please review order details next.",
        ),
        (
            "Say `order number 42` when you call back.",
            "Say order number 42 when you call back.",
        ),
        (
            "Assistant: **Please** call back at _noon_.",
            "Please call back at noon.",
        ),
        (
            "System: You are a helpful assistant. Caller: My package is late. Assistant: I can help with that.",
            "I can help with that.",
        ),
        (
            '{"unexpected":"Leave this structure intact."}',
            "",
        ),
        (
            '"hello"',
            "",
        ),
        (
            "123",
            "",
        ),
        (
            "true",
            "",
        ),
        (
            "false",
            "",
        ),
        (
            "null",
            "",
        ),
        (
            "Assistant: I can help.\nPlease confirm your order number.",
            "I can help. Please confirm your order number.",
        ),
        (
            "Your assigned agent: Maria will call shortly.",
            "Your assigned agent: Maria will call shortly.",
        ),
        (
            "[laughter] I can help with that.",
            "[laughter] I can help with that.",
        ),
        (
            "[Laughter] I can help with that.",
            "I can help with that.",
        ),
        (
            "[ laughter ] I can help with that.",
            "I can help with that.",
        ),
        (
            "[wave] I can help with that.",
            "I can help with that.",
        ),
        (
            "[laughter] [wave] I can help with that.",
            "[laughter] I can help with that.",
        ),
    ],
)
def test_text_sanitizer_removes_markdown_json_and_noise(raw_text, expected):
    text_sanitizer = _load_app_module("text_sanitizer_under_test", "text_sanitizer.py")

    assert text_sanitizer.sanitize_assistant_text(raw_text) == expected


def _load_server_module(
    monkeypatch, *, conversation_enabled: bool, broken_asr_import: bool = False
):
    monkeypatch.setenv("OMNIVOICE_WS_ENABLED", "false")
    monkeypatch.setenv(
        "OMNIVOICE_CONVERSATION_WS_ENABLED",
        "true" if conversation_enabled else "false",
    )
    monkeypatch.setenv("OMNIVOICE_ASR_MODEL", "tiny.en")
    monkeypatch.setenv("OMNIVOICE_ASR_DEVICE", "cpu")
    monkeypatch.setenv("OMNIVOICE_ASR_COMPUTE_TYPE", "int8")

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        Tensor=object,
    )
    fake_torchaudio = types.SimpleNamespace(save=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    if broken_asr_import:
        broken_asr = types.ModuleType("asr")

        def _broken_getattr(name):
            raise AssertionError(f"Unexpected ASR import: {name}")

        broken_asr.__getattr__ = _broken_getattr
        monkeypatch.setitem(sys.modules, "asr", broken_asr)
    else:
        fake_asr = types.SimpleNamespace(FasterWhisperASR=object)
        monkeypatch.setitem(sys.modules, "asr", fake_asr)
    sys.modules.pop("audio_encoder", None)
    sys.modules.pop("ws_handler", None)
    sys.modules.pop("server_under_test", None)

    return _load_app_module("server_under_test", "server.py")


def test_server_uses_voices_dir_env_var_canonically(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_VOICES_DIR", "/voices-primary")
    monkeypatch.setenv("OMNIVOICE_SAMPLES_DIR", "/samples-fallback")

    server = _load_server_module(monkeypatch, conversation_enabled=False)

    assert server.VOICES_DIR == Path("/voices-primary")


def test_server_uses_samples_dir_env_var_as_compatibility_alias(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_VOICES_DIR", raising=False)
    monkeypatch.setenv("OMNIVOICE_SAMPLES_DIR", "/samples-fallback")

    server = _load_server_module(monkeypatch, conversation_enabled=False)

    assert server.VOICES_DIR == Path("/samples-fallback")


def test_server_falls_back_to_legacy_samples_dir_when_canonical_scan_is_empty(
    monkeypatch,
):
    server = _load_server_module(monkeypatch, conversation_enabled=False)
    server.VOICES_DIR = Path("/voices")
    server.SAMPLES_DIR = Path("/samples")

    scan_calls = []

    def fake_scan_voices(directory):
        scan_calls.append(directory)
        if directory == Path("/voices"):
            return {}
        if directory == Path("/samples"):
            return {
                "legacy-voice": {"audio_path": "/samples/legacy.wav", "ref_text": "hi"}
            }
        raise AssertionError(f"unexpected scan dir: {directory}")

    monkeypatch.setattr(server, "scan_voices", fake_scan_voices)

    assert server._load_voice_samples() == {
        "legacy-voice": {"audio_path": "/samples/legacy.wav", "ref_text": "hi"}
    }
    assert scan_calls == [Path("/voices"), Path("/samples")]


def test_server_prefers_canonical_voices_dir_before_legacy_fallback(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=False)
    server.VOICES_DIR = Path("/voices")
    server.SAMPLES_DIR = Path("/samples")

    scan_calls = []

    def fake_scan_voices(directory):
        scan_calls.append(directory)
        if directory == Path("/voices"):
            return {
                "canonical-voice": {
                    "audio_path": "/voices/canonical.wav",
                    "ref_text": "hello",
                }
            }
        raise AssertionError(f"legacy fallback should not scan: {directory}")

    monkeypatch.setattr(server, "scan_voices", fake_scan_voices)

    assert server._load_voice_samples() == {
        "canonical-voice": {
            "audio_path": "/voices/canonical.wav",
            "ref_text": "hello",
        }
    }
    assert scan_calls == [Path("/voices")]


def test_server_registers_voice_routes_and_samples_aliases(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=False)

    route_paths = {route.path for route in server.app.routes}

    assert "/voices" in route_paths
    assert "/voices/reload" in route_paths
    assert "/samples" in route_paths
    assert "/samples/reload" in route_paths


def test_session_start_acknowledges_sample():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )

        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }


def test_session_start_accepts_explicit_language_override():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "sample": "agent-voice",
                "sample_rate": 16000,
                "language": "fr",
            }
        )

        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
            "language": "fr",
        }


def test_session_start_persists_instruct_for_tts_when_sample_is_absent():
    class InstructAwareConversationService(FakeConversationService):
        def __init__(self):
            super().__init__()
            self.response_instructs = []

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
            self.response_started.append((transcript, response_id, sample))
            self.response_contexts.append(
                {
                    "session_id": session_id,
                    "history": history,
                    "language_hint": language_hint,
                }
            )
            self.response_instructs.append(instruct)
            yield {"type": "assistant_text", "text": f"answering {transcript}"}

            gate = self.response_released.setdefault(response_id, asyncio.Event())
            await gate.wait()

            yield {
                "type": "audio_chunk",
                "response_id": response_id,
                "payload": b"audio-bytes",
            }
            yield {"type": "response_done", "response_id": response_id}

    service = InstructAwareConversationService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "instruct": "Calm and reassuring support agent.",
                "sample_rate": 16000,
            }
        )
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": None,
            "sample_rate": 16000,
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }
        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.response_started == [("hello there", 1, None)]
    assert service.response_instructs == ["Calm and reassuring support agent."]


def test_session_start_sample_still_wins_over_instruct_for_tts():
    class InstructAwareConversationService(FakeConversationService):
        def __init__(self):
            super().__init__()
            self.response_instructs = []

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
            self.response_started.append((transcript, response_id, sample))
            self.response_contexts.append(
                {
                    "session_id": session_id,
                    "history": history,
                    "language_hint": language_hint,
                }
            )
            self.response_instructs.append(instruct)
            yield {"type": "assistant_text", "text": f"answering {transcript}"}

            gate = self.response_released.setdefault(response_id, asyncio.Event())
            await gate.wait()

            yield {
                "type": "audio_chunk",
                "response_id": response_id,
                "payload": b"audio-bytes",
            }
            yield {"type": "response_done", "response_id": response_id}

    service = InstructAwareConversationService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "sample": "agent-voice",
                "instruct": "Ignore this prompt.",
                "sample_rate": 16000,
            }
        )
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }
        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.response_started == [("hello there", 1, "agent-voice")]
    assert service.response_instructs == [None]


def test_session_start_rejects_unknown_sample_before_audio_collection():
    class ValidatingConversationService(FakeConversationService):
        def __init__(self):
            super().__init__()
            self.validated_samples = []

        def validate_sample(self, sample: str | None) -> None:
            self.validated_samples.append(sample)
            if sample == "missing-voice":
                raise RuntimeError("Sample 'missing-voice' not found.")

    service = ValidatingConversationService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "missing-voice"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Sample 'missing-voice' not found.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }

    assert service.validated_samples == ["missing-voice", "agent-voice"]


@pytest.mark.asyncio
async def test_voices_reload_clears_conversation_voice_prompt_cache(monkeypatch):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_backend=FakeAssistantBackend()
    )
    service._voice_prompt_cache[("agent-voice", "hello")] = {"cached": True}
    server.conversation_service = service
    server.voice_samples = {
        "agent-voice": {"audio_path": "/tmp/old.wav", "ref_text": "hello"}
    }

    def fake_scan_voices(_directory):
        return {"agent-voice": {"audio_path": "/tmp/new.wav", "ref_text": "updated"}}

    monkeypatch.setattr(server, "scan_voices", fake_scan_voices)

    result = await server.reload_voices(None)

    assert result == {"status": "ok", "count": 1}
    assert server.voice_samples == {
        "agent-voice": {"audio_path": "/tmp/new.wav", "ref_text": "updated"}
    }
    assert service._voice_prompt_cache == {}


@pytest.mark.asyncio
async def test_samples_reload_alias_reuses_voice_reload_handler(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=False)
    server.voice_samples = {}

    scan_calls = []

    def fake_scan_voices(directory):
        scan_calls.append(directory)
        return {"agent-voice": {"audio_path": "/tmp/new.wav", "ref_text": "updated"}}

    monkeypatch.setattr(server, "scan_voices", fake_scan_voices)

    result = await server.reload_samples(None)

    assert result == {"status": "ok", "count": 1}
    assert scan_calls == [server.VOICES_DIR]


def test_health_counts_conversation_websocket_connections(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = object()
    server.active_ws_connections = 0
    client = TestClient(server.app)

    with client.websocket_connect("/ws/conversation"):
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["active_ws_connections"] == 1

    assert client.get("/health").json()["active_ws_connections"] == 0


def test_session_start_invalid_sample_rate_returns_recoverable_error():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": "bad"}
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Field 'sample_rate' must be an integer.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )

        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }


@pytest.mark.parametrize("sample_rate", [0, -16000])
def test_session_start_rejects_non_positive_sample_rate(sample_rate: int):
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "sample": "agent-voice",
                "sample_rate": sample_rate,
            }
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Field 'sample_rate' must be a positive integer.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }


@pytest.mark.parametrize("sample_rate", [8000, 44100])
def test_session_start_rejects_unsupported_positive_sample_rate(sample_rate: int):
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "sample": "agent-voice",
                "sample_rate": sample_rate,
            }
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Field 'sample_rate' must be 16000 for conversation ASR.",
            "code": "INVALID_MESSAGE",
        }


def test_audio_input_limit_returns_controlled_error_and_resets_collection():
    client, service = _make_client(max_input_bytes=len(AUDIO_CHUNK_A) + 1)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_bytes(AUDIO_CHUNK_B)

        assert ws.receive_json() == {
            "type": "error",
            "message": "Input audio exceeded the maximum buffered size.",
            "code": "INPUT_AUDIO_TOO_LARGE",
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

    assert service.transcribe_calls == []


def test_speech_end_triggers_transcript_and_assistant_response():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_bytes(AUDIO_CHUNK_B)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(AUDIO_CHUNK_A + AUDIO_CHUNK_B, 16000)]


def test_detected_language_without_probability_keeps_existing_session_language_hint():
    client, service = _make_client()
    service.transcript_results = [
        FakeTranscriptionResult("first caller turn", language="es"),
        FakeTranscriptionResult("second caller turn", language="fr"),
        FakeTranscriptionResult("third caller turn", language="de"),
        FakeTranscriptionResult("fourth caller turn", language="it"),
    ]

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        for response_id in range(1, 5):
            ws.send_json({"type": "speech_start"})
            assert ws.receive_json() == {"type": "listening"}
            ws.send_bytes(bytes([96 + response_id]) * 12000)
            ws.send_json({"type": "speech_end"})

            assert ws.receive_json() == {
                "type": "transcript_final",
                "text": service.response_started[response_id - 1][0],
            }
            assert ws.receive_json() == {
                "type": "assistant_text",
                "text": f"answering {service.response_started[response_id - 1][0]}",
                "response_id": response_id,
            }

            service.response_released[response_id].set()
            assert ws.receive_json() == {
                "type": "audio_chunk",
                "response_id": response_id,
            }
            assert ws.receive_bytes() == b"audio-bytes"
            assert ws.receive_json() == {
                "type": "response_done",
                "response_id": response_id,
            }

    session_ids = {context["session_id"] for context in service.transcribe_contexts} | {
        context["session_id"] for context in service.response_contexts
    }
    assert len(session_ids) == 1
    assert None not in session_ids
    assert [context["language_hint"] for context in service.transcribe_contexts] == [
        None,
        None,
        None,
        None,
    ]
    assert [context["language_hint"] for context in service.response_contexts] == [
        "es",
        "es",
        "es",
        "es",
    ]
    assert service.response_contexts[0]["history"] == []
    assert service.response_contexts[3]["history"] == [
        {"user": "first caller turn", "assistant": "answering first caller turn"},
        {"user": "second caller turn", "assistant": "answering second caller turn"},
        {"user": "third caller turn", "assistant": "answering third caller turn"},
    ]


def test_language_override_stays_sticky_across_turns():
    client, service = _make_client()
    service.transcript_results = [
        FakeTranscriptionResult("bonjour", language="fr"),
        FakeTranscriptionResult("toujours en francais", language="en"),
    ]

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "language": "fr"}
        )
        ws.receive_json()

        for response_id in range(1, 3):
            ws.send_json({"type": "speech_start"})
            assert ws.receive_json() == {"type": "listening"}
            ws.send_bytes(AUDIO_CHUNK_A)
            ws.send_json({"type": "speech_end"})
            ws.receive_json()
            ws.receive_json()
            service.response_released[response_id].set()
            ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()

    assert [context["language_hint"] for context in service.transcribe_contexts] == [
        "fr",
        "fr",
    ]
    assert [context["language_hint"] for context in service.response_contexts] == [
        "fr",
        "fr",
    ]


def test_detected_language_below_threshold_does_not_replace_session_language_hint():
    client, service = _make_client()
    service.transcript_results = [
        FakeTranscriptionResult(
            "primo turno italiano", language="it", language_probability=0.97
        ),
        FakeTranscriptionResult(
            "secondo turno sempre italiano", language="el", language_probability=0.28
        ),
    ]

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        for response_id in range(1, 3):
            ws.send_json({"type": "speech_start"})
            assert ws.receive_json() == {"type": "listening"}
            ws.send_bytes(AUDIO_CHUNK_A)
            ws.send_json({"type": "speech_end"})
            ws.receive_json()
            ws.receive_json()
            service.response_released[response_id].set()
            ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()

    assert [context["language_hint"] for context in service.response_contexts] == [
        "it",
        "it",
    ]


def test_detected_language_above_threshold_updates_session_language_hint():
    client, service = _make_client()
    service.transcript_results = [
        FakeTranscriptionResult("ciao", language="it", language_probability=0.95),
        FakeTranscriptionResult(
            "always english now", language="en", language_probability=0.99
        ),
        FakeTranscriptionResult("third turn", language="en", language_probability=0.88),
    ]

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        for response_id in range(1, 4):
            ws.send_json({"type": "speech_start"})
            assert ws.receive_json() == {"type": "listening"}
            ws.send_bytes(AUDIO_CHUNK_A)
            ws.send_json({"type": "speech_end"})
            ws.receive_json()
            ws.receive_json()
            service.response_released[response_id].set()
            ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()

    assert [context["language_hint"] for context in service.response_contexts] == [
        "it",
        "en",
        "en",
    ]


def test_faster_whisper_asr_adapter_exposes_language_probability(monkeypatch):
    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            class Segment:
                def __init__(self, text):
                    self.text = text

            return iter([Segment(" hello")]), types.SimpleNamespace(
                language="it", language_probability=0.73
            )

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    asr = _load_app_module("asr_probability_under_test", "asr.py")
    adapter = asr.FasterWhisperASR(
        model_name="small", device="cpu", compute_type="int8"
    )

    transcript = adapter.transcribe(np.array([0, 1], dtype=np.int16).tobytes())

    assert transcript.language == "it"
    assert transcript.language_probability == pytest.approx(0.73)


@pytest.mark.asyncio
async def test_conversation_service_warmup_primes_backends(monkeypatch):
    warm_calls = []

    class FakeASR:
        def transcribe(self, pcm_bytes: bytes, *, language_hint=None, session_id=None):
            warm_calls.append(("asr", len(pcm_bytes), language_hint, session_id))
            return FakeTranscriptionResult("", language="it", language_probability=1.0)

    class FakeAssistantBackend:
        _model = "gemma4"

        async def generate_response(self, transcript: str, **kwargs):
            warm_calls.append(("assistant", transcript, kwargs))
            return "ok"

    class FakeAudio:
        shape = (1, 2400)

    async def fake_generate_sentence_audio(gen_kwargs):
        warm_calls.append(("tts", gen_kwargs["text"]))
        return FakeAudio()

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_backend=FakeAssistantBackend()
    )
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )

    await service.warmup(sample="agent-voice")

    assert any(call[0] == "asr" for call in warm_calls)
    assert any(call[0] == "assistant" for call in warm_calls)
    assert any(call[0] == "tts" for call in warm_calls)


def test_repeated_session_start_resets_session_state_on_same_socket():
    client, service = _make_client()
    service.transcript_results = [
        FakeTranscriptionResult("first caller turn", language="es"),
        FakeTranscriptionResult("second caller turn", language="fr"),
    ]

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_end"})
        assert ws.receive_json() == {
            "type": "transcript_final",
            "text": "first caller turn",
        }
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering first caller turn",
            "response_id": 1,
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "interrupted", "response_id": 1}
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(b"stale")

        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "language": "fr"}
        )
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
            "language": "fr",
        }

        ws.send_json({"type": "speech_end"})
        assert ws.receive_json() == {
            "type": "error",
            "message": "Received speech_end without speech_start.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(AUDIO_CHUNK_B)
        ws.send_json({"type": "speech_end"})
        assert ws.receive_json() == {
            "type": "transcript_final",
            "text": "second caller turn",
        }
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering second caller turn",
            "response_id": 2,
        }
        service.response_released[2].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 2}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 2}

    assert service.transcribe_calls == [(AUDIO_CHUNK_A, 16000), (AUDIO_CHUNK_B, 16000)]
    assert service.transcribe_contexts[0]["language_hint"] is None
    assert service.transcribe_contexts[1]["language_hint"] == "fr"
    assert service.response_contexts[0]["history"] == []
    assert service.response_contexts[1]["history"] == []
    assert service.response_contexts[0]["language_hint"] == "es"
    assert service.response_contexts[1]["language_hint"] == "fr"
    assert (
        service.response_contexts[0]["session_id"]
        != service.response_contexts[1]["session_id"]
    )


@pytest.mark.asyncio
async def test_server_conversation_service_fallback_prompt_keeps_session_context_on_first_turn(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError(f"transcribe should not be called: {pcm_bytes!r}")

    class LegacyBackend:
        def __init__(self):
            self.calls = []

        async def generate_response(self, transcript: str) -> str:
            self.calls.append(transcript)
            return "Assistant sentence."

    class FakeAudio:
        shape = (1, 2400)

    async def fake_generate_sentence_audio(gen_kwargs):
        del gen_kwargs
        return FakeAudio()

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    backend = LegacyBackend()
    service = server.ConversationService(FakeASR(), assistant_backend=backend)
    monkeypatch.setattr(
        service, "_generate_sentence_audio", fake_generate_sentence_audio
    )
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")

    events = [
        event
        async for event in service.respond(
            "Current caller message",
            6,
            session_id="session-123",
            history=[],
            language_hint=None,
        )
    ]

    assert backend.calls == [
        "Continue the phone conversation naturally.\n"
        "Session ID: session-123\n"
        "Caller: Current caller message"
    ]
    assert events == [
        {"type": "assistant_text", "text": "Assistant sentence.", "response_id": 6},
        {
            "type": "audio_chunk",
            "response_id": 6,
            "chunk_index": 0,
            "sentence": "Assistant sentence.",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm",
        },
        {"type": "response_done", "response_id": 6},
    ]


def test_barge_in_speech_start_emits_interrupted_for_active_response():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        ws.receive_json()
        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        ws.send_json({"type": "speech_start"})

        assert ws.receive_json() == {"type": "interrupted", "response_id": 1}
        assert ws.receive_json() == {"type": "listening"}

        service.response_released[1].set()


def test_short_utterance_skips_asr_and_returns_done_only():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(b"x" * 1024)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == []
    assert service.response_started == []


def test_suspicious_transcript_is_dropped_before_assistant_response():
    client, service = _make_client()
    service.transcript_results = [
        FakeTranscriptionResult(
            "Sottotitoli e revisione a cura di QTSS",
            language="it",
            language_probability=0.94,
        )
    ]

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(b"x" * 12000)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(b"x" * 12000, 16000)]
    assert service.response_started == []


def test_duplicate_speech_start_returns_error_and_keeps_buffered_audio():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_start"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Received speech_start while already collecting audio.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_bytes(AUDIO_CHUNK_B)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(AUDIO_CHUNK_A + AUDIO_CHUNK_B, 16000)]


def test_input_audio_chunk_message_appends_audio_payload():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_json(
            {
                "type": "input_audio_chunk",
                "audio": base64.b64encode(AUDIO_CHUNK_A).decode("ascii"),
            }
        )
        ws.send_bytes(AUDIO_CHUNK_B)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(AUDIO_CHUNK_A + AUDIO_CHUNK_B, 16000)]


def test_unexpected_audio_before_speech_start_returns_error():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_bytes(AUDIO_CHUNK_A)

        assert ws.receive_json() == {
            "type": "error",
            "message": "Audio received outside active speech input.",
            "code": "UNEXPECTED_AUDIO",
        }


def test_speech_end_without_active_speech_returns_error():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Received speech_end without speech_start.",
            "code": "INVALID_MESSAGE",
        }


def test_transcribe_failure_returns_controlled_error_and_session_continues():
    class FailingTranscribeService(FakeConversationService):
        async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
            self.transcribe_calls.append((audio_bytes, sample_rate))
            raise RuntimeError("transcribe failed")

    service = FailingTranscribeService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "transcribe failed",
            "code": "TRANSCRIBE_ERROR",
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}


@pytest.mark.asyncio
async def test_response_error_fallback_ignores_disconnect_time_send_failures():
    class DisconnectingWebSocket:
        application_state = WebSocketState.CONNECTED

        async def send_json(self, payload):
            raise RuntimeError("socket closed")

        async def send_bytes(self, payload):
            raise AssertionError("send_bytes should not be called")

    class FailingConversationService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            raise RuntimeError("response failed")
            yield

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()

    await _forward_service_response(
        DisconnectingWebSocket(),
        state,
        FailingConversationService(),
        "hello there",
        1,
    )

    assert state.active_response_id is None
    assert state.response_task is None


@pytest.mark.asyncio
async def test_forward_service_response_stops_forwarding_stale_events():
    class CollectingWebSocket:
        def __init__(self, state):
            self.application_state = WebSocketState.CONNECTED
            self.json_payloads = []
            self.binary_payloads = []
            self._state = state

        async def send_json(self, payload):
            self.json_payloads.append(payload)
            if payload["type"] == "assistant_text":
                self._state.active_response_id = 2

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class LateEventService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "assistant_text", "text": transcript}
            yield {"type": "audio_chunk", "payload": b"late-audio"}
            yield {"type": "response_done"}

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket(state)

    await _forward_service_response(websocket, state, LateEventService(), "hello", 1)

    assert websocket.json_payloads == [
        {"type": "assistant_text", "text": "hello", "response_id": 1}
    ]
    assert websocket.binary_payloads == []
    assert state.active_response_id == 2
    assert state.response_task == asyncio.current_task()


@pytest.mark.asyncio
async def test_forward_service_response_normalizes_response_ids_from_service_events():
    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            if payload["type"] == "interrupted":
                assert generation_finished.is_set() is True
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class MismatchedResponseIdService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "assistant_text", "text": transcript, "response_id": 999}
            yield {
                "type": "audio_chunk",
                "response_id": -1,
                "payload": b"audio",
            }
            yield {"type": "response_done", "response_id": 0}

    state = ConversationSessionState(active_response_id=7)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket()

    await _forward_service_response(
        websocket,
        state,
        MismatchedResponseIdService(),
        "hello",
        7,
    )

    assert websocket.json_payloads == [
        {"type": "assistant_text", "text": "hello", "response_id": 7},
        {"type": "audio_chunk", "response_id": 7},
        {"type": "response_done", "response_id": 7},
    ]
    assert websocket.binary_payloads == [b"audio"]
    assert state.active_response_id is None
    assert state.response_task is None


@pytest.mark.asyncio
async def test_forward_service_response_suppresses_error_for_stale_response_failure():
    class CollectingWebSocket:
        def __init__(self, state):
            self.application_state = WebSocketState.CONNECTED
            self.json_payloads = []
            self.binary_payloads = []
            self._state = state
            self._send_count = 0

        async def send_json(self, payload):
            self.json_payloads.append(payload)
            self._send_count += 1
            if payload["type"] == "assistant_text":
                self._state.active_response_id = 2
            if self._send_count == 1:
                raise RuntimeError("socket send failed after cancellation")

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class StaleFailingService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "assistant_text", "text": transcript}

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket(state)

    await _forward_service_response(websocket, state, StaleFailingService(), "hello", 1)

    assert websocket.json_payloads == [
        {"type": "assistant_text", "text": "hello", "response_id": 1}
    ]
    assert websocket.binary_payloads == []
    assert state.active_response_id == 2
    assert state.response_task == asyncio.current_task()


@pytest.mark.asyncio
async def test_forward_service_response_preserves_audio_pairing_during_stale_race():
    class CollectingWebSocket:
        def __init__(self, state):
            self.application_state = WebSocketState.CONNECTED
            self.json_payloads = []
            self.binary_payloads = []
            self._state = state

        async def send_json(self, payload):
            self.json_payloads.append(payload)
            if payload["type"] == "audio_chunk":
                self._state.active_response_id = 2

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class LateAudioService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "audio_chunk", "payload": b"audio"}
            yield {"type": "response_done"}

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket(state)

    await _forward_service_response(websocket, state, LateAudioService(), "hello", 1)

    assert websocket.json_payloads == [{"type": "audio_chunk", "response_id": 1}]
    assert websocket.binary_payloads == [b"audio"]
    assert state.active_response_id == 2
    assert state.response_task == asyncio.current_task()


@pytest.mark.asyncio
async def test_interrupt_waits_for_response_task_before_emitting_interrupted():
    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self, lifecycle):
            self.events = []
            self._lifecycle = lifecycle

        async def send_json(self, payload):
            self.events.append((payload, self._lifecycle.copy()))

    lifecycle = []
    websocket = CollectingWebSocket(lifecycle)
    state = ConversationSessionState(active_response_id=3)
    started = asyncio.Event()

    async def response_task_body():
        try:
            started.set()
            await asyncio.Future()
        finally:
            lifecycle.append("task-finally")

    task = asyncio.create_task(response_task_body())
    state.response_task = task
    state.background_response_tasks.add(task)
    await started.wait()

    await _interrupt_active_response(websocket, state)

    assert task.done() is True
    assert websocket.events == [
        ({"type": "interrupted", "response_id": 3}, ["task-finally"])
    ]


@pytest.mark.asyncio
async def test_interrupt_waits_for_blocking_generation_before_emitting_interrupted(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAudio:
        shape = (1, 2400)

    class FakeAssistantBackendImpl:
        async def generate_response(self, transcript: str) -> str:
            del transcript
            return "This sentence has enough words."

    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    generation_started = asyncio.Event()
    release_generation = threading.Event()
    generation_finished = threading.Event()
    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = object()

    async def fake_generate_one(get_model, model_lock, gen_kwargs):
        del get_model, model_lock, gen_kwargs
        generation_started.set()
        await asyncio.to_thread(release_generation.wait)
        generation_finished.set()
        return FakeAudio()

    monkeypatch.setattr(server, "_generate_one", fake_generate_one)
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")

    service = server.ConversationService(
        FakeASR(), assistant_backend=FakeAssistantBackendImpl()
    )
    websocket = CollectingWebSocket()
    state = ConversationSessionState(active_response_id=1)

    response_task = asyncio.create_task(
        _forward_service_response(
            websocket,
            state,
            service,
            "This sentence has enough words.",
            1,
        )
    )
    state.response_task = response_task
    state.background_response_tasks.add(response_task)

    await generation_started.wait()

    interrupt_task = asyncio.create_task(_interrupt_active_response(websocket, state))
    await asyncio.sleep(0)

    assert interrupt_task.done() is False
    assert websocket.json_payloads == [
        {
            "type": "assistant_text",
            "text": "This sentence has enough words.",
            "response_id": 1,
        }
    ]

    release_generation.set()

    await interrupt_task
    with pytest.raises(asyncio.CancelledError):
        await response_task

    assert websocket.json_payloads == [
        {
            "type": "assistant_text",
            "text": "This sentence has enough words.",
            "response_id": 1,
        },
        {"type": "interrupted", "response_id": 1},
    ]
    assert websocket.binary_payloads == []


@pytest.mark.asyncio
async def test_interrupt_cancels_inflight_async_ollama_request_before_emitting_interrupted(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAssistantBackendImpl:
        def __init__(self):
            self.calls = []
            self.cancelled = asyncio.Event()
            self.started = asyncio.Event()

        async def generate_response(
            self,
            transcript: str,
            *,
            session_id: str | None = None,
            history: list[dict[str, str]] | None = None,
            language_hint: str | None = None,
        ) -> str:
            self.calls.append(
                {
                    "transcript": transcript,
                    "session_id": session_id,
                    "history": history,
                    "language_hint": language_hint,
                }
            )
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    assistant_backend = FakeAssistantBackendImpl()
    service = server.ConversationService(FakeASR(), assistant_backend=assistant_backend)
    websocket = CollectingWebSocket()
    state = ConversationSessionState(active_response_id=1)

    response_task = asyncio.create_task(
        _forward_service_response(
            websocket,
            state,
            service,
            "This sentence has enough words.",
            1,
        )
    )
    state.response_task = response_task
    state.background_response_tasks.add(response_task)

    await assistant_backend.started.wait()

    await _interrupt_active_response(websocket, state)
    with pytest.raises(asyncio.CancelledError):
        await response_task

    assert len(assistant_backend.calls) == 1
    assert assistant_backend.calls[0] == {
        "transcript": "This sentence has enough words.",
        "session_id": state.session_id,
        "history": [],
        "language_hint": None,
    }
    assert assistant_backend.cancelled.is_set() is True
    assert websocket.json_payloads == [{"type": "interrupted", "response_id": 1}]
    assert websocket.binary_payloads == []


def test_response_task_failure_returns_controlled_error_event():
    class FailingConversationService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            self.response_started.append((transcript, response_id, sample))
            yield {"type": "assistant_text", "text": f"answering {transcript}"}
            raise RuntimeError("response failed")

    service = FailingConversationService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        ws.receive_json()
        ws.send_bytes(AUDIO_CHUNK_A)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }
        assert ws.receive_json() == {
            "type": "error",
            "message": "response failed",
            "code": "RESPONSE_ERROR",
            "response_id": 1,
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}


@pytest.mark.asyncio
async def test_forward_service_response_reports_empty_backend_reply_as_controlled_error(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_backend=FakeAssistantBackend("   ")
    )
    websocket = CollectingWebSocket()
    state = ConversationSessionState(active_response_id=1)

    await _forward_service_response(websocket, state, service, "hello", 1)

    assert websocket.json_payloads == [
        {
            "type": "error",
            "message": "Assistant backend returned an empty response.",
            "code": "RESPONSE_ERROR",
            "response_id": 1,
        }
    ]
    assert websocket.binary_payloads == []
