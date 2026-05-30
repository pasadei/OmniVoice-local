from pathlib import Path
import sys
import struct
import types
import wave

import numpy as np

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local test fallback
    torch = None
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from audio_encoder import encode_audio_chunk, tensor_to_pcm16


class FakeTensor:
    def __init__(self, values):
        self._values = np.array(values, dtype=np.float32)

    def __getitem__(self, idx):
        return FakeTensor(self._values[idx])

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._values


def _tensor(values):
    if hasattr(torch, "tensor") and hasattr(torch, "float32"):
        return torch.tensor(values, dtype=torch.float32)
    return FakeTensor(values)


def test_tensor_to_pcm16_length_and_type():
    audio = _tensor([[0.0, 0.5, -0.5, 1.0, -1.0]])
    pcm = tensor_to_pcm16(audio)
    assert isinstance(pcm, bytes)
    assert len(pcm) == 10


def test_encode_pcm_matches_expected_samples():
    audio = _tensor([[0.0, 1.0, -1.0]])
    pcm = encode_audio_chunk(audio, output_format="pcm")
    values = struct.unpack("<hhh", pcm)
    assert list(values) == [0, 32767, -32767]


def test_encode_wav_is_valid_chunk_file(tmp_path):
    audio = _tensor(np.zeros((1, 2400), dtype=np.float32))
    wav_bytes = encode_audio_chunk(audio, output_format="wav", sample_rate=24000)
    wav_path = tmp_path / "chunk.wav"
    wav_path.write_bytes(wav_bytes)

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 2400
