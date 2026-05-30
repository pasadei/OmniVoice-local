from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from sentence_splitter import SentenceSplitter


def test_mixed_punctuation_split():
    splitter = SentenceSplitter(max_chars=150, min_chars=10)
    text = "Ciao! Come va? Tutto bene: grazie; perfetto."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert "Ciao!" in chunks[0]


def test_abbreviation_not_split():
    splitter = SentenceSplitter(max_chars=150, min_chars=5)
    text = "Dr. Rossi arriva ora. Poi parla."
    chunks = splitter.split_text(text)
    assert chunks[0].startswith("Dr. Rossi")


def test_non_verbal_tag_preserved():
    splitter = SentenceSplitter(max_chars=50, min_chars=5)
    text = "[laughter] This is a long sentence that should be split, but keep tags intact."
    chunks = splitter.split_text(text)
    assert chunks[0].startswith("[laughter]")
    assert all("[laughter" not in c for c in chunks[1:])


def test_cjk_punctuation():
    splitter = SentenceSplitter(max_chars=150, min_chars=1)
    text = "你好！今天怎么样？我很好。"
    chunks = splitter.split_text(text)
    assert chunks == ["你好！", "今天怎么样？", "我很好。"]


def test_buffer_accumulation_min_chars():
    splitter = SentenceSplitter(max_chars=150, min_chars=20)
    complete, rem = splitter.extract_complete_sentences("Hi. Ok. This is the long part.")
    assert rem == ""
    merged = splitter._post_process(complete)
    assert len(merged) == 1


def test_empty_and_single_char():
    splitter = SentenceSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("a") == ["a"]


def test_without_punctuation():
    splitter = SentenceSplitter()
    complete, rem = splitter.extract_complete_sentences("no punctuation here")
    assert complete == []
    assert rem == "no punctuation here"
