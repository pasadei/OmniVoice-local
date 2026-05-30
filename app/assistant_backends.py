from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


PHONE_AGENT_SYSTEM_PROMPT = (
    "You are a phone agent speaking with a caller. "
    "Be calm, direct, and reassuring. "
    "Keep replies concise, practical, and easy to say out loud. "
    "Do not repeat the caller's words back unless needed for clarity. "
    "Ask at most one short clarifying question when necessary. "
    "Always reply in the caller's language. "
    "If a caller language hint is provided, treat it as authoritative for the reply language. "
    "Output plain text only. "
    "Do not use markdown, bullet points, emojis, JSON, XML, code blocks, "
    "speaker labels, or stage directions. "
    "Only these non-verbal tags are allowed when they genuinely help delivery: "
    "[laughter], [sigh], [confirmation-en], [question-en], [question-ah], "
    "[question-oh], [question-ei], [question-yi], [surprise-ah], [surprise-oh], "
    "[surprise-wa], [surprise-yo], [dissatisfaction-hnn]. "
    "Do not overuse them."
)


@runtime_checkable
class AssistantBackend(Protocol):
    async def generate_response(
        self,
        transcript: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ) -> str: ...


class OllamaAssistantBackend:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        system_prompt: str = PHONE_AGENT_SYSTEM_PROMPT,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt

    async def generate_response(
        self,
        transcript: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=_build_messages(
                self._system_prompt,
                str(transcript).strip(),
                session_id=session_id,
                history=history or [],
                language_hint=language_hint,
            ),
            stream=False,
        )
        return _extract_message_text(response)


class OpenAIAssistantBackend:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        system_prompt: str = PHONE_AGENT_SYSTEM_PROMPT,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt

    async def generate_response(
        self,
        transcript: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=_build_messages(
                self._system_prompt,
                str(transcript).strip(),
                session_id=session_id,
                history=history or [],
                language_hint=language_hint,
            ),
            stream=False,
        )
        return _openai_extract_message_text(response)


def _openai_extract_message_text(response: Any) -> str:
    choice = response.choices[0] if response.choices else None
    if choice is None:
        raise RuntimeError("Assistant backend returned no choices.")
    content = getattr(choice.message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Assistant backend returned no usable message content.")
    return content.strip()


def _build_messages(
    system_prompt: str,
    transcript: str,
    *,
    session_id: str | None,
    history: list[dict[str, str]],
    language_hint: str | None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]

    context_parts = []
    if session_id:
        context_parts.append(f"session_id={session_id}")
    if language_hint:
        context_parts.append(f"caller_language_hint={language_hint}")
    if context_parts:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation context: {'; '.join(context_parts)}.",
            }
        )
    if language_hint:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Reply only in the caller's language for this turn: "
                    f"{_language_display_name(language_hint)} ({language_hint})."
                ),
            }
        )

    for turn in history[-3:]:
        user_text = str(turn.get("user") or "").strip()
        assistant_text = str(turn.get("assistant") or "").strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": transcript})
    return messages


def _extract_message_text(response: Any) -> str:
    content = None

    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
    else:
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and hasattr(message, "get"):
                content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Assistant backend returned no usable message content.")

    return content.strip()


def _language_display_name(language_code: str) -> str:
    names = {
        "de": "German",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "pt": "Portuguese",
    }
    normalized = language_code.strip().lower()
    return names.get(normalized, normalized)
