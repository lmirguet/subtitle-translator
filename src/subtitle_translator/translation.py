from __future__ import annotations

import json
import os

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from subtitle_translator.models import SubtitleSegment, SubtitleTrack, TargetLanguage

DEFAULT_MODEL = "gpt-5.4"

TRANSLATION_SYSTEM_PROMPT = """
You translate subtitle files for video content.

Rules:
- Translate into the requested target language only.
- Preserve the exact subtitle item order and the exact subtitle IDs.
- Translate each subtitle's text as a whole, not line by line in isolation.
- Preserve subtitle formatting markers, line breaks, and ASS/SSA override tags whenever they appear.
- Do not add notes, explanations, or metadata.
- Return JSON only and follow the supplied schema exactly.
""".strip()


class TranslationError(RuntimeError):
    pass


class TranslationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    text: str = Field(min_length=0)


class TranslationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translations: list[TranslationItem]


class SubtitleTranslator:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise TranslationError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        target_language: TargetLanguage,
        source_track: SubtitleTrack,
    ) -> dict[int, str]:
        if not segments:
            raise TranslationError("The selected subtitle track contains no translatable subtitle events.")

        translations = self._translate_with_fallback(segments, target_language, source_track)
        return {item.id: item.text for item in translations}

    def _translate_with_fallback(
        self,
        segments: list[SubtitleSegment],
        target_language: TargetLanguage,
        source_track: SubtitleTrack,
    ) -> list[TranslationItem]:
        try:
            return self._translate_batch(segments, target_language, source_track)
        except BadRequestError as exc:
            if len(segments) == 1 or not _looks_like_size_limit(exc):
                raise TranslationError(f"OpenAI request failed: {exc}") from exc
        except Exception as exc:
            raise TranslationError(f"OpenAI request failed: {exc}") from exc

        left, right = _split_contiguous_segments(segments)
        translated_left = self._translate_with_fallback(left, target_language, source_track)
        translated_right = self._translate_with_fallback(right, target_language, source_track)
        return translated_left + translated_right

    def _translate_batch(
        self,
        segments: list[SubtitleSegment],
        target_language: TargetLanguage,
        source_track: SubtitleTrack,
    ) -> list[TranslationItem]:
        payload = {
            "target_language": target_language.display_name,
            "source_track": {
                "id": source_track.id,
                "language": source_track.language,
                "name": source_track.name,
                "codec": source_track.codec,
                "codec_id": source_track.codec_id,
            },
            "segments": [{"id": segment.id, "text": segment.text} for segment in segments],
        }

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": TRANSLATION_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
                },
            ],
            max_output_tokens=100_000,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "subtitle_translation",
                    "strict": True,
                    "schema": TranslationEnvelope.model_json_schema(),
                }
            },
        )

        raw_output = response.output_text
        if not raw_output:
            raise TranslationError("OpenAI returned an empty response.")

        try:
            envelope = TranslationEnvelope.model_validate(json.loads(raw_output))
        except Exception as exc:
            raise TranslationError(f"OpenAI returned invalid translation JSON: {exc}") from exc

        self._validate_batch_response(segments, envelope)
        return envelope.translations

    @staticmethod
    def _validate_batch_response(segments: list[SubtitleSegment], envelope: TranslationEnvelope) -> None:
        expected_ids = [segment.id for segment in segments]
        received_ids = [item.id for item in envelope.translations]
        if received_ids != expected_ids:
            raise TranslationError(
                "Translated subtitle IDs do not match the source subtitle order. "
                f"Expected {expected_ids[:5]}..., received {received_ids[:5]}..."
            )


def _split_contiguous_segments(segments: list[SubtitleSegment]) -> tuple[list[SubtitleSegment], list[SubtitleSegment]]:
    total_chars = sum(len(segment.text) for segment in segments)
    running_chars = 0
    for index, segment in enumerate(segments[:-1], start=1):
        running_chars += len(segment.text)
        if running_chars >= total_chars / 2:
            return segments[:index], segments[index:]
    midpoint = len(segments) // 2
    return segments[:midpoint], segments[midpoint:]


def _looks_like_size_limit(exc: BadRequestError) -> bool:
    detail = _collect_error_text(exc).lower()
    keywords = (
        "context",
        "too large",
        "too long",
        "maximum context length",
        "max_output_tokens",
        "input length",
    )
    return any(keyword in detail for keyword in keywords)


def _collect_error_text(exc: BadRequestError) -> str:
    parts: list[str] = [str(exc)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for value in error.values():
                if isinstance(value, str):
                    parts.append(value)
    return " ".join(parts)
