from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from time import monotonic

from openai import APITimeoutError, BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from subtitle_translator.models import SubtitleSegment, SubtitleTrack, TargetLanguage

DEFAULT_MODEL = "gpt-5.4"
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_BATCH_SEGMENTS = 800
MAX_ESTIMATED_OUTPUT_TOKENS = 50_000
ESTIMATED_CHARS_PER_TOKEN = 3.0
TRANSLATION_EXPANSION_FACTOR = 1.35
JSON_ENVELOPE_OVERHEAD_CHARS = 32
JSON_ITEM_OVERHEAD_CHARS = 32
Reporter = Callable[[str], None] | None

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


class TranslationContractError(TranslationError):
    pass


class TranslationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    text: str = Field(min_length=0)


class TranslationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translations: list[TranslationItem]


class SubtitleTranslator:
    def __init__(self, model: str = DEFAULT_MODEL, reporter: Reporter = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise TranslationError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self.model = model
        self.reporter = reporter

    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        target_language: TargetLanguage,
        source_track: SubtitleTrack,
    ) -> dict[int, str]:
        if not segments:
            raise TranslationError("The selected subtitle track contains no translatable subtitle events.")

        self._report(
            f"Sending {len(segments)} subtitle lines to OpenAI for translation to {target_language.display_name}."
        )
        planned_batches = _plan_batches(segments)
        if len(planned_batches) > 1:
            self._report(f"Prepared {len(planned_batches)} translation batches.")

        translations: list[TranslationItem] = []
        for index, batch_segments in enumerate(planned_batches, start=1):
            batch_label = _format_batch_label(index, len(planned_batches))
            translations.extend(
                self._translate_with_fallback(batch_segments, target_language, source_track, batch_label)
            )
        self._report("Received translated subtitles from OpenAI.")
        return {item.id: item.text for item in translations}

    def _translate_with_fallback(
        self,
        segments: list[SubtitleSegment],
        target_language: TargetLanguage,
        source_track: SubtitleTrack,
        batch_label: str,
    ) -> list[TranslationItem]:
        try:
            return self._translate_batch(segments, target_language, source_track, batch_label)
        except BadRequestError as exc:
            if len(segments) == 1 or not _looks_like_size_limit(exc):
                raise TranslationError(f"OpenAI request failed: {exc}") from exc
            self._report(
                f"{batch_label}: OpenAI rejected {len(segments)} subtitle lines as too large. Splitting and retrying."
            )
        except APITimeoutError as exc:
            if len(segments) == 1:
                raise TranslationError(
                    f"OpenAI translation request timed out after {int(REQUEST_TIMEOUT_SECONDS)} seconds."
                ) from exc
            self._report(
                f"{batch_label}: OpenAI translation timed out after {int(REQUEST_TIMEOUT_SECONDS)} seconds for {len(segments)} lines. Splitting and retrying."
            )
        except TranslationContractError as exc:
            if len(segments) == 1:
                raise
            self._report(
                f"{batch_label}: OpenAI returned an invalid subtitle mapping for {len(segments)} lines ({exc}). Splitting and retrying."
            )
        except Exception as exc:
            raise TranslationError(f"OpenAI request failed: {exc}") from exc

        left, right = _split_contiguous_segments(segments)
        translated_left = self._translate_with_fallback(
            left,
            target_language,
            source_track,
            _format_retry_batch_label(batch_label, 1, 2),
        )
        translated_right = self._translate_with_fallback(
            right,
            target_language,
            source_track,
            _format_retry_batch_label(batch_label, 2, 2),
        )
        return translated_left + translated_right

    def _translate_batch(
        self,
        segments: list[SubtitleSegment],
        target_language: TargetLanguage,
        source_track: SubtitleTrack,
        batch_label: str,
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

        self._report(f"{batch_label}: Waiting for OpenAI translation response.")
        with _heartbeat(self.reporter, f"{batch_label}: Still waiting for OpenAI translation response"):
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
                max_output_tokens=_request_max_output_tokens(segments),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "subtitle_translation",
                        "strict": True,
                        "schema": TranslationEnvelope.model_json_schema(),
                    }
                }
            )

        raw_output = response.output_text
        if not raw_output:
            raise TranslationContractError("OpenAI returned an empty response.")

        try:
            envelope = TranslationEnvelope.model_validate(json.loads(raw_output))
        except Exception as exc:
            raise TranslationContractError(f"OpenAI returned invalid translation JSON: {exc}") from exc

        self._validate_batch_response(segments, envelope)
        return envelope.translations

    @staticmethod
    def _validate_batch_response(segments: list[SubtitleSegment], envelope: TranslationEnvelope) -> None:
        expected_ids = [segment.id for segment in segments]
        received_ids = [item.id for item in envelope.translations]
        if received_ids != expected_ids:
            mismatch_index = next(
                (index for index, (expected, received) in enumerate(zip(expected_ids, received_ids), start=1) if expected != received),
                None,
            )
            if mismatch_index is None:
                mismatch_index = min(len(expected_ids), len(received_ids)) + 1
            raise TranslationContractError(
                "Translated subtitle IDs do not match the source subtitle order. "
                f"Expected {len(expected_ids)} IDs, received {len(received_ids)} IDs, first mismatch at position {mismatch_index}."
            )

    def _report(self, message: str) -> None:
        if self.reporter is not None:
            self.reporter(message)


def _split_contiguous_segments(segments: list[SubtitleSegment]) -> tuple[list[SubtitleSegment], list[SubtitleSegment]]:
    total_chars = sum(len(segment.text) for segment in segments)
    running_chars = 0
    for index, segment in enumerate(segments[:-1], start=1):
        running_chars += len(segment.text)
        if running_chars >= total_chars / 2:
            return segments[:index], segments[index:]
    midpoint = len(segments) // 2
    return segments[:midpoint], segments[midpoint:]


def _batch_split_reason(segments: list[SubtitleSegment]) -> str | None:
    if len(segments) > MAX_BATCH_SEGMENTS:
        return (
            f"Subtitle batch has {len(segments)} lines. Splitting into smaller contiguous batches "
            f"before calling OpenAI."
        )

    estimated_output_tokens = _estimate_translated_output_tokens(segments)
    if estimated_output_tokens > MAX_ESTIMATED_OUTPUT_TOKENS:
        return (
            f"Estimated translated output is about {estimated_output_tokens} tokens for {len(segments)} lines. "
            f"Splitting into smaller contiguous batches before calling OpenAI."
        )

    return None


def _plan_batches(segments: list[SubtitleSegment]) -> list[list[SubtitleSegment]]:
    split_reason = _batch_split_reason(segments)
    if split_reason is None:
        return [segments]

    left, right = _split_contiguous_segments(segments)
    return _plan_batches(left) + _plan_batches(right)


def _estimate_translated_output_tokens(segments: list[SubtitleSegment]) -> int:
    estimated_chars = JSON_ENVELOPE_OVERHEAD_CHARS
    for segment in segments:
        estimated_translated_chars = math.ceil(len(segment.text) * TRANSLATION_EXPANSION_FACTOR)
        estimated_chars += JSON_ITEM_OVERHEAD_CHARS + len(str(segment.id)) + estimated_translated_chars
    return math.ceil(estimated_chars / ESTIMATED_CHARS_PER_TOKEN)


def _request_max_output_tokens(segments: list[SubtitleSegment]) -> int:
    estimated_output_tokens = _estimate_translated_output_tokens(segments)
    return max(8_000, min(64_000, estimated_output_tokens + 8_000))


def _format_batch_label(index: int, total: int) -> str:
    return f"Batch {index}/{total}"


def _format_retry_batch_label(parent_label: str, index: int, total: int) -> str:
    return f"{parent_label}.{index}/{total}"


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


@contextmanager
def _heartbeat(reporter: Reporter, message: str, interval_seconds: float = 10.0):
    if reporter is None:
        yield
        return

    stop_event = threading.Event()
    started = monotonic()

    def emit() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = int(monotonic() - started)
            reporter(f"{message} ({elapsed}s elapsed).")

    thread = threading.Thread(target=emit, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.2)
