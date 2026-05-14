from __future__ import annotations

from pathlib import Path

import pysubs2

from subtitle_translator.models import SubtitleFormat, SubtitleSegment


class SubtitleProcessingError(RuntimeError):
    pass


def load_subtitles(path: Path, subtitle_format: SubtitleFormat) -> pysubs2.SSAFile:
    try:
        return pysubs2.load(str(path), format_=subtitle_format.value)
    except Exception as exc:  # pragma: no cover - pysubs2 raises multiple parser exceptions
        raise SubtitleProcessingError(f"Failed to parse subtitle file {path.name}: {exc}") from exc


def collect_translatable_segments(subtitles: pysubs2.SSAFile) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    for event_index, event in enumerate(subtitles.events):
        if getattr(event, "type", "Dialogue") == "Comment":
            continue
        text = event.text or ""
        if not text.strip():
            continue
        segments.append(
            SubtitleSegment(
                id=len(segments) + 1,
                event_index=event_index,
                text=text,
            )
        )
    return segments


def apply_translations(
    subtitles: pysubs2.SSAFile,
    segments: list[SubtitleSegment],
    translated_texts: dict[int, str],
) -> None:
    for segment in segments:
        translated_text = translated_texts.get(segment.id)
        if translated_text is None:
            raise SubtitleProcessingError(f"Missing translated text for segment {segment.id}.")
        subtitles.events[segment.event_index].text = translated_text


def save_subtitles(subtitles: pysubs2.SSAFile, path: Path, subtitle_format: SubtitleFormat) -> None:
    try:
        subtitles.save(str(path), format_=subtitle_format.value)
    except Exception as exc:  # pragma: no cover - pysubs2 raises multiple writer exceptions
        raise SubtitleProcessingError(f"Failed to write subtitle file {path.name}: {exc}") from exc
