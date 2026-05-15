from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class TargetLanguage(StrEnum):
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"
    RUSSIAN = "ru"
    POLISH = "pl"

    @property
    def display_name(self) -> str:
        return {
            TargetLanguage.ENGLISH: "English",
            TargetLanguage.FRENCH: "French",
            TargetLanguage.SPANISH: "Spanish",
            TargetLanguage.RUSSIAN: "Russian",
            TargetLanguage.POLISH: "Polish",
        }[self]

    @property
    def mkv_language_code(self) -> str:
        return {
            TargetLanguage.ENGLISH: "eng",
            TargetLanguage.FRENCH: "fra",
            TargetLanguage.SPANISH: "spa",
            TargetLanguage.RUSSIAN: "rus",
            TargetLanguage.POLISH: "pol",
        }[self]


class SubtitleFormat(StrEnum):
    SRT = "srt"
    ASS = "ass"


SUPPORTED_CODEC_IDS: dict[str, SubtitleFormat] = {
    "S_TEXT/UTF8": SubtitleFormat.SRT,
    "S_TEXT/ASS": SubtitleFormat.ASS,
    "S_TEXT/SSA": SubtitleFormat.ASS,
}


@dataclass(frozen=True)
class SubtitleTrack:
    id: int
    codec: str | None
    codec_id: str | None
    language: str | None
    name: str | None
    is_default: bool
    is_forced: bool
    format: SubtitleFormat | None

    @property
    def is_supported(self) -> bool:
        return self.format is not None

    @property
    def extracted_suffix(self) -> str:
        if self.format is None:
            raise ValueError("Unsupported subtitle track has no extractable format.")
        return self.format.value

    def display_label(self) -> str:
        parts = [f"#{self.id}"]
        if self.language:
            parts.append(self.language)
        if self.name:
            parts.append(self.name)
        if self.codec:
            parts.append(self.codec)
        if self.is_default:
            parts.append("default")
        if self.is_forced:
            parts.append("forced")
        if not self.is_supported:
            parts.append("unsupported")
        return " | ".join(parts)


@dataclass(frozen=True)
class SubtitleSegment:
    id: int
    event_index: int
    text: str


@dataclass(frozen=True)
class TranslationOptions:
    input_path: Path
    source_track_id: int
    target_language: TargetLanguage
    output_path: Path | None
    in_place: bool
    model: str
