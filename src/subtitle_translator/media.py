from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from subtitle_translator.models import SUPPORTED_CODEC_IDS, SubtitleFormat, SubtitleTrack, TargetLanguage

Reporter = Callable[[str], None] | None


class DependencyError(RuntimeError):
    pass


class ExternalToolError(RuntimeError):
    pass


def ensure_mkvtoolnix_available() -> None:
    missing = [tool for tool in ("mkvmerge", "mkvextract") if shutil.which(tool) is None]
    if missing:
        missing_list = ", ".join(missing)
        raise DependencyError(
            f"Missing required MKVToolNix tool(s): {missing_list}. Install MKVToolNix and ensure the binaries are in PATH."
        )


def inspect_subtitle_tracks(input_path: Path) -> list[SubtitleTrack]:
    result = _run_tool(
        [
            "mkvmerge",
            "--identify",
            "--identification-format",
            "json",
            str(input_path),
        ]
    )
    payload = json.loads(result.stdout)
    return parse_subtitle_tracks(payload)


def parse_subtitle_tracks(payload: dict) -> list[SubtitleTrack]:
    tracks: list[SubtitleTrack] = []
    for raw_track in payload.get("tracks", []):
        if raw_track.get("type") != "subtitles":
            continue
        properties = raw_track.get("properties") or {}
        codec = raw_track.get("codec")
        codec_id = properties.get("codec_id") or raw_track.get("codec_id")
        tracks.append(
            SubtitleTrack(
                id=int(raw_track["id"]),
                codec=codec,
                codec_id=codec_id,
                language=properties.get("language"),
                name=properties.get("track_name"),
                is_default=bool(properties.get("default_track")),
                is_forced=bool(properties.get("forced_track")),
                format=_detect_subtitle_format(codec_id, codec),
            )
        )
    return tracks


def extract_subtitle_track(
    input_path: Path,
    track: SubtitleTrack,
    working_directory: Path,
    reporter: Reporter = None,
) -> Path:
    if not track.is_supported:
        raise ValueError(f"Track #{track.id} is not a supported text subtitle track.")

    output_path = working_directory / f"track_{track.id}.{track.extracted_suffix}"
    if reporter is not None:
        reporter(f"Extracting subtitle track #{track.id} with mkvextract.")
    _run_tool(
        [
            "mkvextract",
            "tracks",
            str(input_path),
            f"{track.id}:{output_path}",
        ],
        stream_output=True,
    )
    return output_path


def mux_translated_track(
    input_path: Path,
    translated_subtitle_path: Path,
    target_language: TargetLanguage,
    output_path: Path,
    reporter: Reporter = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter(f"Remuxing translated subtitles into {output_path.name} with mkvmerge.")
    _run_tool(
        [
            "mkvmerge",
            "-o",
            str(output_path),
            str(input_path),
            "--language",
            f"0:{target_language.mkv_language_code}",
            "--track-name",
            f"0:{target_language.display_name} (AI translated)",
            "--default-track-flag",
            "0:no",
            str(translated_subtitle_path),
        ],
        stream_output=True,
    )


def _detect_subtitle_format(codec_id: str | None, codec: str | None) -> SubtitleFormat | None:
    if codec_id in SUPPORTED_CODEC_IDS:
        return SUPPORTED_CODEC_IDS[codec_id]

    codec_name = (codec or "").lower()
    if "subrip" in codec_name or "srt" in codec_name:
        return SubtitleFormat.SRT
    if "substation alpha" in codec_name or codec_name.endswith("ass") or codec_name.endswith("ssa"):
        return SubtitleFormat.ASS
    return None


def _run_tool(command: list[str], stream_output: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=not stream_output,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Command not found: {command[0]}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() if result.stderr else ""
        if not detail and result.stdout:
            detail = result.stdout.strip()
        if not detail:
            detail = f"exit code {result.returncode}"
        raise ExternalToolError(f"{command[0]} failed: {detail}")
    return result
