from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from subtitle_translator import media, subtitles
from subtitle_translator.models import SubtitleTrack, TargetLanguage, TranslationOptions
from subtitle_translator.translation import SubtitleTranslator


class ApplicationError(RuntimeError):
    pass


class SubtitleTranslationApp:
    def inspect_tracks(self, input_path: Path) -> list[SubtitleTrack]:
        media.ensure_mkvtoolnix_available()
        return media.inspect_subtitle_tracks(input_path)

    def translate(self, options: TranslationOptions) -> Path:
        media.ensure_mkvtoolnix_available()
        tracks = media.inspect_subtitle_tracks(options.input_path)
        source_track = _resolve_track(tracks, options.source_track_id)

        if not source_track.is_supported or source_track.format is None:
            raise ApplicationError(f"Track #{source_track.id} is not a supported text subtitle track.")

        output_path = _resolve_output_path(options.input_path, options.target_language, options.output_path, options.in_place)

        if not options.in_place and output_path.exists():
            raise ApplicationError(f"Output file already exists: {output_path}")

        translator = SubtitleTranslator(model=options.model)

        with TemporaryDirectory(dir=options.input_path.parent) as tmp_dir:
            working_directory = Path(tmp_dir)
            extracted_subtitle_path = media.extract_subtitle_track(options.input_path, source_track, working_directory)
            subtitle_doc = subtitles.load_subtitles(extracted_subtitle_path, source_track.format)
            segments = subtitles.collect_translatable_segments(subtitle_doc)
            translated_texts = translator.translate_segments(segments, options.target_language, source_track)

            translated_subtitle_path = working_directory / f"translated.{source_track.format.value}"
            subtitles.apply_translations(subtitle_doc, segments, translated_texts)
            subtitles.save_subtitles(subtitle_doc, translated_subtitle_path, source_track.format)

            temp_output_path = (
                working_directory / f"{options.input_path.stem}.translated{options.input_path.suffix}"
                if options.in_place
                else output_path
            )
            media.mux_translated_track(options.input_path, translated_subtitle_path, options.target_language, temp_output_path)

            if options.in_place:
                temp_output_path.replace(options.input_path)
                return options.input_path

        return output_path


def _resolve_track(tracks: list[SubtitleTrack], track_id: int) -> SubtitleTrack:
    for track in tracks:
        if track.id == track_id:
            return track
    raise ApplicationError(f"Subtitle track #{track_id} was not found in the MKV file.")


def _resolve_output_path(
    input_path: Path,
    target_language: TargetLanguage,
    requested_output_path: Path | None,
    in_place: bool,
) -> Path:
    if in_place:
        return input_path
    if requested_output_path is not None:
        return requested_output_path
    return input_path.with_name(f"{input_path.stem}.{target_language.value}.translated{input_path.suffix}")
