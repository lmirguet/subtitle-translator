from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import questionary
import typer
from rich.console import Console

from subtitle_translator.app import ApplicationError, SubtitleTranslationApp
from subtitle_translator.media import DependencyError, ExternalToolError
from subtitle_translator.models import SubtitleTrack, TargetLanguage, TranslationOptions
from subtitle_translator.subtitles import SubtitleProcessingError
from subtitle_translator.translation import DEFAULT_MODEL, TranslationError

console = Console()


def main(
    input_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True, help="Input MKV file."),
    ],
    track_id: Annotated[
        int | None,
        typer.Option("--track-id", help="Subtitle track ID to translate. Skip the interactive track prompt."),
    ] = None,
    target_lang: Annotated[
        TargetLanguage | None,
        typer.Option(
            "--target-lang",
            case_sensitive=False,
            help="Target language: en, fr, es, ru, or pl. Skip the interactive language prompt.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
            help="Output MKV path. Ignored when --in-place is used.",
        ),
    ] = None,
    in_place: Annotated[
        bool,
        typer.Option("--in-place", help="Replace the original MKV after a successful remux."),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI model to use for translation."),
    ] = DEFAULT_MODEL,
) -> None:
    if output is not None and in_place:
        raise typer.BadParameter("--output cannot be used together with --in-place.")
    if input_path.suffix.lower() != ".mkv":
        raise typer.BadParameter("Input file must be an MKV file.")

    app = SubtitleTranslationApp(reporter=_log_message)

    try:
        tracks = app.inspect_tracks(input_path)
        supported_tracks = [track for track in tracks if track.is_supported]
        if not supported_tracks:
            raise ApplicationError("No supported text subtitle tracks were found in this MKV file.")

        selected_track_id = track_id if track_id is not None else _prompt_for_track(supported_tracks)
        selected_language = target_lang or _prompt_for_language()
        selected_in_place = in_place or _prompt_for_output_mode(output)

        if selected_in_place and output is not None:
            raise typer.BadParameter("--output cannot be used together with in-place replacement.")

        options = TranslationOptions(
            input_path=input_path,
            source_track_id=selected_track_id,
            target_language=selected_language,
            output_path=output,
            in_place=selected_in_place,
            model=model,
        )

        console.print(f"[bold]Source track:[/bold] #{selected_track_id}")
        console.print(f"[bold]Target language:[/bold] {selected_language.display_name}")
        console.print(f"[bold]Model:[/bold] {model}")

        if selected_in_place:
            console.print("[bold]Output mode:[/bold] in-place replacement after successful remux")
        elif output is not None:
            console.print(f"[bold]Output file:[/bold] {output}")

        result_path = app.translate(options)
        console.print(f"[green]Done.[/green] Output: {result_path}")
    except (
        ApplicationError,
        DependencyError,
        ExternalToolError,
        SubtitleProcessingError,
        TranslationError,
    ) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def run() -> None:
    typer.run(main)


def _prompt_for_track(tracks: list[SubtitleTrack]) -> int:
    if not _is_interactive():
        raise ApplicationError("No subtitle track was provided and the terminal is not interactive.")

    choice = questionary.select(
        "Choose the subtitle track to translate",
        choices=[
            questionary.Choice(title=track.display_label(), value=track.id)
            for track in tracks
        ],
    ).ask()
    if choice is None:
        raise ApplicationError("Track selection was cancelled.")
    return int(choice)


def _prompt_for_language() -> TargetLanguage:
    if not _is_interactive():
        raise ApplicationError("No target language was provided and the terminal is not interactive.")

    choice = questionary.select(
        "Choose the target language",
        choices=[
            questionary.Choice(title=language.display_name, value=language)
            for language in TargetLanguage
        ],
    ).ask()
    if choice is None:
        raise ApplicationError("Language selection was cancelled.")
    return choice


def _prompt_for_output_mode(output: Path | None) -> bool:
    if output is not None:
        return False
    if not _is_interactive():
        return False

    choice = questionary.select(
        "How should the translated MKV be written?",
        choices=[
            questionary.Choice(title="Write a new MKV file", value=False),
            questionary.Choice(title="Replace the original MKV after success", value=True),
        ],
        default=False,
    ).ask()
    if choice is None:
        raise ApplicationError("Output mode selection was cancelled.")
    return bool(choice)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _log_message(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


if __name__ == "__main__":
    run()
