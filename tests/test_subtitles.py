import pysubs2

from subtitle_translator.subtitles import apply_translations, collect_translatable_segments


def test_collect_translatable_segments_skips_comments_and_blanks() -> None:
    subtitles = pysubs2.SSAFile()
    subtitles.events = [
        pysubs2.SSAEvent(start=0, end=1000, text="Hello"),
        pysubs2.SSAEvent(start=1000, end=2000, text="   "),
        pysubs2.SSAEvent(start=2000, end=3000, text="Comment", type="Comment"),
        pysubs2.SSAEvent(start=3000, end=4000, text="World"),
    ]

    segments = collect_translatable_segments(subtitles)

    assert [segment.id for segment in segments] == [1, 2]
    assert [segment.event_index for segment in segments] == [0, 3]


def test_apply_translations_updates_matching_events() -> None:
    subtitles = pysubs2.SSAFile()
    subtitles.events = [
        pysubs2.SSAEvent(start=0, end=1000, text="Hello"),
        pysubs2.SSAEvent(start=1000, end=2000, text="World"),
    ]
    segments = collect_translatable_segments(subtitles)

    apply_translations(subtitles, segments, {1: "Bonjour", 2: "Monde"})

    assert subtitles.events[0].text == "Bonjour"
    assert subtitles.events[1].text == "Monde"
