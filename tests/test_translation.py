import httpx

from subtitle_translator.models import SubtitleFormat, SubtitleSegment, SubtitleTrack, TargetLanguage
from subtitle_translator.translation import (
    MAX_BATCH_SEGMENTS,
    MAX_ESTIMATED_OUTPUT_TOKENS,
    SubtitleTranslator,
    TranslationContractError,
    TranslationEnvelope,
    TranslationError,
    TranslationItem,
    _estimate_translated_output_tokens,
    _split_contiguous_segments,
)


def test_split_contiguous_segments_keeps_order() -> None:
    segments = [
        SubtitleSegment(id=1, event_index=0, text="a" * 10),
        SubtitleSegment(id=2, event_index=1, text="b" * 30),
        SubtitleSegment(id=3, event_index=2, text="c" * 5),
        SubtitleSegment(id=4, event_index=3, text="d" * 5),
    ]

    left, right = _split_contiguous_segments(segments)

    assert [segment.id for segment in left] == [1, 2]
    assert [segment.id for segment in right] == [3, 4]


def test_validate_batch_response_rejects_wrong_order() -> None:
    segments = [
        SubtitleSegment(id=1, event_index=0, text="hello"),
        SubtitleSegment(id=2, event_index=1, text="world"),
    ]
    envelope = TranslationEnvelope(
        translations=[
            TranslationItem(id=2, text="monde"),
            TranslationItem(id=1, text="bonjour"),
        ]
    )

    try:
        from subtitle_translator.translation import SubtitleTranslator

        SubtitleTranslator._validate_batch_response(segments, envelope)
    except TranslationError:
        pass
    else:
        raise AssertionError("Expected TranslationError when translated IDs are reordered.")


def test_translate_segments_splits_on_invalid_mapping() -> None:
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = "gpt-5.4"
    translator.reporter = None

    calls: list[int] = []

    def fake_translate_batch(segments, target_language, source_track, batch_label):
        calls.append(len(segments))
        if len(segments) > 1:
            raise TranslationContractError("bad order")
        return [TranslationItem(id=segments[0].id, text=f"t{segments[0].id}")]

    translator._translate_batch = fake_translate_batch

    segments = [
        SubtitleSegment(id=1, event_index=0, text="hello"),
        SubtitleSegment(id=2, event_index=1, text="world"),
    ]
    track = SubtitleTrack(
        id=7,
        codec="SubRip/SRT",
        codec_id="S_TEXT/UTF8",
        language="kor",
        name=None,
        is_default=False,
        is_forced=False,
        format=SubtitleFormat.SRT,
    )

    result = SubtitleTranslator.translate_segments(translator, segments, TargetLanguage.FRENCH, track)

    assert result == {1: "t1", 2: "t2"}
    assert calls == [2, 1, 1]


def test_translate_segments_presplits_large_batches() -> None:
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = "gpt-5.4"
    translator.reporter = None

    calls: list[int] = []

    def fake_translate_batch(segments, target_language, source_track, batch_label):
        calls.append(len(segments))
        return [TranslationItem(id=segment.id, text=f"t{segment.id}") for segment in segments]

    translator._translate_batch = fake_translate_batch

    segments = [
        SubtitleSegment(id=index + 1, event_index=index, text="x")
        for index in range(MAX_BATCH_SEGMENTS + 1)
    ]
    track = SubtitleTrack(
        id=7,
        codec="SubRip/SRT",
        codec_id="S_TEXT/UTF8",
        language="kor",
        name=None,
        is_default=False,
        is_forced=False,
        format=SubtitleFormat.SRT,
    )

    result = SubtitleTranslator.translate_segments(translator, segments, TargetLanguage.FRENCH, track)

    assert len(result) == len(segments)
    assert all(call <= MAX_BATCH_SEGMENTS for call in calls)


def test_estimated_output_tokens_trigger_presplit() -> None:
    segments = [
        SubtitleSegment(id=index + 1, event_index=index, text="x" * 170)
        for index in range(700)
    ]

    estimated_output_tokens = _estimate_translated_output_tokens(segments)

    assert estimated_output_tokens > MAX_ESTIMATED_OUTPUT_TOKENS


def test_translate_segments_splits_on_timeout() -> None:
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = "gpt-5.4"
    translator.reporter = None

    calls: list[int] = []

    def fake_translate_batch(segments, target_language, source_track, batch_label):
        calls.append(len(segments))
        if len(segments) > 1:
            raise __import__("openai").APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        return [TranslationItem(id=segments[0].id, text=f"t{segments[0].id}")]

    translator._translate_batch = fake_translate_batch

    segments = [
        SubtitleSegment(id=1, event_index=0, text="hello"),
        SubtitleSegment(id=2, event_index=1, text="world"),
    ]
    track = SubtitleTrack(
        id=7,
        codec="SubRip/SRT",
        codec_id="S_TEXT/UTF8",
        language="kor",
        name=None,
        is_default=False,
        is_forced=False,
        format=SubtitleFormat.SRT,
    )

    result = SubtitleTranslator.translate_segments(translator, segments, TargetLanguage.FRENCH, track)

    assert result == {1: "t1", 2: "t2"}
    assert calls == [2, 1, 1]


def test_wait_log_contains_batch_label() -> None:
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = "gpt-5.4"
    messages: list[str] = []
    translator.reporter = messages.append

    def fake_translate_batch(segments, target_language, source_track, batch_label):
        translator._report(f"{batch_label}: Waiting for OpenAI translation response.")
        return [TranslationItem(id=segment.id, text=f"t{segment.id}") for segment in segments]

    translator._translate_batch = fake_translate_batch

    segments = [
        SubtitleSegment(id=1, event_index=0, text="hello"),
        SubtitleSegment(id=2, event_index=1, text="world"),
    ]
    track = SubtitleTrack(
        id=7,
        codec="SubRip/SRT",
        codec_id="S_TEXT/UTF8",
        language="kor",
        name=None,
        is_default=False,
        is_forced=False,
        format=SubtitleFormat.SRT,
    )

    SubtitleTranslator.translate_segments(translator, segments, TargetLanguage.FRENCH, track)

    assert any("Batch 1/1: Waiting for OpenAI translation response." in message for message in messages)
