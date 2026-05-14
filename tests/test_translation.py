from subtitle_translator.models import SubtitleSegment
from subtitle_translator.translation import TranslationEnvelope, TranslationError, TranslationItem, _split_contiguous_segments


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
