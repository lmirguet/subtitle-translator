from subtitle_translator.media import parse_subtitle_tracks
from subtitle_translator.models import SubtitleFormat


def test_parse_subtitle_tracks_filters_non_subtitles() -> None:
    payload = {
        "tracks": [
            {"id": 0, "type": "video", "codec": "AVC/H.264", "properties": {}},
            {
                "id": 1,
                "type": "subtitles",
                "codec": "SubRip/SRT",
                "properties": {"codec_id": "S_TEXT/UTF8", "language": "eng"},
            },
            {
                "id": 2,
                "type": "subtitles",
                "codec": "HDMV PGS",
                "properties": {"codec_id": "S_HDMV/PGS", "language": "eng"},
            },
        ]
    }

    tracks = parse_subtitle_tracks(payload)

    assert [track.id for track in tracks] == [1, 2]
    assert tracks[0].format == SubtitleFormat.SRT
    assert tracks[0].is_supported is True
    assert tracks[1].format is None
    assert tracks[1].is_supported is False
