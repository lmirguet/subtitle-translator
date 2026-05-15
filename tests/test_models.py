from subtitle_translator.models import TargetLanguage


def test_polish_target_language_metadata() -> None:
    assert TargetLanguage.POLISH.value == "pl"
    assert TargetLanguage.POLISH.display_name == "Polish"
    assert TargetLanguage.POLISH.mkv_language_code == "pol"
