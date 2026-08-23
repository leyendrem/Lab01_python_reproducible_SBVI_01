import pytest

from python_scientific.quality import validate_channel_configuration


def test_complete_configuration():
    result = validate_channel_configuration(
        ["ECG_I", "ECG_II", "RESP"],
        {"ECG_I", "ECG_II", "RESP"},
    )

    assert result["duplicates"] == []
    assert result["missing"] == []
    assert result["additional"] == []


def test_missing_channel():
    result = validate_channel_configuration(
        ["ECG_I", "RESP"],
        {"ECG_I", "ECG_II", "RESP"},
    )

    assert result["missing"] == ["ECG_II"]


def test_duplicate_channel():
    result = validate_channel_configuration(
        ["ECG_I", "RESP", "ECG_I"],
        {"ECG_I", "ECG_II", "RESP"},
    )

    assert result["duplicates"] == ["ECG_I"]


def test_empty_channel_name():
    with pytest.raises(ValueError):
        validate_channel_configuration(
            ["ECG_I", "", "RESP"],
            {"ECG_I", "RESP"},
        )


def test_whitespace_channel_name():
    with pytest.raises(ValueError):
        validate_channel_configuration(
            ["ECG_I", "   ", "RESP"],
            {"ECG_I", "RESP"},
        )
