"""Protect baseline module from decorative Unicode symbols."""

from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "baselines.py"


def test_baselines_have_no_decorative_emoji():
    source = SOURCE.read_text(encoding="utf-8")
    assert not any(
        0x1F000 <= ord(character) <= 0x1FAFF
        or 0x2600 <= ord(character) <= 0x27BF
        or ord(character) in (0xFE0F, 0x200D)
        for character in source
    )


def test_baseline_status_strings_remain_readable():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"PASS" if' in source
    assert 'else "FAIL"' in source
