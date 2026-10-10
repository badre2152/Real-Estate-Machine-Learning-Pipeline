"""Ensure Python sources do not contain decorative emoji."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_python_sources_have_no_decorative_emoji():
    offenders = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        if any(
            0x1F000 <= ord(char) <= 0x1FAFF
            or 0x2600 <= ord(char) <= 0x27BF
            or ord(char) in (0xFE0F, 0x200D)
            for char in content
        ):
            offenders.append(path.relative_to(SRC_DIR).as_posix())
    assert not offenders, f"Decorative emoji found in: {offenders}"


def test_baseline_status_strings_remain_readable():
    source = (SRC_DIR / "baselines.py").read_text(encoding="utf-8")
    assert '"PASS" if' in source
    assert 'else "FAIL"' in source
