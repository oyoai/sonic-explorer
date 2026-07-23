import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from components.animated_stats import animated_stat_row


def test_animated_stat_row_includes_every_label_and_target_value():
    html = animated_stat_row([("Songs in library", 1400), ("Genres", 8)])
    assert "Songs in library" in html
    assert "Genres" in html
    assert "1400" in html
    assert "8" in html


def test_animated_stat_row_escapes_html_in_labels():
    """Labels are hardcoded by this codebase today, not user input -- but
    html.escape costs nothing and this pins that it's actually applied."""
    html_out = animated_stat_row([("<script>alert(1)</script>", 5)])
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_animated_stat_row_empty_list_does_not_raise():
    html = animated_stat_row([])
    assert "animateCount" in html  # the JS function definition is still present, just no calls need it


def test_animated_stat_row_produces_one_animate_call_per_stat():
    html = animated_stat_row([("A", 1), ("B", 2), ("C", 3)])
    assert html.count("animateCount(") == 4  # 3 real calls + 1 function definition line
