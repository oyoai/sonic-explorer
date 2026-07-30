"""AppTest smoke test for Moment Matcher, including the whole-song
(song-level aggregation) mode added alongside the existing per-moment mode.
Must go through app.py + switch_page for consistency with the rest of the
suite's multipage-registry requirement."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_moment_matcher() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/5_Moment_Matcher.py")
    at.run()
    return at


def test_moment_matcher_page_runs_without_exceptions():
    at = _run_moment_matcher()
    assert not at.exception


def test_moment_matcher_defaults_to_moment_granularity():
    at = _run_moment_matcher()
    granularity_radio = next(r for r in at.radio if r.label == "Match against")
    assert granularity_radio.value == "moment"


def test_moment_matcher_switching_to_whole_song_mode_runs_without_exceptions():
    at = _run_moment_matcher()
    granularity_radio = next(r for r in at.radio if r.label == "Match against")
    granularity_radio.set_value("whole_song").run()
    assert not at.exception


def test_moment_matcher_whole_song_mode_hides_moment_slider():
    at = _run_moment_matcher()
    granularity_radio = next(r for r in at.radio if r.label == "Match against")
    granularity_radio.set_value("whole_song").run()
    assert not at.exception
    assert not any(s.label == "Moment" for s in at.select_slider)


def test_moment_matcher_moment_slider_is_discrete_not_a_continuous_scrub():
    """The moment picker must be a select_slider bound to segment *indices*
    (snapping between the song's actual detected moments), never a plain
    st.slider over a continuous float range."""
    at = _run_moment_matcher()
    assert any(s.label == "Moment" for s in at.select_slider)
    assert not any(s.label == "Moment" for s in at.slider)


def test_moment_matcher_shows_waveform_of_selected_moment():
    """Default mode (moment granularity) must render a waveform plot for
    the selected ~5s clip alongside the audio player, not just the player."""
    at = _run_moment_matcher()
    assert not at.exception
    charts = at.get("plotly_chart")
    assert any("Selected moment" in c.spec for c in charts)


def test_moment_matcher_moment_results_render_as_a_paginated_card():
    """Results must show one at a time in a fixed-height card with
    Prev/Next arrow buttons, not one long expanding list of every match."""
    at = _run_moment_matcher()
    assert not at.exception
    button_labels = [b.label for b in at.button]
    assert "◀" in button_labels
    assert "▶" in button_labels
    caption_texts = " ".join(c.value for c in at.caption)
    assert "Result 1 of" in caption_texts
