"""AppTest smoke test for Moment Matcher, including the whole-song
(song-level aggregation) mode added alongside the existing per-moment mode.
Must go through app.py + switch_page for consistency with the rest of the
suite's multipage-registry requirement."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_moment_matcher() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/5_Moment_Matcher.py")
    at.run()
    return at


@pytest.fixture(scope="module")
def moment_matcher_at() -> AppTest:
    """Shared across the read-only tests below (default moment-granularity
    render, no widget interaction) -- one cold start instead of 5. The two
    whole-song-mode tests mutate the granularity radio and rerun, so they
    keep their own fresh instances."""
    return _run_moment_matcher()


def test_moment_matcher_page_runs_without_exceptions(moment_matcher_at):
    assert not moment_matcher_at.exception


def test_moment_matcher_defaults_to_moment_granularity(moment_matcher_at):
    granularity_radio = next(r for r in moment_matcher_at.radio if r.label == "Match against")
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


def test_moment_matcher_moment_slider_is_discrete_not_a_continuous_scrub(moment_matcher_at):
    """The moment picker must be a select_slider bound to segment *indices*
    (snapping between the song's actual detected moments), never a plain
    st.slider over a continuous float range."""
    assert any(s.label == "Moment" for s in moment_matcher_at.select_slider)
    assert not any(s.label == "Moment" for s in moment_matcher_at.slider)


def test_moment_matcher_shows_waveform_of_selected_moment(moment_matcher_at):
    """Default mode (moment granularity) must render a waveform plot for
    the selected ~5s clip alongside the audio player, not just the player."""
    assert not moment_matcher_at.exception
    charts = moment_matcher_at.get("plotly_chart")
    assert any("Selected moment" in c.spec for c in charts)


def test_moment_matcher_moment_results_render_as_a_paginated_card(moment_matcher_at):
    """Results must show one at a time in a fixed-height card with
    Prev/Next arrow buttons, not one long expanding list of every match."""
    assert not moment_matcher_at.exception
    button_labels = [b.label for b in moment_matcher_at.button]
    assert "◀" in button_labels
    assert "▶" in button_labels
    caption_texts = " ".join(c.value for c in moment_matcher_at.caption)
    assert "Result 1 of" in caption_texts
