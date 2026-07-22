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
    at.switch_page("pages/4_Moment_Matcher.py")
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
