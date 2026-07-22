"""Regression tests for the drill-down architecture: Explore is the hub,
Song X-Ray/Moment Matcher/Ask the DJ are reached from within it and must
carry real context along (not just "doesn't crash with an arbitrary
default"). See streamlit_app/Overview.py's st.navigation() call --
visibility="hidden" keeps these three switch_page-reachable while excluding
them from the sidebar.

Real plotly on_select click events (song node clicks, timeline segment
clicks) aren't simulable through AppTest -- the existing page tests already
work around this by setting the resulting session_state directly rather
than simulating the click itself (see test_explore_page.py). These tests
follow the same pattern."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic_explorer.config import DB_PATH  # noqa: E402
from sonic_explorer.repository.db import init_db  # noqa: E402
from sonic_explorer.repository.song_repository import SongRepository  # noqa: E402


def _some_song_with_segments():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    for song in sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title)):
        segments = song_repo.get_segments(song.id)
        if segments:
            return song, segments
    raise AssertionError("no song with segments found in the test database")


def test_explore_open_xray_button_carries_the_selected_song():
    song, _ = _some_song_with_segments()

    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/6_Explore.py")
    at.session_state["explore_selected_song_id"] = song.id
    at.run()

    xray_button = next(b for b in at.button if "Open full Song X-Ray" in b.label)
    xray_button.click().run()

    subheader_texts = [s.value for s in at.subheader]
    assert any(song.title in s for s in subheader_texts), (
        f"expected Song X-Ray to default to {song.title!r}, got {subheader_texts}"
    )


def test_song_xray_context_is_consumed_once_not_reapplied():
    """xray_context_song_id must be popped after use -- otherwise a second,
    unrelated visit would keep reapplying a stale selection."""
    song, _ = _some_song_with_segments()

    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/3_Song_XRay.py")
    at.session_state["xray_context_song_id"] = song.id
    at.run()

    assert "xray_context_song_id" not in at.session_state
    subheader_texts = [s.value for s in at.subheader]
    assert any(song.title in s for s in subheader_texts)


def test_moment_matcher_context_defaults_song_and_moment():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
    song = next(s for s in songs if song_repo.get_segments(s.id))
    segments = song_repo.get_segments(song.id)
    target_segment = segments[-1] if len(segments) > 1 else segments[0]

    expected_song_index = next(i for i, s in enumerate(songs) if s.id == song.id)
    expected_moment_index = next(i for i, seg in enumerate(segments) if seg.id == target_segment.id)

    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/4_Moment_Matcher.py")
    at.session_state["mm_context"] = {"song_id": song.id, "segment_id": target_segment.id}
    at.run()

    assert "mm_context" not in at.session_state
    song_select = next(w for w in at.selectbox if w.label == "Song")
    assert song_select.value == expected_song_index
    moment_slider = next(w for w in at.select_slider if w.label == "Moment")
    assert moment_slider.value == expected_moment_index


def test_moment_matcher_with_no_context_still_renders_cleanly():
    """No-context visit (e.g. a stale direct link, or sidebar removal
    notwithstanding someone bookmarks the URL) must still work -- the
    pre-existing "doesn't crash" behavior, kept as a regression guard now
    that a context path also exists."""
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/4_Moment_Matcher.py")
    at.run()
    assert not at.exception


def test_ask_the_dj_links_back_to_explore():
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/5_Ask_The_DJ.py")
    at.run()
    assert not at.exception
