"""AppTest smoke test for the landing page (streamlit_app/app.py). Used to
be a bare st.switch_page("pages/0_Methodology.py") passthrough with no
content of its own -- now a real introduction page, tested the same way as
every other page in the app."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_landing() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.run()
    return at


def test_landing_page_runs_without_exceptions():
    at = _run_landing()
    assert not at.exception


def test_landing_page_is_not_a_passthrough():
    """Regression test for the exact bug this page replaced: no
    switch_page-only landing, real intro content instead."""
    at = _run_landing()
    header_texts = [h.value for h in at.header]
    assert "1. What this is" in header_texts


def test_landing_page_has_naive_vs_real_comparison_and_real_related_work():
    """1.1 is now a real naive-vs-audio graph comparison, combining every
    available non-audio metadata signal rather than a genre-only strawman
    (no longer a placeholder); 1.2 has real (web-verified) citations --
    regression guard against either silently reverting to a stub."""
    at = _run_landing()
    subheader_texts = [s.value for s in at.subheader]
    assert any("naive approach" in s for s in subheader_texts)
    assert any("Related work" in s for s in subheader_texts)

    caption_texts = " ".join(c.value for c in at.caption)
    assert "genre + genre hierarchy + album + tags" in caption_texts
    assert "audio embeddings" in caption_texts

    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "Tovstogan" in markdown_texts
    assert "Vohra" in markdown_texts
    assert "VidTune" in markdown_texts
    assert "Audiobrain" not in markdown_texts  # dropped -- couldn't be verified, see the warning box

    warning_texts = " ".join(w.value for w in at.warning)
    assert "dropped" in warning_texts.lower()


def test_landing_page_renders_naive_vs_real_graphs_without_exception():
    """The two side-by-side network graphs (combined metadata baseline vs.
    audio embeddings) in 1.1 must execute against the real repositories with
    no exception -- AppTest has no typed plotly_chart accessor to assert on
    directly here either, so a clean run is the meaningful check."""
    at = _run_landing()
    assert not at.exception


def test_landing_page_reports_real_song_and_genre_counts():
    """Regression guard for the stat-box -> waffle-grid replacement: the
    caption above the grid must show real, computed numbers, not a stale
    hardcoded string."""
    at = _run_landing()
    caption_texts = [c.value for c in at.caption]
    assert any("songs across" in c and "genres" in c for c in caption_texts)


def test_landing_page_no_longer_mentions_embedded_segments():
    """The 'Embedded segments' stat was removed outright -- it referenced a
    concept (segments) never explained this early on the page."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "Embedded segments" not in caption_texts


def test_landing_page_renders_waffle_grid_without_exception():
    """The waffle grid must execute against the real repositories with no
    exception -- AppTest in this Streamlit version has no typed plotly_chart
    accessor to assert on directly, so a clean run is the meaningful check
    here."""
    at = _run_landing()
    assert not at.exception
