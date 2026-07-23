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


def test_landing_page_has_naive_baseline_placeholder_and_real_related_work():
    """1.1 is still an honest stub; 1.2 has real (web-verified) citations now
    -- regression guard against either silently reverting to a placeholder."""
    at = _run_landing()
    subheader_texts = [s.value for s in at.subheader]
    assert any("naive approach" in s for s in subheader_texts)
    assert any("Related work" in s for s in subheader_texts)

    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "Tovstogan" in markdown_texts
    assert "Vohra" in markdown_texts
    assert "VidTune" in markdown_texts
    assert "Audiobrain" not in markdown_texts  # dropped -- couldn't be verified, see the warning box

    warning_texts = " ".join(w.value for w in at.warning)
    assert "dropped" in warning_texts.lower()


def test_landing_page_renders_animated_stats_iframe():
    at = _run_landing()
    iframe_found = any(type(node).__name__ == "UnknownElement" for node in at.main.children.values())
    assert iframe_found


def test_landing_page_renders_genre_and_cluster_visuals_without_exception():
    """Both new Plotly visuals (genre-breakdown bar, cluster density preview)
    must execute against the real repositories with no exception -- AppTest
    in this Streamlit version has no typed plotly_chart accessor to assert
    on directly, so a clean run is the meaningful check here."""
    at = _run_landing()
    assert not at.exception
