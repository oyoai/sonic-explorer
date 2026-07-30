"""AppTest smoke test for the Overview/landing page. Content follows a real
content spec (Problem / Existing solutions / Research question) -- the
concrete "existing systems fail" evidence is two real metadata-vs-real
similarity pairs (moved here from Approach's old step 1, since this is where
the point actually belongs), not a Spotify screenshot or the metadata
network graph (both dropped from this page in the restructure -- the graph's
fuller treatment already lives in Results)."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_landing() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.run()
    return at


def test_landing_page_runs_without_exceptions():
    at = _run_landing()
    assert not at.exception


def test_landing_page_has_all_three_sections():
    at = _run_landing()
    header_texts = [h.value for h in at.header]
    assert "Problem" in header_texts
    assert "Existing solutions" in header_texts
    assert "Research question" in header_texts


def test_landing_page_problem_has_the_big_quote():
    at = _run_landing()
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "I love this song, find me more like it" in markdown_texts


def test_landing_page_research_question_has_the_big_quote():
    at = _run_landing()
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "Can we find songs that are actually similar to one another" in markdown_texts
    assert "by sound" in markdown_texts


def test_landing_page_has_both_concept_diagrams_with_real_descriptive_captions():
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "Songs similar to this song" in caption_texts
    assert "Users who liked this song also liked" in caption_texts


def test_landing_page_discloses_missing_user_data_for_collaborative_filtering():
    at = _run_landing()
    info_texts = " ".join(i.value for i in at.info)
    assert "honest gap" in info_texts.lower()
    assert "user-level" in info_texts.lower()


def test_landing_page_no_longer_shows_spotify_screenshot_or_metadata_graph():
    """Restructure decision: both dropped from Overview -- the metadata
    graph's fuller treatment (with real cross-genre stats) already lives in
    Results, and the Spotify-screenshot placeholder was replaced by the two
    real similarity pairs, which need no external asset at all."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption).lower()
    assert "spotify" not in caption_texts
    assert "cross a genre boundary" not in caption_texts


def test_landing_page_shows_two_real_pairs_with_real_similarity_scores_and_playback():
    at = _run_landing()
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "sound completely different" in markdown_texts.lower()
    assert "sound remarkably similar" in markdown_texts.lower()
    # 4 songs total across the two pairs, each with its own audio player
    assert len(at.get("audio")) >= 4


def test_landing_page_links_to_approach_next():
    at = _run_landing()
    assert any(b.label.startswith("See the approach") for b in at.button)
