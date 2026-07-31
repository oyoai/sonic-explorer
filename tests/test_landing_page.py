"""AppTest smoke test for the Overview/landing page. Content follows a real
content spec (Problem / Existing solutions / Research question) -- the
concrete "existing systems fail" evidence is two real metadata-vs-real
similarity pairs (moved here from Approach's old step 1, since this is where
the point actually belongs), not a Spotify screenshot or the metadata
network graph (both dropped from this page in the restructure -- the graph's
fuller treatment already lives in Results)."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_landing() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.run()
    return at


@pytest.fixture(scope="module")
def landing_at() -> AppTest:
    """Every test in this file only reads the default render -- one shared
    cold start instead of 9 redundant ones."""
    return _run_landing()


def test_landing_page_runs_without_exceptions(landing_at):
    assert not landing_at.exception


def test_landing_page_has_all_three_sections(landing_at):
    header_texts = [h.value for h in landing_at.header]
    assert "Problem" in header_texts
    assert "Existing solutions" in header_texts
    assert "Research question" in header_texts


def test_landing_page_problem_has_the_big_quote(landing_at):
    markdown_texts = " ".join(m.value for m in landing_at.markdown)
    assert "I love this song, find me more like it" in markdown_texts


def test_landing_page_research_question_has_the_big_quote(landing_at):
    markdown_texts = " ".join(m.value for m in landing_at.markdown)
    assert "Can we find songs that are actually similar to one another" in markdown_texts
    assert "by sound" in markdown_texts


def test_landing_page_has_both_concept_diagrams_with_real_descriptive_captions(landing_at):
    caption_texts = " ".join(c.value for c in landing_at.caption)
    assert "Songs similar to this song" in caption_texts
    assert "Users who liked this song also liked" in caption_texts


def test_landing_page_discloses_missing_user_data_for_collaborative_filtering(landing_at):
    info_texts = " ".join(i.value for i in landing_at.info)
    assert "honest gap" in info_texts.lower()
    assert "user-level" in info_texts.lower()


def test_landing_page_no_longer_shows_spotify_screenshot_or_metadata_graph(landing_at):
    """Restructure decision: both dropped from Overview -- the metadata
    graph's fuller treatment (with real cross-genre stats) already lives in
    Results, and the Spotify-screenshot placeholder was replaced by the two
    real similarity pairs, which need no external asset at all."""
    caption_texts = " ".join(c.value for c in landing_at.caption).lower()
    assert "spotify" not in caption_texts
    assert "cross a genre boundary" not in caption_texts


def test_landing_page_shows_two_real_pairs_with_real_similarity_scores_and_playback(landing_at):
    markdown_texts = " ".join(m.value for m in landing_at.markdown)
    assert "sound completely different" in markdown_texts.lower()
    assert "sound remarkably similar" in markdown_texts.lower()
    # 4 songs total across the two pairs, each with its own audio player
    assert len(landing_at.get("audio")) >= 4


def test_landing_page_links_to_approach_next(landing_at):
    assert any(b.label.startswith("See the approach") for b in landing_at.button)
