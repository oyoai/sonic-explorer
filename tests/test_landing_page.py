"""AppTest smoke test for the Overview/landing page. Content and structure
have changed substantially across this project's restructure -- Problem /
Existing solutions / Proposed solution, no more Related Work section (moved
inline into Methodology), no more waffle grid (moved to Methodology's
dataset section), a new closing link to the Approach page."""

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
    """Regression test for the original bug this page replaced: no
    switch_page-only landing, real intro content instead."""
    at = _run_landing()
    header_texts = [h.value for h in at.header]
    assert "1. Problem" in header_texts


def test_landing_page_has_problem_existing_solutions_proposed_solution():
    at = _run_landing()
    header_texts = [h.value for h in at.header]
    assert "1. Problem" in header_texts
    assert "2. Existing solutions" in header_texts
    assert "3. Proposed solution" in header_texts


def test_landing_page_no_longer_has_related_work_section():
    """Restructure decision: a dedicated Related Work section here would
    visually imply this project was inspired by the cited papers, which
    isn't true -- citations moved inline into Methodology instead."""
    at = _run_landing()
    subheader_texts = " ".join(s.value for s in at.subheader)
    header_texts = " ".join(h.value for h in at.header)
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "Related work" not in subheader_texts
    assert "Related work" not in header_texts
    assert "Tovstogan" not in markdown_texts
    assert "Vohra" not in markdown_texts
    assert "VidTune" not in markdown_texts


def test_landing_page_naive_vs_real_comparison_uses_combined_metadata_baseline():
    """2's naive baseline combines every available non-audio metadata
    signal, not a genre-only strawman."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "genre + genre hierarchy + album + tags" in caption_texts
    assert "audio embeddings" in caption_texts


def test_landing_page_has_tautology_callout_with_real_stats():
    """The naive graph's clean clusters are a structural artifact of its
    edge definition, not evidence of quality -- must be called out
    explicitly with real, computed cross-genre-edge percentages, not left
    to silently imply the naive approach "worked better."."""
    at = _run_landing()
    warning_texts = " ".join(w.value for w in at.warning)
    assert "not evidence" in warning_texts.lower() or "by construction" in warning_texts.lower()
    assert "%" in warning_texts


def test_landing_page_has_audio_playback_demo():
    """The most persuasive part of the comparison: real audio the user can
    hear, not just an annotated graph -- both a naive same-genre-but-
    different-sounding pair and a real cross-genre-but-similar-sounding
    pair, each with playable audio."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "naive calls these" in caption_texts.lower()
    assert "audio calls these" in caption_texts.lower()
    assert len(at.get("audio")) >= 4  # two songs per pair, two pairs


def test_landing_page_links_to_approach_next():
    """Restructure decision: Overview hands off to the new Approach page
    (which picks up exactly where "Proposed solution" leaves off), not
    straight to Methodology."""
    at = _run_landing()
    write_texts = " ".join(m.value for m in at.markdown)
    assert "Approach" in write_texts
