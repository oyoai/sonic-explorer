"""AppTest smoke test for the Overview/landing page. Content and structure
have changed substantially across this project's restructure -- Problem /
Existing solutions / Proposed solution, no more Related Work section (moved
inline into Methodology), no more waffle grid (moved to Methodology's
dataset section), a new closing link to the Approach page. The real
audio-embeddings graph and the audio-playback demo moved to Results --
Overview raises the question visually (bubble diagrams + the static
metadata-baseline graph alone), Results is where the evidence for an answer
belongs.

Problem stays focused purely on the personal-story framing (no embedded
concrete system example); the concrete "what an existing system actually
does" example (a real Spotify screenshot, with an honest placeholder until
one is supplied) lives in Existing solutions instead."""

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


def test_landing_page_header_keeps_the_digging_metaphor():
    at = _run_landing()
    header_texts = [h.value for h in at.header]
    assert any("Unearth" in h for h in header_texts)


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


def test_landing_page_problem_section_has_no_embedded_concrete_example():
    """Problem must stay focused on the personal-story frustration, without a
    concrete system example (real song + real recommendation) embedded in
    it -- that example belongs in Existing solutions, not Problem."""
    at = _run_landing()
    write_texts = " ".join(m.value for m in at.markdown)
    assert "you loved this song" not in write_texts.lower()
    assert "a typical existing system recommends" not in write_texts.lower()


def test_landing_page_existing_solutions_has_a_real_spotify_screenshot_or_honest_placeholder():
    """A real screenshot of Spotify's actual recommendation UI is the
    concrete "what existing systems do" evidence -- not a fabricated mock-up.
    Until the real file is supplied, an explicit placeholder must say so
    rather than silently omitting the example."""
    at = _run_landing()
    info_texts = " ".join(i.value for i in at.info)
    has_real_screenshot = bool(at.get("imgs"))
    has_honest_placeholder = "placeholder" in info_texts.lower() and "spotify" in info_texts.lower()
    assert has_real_screenshot or has_honest_placeholder


def test_landing_page_has_both_concept_bubble_diagrams():
    """Section 2 opens with two illustrative (not real-data) diagrams --
    metadata-based matching and collaborative filtering -- before the real,
    static metadata-baseline network graph further down."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "metadata-based matching" in caption_texts.lower()
    assert "collaborative filtering" in caption_texts.lower()


def test_landing_page_discloses_missing_user_data_for_collaborative_filtering():
    """Collaborative filtering is presented as one of the two dominant
    existing paradigms, but this library has no user-level listen/favorite
    data to build or compare against it -- must be disclosed explicitly, not
    left looking like an oversight."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "honest gap" in caption_texts.lower()
    assert "user-level" in caption_texts.lower()


def test_landing_page_existing_solutions_is_framed_as_exploratory_not_declarative():
    """Restructure decision: section 2 raises an open question (does audio
    actually help?) rather than asserting the metadata baseline was "solved"
    or that the proposed approach already won."""
    at = _run_landing()
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "open question" in markdown_texts.lower()


def test_landing_page_metadata_graph_is_static_not_interactive():
    """Overview's 'quick gist' purpose doesn't need an interactive Plotly
    widget -- staticPlot disables hover/zoom/pan/click. Interactivity lives
    in Results/Explore instead, where deeper engagement is expected."""
    at = _run_landing()
    configs = [c.proto.config for c in at.get("plotly_chart")]
    assert any('"staticPlot": true' in cfg for cfg in configs)


def test_landing_page_no_naive_graph_shown_alone_not_side_by_side_with_real_graph():
    """Restructure decision: the real audio-embeddings graph and the audio
    demo moved to Results (evidence belongs there, once the mechanism has
    been explained) -- Overview shows only the metadata-baseline graph, and
    never the "naive"/old wording."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "metadata baseline" in caption_texts.lower()
    assert "naive" not in caption_texts.lower()
    assert "naive" not in markdown_texts.lower()
    assert "metadata baseline calls these" not in caption_texts.lower()
    assert "audio calls these" not in caption_texts.lower()
    assert len(at.get("audio")) == 0  # no demo-pair audio on Overview -- that's Results' job


def test_landing_page_has_tautology_callout_with_real_stats():
    """The metadata graph's clean clusters are a structural artifact of its
    edge definition, not evidence of quality -- must be called out
    explicitly with real, computed cross-genre-edge percentages, not left
    to silently imply the metadata baseline "worked better."."""
    at = _run_landing()
    warning_texts = " ".join(w.value for w in at.warning)
    assert "not evidence" in warning_texts.lower() or "guaranteed by construction" in warning_texts.lower()
    assert "%" in warning_texts


def test_landing_page_proposed_solution_is_visual_not_a_wordy_paragraph():
    """Section 3 illustrates the facet-based approach with a diagram rather
    than explaining it purely in prose -- kept visual-first, consistent with
    Overview's 'lightweight and highly visual' framing."""
    at = _run_landing()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "genre labels never enter this computation" in caption_texts.lower()
    assert len(at.get("plotly_chart")) == 4  # metadata + collaborative concept diagrams, the real graph, facets


def test_landing_page_links_to_approach_next():
    """Restructure decision: Overview hands off to the new Approach page
    (which picks up exactly where "Proposed solution" leaves off), not
    straight to Methodology."""
    at = _run_landing()
    write_texts = " ".join(m.value for m in at.markdown)
    assert "Approach" in write_texts
