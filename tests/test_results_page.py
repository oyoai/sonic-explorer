"""AppTest smoke test for the Results page -- split out of Methodology so
process (how the library was analyzed and improved) and outcome (the
evaluation numbers) live on separate pages. Must go through Overview.py +
switch_page -- nav_button()'s st.switch_page() needs the full multipage
registry, which only exists when the app is loaded from its root script.

Results is organized into tabs (Facet Evaluation, Calibration & Blend-Weights,
Ask the DJ Gallery, Metadata baseline vs. Real approach, in that order --
comparison last, as the payoff rather than the opener). Streamlit's AppTest
executes every tab's content regardless of which tab is visually selected
(tabs are a layout container, not lazy-loaded), so all four are queryable via
at.tabs[i]."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_results() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/2_Results.py")
    at.run()
    return at


@pytest.fixture(scope="module")
def results_at() -> AppTest:
    """Every test in this file only reads the default render (no widget
    interaction) -- one shared cold start instead of 8 redundant ones."""
    return _run_results()


def test_results_page_runs_without_exceptions(results_at):
    assert not results_at.exception


def test_results_page_has_all_four_tabs_in_order(results_at):
    assert len(results_at.tabs) == 4
    assert "Genre-cohesion outcome" in [h.value for h in results_at.tabs[0].header]
    assert "Calibration study & blend-weight regression" in [h.value for h in results_at.tabs[1].header]
    assert "Ask the DJ Gallery" in [h.value for h in results_at.tabs[2].header]
    assert "Metadata baseline vs. real audio similarity" in [h.value for h in results_at.tabs[3].header]


def test_results_page_facet_evaluation_tab_has_score_distributions(results_at):
    """Score distributions moved here from Methodology -- the methodology
    itself (why this check matters) stays on Methodology; the actual sampled
    numbers/histograms live here."""
    header_texts = [h.value for h in results_at.tabs[0].header]
    assert "Score distributions across the whole library" in header_texts


def test_results_page_comparison_tab_has_metadata_vs_real_graph_and_audio_demo(results_at):
    """The real audio-embeddings network graph and the audio-playback demo
    (metadata-pair vs. real-cross-genre-pair) both moved here from Overview --
    must render with real audio players, not just descriptive text."""
    caption_texts = " ".join(c.value for c in results_at.tabs[3].caption)
    assert "audio embeddings" in caption_texts.lower()
    assert "metadata baseline calls these" in caption_texts.lower()
    assert "audio calls these" in caption_texts.lower()
    assert len(results_at.tabs[3].get("audio")) >= 4  # two songs per pair, two pairs


def _has_blend_real_data(info_texts: list[str]) -> bool:
    """The blend-weight info box's own text, distinguished from the taste
    section's near-identical phrasing below by the absence of the word
    "taste" -- both sections say "N rating(s) from M rater(s)", so a plain
    substring check would conflate them."""
    return any("rating(s) from" in i.lower() and "rater(s)" in i.lower() and "taste" not in i.lower() for i in info_texts)


def _has_taste_real_data(info_texts: list[str]) -> bool:
    return any("taste rating(s) from" in i.lower() and "rater(s)" in i.lower() for i in info_texts)


def test_results_page_calibration_tab_reports_honestly_either_way(results_at):
    """This section is computed LIVE from whatever's actually in
    calibration_ratings (sonic_explorer.evaluation.blend_weight_regression),
    not a hardcoded/precomputed number -- so which of these two honest
    states it shows depends on real, currently-mutable local DB state, not
    something this test can assume a fixed outcome for. Either "no ratings
    yet" (a warning) or "N rating(s) from M rater(s)" (an info box stating
    the real sample size) is correct; showing neither, or a fabricated
    number with no ratings behind it, would not be."""
    warning_texts = [w.value for w in results_at.tabs[1].warning]
    info_texts = [i.value for i in results_at.tabs[1].info]
    no_ratings_yet = any("no ratings yet" in w.lower() and "taste" not in w.lower() for w in warning_texts)
    assert no_ratings_yet or _has_blend_real_data(info_texts)


def test_results_page_calibration_tab_shows_agreement_and_regression_sections_when_data_exists(results_at):
    """When real ratings do exist, both the per-facet agreement rate and
    the blend-weight regression subsections must render something (a real
    chart/table, or -- for the regression specifically, which needs many
    more ratings than a handful to fit at all -- an honest "too few
    ratings" note) rather than either crashing or silently showing
    nothing."""
    info_texts = [i.value for i in results_at.tabs[1].info]
    if not _has_blend_real_data(info_texts):
        return  # nothing to check further -- the honest "no ratings yet" test above covers this state
    markdown_texts = " ".join(m.value for m in results_at.tabs[1].markdown)
    assert "Per-facet agreement rate" in markdown_texts
    assert "Blend-weight regression" in markdown_texts


def test_results_page_calibration_tab_taste_section_reports_honestly_either_way(results_at):
    """Groundwork section, added alongside the similarity blend-weight
    regression -- same live-computation, same honest either/or pattern,
    computed from taste_ratings (sonic_explorer.evaluation.
    taste_weight_regression) rather than a placeholder."""
    warning_texts = [w.value for w in results_at.tabs[1].warning]
    info_texts = [i.value for i in results_at.tabs[1].info]
    no_taste_ratings_yet = any("no taste ratings yet" in w.lower() for w in warning_texts)
    assert no_taste_ratings_yet or _has_taste_real_data(info_texts)


def test_results_page_calibration_tab_taste_section_shows_a_regression_or_honest_note_when_data_exists(results_at):
    info_texts = [i.value for i in results_at.tabs[1].info]
    if not _has_taste_real_data(info_texts):
        return  # nothing to check further -- the honest "no taste ratings yet" test above covers this state
    markdown_texts = " ".join(m.value for m in results_at.tabs[1].markdown)
    caption_texts = " ".join(c.value for c in results_at.tabs[1].caption)
    assert "does a song's own dna predict liked vs. disliked" in markdown_texts.lower()
    # Either a real fitted regression chart, or an honest "too few"/"same
    # side" note -- never silently nothing.
    has_regression_chart = "corpus-normalized to [0, 1]" in caption_texts
    has_honest_note = "too few" in caption_texts.lower() or "same" in caption_texts.lower()
    assert has_regression_chart or has_honest_note


def test_results_page_dj_gallery_tab_includes_sound_recognition_query(results_at):
    """The gallery must include a sound-recognition-specific example
    (search_by_sound_content), not just typical mood/genre queries."""
    button_labels = " ".join(b.label for b in results_at.tabs[2].button)
    assert "crow" in button_labels.lower()


def test_results_page_has_future_work_note_on_three_signal_types(results_at):
    """Honest acknowledgment that an ideal fuller system would combine
    metadata, audio-based similarity, and user-behavioral signals -- not
    something this project implements, just a brief scope disclosure."""
    info_texts = " ".join(i.value for i in results_at.info)
    assert "future work" in info_texts.lower()
    assert "user-level" in info_texts.lower() or "behavioral" in info_texts.lower()
