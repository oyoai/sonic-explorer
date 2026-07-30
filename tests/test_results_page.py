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

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_results() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/2_Results.py")
    at.run()
    return at


def test_results_page_runs_without_exceptions():
    at = _run_results()
    assert not at.exception


def test_results_page_has_all_four_tabs_in_order():
    at = _run_results()
    assert len(at.tabs) == 4
    assert "Genre-cohesion outcome" in [h.value for h in at.tabs[0].header]
    assert "Calibration study & blend-weight regression" in [h.value for h in at.tabs[1].header]
    assert "Ask the DJ Gallery" in [h.value for h in at.tabs[2].header]
    assert "Metadata baseline vs. real audio similarity" in [h.value for h in at.tabs[3].header]


def test_results_page_facet_evaluation_tab_has_score_distributions():
    """Score distributions moved here from Methodology -- the methodology
    itself (why this check matters) stays on Methodology; the actual sampled
    numbers/histograms live here."""
    at = _run_results()
    header_texts = [h.value for h in at.tabs[0].header]
    assert "Score distributions across the whole library" in header_texts


def test_results_page_comparison_tab_has_metadata_vs_real_graph_and_audio_demo():
    """The real audio-embeddings network graph and the audio-playback demo
    (metadata-pair vs. real-cross-genre-pair) both moved here from Overview --
    must render with real audio players, not just descriptive text."""
    at = _run_results()
    caption_texts = " ".join(c.value for c in at.tabs[3].caption)
    assert "audio embeddings" in caption_texts.lower()
    assert "metadata baseline calls these" in caption_texts.lower()
    assert "audio calls these" in caption_texts.lower()
    assert len(at.tabs[3].get("audio")) >= 4  # two songs per pair, two pairs


def test_results_page_calibration_tab_reports_honestly_either_way():
    """This section is computed LIVE from whatever's actually in
    calibration_ratings (sonic_explorer.evaluation.blend_weight_regression),
    not a hardcoded/precomputed number -- so which of these two honest
    states it shows depends on real, currently-mutable local DB state, not
    something this test can assume a fixed outcome for. Either "no ratings
    yet" (a warning) or "N rating(s) from M rater(s)" (an info box stating
    the real sample size) is correct; showing neither, or a fabricated
    number with no ratings behind it, would not be."""
    at = _run_results()
    warning_texts = [w.value for w in at.tabs[1].warning]
    info_texts = [i.value for i in at.tabs[1].info]
    no_ratings_yet = any("no ratings yet" in w.lower() for w in warning_texts)
    has_real_data = any("rating(s) from" in i.lower() and "rater(s)" in i.lower() for i in info_texts)
    assert no_ratings_yet or has_real_data


def test_results_page_calibration_tab_shows_agreement_and_regression_sections_when_data_exists():
    """When real ratings do exist, both the per-facet agreement rate and
    the blend-weight regression subsections must render something (a real
    chart/table, or -- for the regression specifically, which needs many
    more ratings than a handful to fit at all -- an honest "too few
    ratings" note) rather than either crashing or silently showing
    nothing."""
    at = _run_results()
    info_texts = [i.value for i in at.tabs[1].info]
    has_real_data = any("rating(s) from" in i.lower() and "rater(s)" in i.lower() for i in info_texts)
    if not has_real_data:
        return  # nothing to check further -- the honest "no ratings yet" test above covers this state
    markdown_texts = " ".join(m.value for m in at.tabs[1].markdown)
    assert "Per-facet agreement rate" in markdown_texts
    assert "Blend-weight regression" in markdown_texts


def test_results_page_dj_gallery_tab_includes_sound_recognition_query():
    """The gallery must include a sound-recognition-specific example
    (search_by_sound_content), not just typical mood/genre queries."""
    at = _run_results()
    button_labels = " ".join(b.label for b in at.tabs[2].button)
    assert "crow" in button_labels.lower()


def test_results_page_has_future_work_note_on_three_signal_types():
    """Honest acknowledgment that an ideal fuller system would combine
    metadata, audio-based similarity, and user-behavioral signals -- not
    something this project implements, just a brief scope disclosure."""
    at = _run_results()
    info_texts = " ".join(i.value for i in at.info)
    assert "future work" in info_texts.lower()
    assert "user-level" in info_texts.lower() or "behavioral" in info_texts.lower()
