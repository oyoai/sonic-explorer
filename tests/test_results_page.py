"""AppTest smoke test for the Results page -- split out of Methodology so
process (how the library was analyzed and improved) and outcome (the
evaluation numbers) live on separate pages. Must go through Overview.py +
switch_page -- nav_button()'s st.switch_page() needs the full multipage
registry, which only exists when the app is loaded from its root script."""

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


def test_results_page_has_all_four_sections():
    """Restructure: the naive-vs-real audio comparison (with its audio demo)
    moved here from Overview as the new section 1 -- Overview raises the
    question, Results is where the evidence for an answer belongs, once
    Approach/Methodology have explained the mechanism."""
    at = _run_results()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. Naive baseline vs. real audio similarity",
        "2. Genre-cohesion evaluation",
        "3. Genre classifier baseline (CNN)",
        "4. Calibration study & blend-weight regression",
    ]:
        assert expected in header_texts


def test_results_page_has_naive_vs_real_graph_and_audio_demo():
    """The real audio-embeddings network graph and the audio-playback demo
    (naive-pair vs. real-cross-genre-pair) both moved here from Overview --
    must render with real audio players, not just descriptive text."""
    at = _run_results()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "audio embeddings" in caption_texts.lower()
    assert "naive calls these" in caption_texts.lower()
    assert "audio calls these" in caption_texts.lower()
    assert len(at.get("audio")) >= 4  # two songs per pair, two pairs


def test_results_page_reports_cnn_accuracy_against_random_baseline():
    at = _run_results()
    metric_labels = [m.label for m in at.metric]
    assert "Test accuracy" in metric_labels
    assert "Random baseline" in metric_labels


def test_results_page_calibration_section_is_honest_about_pending_status():
    """No fabricated numbers -- calibration data collection hadn't produced
    any ratings yet when this page was built, so the section must say so."""
    at = _run_results()
    warning_texts = [w.value for w in at.warning]
    assert any("no results yet" in w.lower() for w in warning_texts)
