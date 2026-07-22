"""AppTest smoke test for the Results page -- split out of Methodology so
process (how the library was analyzed and improved) and outcome (the
evaluation numbers) live on separate pages. Must go through app.py +
switch_page -- st.page_link needs the full multipage registry, which only
exists when the app is loaded from its root script."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_results() -> AppTest:
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=120)
    at.switch_page("pages/1_Results.py")
    at.run()
    return at


def test_results_page_runs_without_exceptions():
    at = _run_results()
    assert not at.exception


def test_results_page_has_all_three_sections():
    at = _run_results()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. Genre-cohesion evaluation",
        "2. Genre classifier baseline (CNN)",
        "3. Calibration study & blend-weight regression",
    ]:
        assert expected in header_texts


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
