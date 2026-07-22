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


def test_landing_page_has_naive_baseline_and_related_work_placeholders():
    at = _run_landing()
    subheader_texts = [s.value for s in at.subheader]
    assert any("naive approach" in s for s in subheader_texts)
    assert any("Related work" in s for s in subheader_texts)
