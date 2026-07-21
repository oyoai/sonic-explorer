"""AppTest smoke test for the walkthrough landing page. Must go through
app.py + switch_page rather than AppTest.from_file on the page directly --
st.page_link needs the full multipage registry, which only exists when the
app is loaded from its root script (see streamlit_app/app.py)."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_walkthrough() -> AppTest:
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=120)
    at.switch_page("pages/0_Walkthrough.py")
    at.run()
    return at


def test_walkthrough_page_runs_without_exceptions():
    at = _run_walkthrough()
    assert not at.exception


def test_walkthrough_page_has_all_six_sections():
    at = _run_walkthrough()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. Data", "2. Facets", "3. Song DNA & fingerprints",
        "4. Retrieval", "5. Evaluation", "6. Explore the live app",
    ]:
        assert expected in header_texts
