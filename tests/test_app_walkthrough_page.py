"""AppTest smoke test for the App Walkthrough page (interprets the live app's
own views, distinct from Methodology's analysis narrative). Must go through
app.py + switch_page -- st.page_link needs the full multipage registry, which
only exists when the app is loaded from its root script."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_app_walkthrough() -> AppTest:
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=120)
    at.switch_page("pages/1_App_Walkthrough.py")
    at.run()
    return at


def test_app_walkthrough_page_runs_without_exceptions():
    at = _run_app_walkthrough()
    assert not at.exception


def test_app_walkthrough_page_has_all_six_sections():
    at = _run_app_walkthrough()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. Explore -- the network graph", "2. Taste Map -- the whole library, projected",
        "3. Song X-Ray -- one song's anatomy", "4. Moment Matcher -- finding a match, one moment at a time",
        "5. Ask the DJ -- a conversational front-end", "6. All live pages",
    ]:
        assert expected in header_texts
