"""AppTest smoke test for the Methodology landing page. Must go through
app.py + switch_page rather than AppTest.from_file on the page directly --
st.page_link needs the full multipage registry, which only exists when the
app is loaded from its root script (see streamlit_app/app.py)."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_methodology() -> AppTest:
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=120)
    at.switch_page("pages/0_Methodology.py")
    at.run()
    return at


def test_methodology_page_runs_without_exceptions():
    at = _run_methodology()
    assert not at.exception


def test_methodology_page_has_all_seven_sections():
    at = _run_methodology()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. Data", "2. Facets", "3. Song DNA & fingerprints",
        "4. Taste Map -- the whole library at once",
        "5. Retrieval", "6. Evaluation", "7. Next: see it in the app",
    ]:
        assert expected in header_texts


def test_methodology_page_fingerprint_picker_switches_song():
    """The fingerprint picker (added so users can browse multiple structure
    examples, not just one hardcoded case) must actually re-render for a
    different selection rather than silently no-op."""
    at = _run_methodology()
    picker = at.selectbox(key="walkthrough_fp_picker")
    other_option = next(o for o in picker.options if o != picker.value)
    picker.select(other_option).run()
    assert not at.exception
    assert picker.value == other_option
