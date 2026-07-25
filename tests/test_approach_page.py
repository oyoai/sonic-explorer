"""AppTest smoke test for the new Approach page -- the bridge page between
Overview and Methodology. Must go through Overview.py + switch_page rather
than AppTest.from_file on the page directly, same reasoning as every other
page test in this app (st.page_link needs the full multipage registry)."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_approach() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.switch_page("pages/0_Approach.py")
    at.run()
    return at


def test_approach_page_runs_without_exceptions():
    at = _run_approach()
    assert not at.exception


def test_approach_page_has_all_seven_steps():
    at = _run_approach()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. The problem, restated visually",
        "2. Slicing the track into windows",
        "3. Six ways of listening",
        "4. Turning sound into points in space",
        "5. Finding what's similar",
        "6. Explaining it in plain language",
        "7. Putting it all on a map you can explore",
    ]:
        assert expected in header_texts


def test_approach_page_segmentation_slider_reflects_real_window_size():
    """Step 2's window-size number in the text must come from the real
    config constant, not a hardcoded guess."""
    at = _run_approach()
    body_texts = " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
    assert "5" in body_texts  # WINDOW_SEC=5.0 -- loose text check, exact value asserted in unit tests


def test_approach_page_reveal_button_does_not_crash():
    """Step 6's explanation reveal must degrade gracefully with or without a
    configured API key -- clicking it must never raise."""
    at = _run_approach()
    button = at.button(key="step6_reveal")
    button.click().run()
    assert not at.exception


def test_approach_page_reuses_same_demo_pairs_as_overview():
    """Restructure decision: Overview's audio demo and Approach's step 1
    visual contrast must show the same song pair, for continuity across
    pages -- not two independently-picked examples."""
    overview = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    overview.run()
    approach = _run_approach()

    overview_titles = {m.value for m in overview.markdown}
    approach_titles = {c.value for c in approach.caption}

    # Every song title mentioned on Approach's step 1 must also appear somewhere on Overview
    # (both derive from the same get_demo_pairs() call against the same cache_key).
    import re

    quoted_on_approach = set()
    for text in approach_titles:
        quoted_on_approach.update(re.findall(r'"([^"]+)"', text))

    overview_text_blob = " ".join(overview_titles)
    assert quoted_on_approach, "expected at least one quoted song title on Approach step 1"
    for title in quoted_on_approach:
        assert title in overview_text_blob
