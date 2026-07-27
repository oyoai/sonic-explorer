"""AppTest smoke test for the Approach page -- the bridge page between
Overview and Methodology. Must go through Overview.py + switch_page rather
than AppTest.from_file on the page directly, same reasoning as every other
page test in this app (nav_button()'s st.switch_page() needs the full
multipage registry)."""

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


def test_approach_page_has_baseline_and_all_five_steps():
    at = _run_approach()
    header_texts = [h.value for h in at.header]
    for expected in [
        "0. Baseline",
        "1. Slicing the track into windows",
        "2. Seven ways of listening",
        "3. Turning sound into an embedding",
        "4. A second kind of similarity: Song DNA",
        "5. Explaining it in plain language",
    ]:
        assert expected in header_texts


def test_approach_page_baseline_discloses_no_collaborative_filtering():
    at = _run_approach()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "collaborative filtering isn't attempted here" in caption_texts.lower()


def test_approach_page_step1_offers_real_playable_window_clips():
    """Step 1 must give actual separate playable ~5s clips, not just a
    slider highlighting a region of the continuous full-song player."""
    at = _run_approach()
    body_texts = " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
    assert "5" in body_texts  # WINDOW_SEC=5.0 -- loose text check, exact value asserted in unit tests
    assert len(at.get("audio")) >= 4  # at least the window clips themselves


def test_approach_page_step2_shows_all_seven_facets_with_descriptions():
    at = _run_approach()
    caption_texts = " ".join(c.value for c in at.caption)
    for desc_fragment in [
        "Overall timbre, instrumentation, production character",
        "Key, chords, tonal color",
        "Isolated voice timbre and delivery",
        "Isolated drum/percussion pattern and timbre",
        "Isolated bassline tone and pattern",
        "Backing instrumentation with vocals removed",
    ]:
        assert desc_fragment in caption_texts


def test_approach_page_step2_sound_tags_facet_reports_real_index_as_live():
    """Notebook 11's per-segment sound_tags index has actually finished and been
    synced -- this must report it as live with real measured evidence
    (genre-cohesion numbers), not a stale 'pending/in progress' placeholder."""
    at = _run_approach()
    success_texts = " ".join(s.value for s in at.success)
    assert "sound tags is the 7th facet" in success_texts.lower()
    assert "45.7%" in success_texts
    assert "pending" not in success_texts.lower()


def test_approach_page_step2_stem_facets_degrade_to_honest_placeholder_when_missing():
    """Real isolated stem audio doesn't exist in this test environment --
    the four stem facets must say so plainly, not silently omit themselves
    or fabricate a substitute."""
    at = _run_approach()
    info_texts = " ".join(i.value for i in at.info)
    assert "placeholder" in info_texts.lower()
    assert "isolated stem audio" in info_texts.lower()


def test_approach_page_step3_has_embedding_strip_and_close_by_illustration():
    at = _run_approach()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "512-dimensional space" in caption_texts
    assert "illustrative, not real data" in caption_texts.lower()


def test_approach_page_step4_has_song_dna_and_states_the_averaging_limitation():
    at = _run_approach()
    warning_texts = " ".join(w.value for w in at.warning)
    assert "computed once per whole song, not per moment" in warning_texts
    assert "quiet intro and a loud chorus" in warning_texts


def test_approach_page_step5_shows_expandable_real_tags():
    at = _run_approach()
    expander_labels = [e.label for e in at.expander]
    assert any("Detected tags for" in label for label in expander_labels)


def test_approach_page_reveal_button_does_not_crash():
    """Step 5's explanation reveal must degrade gracefully with or without a
    configured API key -- clicking it must never raise."""
    at = _run_approach()
    button = at.button(key="step5_reveal")
    button.click().run()
    assert not at.exception


def test_approach_page_reuses_same_demo_song_as_results_pairs():
    """The demo song threading through steps 1-5 must be picked dynamically
    from the same get_demo_pairs() call Results/Overview already use, not a
    hardcoded title -- a fixed title could easily not exist in a smaller
    deployed subset (this project has hit that exact bug before)."""
    results = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    results.switch_page("pages/2_Results.py")
    results.run()
    approach = _run_approach()

    results_text_blob = " ".join(m.value for m in results.markdown)
    approach_text_blob = " ".join(m.value for m in approach.markdown) + " ".join(c.value for c in approach.caption)

    import re

    quoted_on_approach = set(re.findall(r'"([^"]+)"', approach_text_blob))
    assert quoted_on_approach, "expected at least one quoted song title on Approach"
    assert any(title in results_text_blob for title in quoted_on_approach)
