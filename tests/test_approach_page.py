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
        "0. First, get baseline",
        "1. Slicing the track into windows",
        "2. Seven ways of listening",
        "3. Turning sound into an embedding",
        "4. A second kind of similarity: Song DNA",
        "5. Explaining it in plain language",
    ]:
        assert expected in header_texts


def test_approach_page_baseline_discloses_no_collaborative_filtering():
    at = _run_approach()
    info_texts = " ".join(i.value for i in at.info)
    assert "collaborative filtering isn't attempted here" in info_texts.lower()


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


def test_approach_page_step2_harmony_shows_estimated_key_and_chord_strip():
    """The chromagram alone isn't legible without real music-theory
    literacy -- key and chord estimates (Krumhansl-Schmuckler + template
    matching, both computed fresh on the same chroma) must be shown
    alongside it, not just the raw heatmap."""
    at = _run_approach()
    body_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "estimated key" in body_text.lower()
    assert "krumhansl-schmuckler" in body_text.lower()


def test_approach_page_step2_harmony_seek_uses_the_chord_strip_not_the_heatmap():
    """Real bug fix: go.Heatmap doesn't fire a click-selection event the way
    Bar/Scatter traces do, so the chromagram itself can't drive seek -- the
    page must say so honestly and point at the chord strip as the real
    control instead of silently claiming a broken interaction works."""
    at = _run_approach()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "click a chord segment" in caption_texts.lower()
    assert "don't support click-to-select" in caption_texts.lower()


def test_approach_page_step2_sound_shows_mel_spectrogram_as_supporting_visual_only():
    """Explicit framing required: the mel-spectrogram is context (the raw
    texture Sound is computed from), not a view of CLAP's real embedding --
    and the page must point back at the real way to verify Sound similarity
    (paired comparison), not let the spectrogram stand in for it."""
    at = _run_approach()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "mel-spectrogram" in caption_texts.lower()
    assert "not a view of what clap actually" in caption_texts.lower()
    assert "paired comparison" in caption_texts.lower()


def test_approach_page_step2_stem_facets_use_real_cdoc_stems():
    """Real isolated stem audio for "Freak of Nature (Time Out Dubb)"
    (streamlit_app/static/stem_example/, produced locally via the real
    separate_stems() pipeline, chosen over flekkefjord for having clearly
    audible/distinct vocal/drums/bass) now exists, and it's the same song as
    the pinned demo_song -- Sound, Harmony, and the four stem facets all use
    real audio from one song, not the old honest placeholder this showed
    before extraction."""
    at = _run_approach()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "freak of nature" in caption_texts.lower()
    assert "same song" in caption_texts.lower()
    info_texts = " ".join(i.value for i in at.info)
    assert "placeholder" not in info_texts.lower()


def test_approach_page_step2_sound_tags_shows_tag_list_not_waveform():
    """Sound Tags isn't an amplitude-over-time measurement like the other six
    facets -- it must show a labeled list of real detected tags, computed by
    AST + CLAP's text encoder, not a fabricated waveform chart."""
    at = _run_approach()
    body_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "runs on: full mix. computed by: ast" in body_text.lower()
    assert "hip hop music" in body_text.lower() or "rapping" in body_text.lower()


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


def test_approach_page_demo_song_is_pinned_to_freak_of_nature():
    """The demo song threading through steps 1-5 is deliberately pinned to
    "Freak of Nature (Time Out Dubb)" by C-Doc (not the old dynamic
    real_pair-derived pick, and not flekkefjord -- reassigned to Results'
    Ask the DJ Gallery's crow-sound example instead, since Approach's Step 2
    plays each stem separately and needs clearly audible vocal/drums/bass,
    which flekkefjord's sparse/ambient stems didn't have). Safe to pin here
    (unlike a normal fixed title) because it's force-included in
    scripts/build_deploy_subset.py's REQUIRED_EXAMPLE_TITLES, so it's
    guaranteed present in the deployed subset too."""
    at = _run_approach()
    body_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "freak of nature" in body_text.lower()
