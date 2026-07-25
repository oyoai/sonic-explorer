"""AppTest smoke test for the Methodology page. Must go through Overview.py
+ switch_page rather than AppTest.from_file on the page directly --
st.page_link needs the full multipage registry, which only exists when the
app is loaded from its root script."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_methodology() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/1_Methodology.py")
    at.run()
    return at


def test_methodology_page_runs_without_exceptions():
    at = _run_methodology()
    assert not at.exception


def test_methodology_page_has_all_nine_sections():
    """Restructure: a new Segmentation section, Structure split out of the
    old fingerprints section into its own, per-facet score distributions
    folded into the facets section, curated examples moved under the 2D map
    section, and a new Calibration/XAB methodology section -- Results still
    owns the actual regression numbers."""
    at = _run_methodology()
    header_texts = [h.value for h in at.header]
    for expected in [
        "1. The dataset", "2. Segmentation", "3. The six similarity facets",
        "4. Structure / Abstractivity", "5. Per-song artifacts",
        "6. The 2D map and axis interpretability", "7. Case studies",
        "8. Calibration / XAB methodology", "9. Next: Results",
    ]:
        assert expected in header_texts


def test_methodology_page_has_distribution_subsections():
    """Data-distribution content: raw-metadata EDA (duration, artists) and
    per-facet retrieval score distributions, not just curated examples."""
    at = _run_methodology()
    subheader_texts = [s.value for s in at.subheader]
    assert any("Track duration" in s for s in subheader_texts)
    assert any("Artists" in s for s in subheader_texts)
    assert any("3a. Score distributions" in s for s in subheader_texts)


def test_methodology_page_has_case_study_subsections():
    """Each case study must follow the hypothesis -> test -> honest result
    pattern explicitly requested, not a simplified success summary."""
    at = _run_methodology()
    subheader_texts = [s.value for s in at.subheader]
    assert any("7a. Vocal-facet cross-check" in s for s in subheader_texts)
    assert any("7b. Sound recognition" in s for s in subheader_texts)
    assert any("7c. Harmony whitening" in s for s in subheader_texts)
    assert any("7d. Song-level aggregation" in s for s in subheader_texts)
    assert any("7e. Does segment misalignment explain" in s for s in subheader_texts)


def test_methodology_page_renders_waffle_grid_without_exception():
    """The library snapshot moved here from Overview -- must render against
    the real repositories with no exception (no typed plotly_chart accessor
    to assert on more directly here)."""
    at = _run_methodology()
    assert not at.exception


def test_methodology_page_genre_caption_is_computed_not_asserted():
    """Regression guard for a real bug: the old caption asserted a fixed
    "not evenly represented" claim that didn't match an evenly-stratified
    deployed subset. The caption must now name the real largest/smallest
    genre and their real counts, computed live."""
    at = _run_methodology()
    caption_texts = " ".join(c.value for c in at.caption)
    assert "leads at" in caption_texts or "genre-stratified sample" in caption_texts


def test_methodology_page_segmentation_section_has_dynamic_segment_count():
    """The segment-count number must come from song_repo.count_segments(),
    not a hardcoded figure that would mismatch a different-sized library."""
    at = _run_methodology()
    write_texts = " ".join(m.value for m in at.markdown)
    assert "segments" in write_texts.lower()
    assert "computed live" in write_texts.lower()


def test_methodology_page_dna_example_is_computed_dynamically_not_missing():
    """Regression test for a real deployed-app bug: 5a used to reference two
    hardcoded song titles that didn't exist in the smaller deploy_data
    subset, showing a literal warning instead of the DNA comparison. 5a now
    computes its two examples from whatever library is actually loaded, so
    this must never show that warning against the real repositories."""
    at = _run_methodology()
    warning_texts = " ".join(w.value for w in at.warning)
    assert "not enough songs with computed dna" not in warning_texts.lower()

    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "Slowest / calmest" in markdown_texts
    assert "Fastest / most energetic" in markdown_texts


def test_methodology_page_fingerprint_picker_switches_song():
    """The fingerprint picker must actually re-render for a different
    selection rather than silently no-op."""
    at = _run_methodology()
    picker = at.selectbox(key="walkthrough_fp_picker")
    other_option = next(o for o in picker.options if o != picker.value)
    picker.select(other_option).run()
    assert not at.exception
    assert picker.value == other_option


def test_methodology_page_structure_picker_is_independent_of_fingerprint_picker():
    """Restructure decision: Structure got split out into its own section
    (4) with its own picker, separate from fingerprints/description/tags
    (5b) -- not one shared widget across two sections."""
    at = _run_methodology()
    structure_picker = at.selectbox(key="structure_picker")
    fp_picker = at.selectbox(key="walkthrough_fp_picker")
    assert structure_picker.options
    assert fp_picker.options

    other_option = next(o for o in structure_picker.options if o != structure_picker.value)
    structure_picker.select(other_option).run()
    assert not at.exception


def test_methodology_page_has_inline_citations_not_a_dedicated_section():
    """Restructure decision: no dedicated Related Work section (that would
    visually imply this project was inspired by the cited papers, which
    isn't true) -- citations are woven in locally at the specific points
    where the approach happens to parallel the cited work."""
    at = _run_methodology()
    body_text = " ".join(m.value for m in at.markdown)
    header_texts = " ".join(h.value for h in at.header) + " ".join(s.value for s in at.subheader)
    assert "Related work" not in header_texts
    assert "Tovstogan" in body_text
    assert "Vohra" in body_text
    assert "VidTune" in body_text


def test_methodology_page_has_calibration_xab_methodology_not_outcomes():
    """Section 8 describes the XAB rating methodology (format, why chosen,
    sampling approach) -- the actual regression results/numbers stay in
    Results, not duplicated here."""
    at = _run_methodology()
    body_text = " ".join(m.value for m in at.markdown)
    assert "XAB" in body_text or "forced binary discrimination" in body_text.lower()
    assert "350" in body_text


def test_methodology_page_links_to_results_next():
    at = _run_methodology()
    body_text = " ".join(m.value for m in at.markdown)
    assert "Results" in body_text
