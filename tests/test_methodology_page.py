"""AppTest smoke test for the Methodology page. Must go through Overview.py
+ switch_page rather than AppTest.from_file on the page directly --
st.page_link needs the full multipage registry, which only exists when the
app is loaded from its root script."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_methodology() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/1_Methodology.py")
    at.run()
    return at


@pytest.fixture(scope="module")
def methodology_at() -> AppTest:
    """Cold-starting the full multipage app is the expensive part of every
    test in this file; most of them only ever READ the resulting page
    (headers/captions/markdown text), never mutate a widget. Sharing one
    real render across those read-only tests (module-scoped: computed once,
    reused for the rest of this file) cuts ~13 redundant cold starts down
    to 1 without changing what any test actually asserts -- the two tests
    that DO interact with a picker widget (below) still get their own
    fresh, independent _run_methodology() instance, exactly as before, so
    mutating one never leaks into another test's expectations."""
    return _run_methodology()


def test_methodology_page_runs_without_exceptions(methodology_at):
    assert not methodology_at.exception


def test_methodology_page_has_all_nine_sections(methodology_at):
    """Restructure: a new Segmentation section, Structure split out of the
    old fingerprints section into its own, per-facet score distributions
    folded into the facets section, curated examples moved under the 2D map
    section, and a new Calibration/XAB methodology section -- Results still
    owns the actual regression numbers."""
    header_texts = [h.value for h in methodology_at.header]
    for expected in [
        "1. The dataset", "2. Segmentation", "3. The seven similarity facets",
        "4. Structure / Abstractivity", "5. Per-song artifacts",
        "6. The 2D map and axis interpretability", "7. Case studies",
        "8. Calibration / XAB methodology", "9. Next: Engineering",
    ]:
        assert expected in header_texts


def test_methodology_page_has_distribution_subsections(methodology_at):
    """Data-distribution content: raw-metadata EDA (duration, artists) and
    per-facet retrieval score distributions, not just curated examples."""
    subheader_texts = [s.value for s in methodology_at.subheader]
    assert any("Track duration" in s for s in subheader_texts)
    assert any("Artists" in s for s in subheader_texts)
    assert any("3a. Score distributions" in s for s in subheader_texts)


def test_methodology_page_has_case_study_subsections(methodology_at):
    """Each case study must follow the hypothesis -> test -> honest result
    pattern explicitly requested, not a simplified success summary."""
    subheader_texts = [s.value for s in methodology_at.subheader]
    assert any("7a. Vocal-facet cross-check" in s for s in subheader_texts)
    assert any("7b. Sound recognition" in s for s in subheader_texts)
    assert any("7c. Harmony whitening" in s for s in subheader_texts)
    assert any("7d. Song-level aggregation" in s for s in subheader_texts)
    assert any("7e. Does segment misalignment explain" in s for s in subheader_texts)
    assert any("7f. CLAP gain sensitivity" in s for s in subheader_texts)


def test_methodology_page_clap_gain_sensitivity_section_shows_real_measured_numbers(methodology_at):
    """Real regression coverage: the measured gain-sensitivity numbers (not
    just the section existing) must actually be on the page -- this is
    presented as a real empirical finding driving a real decision (loudness-
    normalize before perturbation testing), not just an implementation
    detail, so the numbers themselves need to be checkable, not asserted."""
    assert not methodology_at.exception
    caption_texts = " ".join(c.value for c in methodology_at.caption)
    assert "loudness-invariant at ±3dB" in caption_texts
    assert "0.70" in caption_texts  # the worst-case similarity at +-12dB
    info_texts = " ".join(i.value for i in methodology_at.info)
    assert "normalize_peak" in info_texts
    charts = methodology_at.get("plotly_chart")
    assert any("clap_gain_sensitivity_chart" in c.proto.id for c in charts)


def test_methodology_page_renders_waffle_grid_without_exception(methodology_at):
    """The library snapshot moved here from Overview -- must render against
    the real repositories with no exception (no typed plotly_chart accessor
    to assert on more directly here)."""
    assert not methodology_at.exception


def test_methodology_page_genre_caption_is_computed_not_asserted(methodology_at):
    """Regression guard for a real bug: the old caption asserted a fixed
    "not evenly represented" claim that didn't match an evenly-stratified
    deployed subset. The caption must now name the real largest/smallest
    genre and their real counts, computed live."""
    caption_texts = " ".join(c.value for c in methodology_at.caption)
    assert "leads at" in caption_texts or "genre-stratified sample" in caption_texts


def test_methodology_page_segmentation_section_has_dynamic_segment_count(methodology_at):
    """The segment-count number must come from song_repo.count_segments(),
    not a hardcoded figure that would mismatch a different-sized library."""
    write_texts = " ".join(m.value for m in methodology_at.markdown)
    assert "segments" in write_texts.lower()
    assert "computed live" in write_texts.lower()


def test_methodology_page_dna_example_is_computed_dynamically_not_missing(methodology_at):
    """Regression test for a real deployed-app bug: 5a used to reference two
    hardcoded song titles that didn't exist in the smaller deploy_data
    subset, showing a literal warning instead of the DNA comparison. 5a now
    computes its two examples from whatever library is actually loaded, so
    this must never show that warning against the real repositories."""
    warning_texts = " ".join(w.value for w in methodology_at.warning)
    assert "not enough songs with computed dna" not in warning_texts.lower()

    markdown_texts = " ".join(m.value for m in methodology_at.markdown)
    assert "Slowest / calmest" in markdown_texts
    assert "Fastest / most energetic" in markdown_texts


def test_methodology_page_fingerprint_picker_switches_song():
    """The fingerprint picker must actually re-render for a different
    selection rather than silently no-op. Needs its own fresh instance
    (not the shared methodology_at fixture above) -- it mutates a widget
    and re-runs, which would leak into every other test sharing that
    instance."""
    at = _run_methodology()
    picker = at.selectbox(key="walkthrough_fp_picker")
    other_option = next(o for o in picker.options if o != picker.value)
    picker.select(other_option).run()
    assert not at.exception
    assert picker.value == other_option


def test_methodology_page_structure_picker_is_independent_of_fingerprint_picker():
    """Restructure decision: Structure got split out into its own section
    (4) with its own picker, separate from fingerprints/description/tags
    (5b) -- not one shared widget across two sections. Needs its own fresh
    instance for the same reason as the test above."""
    at = _run_methodology()
    structure_picker = at.selectbox(key="structure_picker")
    fp_picker = at.selectbox(key="walkthrough_fp_picker")
    assert structure_picker.options
    assert fp_picker.options

    other_option = next(o for o in structure_picker.options if o != structure_picker.value)
    structure_picker.select(other_option).run()
    assert not at.exception


def test_methodology_page_has_inline_citations_not_a_dedicated_section(methodology_at):
    """Restructure decision: no dedicated Related Work section (that would
    visually imply this project was inspired by the cited papers, which
    isn't true) -- citations are woven in locally at the specific points
    where the approach happens to parallel the cited work."""
    body_text = " ".join(m.value for m in methodology_at.markdown)
    header_texts = " ".join(h.value for h in methodology_at.header) + " ".join(
        s.value for s in methodology_at.subheader
    )
    assert "Related work" not in header_texts
    assert "Tovstogan" in body_text
    assert "Vohra" in body_text
    assert "VidTune" in body_text


def test_methodology_page_has_calibration_xab_methodology_not_outcomes(methodology_at):
    """Section 8 describes the XAB rating methodology (format, why chosen,
    sampling approach) -- the actual regression results/numbers stay in
    Results, not duplicated here."""
    body_text = " ".join(m.value for m in methodology_at.markdown)
    assert "XAB" in body_text or "forced binary discrimination" in body_text.lower()
    assert "350" in body_text


def test_methodology_page_names_sound_only_sampling_as_a_known_limitation(methodology_at):
    """Real methodological honesty check: the main calibration pool only
    ever samples candidates via Sound-facet retrieval, a real coverage bias
    (see calibration_triplets.py / pages/9_Calibration.py) -- this must be
    named explicitly, not left implicit in "currently calibrating Sound."""
    warning_texts = " ".join(w.value for w in methodology_at.warning)
    assert "coverage bias" in warning_texts.lower()
    assert "harmony" in warning_texts.lower()


def test_methodology_page_links_to_results_next(methodology_at):
    body_text = " ".join(m.value for m in methodology_at.markdown)
    assert "Results" in body_text
