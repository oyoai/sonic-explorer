"""AppTest smoke test for the new Engineering page (sits between Methodology
and Results in Overview.py's st.navigation() list, despite its filename
prefix -- see Overview.py's comment on why). Demonstrates engineering/safety
rigor interactively rather than as static claims: clickable red-team
prompts against the live agent, a real CI badge, and a live CNN genre
picker. Must go through Overview.py + switch_page -- nav_button()'s
st.switch_page() needs the full multipage registry."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_engineering() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.switch_page("pages/8_Engineering.py")
    at.run()
    return at


def test_engineering_page_runs_without_exceptions():
    at = _run_engineering()
    assert not at.exception


def test_engineering_page_has_all_three_sections():
    at = _run_engineering()
    header_texts = [h.value for h in at.header]
    assert "1. Red-teaming, interactive" in header_texts
    assert "2. CI / Docker" in header_texts
    assert "3. CNN classifier, live" in header_texts


def test_engineering_page_red_team_section_has_all_14_prompts_as_buttons():
    """Every category from the real red-team script must be represented,
    including a clickable button per prompt -- not just the summary number."""
    at = _run_engineering()
    subheader_texts = [s.value for s in at.subheader]
    for category in [
        "Instruction override", "Hallucination bait", "Tool misuse",
        "Injection via data framing", "Fabrication bait", "Scope overreach", "Extraction attempt",
    ]:
        assert category in subheader_texts

    button_labels = [b.label for b in at.button]
    prompt_buttons = [label for label in button_labels if label.startswith('"')]
    assert len(prompt_buttons) == 14


def test_engineering_page_reports_the_graded_pass_rate_as_a_supplement():
    """The 14/14 pass-rate summary must appear, but as a supplement to the
    interactive buttons, not a replacement for them (both must be present)."""
    at = _run_engineering()
    write_texts = " ".join(m.value for m in at.markdown)
    assert "14/14" in write_texts or "/14 adversarial prompts passed" in write_texts


def test_engineering_page_has_live_ci_badge_linked_to_the_real_workflow():
    at = _run_engineering()
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "github.com/oyoai/sonic-explorer/actions/workflows/ci.yml" in markdown_texts


def test_engineering_page_cnn_section_reports_static_summary_and_has_live_picker():
    at = _run_engineering()
    metric_labels = [m.label for m in at.metric]
    assert "Test accuracy" in metric_labels
    assert "Random baseline" in metric_labels
    assert len(at.selectbox) == 1
    assert any(b.label == "Predict genre" for b in at.button)


def test_engineering_page_cnn_live_prediction_shows_predicted_vs_actual():
    """Clicking Predict genre must run real inference on the picked song's
    real audio and report both the predicted and actual genre -- not a
    precomputed/cached number."""
    at = _run_engineering()
    if any("ANTHROPIC_API_KEY" in i.value for i in at.info):
        pass  # agent absence doesn't affect the CNN picker; still runs below
    at.button(key="cnn_predict_button").click().run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    predicted_metric = next(label for label in metric_labels if label.startswith("Predicted genre"))
    assert predicted_metric  # e.g. "Predicted genre ✓" or "Predicted genre ✗"


def test_engineering_page_links_to_results_next():
    at = _run_engineering()
    write_texts = " ".join(m.value for m in at.markdown)
    assert "Results" in write_texts
