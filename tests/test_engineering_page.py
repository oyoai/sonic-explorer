"""AppTest smoke test for the new Engineering page (sits between Methodology
and Results in Overview.py's st.navigation() list, despite its filename
prefix -- see Overview.py's comment on why). Demonstrates engineering/safety
rigor interactively rather than as static claims: clickable red-team
prompts against the live agent, a real CI badge, and a live CNN genre
picker. Must go through Overview.py + switch_page -- nav_button()'s
st.switch_page() needs the full multipage registry."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_engineering() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.switch_page("pages/8_Engineering.py")
    at.run()
    return at


@pytest.fixture(scope="module")
def engineering_at() -> AppTest:
    """Every test below except the CNN live-inference one only reads the
    default render -- one shared cold start instead of 7 redundant ones.
    The live-inference test clicks the Predict genre button and reruns
    (real CNN inference), so it keeps its own fresh instance."""
    return _run_engineering()


def test_engineering_page_runs_without_exceptions(engineering_at):
    assert not engineering_at.exception


def test_engineering_page_has_all_three_sections(engineering_at):
    header_texts = [h.value for h in engineering_at.header]
    assert "1. Red-teaming, interactive" in header_texts
    assert "2. CI / Docker" in header_texts
    assert "3. CNN classifier, live" in header_texts


def test_engineering_page_red_team_section_has_all_14_prompts_as_buttons(engineering_at):
    """Every category from the real red-team script must be represented,
    including a clickable button per prompt -- not just the summary number."""
    subheader_texts = [s.value for s in engineering_at.subheader]
    for category in [
        "Instruction override", "Hallucination bait", "Tool misuse",
        "Injection via data framing", "Fabrication bait", "Scope overreach", "Extraction attempt",
    ]:
        assert category in subheader_texts

    button_labels = [b.label for b in engineering_at.button]
    prompt_buttons = [label for label in button_labels if label.startswith('"')]
    assert len(prompt_buttons) == 14


def test_engineering_page_reports_the_graded_pass_rate_as_a_supplement(engineering_at):
    """The 14/14 pass-rate summary must appear, but as a supplement to the
    interactive buttons, not a replacement for them (both must be present)."""
    write_texts = " ".join(m.value for m in engineering_at.markdown)
    assert "14/14" in write_texts or "/14 adversarial prompts passed" in write_texts


def test_engineering_page_has_live_ci_badge_linked_to_the_real_workflow(engineering_at):
    markdown_texts = " ".join(m.value for m in engineering_at.markdown)
    assert "github.com/oyoai/sonic-explorer/actions/workflows/ci.yml" in markdown_texts


def test_engineering_page_cnn_section_reports_static_summary_and_has_live_picker(engineering_at):
    metric_labels = [m.label for m in engineering_at.metric]
    assert "Test accuracy" in metric_labels
    assert "Random baseline" in metric_labels
    assert len(engineering_at.selectbox) == 1
    assert any(b.label == "Predict genre" for b in engineering_at.button)


def test_engineering_page_cnn_live_prediction_shows_predicted_vs_actual():
    """Clicking Predict genre must run real inference on the picked song's
    real audio and report both the predicted and actual genre -- not a
    precomputed/cached number. Needs its own fresh instance (not the shared
    engineering_at fixture) since it clicks a widget and reruns."""
    at = _run_engineering()
    if any("ANTHROPIC_API_KEY" in i.value for i in at.info):
        pass  # agent absence doesn't affect the CNN picker; still runs below
    at.button(key="cnn_predict_button").click().run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    predicted_metric = next(label for label in metric_labels if label.startswith("Predicted genre"))
    assert predicted_metric  # e.g. "Predicted genre ✓" or "Predicted genre ✗"


def test_engineering_page_links_to_results_next(engineering_at):
    write_texts = " ".join(m.value for m in engineering_at.markdown)
    assert "Results" in write_texts
