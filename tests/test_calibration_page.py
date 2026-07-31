"""AppTest smoke test for the in-app Calibration page (replaces the old
standalone scripts/calibration_rating_app.py). Must go through Overview.py +
switch_page for consistency with the rest of the suite's multipage-registry
requirement.

Deliberately read-only: unlike every other AppTest-backed page test in this
suite, clicking this page's rating buttons performs a REAL, persistent write
(CalibrationRepository.add_choice / TasteRepository.add_rating) against the
actual local DB_PATH -- there's no test-DB isolation fixture in this repo
(every other page's AppTest coverage is read-only against the real local
library, which is why that's never mattered before). Simulating a button
click here would write real rows into the real calibration/taste tables,
contaminating the actual calibration data collection this feature exists to
support cleanly -- exactly the kind of contamination the guest/real table
split is designed to prevent, just from the test suite instead of a random
visitor. Write behavior (add_choice/add_rating, rater filtering, guest-table
isolation) is instead covered in full isolation (in-memory DB) by
test_calibration_repository.py and test_taste_repository.py."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_calibration() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=120)
    at.switch_page("pages/9_Calibration.py")
    at.run()
    return at


@pytest.fixture(scope="module")
def calibration_at() -> AppTest:
    """Shared across the read-only tests below -- one cold start instead of
    4. The profile-select test that reruns after set_value(), and the
    deploy-subset test that monkeypatches config before rendering, both
    keep their own fresh instances."""
    return _run_calibration()


def test_calibration_page_runs_without_exceptions(calibration_at):
    assert not calibration_at.exception


def test_calibration_page_defaults_to_similarity_round(calibration_at):
    subheader_texts = [s.value for s in calibration_at.subheader]
    assert any("Similarity round" in s for s in subheader_texts)


def test_calibration_page_shows_real_profiles_plus_guest_locally(calibration_at):
    """This suite runs against the real local data/ library (is_deploy_
    subset() is False there), so the profile selector must offer both real
    profiles and Guest/Test."""
    profile_select = next(s for s in calibration_at.selectbox if s.label == "Rating profile")
    assert profile_select.options == ["profile1", "profile2", "Guest / Test"]


def test_calibration_page_shows_only_guest_when_deploy_subset(monkeypatch):
    """Purely a boolean-check patch -- resources.DATA_DIR only feeds
    is_deploy_subset()'s comparison here, it does not redirect DB_PATH/
    ARTIFACTS_DIR (those are fixed at import time), so this stays read-only
    against the real local DB like every other test in this file."""
    import resources

    monkeypatch.setattr(resources, "DATA_DIR", Path("deploy_data"))

    at = _run_calibration()

    profile_select = next(s for s in at.selectbox if s.label == "Rating profile")
    assert profile_select.options == ["Guest / Test"]


def test_calibration_page_guest_profile_shows_isolation_caption():
    at = _run_calibration()
    profile_select = next(s for s in at.selectbox if s.label == "Rating profile")
    profile_select.set_value("Guest / Test").run()

    assert not at.exception
    caption_texts = " ".join(c.value for c in at.caption)
    assert "kept separate" in caption_texts.lower()


def test_calibration_page_shows_harmony_sanity_check_pool(calibration_at):
    """The small opt-in supplemental pool (sampled via Harmony retrieval,
    not Sound) exists as a separate section, not mixed into the main
    similarity batch -- read-only check, doesn't click into it."""
    assert not calibration_at.exception
    expander_labels = [e.label for e in calibration_at.expander]
    assert any("Harmony sanity-check pool" in label for label in expander_labels)
