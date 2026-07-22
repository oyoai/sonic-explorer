"""AppTest smoke test for the Explore page -- now two view modes (network
graph, 2D map) merged into one page after retiring the standalone Taste Map
page. Must go through app.py + switch_page -- some interactions on this page
render markdown/captions Streamlit only resolves with the full app context."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_explore() -> AppTest:
    at = AppTest.from_file("streamlit_app/app.py", default_timeout=120)
    at.switch_page("pages/6_Explore.py")
    at.run()
    return at


def test_explore_page_runs_without_exceptions():
    at = _run_explore()
    assert not at.exception


def test_explore_page_switching_to_map_view_renders_without_exceptions():
    at = _run_explore()
    view_radio = next(r for r in at.radio if r.label == "View")
    view_radio.set_value("map").run()
    assert not at.exception
    projection_radio = next(r for r in at.radio if r.label == "Projection")
    assert projection_radio.value == "pca"


def test_explore_page_map_view_has_axis_inspection_expander():
    at = _run_explore()
    view_radio = next(r for r in at.radio if r.label == "View")
    view_radio.set_value("map").run()
    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    assert any("Inspect these axes" in label for label in expander_labels)
