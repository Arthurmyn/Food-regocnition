"""The Streamlit pages must run without errors, including after an interrupted evaluation."""
import os

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from common import ROOT

EXAMPLE = os.path.join(ROOT, "web", "assets", "example.jpg")
VIEWS = ["app/app.py", "app/views/hotdog.py", "app/views/classifier.py", "app/views/synthetic.py"]


@pytest.mark.parametrize("path", VIEWS)
def test_page_runs_without_exceptions(path):
    at = AppTest.from_file(os.path.join(ROOT, path), default_timeout=120).run()
    assert not at.exception


def test_hotdog_recovers_when_an_evaluation_was_interrupted():
    """Regression: the photo was saved but the result never was (rerun interrupted on a slow server).
    The page used to crash with AttributeError: st.session_state has no attribute 'hd_probs'."""
    at = AppTest.from_file(os.path.join(ROOT, "app/views/hotdog.py"), default_timeout=120)
    at.session_state["hd_data"] = open(EXAMPLE, "rb").read()
    at.run()
    assert not at.exception
    assert "hd_probs" in at.session_state and at.session_state["hd_done"]


def test_hotdog_does_not_recompute_a_finished_evaluation():
    at = AppTest.from_file(os.path.join(ROOT, "app/views/hotdog.py"), default_timeout=120)
    at.session_state["hd_data"] = open(EXAMPLE, "rb").read()
    at.run()
    done, probs = at.session_state["hd_done"], list(at.session_state["hd_probs"])
    at.run()
    assert at.session_state["hd_done"] == done and list(at.session_state["hd_probs"]) == probs


def test_predict_stops_timing_when_the_time_budget_is_spent():
    """On a slow CPU the latency loop must not run all 10 passes: at least 3, then stop at the budget."""
    import core
    calls = []
    handle = core.get_model().register_forward_hook(lambda *a: calls.append(1))
    try:
        core.predict(Image.open(EXAMPLE).convert("RGB"), runs=10, budget_s=0.0)
    finally:
        handle.remove()
    assert len(calls) == 1 + 3          # one warm-up pass + the minimum of three timed passes
