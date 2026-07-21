import numpy as np
import pytest

from sonic_explorer.pipeline.vocal_presence import MIN_VOCAL_CONFIDENCE, best_vocal_label_score, has_vocal_content


def fake_tagger(predictions):
    def _tag(audio, sr):
        return predictions
    return _tag


def test_has_vocal_content_true_when_singing_label_present():
    tagger = fake_tagger([
        {"label": "Music", "score": 0.89},
        {"label": "Singing", "score": 0.02},
    ])
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger) is True


def test_has_vocal_content_false_when_no_vocal_labels():
    tagger = fake_tagger([
        {"label": "Music", "score": 0.33},
        {"label": "Cello", "score": 0.25},
        {"label": "Bowed string instrument", "score": 0.09},
    ])
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger) is False


def test_has_vocal_content_ignores_label_below_min_confidence():
    tagger = fake_tagger([{"label": "Singing", "score": 0.005}])
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger) is False


def test_has_vocal_content_respects_custom_threshold():
    tagger = fake_tagger([{"label": "Speech", "score": 0.02}])
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger, min_confidence=0.05) is False
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger, min_confidence=0.01) is True


def test_has_vocal_content_matches_case_insensitively():
    tagger = fake_tagger([{"label": "MALE SPEECH, MAN SPEAKING", "score": 0.5}])
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger) is True


def test_has_vocal_content_excludes_singing_bowl_false_match():
    """'Singing bowl' is a meditation instrument, not human vocalization --
    must not match the 'singing' substring."""
    tagger = fake_tagger([{"label": "Singing bowl", "score": 0.5}])
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=tagger) is False


def test_has_vocal_content_empty_predictions_is_false():
    assert has_vocal_content(np.zeros(1000), 16000, tagger_fn=fake_tagger([])) is False


def test_min_vocal_confidence_matches_validated_per_segment_threshold():
    # Pins the 9-song per-segment validation result (real-vocal segments >=
    # 0.020, confirmed non-vocal segments <= 0.016) -- this threshold is
    # specific to per-segment scoring, not the abandoned whole-clip design.
    assert MIN_VOCAL_CONFIDENCE == pytest.approx(0.018)


def test_best_vocal_label_score_returns_highest_matching_label():
    predictions = [
        {"label": "Music", "score": 0.5},
        {"label": "Speech", "score": 0.02},
        {"label": "Singing", "score": 0.05},
    ]
    score, label = best_vocal_label_score(predictions)
    assert score == pytest.approx(0.05)
    assert label == "Singing"


def test_best_vocal_label_score_returns_zero_when_no_vocal_labels():
    score, label = best_vocal_label_score([{"label": "Cello", "score": 0.9}])
    assert score == 0.0
    assert label is None


def test_best_vocal_label_score_excludes_singing_bowl():
    score, label = best_vocal_label_score([{"label": "Singing bowl", "score": 0.9}])
    assert score == 0.0
    assert label is None
