import numpy as np

from sonic_explorer.pipeline.vocal_presence import MIN_VOCAL_CONFIDENCE, has_vocal_content


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


def test_min_vocal_confidence_is_low_by_design():
    # Documents the deliberate choice (see module docstring): a false "no
    # vocals" verdict deletes real content, so the threshold stays permissive.
    assert MIN_VOCAL_CONFIDENCE <= 0.02
