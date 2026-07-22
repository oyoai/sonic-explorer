import numpy as np

from sonic_explorer.pipeline.sound_tagging import get_descriptive_tags


def fake_tagger(predictions):
    def _tag(audio, sr):
        return predictions
    return _tag


def test_get_descriptive_tags_returns_sorted_highest_first():
    tagger = fake_tagger([
        {"label": "Music", "score": 0.5},
        {"label": "Cello", "score": 0.25},
        {"label": "Bowed string instrument", "score": 0.09},
    ])
    tags = get_descriptive_tags(np.zeros(1000), 16000, tagger_fn=tagger)
    assert tags == [("Music", 0.5), ("Cello", 0.25), ("Bowed string instrument", 0.09)]


def test_get_descriptive_tags_excludes_below_confidence_floor():
    tagger = fake_tagger([
        {"label": "Music", "score": 0.5},
        {"label": "Noise", "score": 0.001},
    ])
    tags = get_descriptive_tags(np.zeros(1000), 16000, tagger_fn=tagger)
    assert tags == [("Music", 0.5)]


def test_get_descriptive_tags_respects_top_k():
    tagger = fake_tagger([{"label": f"Tag{i}", "score": 1.0 - i * 0.01} for i in range(10)])
    tags = get_descriptive_tags(np.zeros(1000), 16000, tagger_fn=tagger, top_k=3)
    assert len(tags) == 3
    assert tags[0][0] == "Tag0"


def test_get_descriptive_tags_no_category_filtering_unlike_vocal_presence():
    """Unlike vocal_presence.has_vocal_content, this has no vocal-keyword
    filter -- instrument/texture tags pass through untouched."""
    tagger = fake_tagger([{"label": "Gong", "score": 0.5}, {"label": "Ambient music", "score": 0.3}])
    tags = get_descriptive_tags(np.zeros(1000), 16000, tagger_fn=tagger)
    assert ("Gong", 0.5) in tags
    assert ("Ambient music", 0.3) in tags


def test_get_descriptive_tags_empty_predictions_returns_empty_list():
    assert get_descriptive_tags(np.zeros(1000), 16000, tagger_fn=fake_tagger([])) == []
