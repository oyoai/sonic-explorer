"""General-purpose AST/AudioSet tag extraction -- the descriptive-tagging
capability demonstrated in Methodology 7b (e.g. "3rd Chair" -> Cello/Bowed
string instrument/Violin), now exposed as a reusable function rather than
the vocal-specific keyword filter in pipeline/vocal_presence.py. Returns the
top-K tags above a confidence floor, unfiltered by category -- the caller
(llm/explain.py's description synthesis) decides what to do with them.

Same lazy-import/injectable-tagger discipline as vocal_presence.py: runs on
CPU, never imported by the deployed app or the main test suite.

serialize_tags/deserialize_tags are the one exception to "never imported by
the deployed app" -- they're pure json, no torch/transformers import, safe
for the deployed app's agent_tools.search_by_sound_content to call against
the sound_tags column without pulling in the tagger itself."""

import json
from typing import Callable

import numpy as np

from sonic_explorer.pipeline.vocal_presence import AST_MODEL_NAME, AST_SAMPLE_RATE  # re-exported for callers

TaggerFn = Callable[[np.ndarray, int], list[dict]]  # (audio, sr) -> [{"label": str, "score": float}, ...]

# Below this, a tag is more likely residual noise than a real description of
# the clip -- see evaluation/retrieval_diagnostics.py-style validation logic
# in pipeline/vocal_presence.py for why a low-but-nonzero floor still matters
# more than a high one for a *descriptive* (not gating) use case: an overly
# strict floor would silently return an empty tag list for quieter/subtler
# clips, which is worse for description synthesis than including a slightly
# weak but real signal.
MIN_TAG_CONFIDENCE = 0.01
DEFAULT_TOP_K = 6

_tagger_fn: TaggerFn | None = None


def _ensure_tagger_loaded() -> TaggerFn:
    global _tagger_fn
    if _tagger_fn is not None:
        return _tagger_fn

    from transformers import pipeline

    classifier = pipeline("audio-classification", model=AST_MODEL_NAME, top_k=15)

    def _tag(audio: np.ndarray, sr: int) -> list[dict]:
        return classifier(np.asarray(audio, dtype=np.float32), sampling_rate=sr)

    _tagger_fn = _tag
    return _tagger_fn


def get_descriptive_tags(
    audio: np.ndarray,
    sr: int,
    tagger_fn: TaggerFn | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_confidence: float = MIN_TAG_CONFIDENCE,
) -> list[tuple[str, float]]:
    """Top-`top_k` (label, score) pairs at or above min_confidence, sorted
    highest-first, unfiltered by category (instruments, sound events,
    music-genre-adjacent AudioSet classes -- whatever AST actually returns)."""
    tag_fn = tagger_fn if tagger_fn is not None else _ensure_tagger_loaded()
    predictions = tag_fn(audio, sr)
    above_floor = [p for p in predictions if p["score"] >= min_confidence]
    above_floor.sort(key=lambda p: -p["score"])
    return [(p["label"], float(p["score"])) for p in above_floor[:top_k]]


def serialize_tags(tags: list[tuple[str, float]]) -> str:
    return json.dumps([[label, score] for label, score in tags])


def deserialize_tags(sound_tags: str | None) -> list[tuple[str, float]]:
    """[] for None/empty input -- callers (e.g. agent_tools.search_by_sound_content)
    can iterate the result unconditionally without a None-check per song."""
    if not sound_tags:
        return []
    return [(label, float(score)) for label, score in json.loads(sound_tags)]
