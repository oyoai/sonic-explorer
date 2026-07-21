"""Per-segment singing/speech presence check via a pretrained AudioSet tagger
(AST) -- an independent, out-of-band cross-check on the vocal facet's Demucs
separation output.

Why this exists: Demucs' vocal stem can carry substantial energy from
non-vocal content -- a real, observed case is sustained melodic string
instruments (cello, violin) getting misclassified as voice, since they
occupy a similar frequency/timbral range to singing. embed_stems.py's energy
gate only catches near-silence; it has no way to catch a stem that's
genuinely non-silent but still isn't voice. This runs on the ORIGINAL MIX
audio, not the isolated stem -- AST was trained on realistic full mixes.

Segment-level, not song-level: an early whole-clip design (score the full
30s at once) failed validation -- dominant instrumental content in a mixed
clip swamps genuinely-present-but-quieter vocals into the same noise floor
that residual background "vocal" mass sits at in truly instrumental tracks,
so the confirmed bleed case scored *higher* than real vocal songs. Checking
each ~5s segment individually (matching the vocal facet's own indexing
granularity) fixed this. It also matches how the facet is actually
consumed: Moment Matcher queries a specific moment, not a whole song, so a
real vocal song's instrumental bridge/intro should be excluded from vocal
retrieval even though the rest of the song has real vocals -- the segment,
not the song, is the right unit to gate.

Runs entirely on CPU (confirmed via manual spikes against real library
songs) -- unlike Demucs separation, no GPU/Colab dependency. tagger_fn is
injectable so this stays fully testable without a real transformers/torch
pipeline -- same duck-typing discipline as pipeline/separation.py's
separate_fn. torch/transformers are lazily imported (see pipeline/
separation.py's docstring for why) -- already available via this project's
[colab] extra, no new dependency group needed since this never runs as part
of the deployed app itself.
"""

from typing import Callable

import numpy as np

AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000  # what the spikes validated against -- load audio at this rate directly
                          # rather than relying on the HF pipeline's internal resampling from a
                          # different rate (e.g. CLAP_SR=48000), an untested code path

# AudioSet label substrings that indicate singing/speech content.
VOCAL_LABEL_KEYWORDS = (
    "speech", "singing", "vocal", "voice", "rapping", "choir", "a capella",
    "chant", "yodeling", "beatboxing", "humming", "whispering",
)
# AudioSet labels that would otherwise false-match a VOCAL_LABEL_KEYWORDS
# substring despite not being human vocalization ("Singing bowl" is a
# meditation instrument, matched by the "singing" substring) -- checked
# before the keyword match, not after.
VOCAL_LABEL_EXCLUSIONS = ("singing bowl",)

# Validated per-segment threshold (9-song spike): real-vocal segments scored
# >= 0.020, confirmed non-vocal segments (including the "3rd Chair" bleed
# case) scored <= 0.016 -- 0.018 sits cleanly in that gap. This threshold is
# NOT valid for whole-clip scoring (a different, abandoned design) -- see
# the module docstring.
MIN_VOCAL_CONFIDENCE = 0.018

TaggerFn = Callable[[np.ndarray, int], list[dict]]  # (audio, sr) -> [{"label": str, "score": float}, ...]

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


def best_vocal_label_score(predictions: list[dict]) -> tuple[float, str | None]:
    """The highest-scoring genuine vocal-keyword prediction, (0.0, None) if
    none present. Exposed separately from has_vocal_content() so callers
    that want the raw continuous score (e.g. for distribution/threshold
    analysis) don't have to duplicate the keyword-matching logic."""
    candidates = [
        p for p in predictions
        if any(kw in p["label"].lower() for kw in VOCAL_LABEL_KEYWORDS)
        and not any(excl in p["label"].lower() for excl in VOCAL_LABEL_EXCLUSIONS)
    ]
    if not candidates:
        return 0.0, None
    best = max(candidates, key=lambda p: p["score"])
    return best["score"], best["label"]


def has_vocal_content(
    audio: np.ndarray,
    sr: int,
    tagger_fn: TaggerFn | None = None,
    min_confidence: float = MIN_VOCAL_CONFIDENCE,
) -> bool:
    """True if the tagger's predictions include any singing/speech-related
    label at or above min_confidence. tagger_fn defaults to the real AST
    pipeline (lazily loaded on first real use); pass a fake for tests.
    Intended to be called per-segment (~5s window), not on a whole song --
    see module docstring."""
    tag_fn = tagger_fn if tagger_fn is not None else _ensure_tagger_loaded()
    predictions = tag_fn(audio, sr)
    score, _ = best_vocal_label_score(predictions)
    return score >= min_confidence
