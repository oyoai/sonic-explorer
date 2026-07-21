"""Song-level singing/speech presence check via a pretrained AudioSet tagger
(AST) -- an independent, out-of-band cross-check on the vocal facet's Demucs
separation output.

Why this exists: Demucs' vocal stem can carry substantial energy from
non-vocal content -- a real, observed case is sustained melodic string
instruments (cello, violin) getting misclassified as voice, since they
occupy a similar frequency/timbral range to singing. embed_stems.py's energy
gate only catches near-silence; it has no way to catch a stem that's
genuinely non-silent but still isn't voice. This runs on the ORIGINAL MIX
audio, not the isolated stem -- AST was trained on realistic full mixes, and
"does this song have singing/speech in it at all" is a cleaner, more
reliable question than classifying an out-of-distribution isolated stem
would be.

Runs entirely on CPU (confirmed via a manual spike against 6 real library
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
AST_SAMPLE_RATE = 16000  # what the spike validated against -- load audio at this rate directly
                          # rather than relying on the HF pipeline's internal resampling from a
                          # different rate (e.g. CLAP_SR=48000), an untested code path

# AudioSet label substrings that indicate singing/speech content. From the
# spike: real vocal tracks show labels like "Singing," "Chant," "Female
# singing" at modest-but-nonzero confidence (0.01-0.09) alongside a dominant
# "Music" tag; songs with no real vocals show literally none of these in
# their top predictions -- a clean, low-noise signal despite the low
# absolute scores.
VOCAL_LABEL_KEYWORDS = (
    "speech", "singing", "vocal", "voice", "rapping", "choir", "a capella",
    "chant", "yodeling", "beatboxing", "humming", "whispering",
)
# AudioSet labels that would otherwise false-match a VOCAL_LABEL_KEYWORDS
# substring despite not being human vocalization ("Singing bowl" is a
# meditation instrument, matched by the "singing" substring) -- checked
# before the keyword match, not after.
_VOCAL_LABEL_EXCLUSIONS = ("singing bowl",)

# Deliberately low: a false "no vocals" verdict silently deletes real content
# from the vocal facet, which is worse than occasionally leaving a borderline
# non-vocal segment in. Catching quiet/background vocals matters more here
# than being strict. Not validated beyond the initial 6-song spike -- see
# scripts/filter_vocal_facet_by_ast.py's broader pre-flight check.
MIN_VOCAL_CONFIDENCE = 0.01

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


def has_vocal_content(
    audio: np.ndarray,
    sr: int,
    tagger_fn: TaggerFn | None = None,
    min_confidence: float = MIN_VOCAL_CONFIDENCE,
) -> bool:
    """True if the tagger's predictions include any singing/speech-related
    label at or above min_confidence. tagger_fn defaults to the real AST
    pipeline (lazily loaded on first real use); pass a fake for tests."""
    tag_fn = tagger_fn if tagger_fn is not None else _ensure_tagger_loaded()
    predictions = tag_fn(audio, sr)
    return any(
        p["score"] >= min_confidence
        and not any(excl in p["label"].lower() for excl in _VOCAL_LABEL_EXCLUSIONS)
        and any(kw in p["label"].lower() for kw in VOCAL_LABEL_KEYWORDS)
        for p in predictions
    )
