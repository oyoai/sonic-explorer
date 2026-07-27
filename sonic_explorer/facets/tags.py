"""Sound-tags facet: per-segment AST-detected sound/instrument tags, embedded
via CLAP's TEXT encoder -- not the audio encoder every other facet uses. This
puts the facet's vectors in the same 512-dim joint space Sound's CLAP-audio
embeddings already live in, reusing the exact model already loaded for that
facet rather than adding a new text-embedding dependency.

Two-stage, unlike every other facet: (1) AST tags the segment's raw audio
(pipeline/sound_tagging.py's get_descriptive_tags() -- the same function
that already powers songs.sound_tags/Methodology 7b, but that's only ever
been run on a song-level 10s middle slice; this is the first time it runs
per-segment on ~5s windows, see notebooks/11_sound_tags_facet.ipynb's smoke
test), (2) the resulting tag text is embedded with CLAP's text encoder.
embed() does both steps internally so this facet still satisfies the same
embed(audio, sr) contract every other facet uses -- callers (the batch
pipeline, tests) don't need to know it's two models under the hood."""

import numpy as np

from sonic_explorer.config import CLAP_DIM, CLAP_MODEL_NAME
from sonic_explorer.facets.base import Facet
from sonic_explorer.pipeline.sound_tagging import AST_SAMPLE_RATE, get_descriptive_tags


# Empirically found (real per-segment smoke test on ~5s windows, before
# committing to the full batch run) to dominate as the #1-ranked tag on
# almost every segment regardless of content -- 0.31-0.50 confidence on
# genuinely different songs/moments. Embedding text that starts with
# "Music, ..." on nearly every segment would flood the embedding space with
# a shared, uninformative token and dilute the actual distinguishing
# signal (Cello vs. Female singing vs. Gong). Same curation choice already
# made by hand for Methodology's AST_CAPABILITY_EXAMPLES display -- here
# it's load-bearing for the embedding itself, not just presentation.
GENERIC_TAG_LABELS = frozenset({"Music", "Musical instrument"})


def tags_to_text(tags: list[tuple[str, float]]) -> str:
    """Comma-joined tag labels, highest-confidence first (already the order
    get_descriptive_tags returns), generic umbrella labels excluded -- e.g.
    "Cello, Bowed string instrument, Violin, fiddle". Empty input, or input
    that's entirely generic labels, returns ""."""
    return ", ".join(label for label, _ in tags if label not in GENERIC_TAG_LABELS)


class NoTagsDetected(Exception):
    """Raised by embed() when a segment produces zero non-generic tags above
    the confidence floor -- the caller's signal to mark_skipped() rather
    than index a vector, same reasoning as embed_stems.py's energy gate:
    embedding a fake placeholder string would make every such segment
    (from unrelated songs) look falsely "similar" to each other. In
    practice rare: real-audio smoke testing found AST returns a real tag
    (even "Silence" for near-total quiet) far more often than it returns
    nothing at all -- unlike the stem facets' energy gate, which triggers
    often (any purely-instrumental song's vocal stem)."""


class SoundTagsFacet(Facet):
    name = "sound_tags"

    def __init__(self):
        self.dim = CLAP_DIM
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_clap_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(self._device)
        self._processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
        self._model.eval()

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != AST_SAMPLE_RATE:
            raise ValueError(f"AST tagging expects {AST_SAMPLE_RATE}Hz audio, got {sr}Hz")
        tags = get_descriptive_tags(audio, sr)
        text = tags_to_text(tags)
        if not text:
            raise NoTagsDetected("no non-generic tags above the confidence floor for this segment")
        return self.embed_text(text)

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        self._ensure_clap_loaded()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            raw = self._model.get_text_features(**inputs)
        vec = raw[0] if isinstance(raw, torch.Tensor) else raw.pooler_output[0]
        return vec.cpu().numpy().astype(np.float32)

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        """Batched CLAP text embedding -- unlike AST tagging (empirically NOT
        batchable on CPU, see notebooks/11_sound_tags_facet.ipynb's timing
        section), CLAP's text encoder is lightweight enough for short tag
        strings that batching here is a real, if modest, speedup. Called
        once per song by the batch pipeline (all of a song's pending
        segments' tag text at once), not per segment."""
        import torch

        self._ensure_clap_loaded()
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            raw = self._model.get_text_features(**inputs)
        vec = raw if isinstance(raw, torch.Tensor) else raw.pooler_output
        return vec.cpu().numpy().astype(np.float32)
