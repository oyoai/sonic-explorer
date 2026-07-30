import numpy as np
import pytest

from sonic_explorer.analysis.embedding_whitening import fit_whitener
from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.harmony import HARMONY_DIM, HarmonyFacet


def make_sine(duration_sec=5.0, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_embed_returns_expected_shape():
    facet = HarmonyFacet()
    vec = facet.embed(make_sine(), CLAP_SR)
    assert vec.shape == (HARMONY_DIM,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_embed_batch_returns_expected_shape():
    facet = HarmonyFacet()
    windows = [make_sine(duration_sec=5.0, freq=f) for f in (220.0, 440.0, 880.0)]
    vecs = facet.embed_batch(windows, CLAP_SR)
    assert vecs.shape == (3, HARMONY_DIM)


def test_identical_audio_has_similarity_near_one():
    facet = HarmonyFacet()
    audio = make_sine(freq=440.0)
    vec = facet.embed(audio, CLAP_SR)
    assert facet.similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)


def test_same_pitch_class_octave_apart_is_more_similar_than_tritone():
    """Chroma collapses octaves -- A4 (440Hz) and A5 (880Hz) share a pitch
    class and should score meaningfully more similar than A4 vs. a tritone
    away (Eb, ~622Hz), which is about as harmonically distant as two notes
    get on the chroma circle."""
    facet = HarmonyFacet()
    a4 = facet.embed(make_sine(freq=440.0), CLAP_SR)
    a5 = facet.embed(make_sine(freq=880.0), CLAP_SR)
    d_sharp5 = facet.embed(make_sine(freq=622.25), CLAP_SR)

    same_pitch_class_sim = facet.similarity(a4, a5)
    tritone_sim = facet.similarity(a4, d_sharp5)

    assert same_pitch_class_sim > tritone_sim


def test_dim_matches_class_attribute():
    facet = HarmonyFacet()
    assert facet.dim == HARMONY_DIM
    assert facet.embed(make_sine(), CLAP_SR).shape[0] == facet.dim


def test_embed_applies_whitener_when_one_is_loaded():
    """Real regression coverage for a real gap: HarmonyFacet.embed() used to
    always return raw (unwhitened) vectors, even though the FAISS index it
    feeds has been whitened (scripts/whiten_harmony_index.py) -- a fresh
    embedding (a live query, a new song, a robustness/perturbation test)
    landed in a different vector space than what's actually indexed.
    _whitener is set directly here (bypassing the real file-based loader)
    since HARMONY_WHITENER_PATH won't exist in a bare test environment --
    this isolates "does embed() apply a loaded whitener correctly" from
    "is a whitener file present on this machine," which is what
    _load_whitener() itself is for."""
    facet = HarmonyFacet()
    raw = facet.embed(make_sine(), CLAP_SR)

    normalized = raw / np.linalg.norm(raw)
    facet._whitener = fit_whitener([normalized, normalized * 0.9, normalized * 1.1])

    whitened = facet.embed(make_sine(), CLAP_SR)

    assert whitened.shape == (HARMONY_DIM,)
    assert np.linalg.norm(whitened) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(whitened, raw)


def test_embed_without_a_whitener_returns_raw_unwhitened_vector():
    """With no whitener set (the common case until scripts/whiten_harmony_
    index.py has been run against a given data dir), embed() must behave
    exactly as before this change -- the raw chroma mean/std vector,
    un-normalized by the facet itself (L2-normalization happens later, in
    EmbeddingRepository's own add/query path). Set explicitly to None
    rather than relying on no whitener file existing on disk, which this
    test shouldn't depend on either way."""
    facet = HarmonyFacet()
    facet._whitener = None
    vec = facet.embed(make_sine(), CLAP_SR)
    assert vec.shape == (HARMONY_DIM,)
    assert np.linalg.norm(vec) != pytest.approx(1.0, abs=1e-3)
