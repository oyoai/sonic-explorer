import numpy as np

from sonic_explorer.evaluation.linear_probing import probe_dna_from_embeddings


def _make_songs(n_songs=120, dim=32, seed=1):
    rng = np.random.default_rng(seed)
    song_vectors = {sid: rng.normal(size=dim) for sid in range(n_songs)}
    return song_vectors, rng


def test_r2_high_when_target_is_a_real_linear_function_of_the_embedding():
    """The real correctness check: a target that genuinely IS a linear
    function of the embedding (plus small noise) must probe with a high
    cross-validated R^2 -- confirms this is measuring real recoverable
    linear structure, not returning some fixed/uninformative number."""
    song_vectors, rng = _make_songs(n_songs=150, dim=20, seed=3)
    true_weights = rng.normal(size=20)
    dna_by_song = {
        sid: {"tempo_bpm": float(vec @ true_weights + rng.normal(scale=0.05))}
        for sid, vec in song_vectors.items()
    }

    results = probe_dna_from_embeddings(song_vectors, dna_by_song, axes=["tempo_bpm"], cv_folds=5, seed=42)

    assert results["tempo_bpm"].r2_mean > 0.9
    assert results["tempo_bpm"].n_songs == 150


def test_r2_low_when_target_is_unrelated_random_noise():
    """Same embeddings, but the target is pure noise with no relationship
    to them at all -- cross-validated R^2 must land near 0 (or negative,
    which Ridge-CV can legitimately report on pure noise), not spuriously
    high the way in-sample R^2 on a high-dimensional embedding easily
    could without cross-validation."""
    song_vectors, rng = _make_songs(n_songs=150, dim=20, seed=3)
    dna_by_song = {sid: {"tempo_bpm": float(rng.normal())} for sid in song_vectors}

    results = probe_dna_from_embeddings(song_vectors, dna_by_song, axes=["tempo_bpm"], cv_folds=5, seed=42)

    assert results["tempo_bpm"].r2_mean < 0.15


def test_missing_axis_values_excluded_only_from_that_axis():
    song_vectors, rng = _make_songs(n_songs=50, dim=10, seed=5)
    dna_by_song = {sid: {"tempo_bpm": float(rng.normal()), "energy": float(rng.normal())} for sid in song_vectors}
    # Remove "energy" for 10 songs only -- tempo_bpm's probe must still see all 50.
    for sid in list(song_vectors)[:10]:
        del dna_by_song[sid]["energy"]

    results = probe_dna_from_embeddings(song_vectors, dna_by_song, axes=["tempo_bpm", "energy"], cv_folds=5, seed=42)

    assert results["tempo_bpm"].n_songs == 50
    assert results["energy"].n_songs == 40


def test_returns_a_result_per_requested_axis():
    song_vectors, rng = _make_songs(n_songs=60, dim=10, seed=9)
    dna_by_song = {
        sid: {"tempo_bpm": float(rng.normal()), "energy": float(rng.normal()), "brightness": float(rng.normal())}
        for sid in song_vectors
    }

    results = probe_dna_from_embeddings(
        song_vectors, dna_by_song, axes=["tempo_bpm", "energy", "brightness"], cv_folds=5, seed=42,
    )

    assert set(results.keys()) == {"tempo_bpm", "energy", "brightness"}
