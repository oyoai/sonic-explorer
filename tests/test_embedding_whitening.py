import numpy as np
import pytest

from sonic_explorer.analysis.embedding_whitening import fit_whitener


def test_whitening_increases_spread_of_collapsed_corpus():
    # simulate harmony's real problem: a corpus tightly clustered around one
    # direction (small variance) plus a small amount of real signal in a
    # low-variance dimension -- cosine similarity between random pairs should
    # be high before whitening, lower after.
    rng = np.random.default_rng(0)
    base = np.array([10.0, 10.0, 10.0, 0.1, 0.1])
    vectors = [base + rng.normal(scale=[0.05, 0.05, 0.05, 0.5, 0.5]) for _ in range(200)]
    normalized = [v / np.linalg.norm(v) for v in vectors]

    def mean_random_pair_cosine(vecs):
        sims = []
        for _ in range(100):
            i, j = rng.choice(len(vecs), size=2, replace=False)
            sims.append(float(np.dot(vecs[i], vecs[j])))
        return np.mean(sims)

    before = mean_random_pair_cosine(normalized)

    whitener = fit_whitener(normalized)
    whitened = [whitener.transform(v) for v in normalized]
    after = mean_random_pair_cosine(whitened)

    assert after < before


def test_whitener_transform_output_is_unit_normalized():
    vectors = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0]), np.array([3.0, 3.0, 1.0])]
    whitener = fit_whitener(vectors)

    result = whitener.transform(np.array([1.5, 2.5, 2.0]))

    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


def test_fit_whitener_handles_constant_dimension_without_dividing_by_zero():
    vectors = [np.array([1.0, 5.0]), np.array([2.0, 5.0]), np.array([3.0, 5.0])]  # 2nd dim is constant
    whitener = fit_whitener(vectors)

    result = whitener.transform(np.array([2.0, 5.0]))

    assert np.all(np.isfinite(result))
