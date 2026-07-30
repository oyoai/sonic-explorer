"""Turns real human XAB similarity judgments (repository/calibration_
repository.py's calibration_ratings) into per-facet blend weights -- the
regression Methodology section 8 and Results' Calibration tab describe as
planned. For each rated triplet (reference X, candidates A/B, which one the
rater judged more similar), computes every facet's own cosine-similarity
DIFFERENCE between (X, A) and (X, B) -- the feature a logistic regression
predicts the human's A-vs-B choice from. A facet whose sign of that
difference agrees with the human choice more often than chance is
contributing real signal to perceived similarity; the regression's fitted
coefficient turns that into a single interpretable per-facet weight.

Deliberately reports two things side by side, not just the regression: the
sample size actually behind these numbers (this is real, ongoing data
collection -- the honest number belongs on the same screen as any result
computed from it, not buried in a caption), and a simpler, small-sample-
robust per-facet AGREEMENT RATE alongside the regression coefficients --
logistic regression with 6 features and only a handful of labeled examples
is exactly the p >> n regime that overfits/produces unstable coefficients,
so the agreement rate (does this facet's own top choice match the human's,
more often than the other candidate) is the more trustworthy number until
real rating volume catches up. Both are computed fresh from whatever's in
calibration_ratings right now -- no hardcoded/precomputed numbers, since
this table grows with every rating session."""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.facets.registry import default_registry
from sonic_explorer.repository.calibration_repository import CalibrationRepository
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


@dataclass
class BlendWeightResult:
    n_ratings: int
    n_raters: int
    facet_names: list[str]
    # None when there isn't enough data (or scikit-learn couldn't fit at
    # all, e.g. every rating picked the same side) to report a coefficient
    # -- an honest "not computed," not a fabricated 0.0.
    regression_weights: dict[str, float] | None
    regression_note: str | None
    agreement_rate: dict[str, float]  # facet_name -> fraction of usable ratings that facet "agreed" with
    agreement_n: dict[str, int]  # facet_name -> how many ratings actually had this facet available for all 3 segments


def _facet_diff(embedding_repo: EmbeddingRepository, facet_name: str, x_id: int, a_id: int, b_id: int) -> float | None:
    """cos(X, A) - cos(X, B) for one facet, or None if any of the three
    segments doesn't have this facet embedded (status != "done") --
    vectors are already L2-normalized when indexed (see EmbeddingRepository.
    add_to_index), so cosine similarity is just the dot product, no extra
    normalization needed here."""
    for seg_id in (x_id, a_id, b_id):
        if embedding_repo.status(seg_id, facet_name) != "done":
            return None
    x_vec = embedding_repo.get_vector(facet_name, x_id)
    a_vec = embedding_repo.get_vector(facet_name, a_id)
    b_vec = embedding_repo.get_vector(facet_name, b_id)
    return float(np.dot(x_vec, a_vec) - np.dot(x_vec, b_vec))


def compute_blend_weights(
    calibration_repo: CalibrationRepository, song_repo: SongRepository, embedding_repo: EmbeddingRepository,
) -> BlendWeightResult:
    ratings = calibration_repo.get_all_ratings()
    facet_names = default_registry().names()
    n_raters = len({r["rater"] for r in ratings if r["rater"]})

    # Per-facet diff + human choice (y=1 means "A won"), independently per
    # facet so one facet missing an embedding for one triplet doesn't
    # throw away that triplet's signal for every OTHER facet too.
    facet_diffs: dict[str, list[float]] = {f: [] for f in facet_names}
    facet_labels: dict[str, list[int]] = {f: [] for f in facet_names}
    for row in ratings:
        y = 1 if row["choice"] == "a" else 0
        for facet_name in facet_names:
            diff = _facet_diff(embedding_repo, facet_name, row["segment_x_id"], row["segment_a_id"], row["segment_b_id"])
            if diff is not None:
                facet_diffs[facet_name].append(diff)
                facet_labels[facet_name].append(y)

    agreement_rate: dict[str, float] = {}
    agreement_n: dict[str, int] = {}
    for facet_name in facet_names:
        diffs, labels = facet_diffs[facet_name], facet_labels[facet_name]
        agreement_n[facet_name] = len(diffs)
        if not diffs:
            agreement_rate[facet_name] = float("nan")
            continue
        # "Agrees" = the facet's own higher-similarity candidate is the one
        # the human picked. diff > 0 means the facet itself preferred A
        # (label 1); diff < 0 means it preferred B (label 0); diff == 0 is
        # a genuine tie, counted as neither agreeing nor disagreeing (half
        # credit), rather than arbitrarily favoring one side.
        agreements = sum(
            1.0 if (d > 0) == bool(y) else (0.5 if d == 0 else 0.0)
            for d, y in zip(diffs, labels, strict=True)
        )
        agreement_rate[facet_name] = agreements / len(diffs)

    # Regression needs every facet present for the SAME set of triplets
    # (one feature matrix, one target vector) -- restricts to whichever
    # ratings have all facets embedded, rather than imputing missing
    # features with a guessed value.
    complete_rows = []
    for row in ratings:
        row_diffs = []
        ok = True
        for facet_name in facet_names:
            diff = _facet_diff(embedding_repo, facet_name, row["segment_x_id"], row["segment_a_id"], row["segment_b_id"])
            if diff is None:
                ok = False
                break
            row_diffs.append(diff)
        if ok:
            complete_rows.append((row_diffs, 1 if row["choice"] == "a" else 0))

    regression_weights: dict[str, float] | None = None
    regression_note: str | None = None
    if len(complete_rows) < 10:
        regression_note = (
            f"Only {len(complete_rows)} rating(s) have every facet embedded for all three segments -- "
            "too few to fit a meaningful regression (a rule of thumb is at least several times as many "
            "examples as features; there are 7 facets here). Showing agreement rate only until more "
            "ratings exist."
        )
    else:
        X = np.array([r[0] for r in complete_rows])
        y = np.array([r[1] for r in complete_rows])
        if len(set(y.tolist())) < 2:
            regression_note = "Every rating so far picked the same side (all A or all B) -- a regression needs both to fit."
        else:
            from sklearn.linear_model import LogisticRegression

            # Strong L2 regularization (small C) -- p (7 facets) is close to
            # or larger than n at realistic early rating volumes, exactly
            # the overfitting regime a plain unregularized fit would blow
            # up in (same reasoning as the CLAP linear-probing evaluation's
            # choice of Ridge over OLS elsewhere in this codebase).
            model = LogisticRegression(C=0.5, max_iter=1000)
            model.fit(X, y)
            regression_weights = dict(zip(facet_names, model.coef_[0].tolist(), strict=True))

    return BlendWeightResult(
        n_ratings=len(ratings),
        n_raters=n_raters,
        facet_names=facet_names,
        regression_weights=regression_weights,
        regression_note=regression_note,
        agreement_rate=agreement_rate,
        agreement_n=agreement_n,
    )
