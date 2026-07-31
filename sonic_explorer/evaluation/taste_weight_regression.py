"""Turns real human taste judgments (repository/taste_repository.py's
taste_ratings: liked/disliked per segment, a deliberately separate
judgment from CalibrationRepository's similarity XAB -- see db.py's
schema comment) into a per-DNA-axis logistic regression: does a segment's
own audio DNA predict whether it was liked?

DNA is a per-SONG scalar (tempo_bpm/energy/brightness/harmonic_complexity/
rhythmic_density), not computed per-moment -- the same disclosed
limitation Approach's Step 4 and Methodology's Song DNA section already
state ("computed once per whole song, not per moment"), not a new gap
introduced here. A liked/disliked SEGMENT is scored against its PARENT
SONG's DNA -- the closest real signal available without a new per-segment
DNA computation pass.

Deliberately mirrors blend_weight_regression.py's pattern: same sample-
size disclosure up front, same small-sample-robust honest fallback (an
explicit "too few ratings" note, never a fabricated number) instead of a
result, same L2-regularized LogisticRegression choice for the same
p-close-to-n overfitting reason (5 DNA axes here vs. 7 facets there).
One real difference: DNA axes are raw physical units on wildly different
scales (tempo in BPM vs. energy in roughly [0, 0.5]), unlike
blend_weight_regression's already-comparable cosine-similarity-diff
features -- so features here go through DNANormalizer's existing
corpus-relative [0, 1] scaling first (the same normalizer the Song DNA
radar chart and nearest_songs_by_dna already use), rather than adding a
second, redundant scaling mechanism just for this."""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.analysis.song_dna import AXES, DNANormalizer
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.repository.taste_repository import TasteRepository


@dataclass
class TasteWeightResult:
    n_ratings: int
    n_raters: int
    n_liked: int
    n_disliked: int
    axis_names: list[str]
    # None when there isn't enough data (or every rating landed on the
    # same side) to report a coefficient -- an honest "not computed," not
    # a fabricated 0.0, same convention as BlendWeightResult.
    regression_weights: dict[str, float] | None
    regression_note: str | None


def compute_taste_weights(
    taste_repo: TasteRepository, song_repo: SongRepository, dna_normalizer: DNANormalizer,
) -> TasteWeightResult:
    ratings = taste_repo.get_all_ratings()
    n_raters = len({r["rater"] for r in ratings if r["rater"]})
    n_liked = sum(1 for r in ratings if r["liked"])
    n_disliked = len(ratings) - n_liked

    # Resolve each rated segment to its parent song's DNA -- taste_ratings.
    # segment_id is FOREIGN KEY-constrained, so every row here is guaranteed
    # to resolve to a real segment/song, no defensive None-check needed.
    # Missing individual DNA values (not the whole song) still degrade
    # gracefully: DNANormalizer.normalize() already returns 0.0 for a None
    # raw value, same as every other consumer of it.
    rows: list[tuple[list[float], int]] = []
    for row in ratings:
        segment = song_repo.get_segment(row["segment_id"])
        song = song_repo.get_song(segment.song_id)
        normalized = dna_normalizer.normalize({axis: getattr(song, axis) for axis in AXES})
        rows.append(([normalized[axis] for axis in AXES], int(row["liked"])))

    regression_weights: dict[str, float] | None = None
    regression_note: str | None = None
    if len(rows) < 10:
        regression_note = (
            f"Only {len(rows)} rating(s) resolve to a real parent song -- too few to fit a meaningful "
            "regression (a rule of thumb is at least several times as many examples as features; there "
            f"are {len(AXES)} DNA axes here). Showing raw like/dislike counts only until more ratings exist."
        )
    else:
        X = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        if len(set(y.tolist())) < 2:
            regression_note = (
                "Every rating so far was the same (all liked or all disliked) -- a regression needs both to fit."
            )
        else:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=0.5, max_iter=1000)
            model.fit(X, y)
            regression_weights = dict(zip(AXES, model.coef_[0].tolist(), strict=True))

    return TasteWeightResult(
        n_ratings=len(ratings), n_raters=n_raters, n_liked=n_liked, n_disliked=n_disliked,
        axis_names=AXES, regression_weights=regression_weights, regression_note=regression_note,
    )
