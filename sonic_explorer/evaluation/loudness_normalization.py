"""Peak loudness normalization -- exists specifically so a robustness/
perturbation test can compare "original vs. perturbed" audio on equal
loudness footing, not for the main pipeline's own feature extraction (which,
per a direct pipeline-normalization audit, has never normalized raw gain
before extraction anywhere, and still doesn't -- that's a separate, larger
decision this module doesn't make).

Measured directly (scripts/measure_clap_gain_sensitivity.py -- see the
Methodology page for the full write-up): CLAP is essentially loudness-
invariant up to about +-6dB of pure gain change (mean cosine similarity
~0.97-0.99) but shows real drift at +-12dB (min similarity down to
0.70-0.90 across the sampled clips). A perturbation whose SIDE EFFECT is a
loudness change -- compression is the obvious one, but not the only one --
would otherwise show up as embedding drift indistinguishable from the
perturbation's actual intended effect. Normalizing both the original and
the perturbed clip to the same peak level right before feature extraction
isolates that: any remaining similarity drop is attributable to the
perturbation itself, not an unmeasured loudness artifact riding along
with it.

Peak normalization specifically (not RMS matching): simple, deterministic,
and matches the pure-gain-scaling model clap_gain_sensitivity.py's own
measurement used -- the target this module is calibrated against."""

import numpy as np


def normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Scales audio so its max absolute sample hits target_db dBFS (default
    -1dB -- a standard bit of headroom, not a hard 0dBFS ceiling that could
    tip into clipping after any downstream processing). Silence (all-zero
    audio) is returned unchanged: there's no peak to scale against, and
    scaling zero by any factor is still zero."""
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio.astype(np.float32)
    target_amplitude = 10.0 ** (target_db / 20.0)
    factor = target_amplitude / peak
    return (audio * factor).astype(np.float32)
