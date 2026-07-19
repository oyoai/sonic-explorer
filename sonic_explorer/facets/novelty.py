"""Structural confidence via novelty detection: a Gaussian-tapered checkerboard
kernel slid along the diagonal of a dense self-similarity matrix (Foote, 2000,
"Automatic Audio Segmentation Using a Measure of Audio Novelty") produces a 1D
curve where sharp peaks mark genuine section boundaries and a flat/noisy curve
means the song doesn't repeat in clear sections (spec 2.2's "structural
confidence" / "Abstractivity"). Used to decide whether Song X-Ray shows the
segmented timeline or, honestly, the continuous curve itself for abstract/
through-composed tracks.

Deliberately uses a fresh dense cosine-similarity matrix here, not the sparse
k-nearest-neighbor affinity matrix from structure.py's recurrence_matrix (built
for finding repeats, not for checkerboard novelty -- its k-NN sparsity would
degrade the convolution)."""

import numpy as np

# Each side of the kernel covers ~4s -- phrase/section scale, not per-beat
# noise. Converted to frames at analyze-time since frame duration differs
# hugely between beat-synced (~0.4-0.8s/frame) and framewise-fallback
# (~0.02s/frame) chroma.
TARGET_KERNEL_SECONDS = 4.0

# The novelty curve is normalized to [0, 1] -- a peak must clear this
# prominence to count as a real boundary, not curve noise.
MIN_PEAK_PROMINENCE = 0.1


def compute_novelty_curve(chroma_sync: np.ndarray, duration_sec: float) -> np.ndarray:
    """One value per frame in chroma_sync, normalized to [0, 1]. High values
    mark likely structural boundaries."""
    n = chroma_sync.shape[-1]
    if n < 3:
        return np.zeros(n, dtype=np.float32)

    avg_frame_sec = duration_sec / n
    k = max(1, min(n // 4, round(TARGET_KERNEL_SECONDS / avg_frame_sec))) if avg_frame_sec > 0 else 1

    normalized = chroma_sync / (np.linalg.norm(chroma_sync, axis=0, keepdims=True) + 1e-8)
    similarity = normalized.T @ normalized  # dense (n, n) cosine similarity, unlike the sparse recurrence matrix

    axis = np.arange(-k, k)
    gaussian_1d = np.exp(-0.5 * (axis / (k / 2 + 1e-8)) ** 2)
    taper = np.outer(gaussian_1d, gaussian_1d)
    quadrant_sign = np.sign(np.outer(axis, axis))
    quadrant_sign[quadrant_sign == 0] = 1.0  # the kernel's own center row/col -- treat as "self" quadrant
    kernel = quadrant_sign * taper

    padded = np.pad(similarity, k, mode="edge")
    novelty = np.array(
        [np.sum(padded[t : t + 2 * k, t : t + 2 * k] * kernel) for t in range(n)], dtype=np.float32
    )

    novelty -= novelty.min()
    peak = novelty.max()
    if peak > 0:
        novelty /= peak
    return novelty


def find_structural_peaks(novelty_curve: np.ndarray, min_prominence: float = MIN_PEAK_PROMINENCE) -> np.ndarray:
    """Frame indices of novelty-curve peaks that clear min_prominence -- these
    are the boundaries "promoted" per the spec's threshold framing. Empty
    means the song doesn't show clear structural boundaries at all."""
    from scipy.signal import find_peaks

    if len(novelty_curve) < 3:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(novelty_curve, prominence=min_prominence)
    return peaks


def compute_structural_confidence(novelty_curve: np.ndarray) -> float:
    """A continuous flatness/confidence measure -- the novelty curve's own
    standard deviation. Near 0 for a flat/noisy curve (no clear structure,
    high "abstractivity"); higher for a curve with sharp, clear peaks."""
    return float(np.std(novelty_curve)) if len(novelty_curve) > 0 else 0.0
