"""Song DNA: a handful of per-song aggregate audio statistics -- tempo, energy,
brightness, harmonic complexity, rhythmic density -- used for the radar-chart
"song DNA" overlay comparison (spec 2.2). Computed once per song in the
structure batch pipeline (reuses the audio already loaded there -- see
pipeline/build_structure_library.py) and persisted as plain columns on the
songs table, since they're simple scalars, not array artifacts like the
fingerprints or structure matrix.

Deliberately built from cheap, well-established librosa features rather than a
new model -- harmonic complexity and rhythmic density in particular are proxies
(chroma entropy, onset rate), not a dedicated harmony/rhythm facet."""

import numpy as np


def compute_raw_song_dna(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Returns a dict with keys tempo_bpm, energy, brightness,
    harmonic_complexity, rhythmic_density -- all raw (unnormalized) scalars.
    See analysis/song_dna.py for normalizing these to a comparable [0,1] scale
    across the library, which is what the radar chart actually plots."""
    import librosa

    duration_sec = len(audio) / sr

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_bpm = float(np.asarray(tempo).ravel()[0])

    rms = librosa.feature.rms(y=audio)[0]
    energy = float(rms.mean())

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    brightness = float(centroid.mean())

    # harmonic complexity: Shannon entropy of the mean chroma vector, normalized
    # by the max possible entropy (log of 12 pitch classes) -- a tonally-focused
    # song (e.g. a drone, a simple riff) concentrates energy in few pitch
    # classes (low entropy); a harmonically busy song spreads across more (high
    # entropy). A proxy, not a dedicated harmony-facet measurement.
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    mean_chroma = chroma.mean(axis=1)
    chroma_probs = mean_chroma / (mean_chroma.sum() + 1e-8)
    chroma_entropy = float(-np.sum(chroma_probs * np.log(chroma_probs + 1e-8)))
    max_entropy = float(np.log(chroma.shape[0]))
    harmonic_complexity = chroma_entropy / max_entropy if max_entropy > 0 else 0.0

    # rhythmic density: onset rate (onsets per second) -- a busier rhythm
    # section triggers more onsets per second than a sparse one.
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    rhythmic_density = float(len(onset_frames) / duration_sec) if duration_sec > 0 else 0.0

    return {
        "tempo_bpm": tempo_bpm,
        "energy": energy,
        "brightness": brightness,
        "harmonic_complexity": harmonic_complexity,
        "rhythmic_density": rhythmic_density,
    }
