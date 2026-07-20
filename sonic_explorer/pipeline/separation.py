"""Demucs-based source separation -- splits a song's full-mix audio into
vocal/drums/bass/instrumental stems before each is independently segmented
and embedded by the corresponding stem facet (facets/stems.py).

Heavy, GPU-preferred neural inference -- same compute-location pattern as
CLAP (the sound facet): runs on Colab
(notebooks/03_stem_separation_and_embed.ipynb), not in the local CPU
pipelines like structure/harmony. torch/demucs are lazily imported so this
module stays importable in a plain local install with no [colab] extra.

Written from documented API knowledge (demucs.api.Separator, added in demucs
4.0 specifically for this kind of programmatic use) rather than tested
locally -- this dev machine has no GPU and demucs/torch aren't installed here.
Smoke-tested on real Colab GPU against real FMA audio (an experimental/
instrumental track correctly produced a near-empty vocal stem, audibly
separated drums/bass/other) before any full-library batch run.
"""

import numpy as np

DEMUCS_MODEL_NAME = "htdemucs"
STEM_NAMES = ["vocal", "drums", "bass", "instrumental"]
# demucs' own stem names -> ours (htdemucs' 4 sources are drums/bass/other/vocals)
_DEMUCS_TO_OURS = {"vocals": "vocal", "drums": "drums", "bass": "bass", "other": "instrumental"}

_separator = None


def _ensure_separator_loaded():
    global _separator
    if _separator is not None:
        return _separator
    from demucs.api import Separator

    _separator = Separator(model=DEMUCS_MODEL_NAME)
    return _separator


def separate_stems(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Returns {"vocal": ..., "drums": ..., "bass": ..., "instrumental": ...},
    each a mono float32 waveform at the caller's sample rate `sr`."""
    import torch
    import torchaudio

    separator = _ensure_separator_loaded()

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).repeat(2, 1)  # mono -> fake stereo, what Demucs expects

    _, separated = separator.separate_tensor(waveform, sr)

    stems: dict[str, np.ndarray] = {}
    for demucs_name, our_name in _DEMUCS_TO_OURS.items():
        stem_audio = separated[demucs_name]
        if stem_audio.shape[-1] != waveform.shape[-1]:
            # separate_tensor may return audio at the model's native rate
            # (44.1kHz) rather than the input `sr` -- resample back so every
            # stem lines up sample-for-sample with the original segmentation.
            native_len = stem_audio.shape[-1]
            native_sr = round(sr * native_len / waveform.shape[-1])
            stem_audio = torchaudio.functional.resample(stem_audio, native_sr, sr)
        mono = stem_audio.mean(dim=0)
        stems[our_name] = mono.numpy().astype(np.float32)
    return stems
