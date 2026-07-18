"""Sound/timbre facet. CLAP by default (promoted from notebooks/audio_deep_dive.ipynb,
cells 29-30); MFCC fallback for environments without GPU/torch (heavier deps are lazily
imported so this module is importable in a plain local install with no [colab] extra).
"""

import numpy as np

from sonic_explorer.config import CLAP_DIM, CLAP_MODEL_NAME, CLAP_SR
from sonic_explorer.facets.base import Facet

MFCC_DIM = 40  # 20 MFCC coefficients x (mean, std)


class SoundFacet(Facet):
    name = "sound"

    def __init__(self, use_clap: bool = True):
        self.use_clap = use_clap
        self.dim = CLAP_DIM if use_clap else MFCC_DIM
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
        if self.use_clap:
            return self._embed_clap(audio, sr)
        return self._embed_mfcc(audio, sr)

    def _embed_clap(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if sr != CLAP_SR:
            raise ValueError(f"CLAP expects {CLAP_SR}Hz audio, got {sr}Hz -- resample before calling embed()")
        self._ensure_clap_loaded()
        inputs = self._processor(audio=[audio.astype(np.float32)], sampling_rate=sr, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            raw = self._model.get_audio_features(**inputs)
        vec = raw[0] if isinstance(raw, torch.Tensor) else raw.pooler_output[0]
        return vec.cpu().numpy().astype(np.float32)

    def embed_batch(self, audio_windows: list[np.ndarray], sr: int, batch_size: int = 8) -> np.ndarray:
        """Batched CLAP embedding -- promoted from the notebook's cell-30 loop."""
        import torch

        if sr != CLAP_SR:
            raise ValueError(f"CLAP expects {CLAP_SR}Hz audio, got {sr}Hz -- resample before calling embed_batch()")
        self._ensure_clap_loaded()
        out = []
        with torch.no_grad():
            for i in range(0, len(audio_windows), batch_size):
                batch = [w.astype(np.float32) for w in audio_windows[i : i + batch_size]]
                inputs = self._processor(audio=batch, sampling_rate=sr, return_tensors="pt").to(self._device)
                raw = self._model.get_audio_features(**inputs)
                vec = raw if isinstance(raw, torch.Tensor) else raw.pooler_output
                out.append(vec.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def _embed_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
