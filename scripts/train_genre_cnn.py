"""Trains the small CNN genre classifier (analysis/genre_cnn.py) on the real
local library -- a genuine trained-model baseline, not another pretrained-
embedding-and-retrieval story like everything else in this project. Reports
test accuracy against genre-cohesion@10's per-facet numbers (already in
Methodology 6) as the "hand-crafted/pretrained vs. actually-trained" parallel
comparison spec section 9 calls for.

Caches extracted mel-spectrograms to disk (scripts/genre_cnn_features.npz)
since extraction over ~1400 real audio files is the expensive part (real
librosa decode + mel computation per song) -- re-running with different
hyperparameters shouldn't have to redo it. Safe to re-run: delete the cache
file to force re-extraction (e.g. after a library resync).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import torch

from sonic_explorer.analysis.genre_cnn import SmallGenreCNN, evaluate, train_one_epoch
from sonic_explorer.analysis.mel_features import N_FRAMES, N_MELS, extract_mel_spectrogram
from sonic_explorer.config import ARTIFACTS_DIR, CLAP_SR, DB_PATH, audio_path_for
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository

FEATURES_CACHE = Path(__file__).resolve().parent / "genre_cnn_features.npz"
MODEL_OUT = Path(__file__).resolve().parent / "genre_cnn_model.pt"
RESULTS_OUT = Path(__file__).resolve().parent / "genre_cnn_results.json"

SEED = 42
N_EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 0.15 is test


def build_or_load_features():
    if FEATURES_CACHE.exists():
        print(f"Loading cached features from {FEATURES_CACHE}")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        return data["X"], data["y"], list(data["classes"])

    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    songs = [s for s in song_repo.list_songs() if s.genre_top]
    classes = sorted({s.genre_top for s in songs})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(songs)} songs across {len(classes)} genres: {classes}")

    X = np.zeros((len(songs), N_MELS, N_FRAMES), dtype=np.float32)
    y = np.zeros(len(songs), dtype=np.int64)
    failed = 0
    start = time.time()
    for i, song in enumerate(songs):
        try:
            audio, sr = librosa.load(str(audio_path_for(song)), sr=CLAP_SR, mono=True)
            X[i] = extract_mel_spectrogram(audio, sr)
            y[i] = class_to_idx[song.genre_top]
        except Exception as exc:
            print(f"  failed on {song.title!r}: {exc}")
            failed += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  ...{i + 1}/{len(songs)} extracted ({elapsed:.0f}s elapsed, "
                  f"~{elapsed / (i + 1) * (len(songs) - i - 1):.0f}s remaining)")

    print(f"Extraction done, {failed} failures. Caching to {FEATURES_CACHE}")
    np.savez_compressed(FEATURES_CACHE, X=X, y=y, classes=np.array(classes))
    return X, y, classes


def stratified_split(y: np.ndarray, seed: int = SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * TRAIN_FRAC)
        n_val = int(len(idx) * VAL_FRAC)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def make_loader(X, y, indices, batch_size, shuffle):
    X_t = torch.from_numpy(X[indices]).unsqueeze(1)  # (N, 1, n_mels, n_frames)
    y_t = torch.from_numpy(y[indices])
    order = np.arange(len(indices))
    if shuffle:
        np.random.default_rng(SEED).shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        batches.append((X_t[batch_idx], y_t[batch_idx]))
    return batches


def main():
    X, y, classes = build_or_load_features()
    train_idx, val_idx, test_idx = stratified_split(y)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    train_loader = make_loader(X, y, train_idx, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X, y, val_idx, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X, y, test_idx, BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    model = SmallGenreCNN(n_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None
    history = []
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_OUT)

    test_loss, test_acc = evaluate(model, test_loader, device)
    random_baseline = 1.0 / len(classes)
    print(f"\n=== Final: test_loss={test_loss:.4f} test_accuracy={test_acc:.3f} "
          f"(random baseline {random_baseline:.3f}) ===")

    results = {
        "classes": classes, "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "test_accuracy": test_acc, "test_loss": test_loss, "random_baseline": random_baseline,
        "best_val_accuracy": best_val_acc, "history": history,
    }
    RESULTS_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {RESULTS_OUT}")
    print(f"Model weights written to {MODEL_OUT}")


if __name__ == "__main__":
    main()
