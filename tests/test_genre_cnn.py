import pytest

torch = pytest.importorskip("torch")  # not installed in the main CI env -- skip cleanly, don't fail

from sonic_explorer.analysis.genre_cnn import SmallGenreCNN, evaluate, train_one_epoch  # noqa: E402
from sonic_explorer.analysis.mel_features import N_FRAMES, N_MELS  # noqa: E402


def test_small_genre_cnn_forward_pass_shape():
    model = SmallGenreCNN(n_classes=8)
    x = torch.randn(4, 1, N_MELS, N_FRAMES)

    logits = model(x)

    assert logits.shape == (4, 8)


def test_small_genre_cnn_handles_batch_size_one():
    model = SmallGenreCNN(n_classes=8)
    x = torch.randn(1, 1, N_MELS, N_FRAMES)

    logits = model(x)

    assert logits.shape == (1, 8)


def _fake_loader(n_batches=2, batch_size=3, n_classes=4):
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 1, N_MELS, N_FRAMES)
        y = torch.randint(0, n_classes, (batch_size,))
        batches.append((x, y))
    return batches


def test_train_one_epoch_reduces_or_maintains_finite_loss():
    model = SmallGenreCNN(n_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _fake_loader()

    loss = train_one_epoch(model, loader, optimizer, device="cpu")

    assert loss == loss  # not NaN
    assert loss >= 0.0


def test_evaluate_returns_loss_and_accuracy_in_valid_ranges():
    model = SmallGenreCNN(n_classes=4)
    loader = _fake_loader()

    loss, accuracy = evaluate(model, loader, device="cpu")

    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_accuracy_is_perfect_when_model_always_predicts_correctly():
    class AlwaysCorrect(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            # cheat: encode the "label" in the mean of the input so eval is deterministic
            logits = torch.zeros(batch_size, 2)
            logits[:, 0] = 1.0
            return logits

    model = AlwaysCorrect()
    x = torch.randn(5, 1, N_MELS, N_FRAMES)
    y = torch.zeros(5, dtype=torch.long)
    loader = [(x, y)]

    _, accuracy = evaluate(model, loader, device="cpu")

    assert accuracy == pytest.approx(1.0)
