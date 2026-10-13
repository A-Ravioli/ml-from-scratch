"""Checks for capstone.py: functional tests plus minimum metric thresholds."""
# TEMPLATE: replace these example checks with the capstone's checks.
import numpy as np


def _blobs(seed: int, n: int = 300):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 4.0], [-4.0, 4.0]])
    y = rng.integers(0, 3, size=n)
    return centers[y] + rng.normal(size=(n, 2)), y


def test_stage1_accuracy_threshold(nb):
    X_train, y_train = _blobs(0)
    X_test, y_test = _blobs(1)
    pred = np.asarray(nb.fit_predict(X_train, y_train, X_test))
    assert pred.shape == y_test.shape, f"expected predictions of shape {y_test.shape}, got {pred.shape}"
    acc = (pred == y_test).mean()
    assert acc >= 0.9, f"held-out accuracy should be at least 0.90, got {acc:.3f}"
