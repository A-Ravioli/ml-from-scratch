"""Small public datasets for labs and capstones: downloaded once, cached, offline-safe.

    from mlfs import datasets
    d = datasets.mnist()            # dict with X_train, y_train, X_test, y_test
    text = datasets.tiny_shakespeare()

Files are cached under $MLFS_DATA (default ~/.cache/mlfs). With MLFS_OFFLINE=1 (set by
`scripts/verify.py`) or `offline=True`, every loader returns a small, deterministic synthetic
stand-in with the same structure, shapes (except n) and dtypes, so notebooks run anywhere.
Tests never call these; they use synthetic data of their own.
"""
from __future__ import annotations

import gzip
import io
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URLS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "tiny_shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "sms_spam": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "olivetti_faces": "https://ndownloader.figshare.com/files/5976027",
    "california_housing": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
}


def cache_dir() -> Path:
    path = Path(os.environ.get("MLFS_DATA", Path.home() / ".cache" / "mlfs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_offline(offline: bool | None = None) -> bool:
    if offline is not None:
        return offline
    return os.environ.get("MLFS_OFFLINE", "") not in ("", "0", "false", "no")


def fetch(url: str, filename: str) -> Path:
    """Download `url` to the cache as `filename` (once). Returns the cached path."""
    path = cache_dir() / filename
    if path.exists():
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"mlfs.datasets: downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


# --- synthetic stand-ins ------------------------------------------------------------------


def _synthetic_images(n: int, shape: tuple[int, ...], classes: int, seed: int):
    """Class-dependent smooth templates plus noise, as uint8 images. Learnable, not trivial."""
    rng = np.random.default_rng(seed)
    templates = rng.normal(size=(classes, *shape))
    for axis in range(2):  # smooth along the spatial axes so they look like blobs
        for _ in range(3):
            templates = (templates + np.roll(templates, 1, axis=axis + 1)
                         + np.roll(templates, -1, axis=axis + 1)) / 3
    y = rng.integers(0, classes, size=n)
    x = templates[y] / templates.std() + 0.6 * rng.normal(size=(n, *shape))
    x = np.clip(127.5 + 50 * x, 0, 255).astype(np.uint8)
    return x, y.astype(np.int64)


def _split(x, y, n_test):
    return {"X_train": x[n_test:], "y_train": y[n_test:], "X_test": x[:n_test], "y_test": y[:n_test]}


_WORDS = ("the of and to in a is that for it as was with be by on not he i this are or his "
          "from at which but have an they you were her she there would their we him been has "
          "when who will more no if out so said what up its about into than them can only").split()


def _synthetic_text(n_words: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    probs = 1 / np.arange(1, len(_WORDS) + 1)  # zipf-like word frequencies
    words = rng.choice(_WORDS, size=n_words, p=probs / probs.sum())
    lines, i = [], 0
    while i < n_words:
        k = int(rng.integers(5, 12))
        lines.append(" ".join(words[i:i + k]).capitalize() + ".")
        i += k
    return "\n".join(lines) + "\n"


# --- loaders ------------------------------------------------------------------------------


def mnist(*, offline: bool | None = None) -> dict[str, np.ndarray]:
    """Handwritten digits. X_*: uint8 (n, 28, 28); y_*: int64 (n,) in 0..9. 60k/10k."""
    if is_offline(offline):
        x, y = _synthetic_images(2_400, (28, 28), 10, seed=0)
        return _split(x, y, 400)

    def read(name, offset, shape):
        raw = gzip.decompress(fetch(URLS["mnist"] + name, "mnist-" + name).read_bytes())
        return np.frombuffer(raw, np.uint8, offset=offset).reshape(shape)

    return {
        "X_train": read("train-images-idx3-ubyte.gz", 16, (-1, 28, 28)),
        "y_train": read("train-labels-idx1-ubyte.gz", 8, (-1,)).astype(np.int64),
        "X_test": read("t10k-images-idx3-ubyte.gz", 16, (-1, 28, 28)),
        "y_test": read("t10k-labels-idx1-ubyte.gz", 8, (-1,)).astype(np.int64),
    }


CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")


def cifar10(*, offline: bool | None = None) -> dict[str, np.ndarray]:
    """Small color images. X_*: uint8 (n, 32, 32, 3) (NHWC); y_*: int64 (n,). 50k/10k."""
    if is_offline(offline):
        x, y = _synthetic_images(2_400, (32, 32, 3), 10, seed=1)
        return _split(x, y, 400)
    import pickle

    path = fetch(URLS["cifar10"], "cifar-10-python.tar.gz")
    xs, ys = {"train": [], "test": []}, {"train": [], "test": []}
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            base = member.name.rsplit("/", 1)[-1]
            split = "train" if base.startswith("data_batch") else "test" if base == "test_batch" else None
            if split is None:
                continue
            batch = pickle.load(tar.extractfile(member), encoding="bytes")
            xs[split].append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys[split].append(np.array(batch[b"labels"], dtype=np.int64))
    return {"X_train": np.concatenate(xs["train"]), "y_train": np.concatenate(ys["train"]),
            "X_test": np.concatenate(xs["test"]), "y_test": np.concatenate(ys["test"])}


def tiny_shakespeare(*, offline: bool | None = None) -> str:
    """~1.1M characters of Shakespeare as one string."""
    if is_offline(offline):
        return _synthetic_text(20_000, seed=2)
    return fetch(URLS["tiny_shakespeare"], "tiny_shakespeare.txt").read_text(encoding="utf-8")


def sms_spam(*, offline: bool | None = None) -> tuple[list[str], np.ndarray]:
    """5,574 SMS messages and labels (int64, 1 = spam)."""
    if is_offline(offline):
        rng = np.random.default_rng(3)
        spam_words = "free win prize cash call now claim urgent txt offer".split()
        texts, labels = [], []
        for _ in range(600):
            is_spam = rng.random() < 0.15
            words = list(rng.choice(_WORDS, size=int(rng.integers(4, 15))))
            if is_spam:
                words += list(rng.choice(spam_words, size=int(rng.integers(2, 5))))
                rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(int(is_spam))
        return texts, np.array(labels, dtype=np.int64)
    path = fetch(URLS["sms_spam"], "sms_spam.zip")
    with zipfile.ZipFile(path) as z:
        raw = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
    pairs = [line.split("\t", 1) for line in raw.splitlines() if "\t" in line]
    return [t for _, t in pairs], np.array([int(lbl == "spam") for lbl, _ in pairs], dtype=np.int64)


def olivetti_faces(*, offline: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """400 grayscale faces of 40 people. images: float64 (400, 64, 64) in [0, 1]; labels: int64."""
    if is_offline(offline):
        x, y = _synthetic_images(200, (64, 64), 20, seed=4)
        return x.astype(np.float64) / 255, y
    from scipy.io import loadmat

    mat = loadmat(fetch(URLS["olivetti_faces"], "olivettifaces.mat"))
    faces = mat["faces"].T.reshape(400, 64, 64).transpose(0, 2, 1).astype(np.float64)
    faces /= faces.max()
    return faces, np.repeat(np.arange(40), 10).astype(np.int64)


def california_housing(*, offline: bool | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """California districts (1990 census): X float64 (n, 8) numeric features, y = median house
    value in $100k, and the feature names. Rows with missing values are dropped."""
    names = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
             "population", "households", "median_income"]
    if is_offline(offline):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(1_000, 8))
        y = 2 + 0.8 * X[:, 7] - 0.3 * X[:, 1] + 0.1 * X[:, 2] + 0.3 * rng.normal(size=1_000)
        return X, y, names
    path = fetch(URLS["california_housing"], "california_housing.csv")
    rows = path.read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    cols = [header.index(n) for n in names]
    target = header.index("median_house_value")
    X, y = [], []
    for line in rows[1:]:
        parts = line.split(",")
        try:
            X.append([float(parts[c]) for c in cols])
            y.append(float(parts[target]) / 1e5)
        except ValueError:  # missing value
            continue
    return np.array(X), np.array(y), names


__all__ = ["cache_dir", "fetch", "is_offline", "mnist", "cifar10", "CIFAR10_CLASSES",
           "tiny_shakespeare", "sms_spam", "olivetti_faces", "california_housing"]
