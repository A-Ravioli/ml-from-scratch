import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.nn import Module  # yours, from ch 14 · topic 04
    from mlfs.tensor import Tensor  # yours, from ch 14 · topic 03
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # regularization · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a network with more parameters than
    training examples can memorize anything; regularization is everything you do to make it
    learn the pattern instead. you'll build the five tools that do most of that work in
    practice, and they join **`mlfs.train`**: the losses and `Dropout` for your `mlfs.nn`,
    the augmentations for any NumPy images, and an `EarlyStopping` that works with any
    training loop, including PyTorch's.

    | stage | you build |
    |---|---|
    | 1 | `l2_penalty`: the classic weight-decay term, as a loss |
    | 2 | `Dropout`: inverted dropout, on in training, off in evaluation |
    | 3 | `label_smoothing_cross_entropy`: cross-entropy against softened targets |
    | 4 | `random_crop` and `random_horizontal_flip`: data augmentation for NCHW images |
    | 5 | `EarlyStopping`: stop when the validation metric stops improving |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def l2_penalty(params, weight_decay):
    """stage 1: the Tensor (weight_decay / 2) * sum of every parameter's squared entries (0-d)."""
    total = Tensor(0.0)
    for p in params:
        total = total + (p * p).sum()
    return total * (0.5 * weight_decay)


@app.cell(hide_code=True)
def _():
    checks.run(1, l2_penalty=l2_penalty)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Dropout(Module):
    """inverted dropout: in training, zero each entry with probability p and scale the rest by 1/(1-p); identity in eval."""

    def __init__(self, p=0.5, *, rng=None):
        """stage 2: store p (ValueError unless 0 <= p <= 1) and a generator self.rng = as_rng(rng)."""
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = float(p)
        self.rng = as_rng(rng)

    def forward(self, x):
        """stage 2: eval or p == 0: x unchanged; p == 1: zeros; else x * mask / (1 - p) with keep-mask rng.random(shape) >= p."""
        x = x if isinstance(x, Tensor) else Tensor(x)
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return x * np.zeros(x.shape)
        keep = self.rng.random(x.shape) >= self.p
        return x * (keep / (1.0 - self.p))


@app.cell(hide_code=True)
def _():
    checks.run(2, Dropout=Dropout)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def label_smoothing_cross_entropy(logits, targets, smoothing=0.1):
    """stage 3: mean cross-entropy against the target distribution (1 - s) * one_hot + s / k, from logits (n, k)."""
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets).astype(np.int64)
    n = logits.shape[0]
    lse = logits.logsumexp(axis=1)
    nll = lse - logits[np.arange(n), targets]   # -log p_y
    uniform = lse - logits.mean(axis=1)         # -(1/k) sum_j log p_j
    return ((1.0 - smoothing) * nll + smoothing * uniform).mean()


@app.cell(hide_code=True)
def _():
    checks.run(3, label_smoothing_cross_entropy=label_smoothing_cross_entropy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def random_crop(images, size, *, padding=0, rng=None):
    """stage 4: zero-pad NCHW images by `padding`, then cut a random size x size window per image (tops, then lefts, from rng.integers)."""
    images = np.asarray(images)
    n, _, h, w = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    hp, wp = h + 2 * padding, w + 2 * padding
    if size > hp or size > wp:
        raise ValueError(f"crop size {size} is larger than the padded image ({hp} x {wp})")
    rng = as_rng(rng)
    tops = rng.integers(0, hp - size + 1, size=n)
    lefts = rng.integers(0, wp - size + 1, size=n)
    rows = tops[:, None] + np.arange(size)        # (n, size)
    cols = lefts[:, None] + np.arange(size)
    return padded[np.arange(n)[:, None, None, None], np.arange(images.shape[1])[None, :, None, None],
                  rows[:, None, :, None], cols[:, None, None, :]]


@app.function
def random_horizontal_flip(images, *, p=0.5, rng=None):
    """stage 4: a copy of the NCHW images where each image is mirrored left-right with probability p (flip = rng.random(n) < p)."""
    images = np.asarray(images)
    flip = as_rng(rng).random(len(images)) < p
    out = images.copy()
    out[flip] = out[flip][..., ::-1]
    return out


@app.cell(hide_code=True)
def _():
    checks.run(4, random_crop=random_crop, random_horizontal_flip=random_horizontal_flip)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class EarlyStopping:
    """tracks a validation metric; step(value) returns True once `patience` steps in a row brought no improvement."""

    def __init__(self, patience=5, *, min_delta=0.0, mode="min"):
        """stage 5: store the settings (ValueError for patience < 1 or a bad mode); best = None, best_step = -1, should_stop = False, num_bad_steps = 0, steps = 0."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        self.patience, self.min_delta, self.mode = int(patience), float(min_delta), mode
        self.best, self.best_step = None, -1
        self.should_stop, self.num_bad_steps, self.steps = False, 0, 0

    def step(self, value):
        """stage 5: an improvement beats best by more than min_delta (the first value always does); returns should_stop."""
        value = float(value)
        if self.best is None:
            improved = not np.isnan(value)
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best, self.best_step, self.num_bad_steps = value, self.steps, 0
        else:
            self.num_bad_steps += 1
        self.steps += 1
        if self.num_bad_steps >= self.patience:
            self.should_stop = True
        return self.should_stop


@app.cell(hide_code=True)
def _():
    checks.run(5, EarlyStopping=EarlyStopping)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.train import Dropout, EarlyStopping` gives you these tools.
    """)
    return


if __name__ == "__main__":
    app.run()
