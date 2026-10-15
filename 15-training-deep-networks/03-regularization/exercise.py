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
    # regularization · build

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
    raise NotImplementedError("stage1: implement l2_penalty")


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
        raise NotImplementedError("stage2: implement Dropout.__init__")

    def forward(self, x):
        """stage 2: eval or p == 0: x unchanged; p == 1: zeros; else x * mask / (1 - p) with keep-mask rng.random(shape) >= p."""
        raise NotImplementedError("stage2: implement Dropout.forward")


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
    raise NotImplementedError("stage3: implement label_smoothing_cross_entropy")


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
    raise NotImplementedError("stage4: implement random_crop")


@app.function
def random_horizontal_flip(images, *, p=0.5, rng=None):
    """stage 4: a copy of the NCHW images where each image is mirrored left-right with probability p (flip = rng.random(n) < p)."""
    raise NotImplementedError("stage4: implement random_horizontal_flip")


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
        raise NotImplementedError("stage5: implement EarlyStopping.__init__")

    def step(self, value):
        """stage 5: an improvement beats best by more than min_delta (the first value always does); returns should_stop."""
        raise NotImplementedError("stage5: implement EarlyStopping.step")


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
