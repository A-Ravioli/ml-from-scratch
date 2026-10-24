import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # contrastive learning · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `info_nce_loss`: the NT-Xent loss over 2B views → `mlfs.ssl` |
    | 2 | `SimCLRAugment`: batched random crops, flips, colour jitter and grayscale → `mlfs.ssl` |
    | 3 | `ProjectionHead`: the MLP the loss is computed on → `mlfs.ssl` |
    | 4 | `train_simclr`: the SimCLR training loop |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def info_nce_loss(z1, z2, *, temperature=0.1):
    """stage 1: NT-Xent: z1[i], z2[i] (B, d) are positives, the other 2B-2 views negatives; cosine / temperature; mean."""
    raise NotImplementedError("stage1: implement info_nce_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, info_nce_loss=info_nce_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SimCLRAugment:
    """SimCLR's augmentations, batched on tensors: every image in the batch gets its own random draw."""

    LUMA = (0.299, 0.587, 0.114)

    def __init__(self, size=32, *, strength=0.5):
        """stage 2: output size (square) and colour-jitter strength s (factors in [1 - 0.8s, 1 + 0.8s])."""
        raise NotImplementedError("stage2: implement SimCLRAugment.__init__")

    def _gray(self, x):
        if x.shape[1] == 1:
            return x
        w = torch.tensor(self.LUMA, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    def __call__(self, x):
        """stage 2: x (B, C, H, W) floats in [0, 1] -> (B, C, size, size) in [0, 1]: crop, flip, jitter (p=0.8), grayscale (p=0.2)."""
        raise NotImplementedError("stage2: implement SimCLRAugment.__call__")


@app.cell(hide_code=True)
def _():
    checks.run(2, SimCLRAugment=SimCLRAugment)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class ProjectionHead(nn.Module):
    """g(h): the MLP between the representation h and the embedding z the loss sees."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        """stage 3: self.net = Linear(in_dim, hidden_dim) -> BatchNorm1d -> ReLU -> Linear(hidden_dim, out_dim)."""
        raise NotImplementedError("stage3: implement ProjectionHead.__init__")

    def forward(self, x):
        """stage 3: (B, in_dim) -> (B, out_dim)."""
        raise NotImplementedError("stage3: implement ProjectionHead.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, ProjectionHead=ProjectionHead)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_simclr(encoder, feature_dim, images, *, epochs=1, batch_size=256, temperature=0.1, lr=1e-3, strength=0.5,
                 hidden_dim=512, proj_dim=128, seed=0, device="cpu"):
    """stage 4: SimCLR on unlabelled images (N, C, H, W) in [0, 1]; trains encoder in place; returns (head, history)."""
    raise NotImplementedError("stage4: implement train_simclr")


@app.cell(hide_code=True)
def _():
    checks.run(4, train_simclr=train_simclr, info_nce_loss=info_nce_loss, ProjectionHead=ProjectionHead,
               SimCLRAugment=SimCLRAugment)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `info_nce_loss`, `SimCLRAugment` and `ProjectionHead` are part
    of `mlfs.ssl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
