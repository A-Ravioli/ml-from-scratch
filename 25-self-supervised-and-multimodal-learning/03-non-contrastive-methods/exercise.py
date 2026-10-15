import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.ssl import SimCLRAugment  # yours, from topic 02
    from mlfs.torchnn import ema_update  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # non-contrastive methods · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `simsiam_loss`, `byol_loss`: negative cosine with a stop-gradient → `mlfs.ssl` |
    | 2 | `collapse_metrics`: the per-dimension std and effective rank of a batch of embeddings |
    | 3 | `SimSiam`: encoder, projector, predictor, and an optional stop-gradient |
    | 4 | `BYOL`: an online network and a momentum (EMA) target |
    | 5 | `train_ssl`: one training loop for both, logging the collapse metric |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def simsiam_loss(p1, p2, z1, z2):
    """stage 1: -(cos(p1, z2.detach()) + cos(p2, z1.detach())) / 2, cosines averaged over the batch."""
    raise NotImplementedError("stage1: implement simsiam_loss")


@app.function
def byol_loss(p, z):
    """stage 1: mean over the batch of 2 - 2 cos(p, z.detach())."""
    raise NotImplementedError("stage1: implement byol_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, simsiam_loss=simsiam_loss, byol_loss=byol_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def collapse_metrics(z):
    """stage 2: {'std': mean per-dim population std of L2-normalised rows, 'effective_rank': exp entropy of normalised singular values}."""
    raise NotImplementedError("stage2: implement collapse_metrics")


@app.cell(hide_code=True)
def _():
    checks.run(2, collapse_metrics=collapse_metrics)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class SimSiam(nn.Module):
    """two views, one network: projector + predictor, and a stop-gradient on the target branch."""

    def __init__(self, encoder, feature_dim, *, proj_dim=128, pred_dim=32, stop_gradient=True):
        """stage 3: encoder; projector Linear-BN-ReLU-Linear-BN (feature_dim -> proj_dim); predictor Linear-BN-ReLU-Linear (proj_dim -> pred_dim -> proj_dim)."""
        raise NotImplementedError("stage3: implement SimSiam.__init__")

    def forward(self, x1, x2):
        """stage 3: (loss, z1): the symmetric SimSiam loss (detached targets only if stop_gradient) and view 1's projection."""
        raise NotImplementedError("stage3: implement SimSiam.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, SimSiam=SimSiam, simsiam_loss=simsiam_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class BYOL(nn.Module):
    """an online network (encoder, projector, predictor) chasing a slowly moving EMA copy of itself."""

    def __init__(self, encoder, feature_dim, *, proj_dim=128, hidden_dim=256, momentum=0.99):
        """stage 4: online encoder/projector/predictor; target_encoder/target_projector deep copies with requires_grad=False."""
        raise NotImplementedError("stage4: implement BYOL.__init__")

    def forward(self, x1, x2):
        """stage 4: (loss, z1): byol_loss(p1, t2) + byol_loss(p2, t1), targets from the target network without grad."""
        raise NotImplementedError("stage4: implement BYOL.forward")

    def update_target(self):
        """stage 4: target <- momentum * target + (1 - momentum) * online (encoder and projector), buffers copied."""
        raise NotImplementedError("stage4: implement BYOL.update_target")


@app.cell(hide_code=True)
def _():
    checks.run(4, BYOL=BYOL, byol_loss=byol_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_ssl(model, images, *, epochs=1, batch_size=256, lr=1e-3, strength=0.5, seed=0, device="cpu"):
    """stage 5: train a SimSiam/BYOL-style model (model(x1, x2) -> (loss, z1)); history {'loss', 'z_std'} per step."""
    raise NotImplementedError("stage5: implement train_ssl")


@app.cell(hide_code=True)
def _():
    checks.run(5, train_ssl=train_ssl, SimSiam=SimSiam, BYOL=BYOL, collapse_metrics=collapse_metrics)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `simsiam_loss` and `byol_loss` are part of `mlfs.ssl`. open
    the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
