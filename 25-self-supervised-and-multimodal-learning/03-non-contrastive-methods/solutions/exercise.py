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
    # non-contrastive methods · build (reference solution)

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
    return -(F.cosine_similarity(p1, z2.detach(), dim=1).mean()
             + F.cosine_similarity(p2, z1.detach(), dim=1).mean()) / 2


@app.function
def byol_loss(p, z):
    """stage 1: mean over the batch of 2 - 2 cos(p, z.detach())."""
    return (2 - 2 * F.cosine_similarity(p, z.detach(), dim=1)).mean()


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
    with torch.no_grad():
        z = z.detach().cpu().double()
        std = F.normalize(z, dim=1).std(dim=0, correction=0).mean().item()
        s = torch.linalg.svdvals(z)
        p = s / s.sum().clamp_min(1e-12) + 1e-12
        erank = torch.exp(-(p * torch.log(p)).sum()).item()
    return {"std": std, "effective_rank": erank}


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
        super().__init__()
        self.encoder = encoder
        self.stop_gradient = stop_gradient
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim))
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim), nn.ReLU(),
            nn.Linear(pred_dim, proj_dim))

    def forward(self, x1, x2):
        """stage 3: (loss, z1): the symmetric SimSiam loss (detached targets only if stop_gradient) and view 1's projection."""
        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        if self.stop_gradient:
            loss = simsiam_loss(p1, p2, z1, z2)
        else:
            loss = -(F.cosine_similarity(p1, z2, dim=1).mean() + F.cosine_similarity(p2, z1, dim=1).mean()) / 2
        return loss, z1


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
        super().__init__()
        self.momentum = momentum
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, proj_dim))
        self.predictor = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, proj_dim))
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad_(False)

    def forward(self, x1, x2):
        """stage 4: (loss, z1): byol_loss(p1, t2) + byol_loss(p2, t1), targets from the target network without grad."""
        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))
        return byol_loss(p1, t2) + byol_loss(p2, t1), z1

    def update_target(self):
        """stage 4: target <- momentum * target + (1 - momentum) * online (encoder and projector), buffers copied."""
        ema_update(self.target_encoder, self.encoder, decay=self.momentum)
        ema_update(self.target_projector, self.projector, decay=self.momentum)


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
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    n = len(X)
    bs = min(batch_size, n)
    model.to(device)
    augment = SimCLRAugment(X.shape[-1], strength=strength)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = {"loss": [], "z_std": []}
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n - bs + 1, bs):
            xb = X[perm[start:start + bs]].to(device)
            loss, z1 = model(augment(xb), augment(xb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hasattr(model, "update_target"):
                model.update_target()
            history["loss"].append(loss.item())
            history["z_std"].append(collapse_metrics(z1)["std"])
    return history


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
