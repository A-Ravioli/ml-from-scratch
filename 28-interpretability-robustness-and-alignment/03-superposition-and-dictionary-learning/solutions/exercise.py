import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # superposition & dictionary learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `sparse_feature_batch`: synthetic data made of sparse features → `mlfs.safety` |
    | 2 | `ToySuperposition`: the toy model of superposition → `mlfs.safety` |
    | 3 | `superposition_loss`, `train_toy_model`, `feature_dimensionality`: train it and measure how features share dimensions |
    | 4 | `SparseAutoencoder`: a sparse dictionary for a model's activations → `mlfs.safety` |
    | 5 | `train_sae`, `mmcs`, `sae_stats`: train the dictionary and judge what it found |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sparse_feature_batch(n, n_features, sparsity, *, generator=None):
    """stage 1: (n, n_features) float32: uniform [0, 1) values, each independently zero with probability sparsity."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    values = torch.rand(n, n_features, generator=generator)
    keep = torch.rand(n, n_features, generator=generator) >= sparsity
    return values * keep


@app.cell(hide_code=True)
def _():
    checks.run(1, sparse_feature_batch=sparse_feature_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ToySuperposition(nn.Module):
    """Elhage et al. (2022): n_features squeezed through n_hidden dimensions and read back with a ReLU."""

    def __init__(self, n_features, n_hidden, *, importance=None):
        """stage 2: W (n_hidden, n_features) Xavier-normal, b zeros (n_features,), buffer importance (default ones)."""
        super().__init__()
        self.n_features, self.n_hidden = n_features, n_hidden
        self.W = nn.Parameter(torch.empty(n_hidden, n_features))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(n_features))
        imp = torch.ones(n_features) if importance is None else torch.as_tensor(importance, dtype=torch.float32)
        if imp.shape != (n_features,):
            raise ValueError(f"importance must have shape ({n_features},), got {tuple(imp.shape)}")
        self.register_buffer("importance", imp.clone())

    def forward(self, x):
        """stage 2: x (B, n_features) -> ReLU(W^T W x + b), (B, n_features)."""
        return torch.relu((x @ self.W.T) @ self.W + self.b)


@app.cell(hide_code=True)
def _():
    checks.run(2, ToySuperposition=ToySuperposition, sparse_feature_batch=sparse_feature_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def superposition_loss(model, x):
    """stage 3: importance-weighted squared error: mean over the batch of sum_i importance_i (x_i - model(x)_i)^2."""
    return (model.importance * (x - model(x)) ** 2).sum(-1).mean()


@app.function
def train_toy_model(model, sparsity, *, steps=3000, batch_size=1024, lr=1e-3, generator=None):
    """stage 3: Adam on fresh sparse_feature_batch draws each step; returns the list of per-step losses (floats)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        x = sparse_feature_batch(batch_size, model.n_features, sparsity, generator=generator)
        loss = superposition_loss(model, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.function
def feature_dimensionality(W):
    """stage 3: D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2 for the columns W_i of W (n_hidden, n_features); 0 for zero columns."""
    W = torch.as_tensor(W).detach().float()
    norms = W.norm(dim=0)
    unit = W / norms.clamp_min(1e-12)
    overlap = ((unit.T @ W) ** 2).sum(dim=1)
    D = norms ** 2 / overlap.clamp_min(1e-24)
    return torch.where(norms > 1e-8, D, torch.zeros_like(D))


@app.cell(hide_code=True)
def _():
    checks.run(3, superposition_loss=superposition_loss, train_toy_model=train_toy_model,
               feature_dimensionality=feature_dimensionality, ToySuperposition=ToySuperposition,
               sparse_feature_batch=sparse_feature_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class SparseAutoencoder(nn.Module):
    """a one-layer sparse autoencoder: ReLU features over an overcomplete dictionary of unit-norm directions."""

    def __init__(self, d_in, d_hidden, *, l1_coeff=0.001):
        """stage 4: W_dec (d_in, d_hidden) random unit columns, W_enc = W_dec copy, b_enc (d_hidden,), b_dec (d_in,) zeros."""
        super().__init__()
        self.d_in, self.d_hidden, self.l1_coeff = d_in, d_hidden, l1_coeff
        dec = torch.randn(d_in, d_hidden)
        dec = dec / dec.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(dec)
        self.W_enc = nn.Parameter(dec.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def dictionary(self):
        """stage 4: the decoder directions with unit-norm columns, (d_in, d_hidden)."""
        return self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def encode(self, x):
        """stage 4: f = ReLU((x - b_dec) W_enc + b_enc), (B, d_hidden)."""
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        """stage 4: x_hat = f dictionary()^T + b_dec, (B, d_in)."""
        return f @ self.dictionary().T + self.b_dec

    def forward(self, x):
        """stage 4: (x_hat, f, loss) with loss = mean_b ||x_hat - x||^2 + l1_coeff mean_b ||f||_1."""
        f = self.encode(x)
        x_hat = self.decode(f)
        loss = ((x_hat - x) ** 2).sum(-1).mean() + self.l1_coeff * f.abs().sum(-1).mean()
        return x_hat, f, loss


@app.cell(hide_code=True)
def _():
    checks.run(4, SparseAutoencoder=SparseAutoencoder)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_sae(sae, acts, *, steps=2000, batch_size=256, lr=1e-3, generator=None):
    """stage 5: Adam on minibatches of rows of acts (drawn with generator); returns the list of per-step losses."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses = []
    sae.train()
    for _ in range(steps):
        idx = torch.randint(0, acts.shape[0], (batch_size,), generator=generator)
        _, _, loss = sae(acts[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.function
def mmcs(learned, true):
    """stage 5: mean over the columns of true (d, n) of the max cosine similarity with a column of learned (d, m)."""
    learned = torch.as_tensor(learned).detach().float()
    true = torch.as_tensor(true).detach().float()
    lu = learned / learned.norm(dim=0, keepdim=True).clamp_min(1e-12)
    tu = true / true.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return float((tu.T @ lu).max(dim=1).values.mean())


@app.function
def sae_stats(sae, acts):
    """stage 5: {'l0': mean active features per row, 'dead': fraction never active, 'fvu': ||x - x_hat||^2 / ||x - mean||^2}."""
    with torch.no_grad():
        x_hat, f, _ = sae(acts)
        active = f > 0
        fvu = ((acts - x_hat) ** 2).sum() / ((acts - acts.mean(0)) ** 2).sum()
    return {"l0": float(active.sum(1).float().mean()), "dead": float((~active.any(0)).float().mean()),
            "fvu": float(fvu)}


@app.cell(hide_code=True)
def _():
    checks.run(5, train_sae=train_sae, mmcs=mmcs, sae_stats=sae_stats, SparseAutoencoder=SparseAutoencoder)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import sparse_feature_batch, ToySuperposition, SparseAutoencoder` gives
    you the toy model and the dictionary learner (SAELens-style libraries are unlocked).
    topic 04 turns from reading models to attacking them.
    """)
    return


if __name__ == "__main__":
    app.run()
