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
    from mlfs.generative import bits_per_dim, dequantize, discrete_nll  # yours, from topic 01


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # normalizing flows · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a normalizing flow is an
    invertible network from data to noise. the change-of-variables formula gives its
    exact log-likelihood, if the Jacobian determinant is cheap; coupling layers make it a
    sum. you'll build the coupling layer, RealNVP, a maximum-likelihood training loop, and
    the preprocessing that turns a flow's density on 8-bit images into honest bits/dim.

    | stage | you build |
    |---|---|
    | 1 | `standard_normal_log_prob`, `AffineCoupling`: an invertible layer with a triangular Jacobian → `mlfs.generative` |
    | 2 | `RealNVP`: a stack of couplings with alternating masks; exact `log_prob`, `sample` → `mlfs.generative` |
    | 3 | `fit_flow`: maximum-likelihood training |
    | 4 | `logit_transform`, `flow_bits_per_dim`: from 8-bit pixels to an unbounded space, and back to bits/dim |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def standard_normal_log_prob(z):
    """stage 1: log N(z; 0, I) summed over all non-batch dims -> (B,)."""
    return (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)


@app.class_definition
class AffineCoupling(nn.Module):
    """y = x_m + (1 - m) (x exp(s(x_m)) + t(x_m)) with x_m = m x; forward -> (y, log_det), inverse -> (x, log_det)."""

    def __init__(self, dim, mask, *, hidden_dim=128):
        """stage 1: a buffer mask (dim,) and networks s, t: dim -> hidden -> hidden -> dim (s bounded by a tanh scale)."""
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32).clone())
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.ones(dim))

    def _st(self, x_m):
        """s and t of the masked input; s = scale * tanh(raw) keeps exp(s) in a safe range; both zeroed on the mask."""
        raw_s, t = self.net(x_m).chunk(2, dim=1)
        s = self.scale * torch.tanh(raw_s)
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, x):
        """stage 1: x (B, dim) -> (y, log_det (B,)) with log_det = sum over unmasked dims of s(x_m)."""
        x_m = x * self.mask
        s, t = self._st(x_m)
        y = x_m + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(1)

    def inverse(self, y):
        """stage 1: y -> (x, log_det of the inverse = -sum s(y_m)); y_m = x_m because masked dims pass through."""
        y_m = y * self.mask
        s, t = self._st(y_m)
        x = y_m + (1 - self.mask) * (y - t) * torch.exp(-s)
        return x, -s.sum(1)


@app.cell(hide_code=True)
def _():
    checks.run(1, standard_normal_log_prob=standard_normal_log_prob, AffineCoupling=AffineCoupling)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class RealNVP(nn.Module):
    """data -> latent: n_layers AffineCouplings with masks alternating (i + layer) % 2; standard-normal base."""

    def __init__(self, dim, *, n_layers=8, hidden_dim=128):
        """stage 2: self.layers, an nn.ModuleList of AffineCoupling with mask_k[i] = (i + k) % 2."""
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            AffineCoupling(dim, (torch.arange(dim) + k) % 2, hidden_dim=hidden_dim) for k in range(n_layers))

    def forward(self, x):
        """stage 2: x (B, dim) -> (z, log_det (B,)), the sum of the layers' log-dets."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, z):
        """stage 2: z -> x through the layers' inverses in reverse order."""
        for layer in reversed(self.layers):
            z, _ = layer.inverse(z)
        return z

    def log_prob(self, x):
        """stage 2: exact log p(x) = log N(f(x); 0, I) + log |det df/dx|, (B,)."""
        z, log_det = self(x)
        return standard_normal_log_prob(z) + log_det

    def sample(self, n, *, generator=None):
        """stage 2: inverse(torch.randn((n, dim), generator=generator)) without gradients."""
        dev = next(self.parameters()).device
        with torch.no_grad():
            return self.inverse(torch.randn((n, self.dim), generator=generator).to(dev))


@app.cell(hide_code=True)
def _():
    checks.run(2, RealNVP=RealNVP, AffineCoupling=AffineCoupling, standard_normal_log_prob=standard_normal_log_prob)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def fit_flow(flow, X, *, n_steps=2000, batch_size=256, lr=1e-3, seed=0):
    """stage 3: Adam on the mean negative log_prob of minibatches X[torch.randint(len(X), (batch_size,), generator)],
    with a torch.Generator seeded by `seed`; returns the list of per-step losses (floats)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        idx = torch.randint(len(X), (batch_size,), generator=g)
        loss = -flow.log_prob(X[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(3, fit_flow=fit_flow, RealNVP=RealNVP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def logit_transform(x, *, alpha=0.05):
    """stage 4: x in [0, 1) (B, ...) -> (y, log_det (B,)): p = alpha + (1 - 2 alpha) x, y = log p - log(1 - p),
    log_det = sum log((1 - 2 alpha) / (p (1 - p)))."""
    p = alpha + (1 - 2 * alpha) * x
    y = torch.log(p) - torch.log1p(-p)
    log_det = (math.log(1 - 2 * alpha) - torch.log(p) - torch.log1p(-p)).flatten(1).sum(1)
    return y, log_det


@app.function
def flow_bits_per_dim(flow, x_levels, *, n_levels=256, alpha=0.05, generator=None):
    """stage 4: per-example bits/dim (B,) of integer data x_levels (B, ...) under a flow on logit space:
    dequantize (topic 01) -> logit_transform -> flow.log_prob(y), with y in x_levels' own shape (flatten the data
    yourself for a flow on vectors); log p(dequantized) = log_prob + log_det; then discrete_nll and bits_per_dim over
    D = the number of dimensions per example; no gradients."""
    with torch.no_grad():
        u = dequantize(x_levels, n_levels=n_levels, generator=generator)
        y, ld = logit_transform(u, alpha=alpha)
        log_p = flow.log_prob(y) + ld
        d = y[0].numel()
        return bits_per_dim(discrete_nll(log_p, d, n_levels=n_levels), d)


@app.cell(hide_code=True)
def _():
    checks.run(4, logit_transform=logit_transform, flow_bits_per_dim=flow_bits_per_dim, RealNVP=RealNVP)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import RealNVP` gives you the flow; the capstone trains it as the
    exact-likelihood latent-variable family, and chapter 22's continuous flows are its
    infinitely deep cousin.
    """)
    return


if __name__ == "__main__":
    app.run()
