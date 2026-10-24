import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # energy-based models · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. an energy-based model is any
    network $E_\theta(x)$ that outputs one number per input; it defines the density
    $p_\theta(x) \propto e^{-E_\theta(x)}$ without ever computing the normalizer. you'll
    build the sampler that only needs $\nabla_x E$ (Langevin dynamics), its exact
    Metropolis-corrected cousin, the contrastive-divergence loss, and a training loop
    with a persistent replay buffer.

    | stage | you build |
    |---|---|
    | 1 | `score_from_energy`, `langevin_dynamics`: sampling with only a gradient → `mlfs.diffusion` |
    | 2 | `mala`: Langevin with a Metropolis correction, so the step size no longer biases the samples |
    | 3 | `EnergyMLP`, `ebm_loss`: an energy network and contrastive divergence → `mlfs.diffusion` |
    | 4 | `train_ebm`: persistent short-run MCMC with a replay buffer |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def score_from_energy(energy_fn):
    """stage 1: returns score_fn(x) = -grad_x sum(energy_fn(x)), detached, same shape as x (works under no_grad)."""

    def score_fn(x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(energy_fn(x).sum(), x)
        return -g.detach()

    return score_fn


@app.function
def langevin_dynamics(score_fn, x0, *, step_size=0.01, n_steps=100, generator=None):
    """stage 1: n_steps of x <- x + (step_size / 2) score_fn(x) + sqrt(step_size) z, z = torch.randn(x.shape,
    generator=generator) drawn on the CPU and moved to x's device; x0 is not modified; returns a detached x."""
    x = x0.detach().clone()
    with torch.no_grad():
        for _ in range(n_steps):
            z = torch.randn(x.shape, generator=generator).to(x.device)
            x = x + 0.5 * step_size * score_fn(x) + math.sqrt(step_size) * z
    return x


@app.cell(hide_code=True)
def _():
    checks.run(1, score_from_energy=score_from_energy, langevin_dynamics=langevin_dynamics)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mala(energy_fn, x0, *, step_size=0.01, n_steps=100, generator=None):
    """stage 2: Metropolis-adjusted Langevin on energy_fn (x (B, ...) -> (B,)). per step: z = torch.randn(x.shape,
    generator), proposal x' = x - (h/2) grad E(x) + sqrt(h) z, then u = torch.rand(B, generator); accept row b if
    log u_b < E(x) - E(x') + log q(x | x') - log q(x' | x). returns (x detached, acceptance rate as a float)."""
    h = step_size
    x = x0.detach().clone()
    B = x.shape[0]

    def energy_and_grad(v):
        with torch.enable_grad():
            v = v.detach().requires_grad_(True)
            e = energy_fn(v)
            (g,) = torch.autograd.grad(e.sum(), v)
        return e.detach(), g.detach()

    def log_q(to, frm, grad_frm):
        """log q(to | frm) up to a constant: -||to - frm + (h/2) grad E(frm)||^2 / (2h), per row."""
        return -((to - frm + 0.5 * h * grad_frm) ** 2).flatten(1).sum(1) / (2 * h)

    e, g = energy_and_grad(x)
    accepted = 0.0
    for _ in range(n_steps):
        z = torch.randn(x.shape, generator=generator).to(x.device)
        prop = x - 0.5 * h * g + math.sqrt(h) * z
        e_p, g_p = energy_and_grad(prop)
        log_alpha = e - e_p + log_q(x, prop, g_p) - log_q(prop, x, g)
        u = torch.rand(B, generator=generator).to(x.device)
        acc = torch.log(u) < log_alpha
        shape = (B,) + (1,) * (x.dim() - 1)
        x = torch.where(acc.view(shape), prop, x)
        g = torch.where(acc.view(shape), g_p, g)
        e = torch.where(acc, e_p, e)
        accepted += acc.float().mean().item()
    return x, accepted / max(n_steps, 1)


@app.cell(hide_code=True)
def _():
    checks.run(2, mala=mala)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class EnergyMLP(nn.Module):
    """a scalar energy E(x) for vectors x (B, data_dim) -> (B,), with smooth (SiLU) activations."""

    def __init__(self, data_dim=2, *, hidden_dim=128, n_layers=3):
        """stage 3: an MLP data_dim -> hidden (n_layers hidden layers) -> 1 with smooth activations."""
        super().__init__()
        layers, d = [], data_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """stage 3: x (B, data_dim) -> energies (B,)."""
        return self.net(x).squeeze(-1)


@app.function
def ebm_loss(energy_real, energy_fake, *, alpha=0.1):
    """stage 3: contrastive divergence mean(E_real) - mean(E_fake) + alpha (mean(E_real^2) + mean(E_fake^2))."""
    return (energy_real.mean() - energy_fake.mean()
            + alpha * (energy_real.pow(2).mean() + energy_fake.pow(2).mean()))


@app.cell(hide_code=True)
def _():
    checks.run(3, EnergyMLP=EnergyMLP, ebm_loss=ebm_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_ebm(energy, X, *, n_iters=500, batch_size=128, lr=1e-3, langevin_steps=20, step_size=0.01,
              buffer_size=10000, reinit_prob=0.05, alpha=0.1, seed=0):
    """stage 4: persistent contrastive divergence. g = torch.Generator().manual_seed(seed); a replay buffer of
    buffer_size points uniform on the data's bounding box; each iteration: a real batch X[randint], batch_size buffer
    entries (randint), each re-initialized to fresh box noise with prob reinit_prob, run through langevin_dynamics on
    score_from_energy(energy) (langevin_steps, step_size, generator g) and written back; Adam(lr) on
    ebm_loss(energy(real), energy(fake), alpha). returns dict(loss, e_real, e_fake lists of floats, buffer tensor)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    lo, hi = X.min(0).values, X.max(0).values

    def box_noise(n):
        return lo + (hi - lo) * torch.rand((n, X.shape[1]), generator=g)

    buffer = box_noise(buffer_size)
    opt = torch.optim.Adam(energy.parameters(), lr=lr)
    score = score_from_energy(energy)
    hist = {"loss": [], "e_real": [], "e_fake": []}
    for _ in range(n_iters):
        real = X[torch.randint(len(X), (batch_size,), generator=g)]
        idx = torch.randint(buffer_size, (batch_size,), generator=g)
        start = buffer[idx]
        fresh = torch.rand(batch_size, generator=g) < reinit_prob
        start = torch.where(fresh[:, None], box_noise(batch_size), start)
        fake = langevin_dynamics(score, start, step_size=step_size, n_steps=langevin_steps, generator=g)
        buffer[idx] = fake
        e_real, e_fake = energy(real), energy(fake)
        loss = ebm_loss(e_real, e_fake, alpha=alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["e_real"].append(e_real.mean().item())
        hist["e_fake"].append(e_fake.mean().item())
    hist["buffer"] = buffer
    return hist


@app.cell(hide_code=True)
def _():
    checks.run(4, train_ebm=train_ebm, EnergyMLP=EnergyMLP)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import langevin_dynamics` gives you the sampler; topic 02 feeds it
    a learned score instead of an energy gradient, and every diffusion sampler in this
    chapter is a descendant of it.
    """)
    return


if __name__ == "__main__":
    app.run()
