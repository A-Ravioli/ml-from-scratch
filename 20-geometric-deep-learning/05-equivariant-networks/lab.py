import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import notebooks, ui
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    N_ATOMS = 6
    TYPES = torch.tensor([0, 0, 1, 1, 2, 2])  # every molecule has the same atoms, listed in the same order
    W = torch.tensor([[1.0, -0.5, 0.8], [-0.5, 1.5, -1.0], [0.8, -1.0, 0.6]])  # pair weights by atom type


@app.function
def random_rotations(n, rng, proper=True):
    """n Haar-random 3x3 orthogonal matrices (rotations if proper), float32 tensor (n, 3, 3)."""
    q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    if proper:
        q[np.linalg.det(q) < 0, :, 0] *= -1
    return torch.tensor(q, dtype=torch.float32)


@app.function
def canonical_frame(pos):
    """rotate each molecule (B, N, 3) into its principal-axes frame (largest spread on x), a proper rotation with a
    sign convention: like a database that stores every molecule in a standard orientation."""
    c = pos - pos.mean(1, keepdim=True)
    _, vecs = torch.linalg.eigh(c.transpose(1, 2) @ c)
    vecs = vecs.flip(-1)
    proj = c @ vecs
    signs = torch.sign((proj ** 3).sum(1, keepdim=True))
    signs[signs == 0] = 1
    vecs = vecs * signs
    vecs[torch.linalg.det(vecs) < 0, :, 2] *= -1  # proper rotation: never mirror the molecule
    return c @ vecs


@app.function
def energy(pos):
    """a smooth pair potential: sum over atom pairs of W[type_i, type_j] * exp(-r_ij^2 / 2)."""
    d2 = torch.cdist(pos, pos) ** 2
    w = W[TYPES][:, TYPES]
    iu = torch.triu_indices(N_ATOMS, N_ATOMS, 1)
    return (w[iu[0], iu[1]] * torch.exp(-d2[:, iu[0], iu[1]] / 2)).sum(1)


@app.function
def make_molecules(n, seed, canonical=True):
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.normal(size=(n, N_ATOMS, 3)), dtype=torch.float32) * 1.2
    if canonical:
        pos = canonical_frame(pos)
    return pos, energy(pos)


@app.function
def egnn_inputs(pos):
    """batch of B molecules (B, N, 3) -> (x, pos, edge_index, batch) on fully connected graphs."""
    B = pos.shape[0]
    flat = pos.reshape(B * N_ATOMS, 3)
    batch = torch.arange(B).repeat_interleave(N_ATOMS)
    x = F.one_hot(TYPES.repeat(B), 3).float()
    return x, flat, ex.radius_graph(flat, float("inf"), batch), batch


@app.class_definition
class CoordMLP(torch.nn.Module):
    """an MLP on the flattened coordinates (atom order fixed, so the types are implicit)."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = MLP(3 * N_ATOMS, [128, 128], 1)

    def forward(self, pos):
        return self.net(pos.reshape(len(pos), -1))[:, 0]


@app.class_definition
class EGNNModel(torch.nn.Module):
    """your EGNN (3 layers, width 32) on the fully connected molecule."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = ex.EGNN(3, 32, 1, n_layers=3)

    def forward(self, pos):
        x, p, ei, b = egnn_inputs(pos)
        return self.net(x, p, ei, b, num_graphs=len(pos))[:, 0]


@app.function
def train_model(model, pos, y, *, steps=600, batch_size=128, augment=False, seed=0):
    """Adam on MSE of standardized targets; augment = a fresh random rotation of every training molecule each step."""
    rng = np.random.default_rng(seed)
    mu, sd = float(y.mean()), float(y.std())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(pos), min(batch_size, len(pos))))
        p = pos[idx]
        if augment:
            p = p @ random_rotations(len(p), rng).transpose(1, 2)
        loss = F.mse_loss(model(p), (y[idx] - mu) / sd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    model.mu, model.sd = mu, sd
    return model


@app.function
def mae(model, pos, y):
    with torch.no_grad():
        return float((model(pos) * model.sd + model.mu - y).abs().mean())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # equivariant networks · lab

    every experiment here uses **your** `EGNN` and `radius_graph`. the "molecules" have six
    atoms of three types (always listed in the same order) at random positions, and their
    energy is a smooth pair potential: a sum over atom pairs of a type-dependent weight times
    $e^{-r^2/2}$. the energy depends only on interatomic distances, so it's invariant to
    rotations, reflections, translations and relabelling the atoms. like many real
    datasets (QM9 among them), ours stores every molecule in a **standard orientation**: its
    principal axes along $x, y, z$. for each experiment: write down a prediction, lock it in,
    run it, then explain what happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: turn the test molecules

    an MLP on the 18 coordinates and your EGNN, each trained on 2,000 molecules in standard
    orientation, then tested on 1,000 new molecules: in standard orientation, and randomly
    rotated. a third model is the MLP trained with **rotation augmentation** (every training
    molecule randomly rotated at every step). the table shows the mean absolute error; for
    scale, predicting the mean energy gives an error of about 0.5.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how does each model's error change when the test molecules are rotated? which model has the lowest error "
        "on molecules in standard orientation, and by how much?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _pos, _y = make_molecules(2000, seed=0)
    _pt, _yt = make_molecules(1000, seed=1)
    _rot = _pt @ random_rotations(len(_pt), np.random.default_rng(2)).transpose(1, 2)
    _base = float((_yt - _y.mean()).abs().mean())
    exp1 = {
        "MLP": train_model(CoordMLP(), _pos, _y),
        "MLP + rotation augmentation": train_model(CoordMLP(), _pos, _y, augment=True),
        "EGNN": train_model(EGNNModel(), _pos, _y),
    }
    _rows = [f"| {k} | {mae(m, _pt, _yt):.4f} | {mae(m, _rot, _yt):.4f} |" for k, m in exp1.items()]
    mo.md("| model | standard orientation | randomly rotated |\n|---|---|---|\n" + "\n".join(_rows)
          + f"\n\npredicting the mean: {_base:.4f}")
    return (exp1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what did the plain MLP actually learn, given that it only ever saw molecules
    in one orientation? why does augmentation help with rotated molecules but cost accuracy on
    upright ones? the EGNN's two columns agree to every printed digit: why is that guaranteed,
    rather than learned?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many molecules do you need?

    test error on randomly rotated molecules against the number of training molecules, for
    the augmented MLP and the EGNN (same 600 training steps each).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 50 training molecules, how many times larger is the augmented MLP's error than the EGNN's? does the gap "
        "close at 3,200?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _sizes = (50, 200, 800, 3200)
    _pt, _yt = make_molecules(1000, seed=1)
    _rot = _pt @ random_rotations(len(_pt), np.random.default_rng(2)).transpose(1, 2)
    _res = {"MLP + augmentation": [], "EGNN": []}
    for _n in _sizes:
        _pos, _y = make_molecules(_n, seed=10 + _n)
        _res["MLP + augmentation"].append(mae(train_model(CoordMLP(), _pos, _y, augment=True), _rot, _yt))
        _res["EGNN"].append(mae(train_model(EGNNModel(), _pos, _y), _rot, _yt))
    _fig, _ax = plt.subplots(figsize=(5.5, 3.6))
    for _k, _v in _res.items():
        _ax.loglog(_sizes, _v, "o-", label=_k)
    _ax.set(xlabel="training molecules", ylabel="test MAE (rotated molecules)", title="sample efficiency")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| model | " + " | ".join(map(str, _sizes)) + " |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {k} | " + " | ".join(f"{a:.4f}" for a in v) + " |" for k, v in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** augmentation shows the MLP rotated copies of each molecule; how many does it
    need to see before it "knows" the energy doesn't depend on orientation, and where does
    that knowledge live? why does building the symmetry in help most when data is scarce?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: left hand, right hand

    a new target: four atoms of four different types at random positions, and the **signed
    volume** $\det[x_1 - x_0,\ x_2 - x_0,\ x_3 - x_0]$. it doesn't change under rotations or
    translations, but a mirror image flips its sign: it tells a molecule from its mirror
    image, which is what chirality means (and why one hand of a drug can work while the other
    is toxic). we train the augmented MLP and an EGNN on 2,000 such molecules and test on
    1,000; then we feed each test molecule's mirror image.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the signed volume is a simple function of the coordinates and is rotation-invariant. which model learns it? "
        "what does each predict for a molecule and its mirror image?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _chiral(n, seed):
        _rng = np.random.default_rng(seed)
        _p = torch.tensor(_rng.normal(size=(n, 4, 3)), dtype=torch.float32)
        _v = torch.linalg.det(_p[:, 1:] - _p[:, :1])
        return _p, _v

    class _MLP4(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = MLP(12, [128, 128], 1)

        def forward(self, p):
            return self.net(p.reshape(len(p), -1))[:, 0]

    class _EGNN4(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = ex.EGNN(4, 32, 1, n_layers=3)

        def forward(self, p):
            _B = len(p)
            _flat = p.reshape(_B * 4, 3)
            _b = torch.arange(_B).repeat_interleave(4)
            _x = torch.eye(4).repeat(_B, 1)
            return self.net(_x, _flat, ex.radius_graph(_flat, float("inf"), _b), _b, num_graphs=_B)[:, 0]

    _p, _v = _chiral(2000, 5)
    _pt, _vt = _chiral(1000, 6)
    _mirror = _pt * torch.tensor([-1.0, 1.0, 1.0])
    _rows, _scatter = [], {}
    for _name, _m in (("MLP + rotation augmentation", _MLP4()), ("EGNN", _EGNN4())):
        _m = train_model(_m, _p, _v, augment=True)
        with torch.no_grad():
            _pred = _m(_pt) * _m.sd + _m.mu
            _pm = _m(_mirror) * _m.sd + _m.mu
        _r2 = 1 - float(((_pred - _vt) ** 2).mean() / _vt.var())
        _rows.append(f"| {_name} | {_r2:.3f} | {float((_pred - _pm).abs().max()):.2e} | {float(_pred.std()):.3f} |")
        _scatter[_name] = _pred
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _name, _pred in _scatter.items():
        _ax.scatter(_vt, _pred, s=6, alpha=0.5, label=_name)
    _ax.plot([-6, 6], [-6, 6], "k--", lw=1)
    _ax.set(xlabel="true signed volume", ylabel="predicted", xlim=(-6, 6), ylim=(-6, 6), title="chirality")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("| model | test R² | max \\|f(mol) − f(mirror)\\| | std of predictions |\n|---|---|---|---|\n"
                     + "\n".join(_rows) + f"\n\nstd of the true signed volume: {float(_vt.std()):.3f}"), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the EGNN is invariant to all of $E(3)$, which includes reflections. what
    must it predict for a molecule and its mirror image, and what's the best such a function
    can do on a target that flips sign? which symmetry group should a model of chiral
    molecules respect instead, and what would you change in the EGNN to get it?
    """)
    return


if __name__ == "__main__":
    app.run()
