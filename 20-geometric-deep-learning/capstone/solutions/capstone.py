import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks
    from mlfs.gnn import EGNNLayer, GCNConv, GINConv, global_add_pool  # yours, from topics 03-05
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03

    SIGMA = np.array([[1.0, 1.1], [1.1, 1.2]])  # Lennard-Jones sigma for species pairs (A-A, A-B, B-B)
    EPS = np.array([[1.0, math.sqrt(0.6)], [math.sqrt(0.6), 0.6]])  # well depths
    KINDS = ("gcn", "gin", "gin_pos", "egnn")
    R_CUT = 2.0  # the potential is truncated (and shifted to 0) at this distance; also the default graph cutoff


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · capstone: GNNs on molecular energies (reference solution)

    read [`brief.md`](brief.md) first. this notebook builds the benchmark (Lennard-Jones
    clusters of two atom species), four models (GCN, GIN, GIN with raw coordinates, EGNN), a
    training loop, an invariance audit and bootstrap intervals, then runs the comparison. the
    checks in `test_capstone.py` pin down the specification; the design is yours.

    | stage | you build |
    |---|---|
    | 1 | `lj_energy`, `make_dataset`: the benchmark |
    | 2 | `collate`: a batch of molecules as one radius graph |
    | 3 | `build_model`: GCN, GIN, GIN + coordinates, EGNN |
    | 4 | `fit`, `evaluate`: training and per-molecule errors |
    | 5 | `invariance_report`, `mae_ci`, `mae_difference_ci`: the audit and the uncertainty |
    """)
    return


@app.function
def lj_energy(pos, species):
    """stage 1: truncated, shifted Lennard-Jones energy: sum over pairs with r < R_CUT of V_ab(r) - V_ab(R_CUT),
    V_ab(r) = 4 eps_ab ((sigma_ab/r)^12 - (sigma_ab/r)^6); a float (float64 math)."""
    pos = np.asarray(pos, dtype=np.float64)
    species = np.asarray(species)
    iu = np.triu_indices(len(pos), 1)
    r = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
    a, b = species[iu[0]], species[iu[1]]

    def v(dist):
        s6 = (SIGMA[a, b] / dist) ** 6
        return 4 * EPS[a, b] * (s6 * s6 - s6)

    return float(np.where(r < R_CUT, v(r) - v(R_CUT), 0.0).sum())


@app.function
def make_dataset(n_molecules, *, n_atoms=(5, 12), seed=0):
    """stage 1: list of dicts {pos (n, 3) float64 tensor, centred; species (n,) int64; energy float}, by random
    sequential growth: each new atom at 1.0-1.5 sigma from a random existing atom, never closer than 0.95 sigma to any."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_molecules):
        n = int(rng.integers(n_atoms[0], n_atoms[1] + 1))
        species = rng.integers(0, 2, n)
        pos = [np.zeros(3)]
        tries = 0
        while len(pos) < n:
            tries += 1
            if tries > 10_000:
                raise RuntimeError("could not place an atom; the cluster is too crowded")
            k = len(pos)
            j = int(rng.integers(k))
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            cand = pos[j] + rng.uniform(1.0, 1.5) * SIGMA[species[k], species[j]] * d
            dists = np.linalg.norm(np.array(pos) - cand, axis=1)
            if np.all(dists >= 0.95 * SIGMA[species[k], species[:k]]):
                pos.append(cand)
        p = np.array(pos)
        p -= p.mean(axis=0)
        out.append({"pos": torch.tensor(p, dtype=torch.float64), "species": torch.tensor(species, dtype=torch.int64),
                    "energy": lj_energy(p, species)})
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", lj_energy=lj_energy, make_dataset=make_dataset)
    return


@app.function
def collate(molecules, *, cutoff=R_CUT, dtype=torch.float32):
    """stage 2: one disjoint graph: x one-hot species (N, 2), pos (N, 3), edge_index (radius graph within each molecule,
    both directions, sorted per molecule), batch (N,), y energies (B,), num_graphs; floats in `dtype`."""
    xs, ps, eis, bs, off = [], [], [], [], 0
    for g, m in enumerate(molecules):
        p = m["pos"].to(dtype)
        n = len(p)
        keep = torch.cdist(p, p) < cutoff
        keep.fill_diagonal_(False)
        src, dst = keep.nonzero(as_tuple=True)
        xs.append(F.one_hot(m["species"], 2).to(dtype))
        ps.append(p)
        eis.append(torch.stack([src, dst]) + off)
        bs.append(torch.full((n,), g, dtype=torch.int64))
        off += n
    return {"x": torch.cat(xs), "pos": torch.cat(ps), "edge_index": torch.cat(eis, dim=1).to(torch.int64),
            "batch": torch.cat(bs), "y": torch.tensor([m["energy"] for m in molecules], dtype=dtype),
            "num_graphs": len(molecules)}


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", collate=collate, make_dataset=make_dataset)
    return


@app.class_definition
class EnergyModel(nn.Module):
    """embedding -> n_layers message-passing layers -> per-atom MLP -> sum: E = sum_i (scale * e_i + per_atom_mean)."""

    def __init__(self, kind, hidden=64, n_layers=3):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.embed = nn.Linear(5 if kind == "gin_pos" else 2, hidden)
        if kind == "gcn":
            self.layers = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        elif kind in ("gin", "gin_pos"):
            self.layers = nn.ModuleList([GINConv(hidden, hidden) for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("atom_mean", torch.zeros(()))
        self.register_buffer("scale", torch.ones(()))

    def forward(self, batch):
        x, pos, ei = batch["x"], batch["pos"], batch["edge_index"]
        h = self.embed(torch.cat([x, pos], dim=1) if self.kind == "gin_pos" else x)
        for layer in self.layers:
            if self.kind == "egnn":
                h, pos = layer(h, pos, ei)
            else:
                h = h + F.relu(layer(h, ei))
        e = self.scale * self.head(h)[:, 0] + self.atom_mean
        return global_add_pool(e[:, None], batch["batch"], batch["num_graphs"])[:, 0]


@app.function
def build_model(kind, *, hidden=64, n_layers=3, seed=0):
    """stage 3: an nn.Module for kind in {'gcn', 'gin', 'gin_pos', 'egnn'}; model(collate(mols)) -> (B,) energies."""
    torch.manual_seed(seed)
    return EnergyModel(kind, hidden, n_layers)


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", build_model=build_model, collate=collate, make_dataset=make_dataset)
    return


@app.function
def fit(model, molecules, *, epochs=60, batch_size=32, lr=3e-3, cutoff=R_CUT, seed=0):
    """stage 4: train on molecules (per-atom target normalization, Adam, cosine schedule, grad clipping); returns model."""
    g = torch.Generator().manual_seed(seed)
    n_at = np.array([len(m["species"]) for m in molecules], dtype=np.float64)
    e = np.array([m["energy"] for m in molecules])
    mu = float(e.sum() / n_at.sum())
    sd = float(np.std(e - mu * n_at)) or 1.0
    with torch.no_grad():
        model.atom_mean.fill_(mu)
        model.scale.fill_(sd / max(1.0, float(np.mean(n_at))) ** 0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(molecules) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    batches = [collate(molecules[i:i + batch_size], cutoff=cutoff) for i in range(0, len(molecules), batch_size)]
    model.train()
    for _ in range(epochs):
        for k in torch.randperm(len(batches), generator=g).tolist():
            b = batches[k]
            loss = F.mse_loss(model(b) / sd, b["y"] / sd)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


@app.function
def evaluate(model, molecules, *, cutoff=R_CUT, batch_size=256):
    """stage 4: per-molecule absolute errors |prediction - energy|, a float64 ndarray (n,)."""
    errs = []
    with torch.no_grad():
        for i in range(0, len(molecules), batch_size):
            b = collate(molecules[i:i + batch_size], cutoff=cutoff)
            errs.append((model(b).double() - b["y"].double()).abs().numpy())
    return np.concatenate(errs)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", fit=fit, evaluate=evaluate, build_model=build_model,
               make_dataset=make_dataset)
    return


@app.function
def invariance_report(model, molecules, *, n_transforms=3, cutoff=R_CUT, seed=0):
    """stage 5: in float64, the largest |change in predicted energy| under random rotations, reflections,
    translations and atom relabellings: dict with keys 'rotation', 'reflection', 'translation', 'permutation'."""
    m64 = copy.deepcopy(model).double().eval()
    rng = np.random.default_rng(seed)

    def predict(mols):
        with torch.no_grad():
            return m64(collate(mols, cutoff=cutoff, dtype=torch.float64))

    base = predict(molecules)
    out = {k: 0.0 for k in ("rotation", "reflection", "translation", "permutation")}
    for _ in range(n_transforms):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        R = torch.tensor(q)
        M = R @ torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
        t = torch.tensor(rng.normal(size=3) * 3.0)
        variants = {
            "rotation": [dict(m, pos=m["pos"] @ R.T) for m in molecules],
            "reflection": [dict(m, pos=m["pos"] @ M.T) for m in molecules],
            "translation": [dict(m, pos=m["pos"] + t) for m in molecules],
        }
        perms = [torch.tensor(rng.permutation(len(m["species"]))) for m in molecules]
        variants["permutation"] = [dict(m, pos=m["pos"][p], species=m["species"][p]) for m, p in zip(molecules, perms)]
        for k, mols in variants.items():
            out[k] = max(out[k], float((predict(mols) - base).abs().max()))
    return out


@app.function
def mae_ci(errors, *, n_boot=2000, alpha=0.05, seed=0):
    """stage 5: (MAE, lo, hi): the mean absolute error and its percentile bootstrap interval (mlfs.stats.bootstrap_ci)."""
    errors = np.asarray(errors, dtype=np.float64)
    lo, hi = bootstrap_ci(errors, np.mean, n_boot=n_boot, alpha=alpha, rng=seed)
    return float(errors.mean()), lo, hi


@app.function
def mae_difference_ci(errors_a, errors_b, *, n_boot=2000, alpha=0.05, seed=0):
    """stage 5: (MAE_a - MAE_b, lo, hi) with a PAIRED bootstrap over test molecules (both arrays resampled jointly)."""
    a, b = np.asarray(errors_a, dtype=np.float64), np.asarray(errors_b, dtype=np.float64)
    lo, hi = bootstrap_ci((a, b), lambda x, y: float(np.mean(x) - np.mean(y)), n_boot=n_boot, alpha=alpha, rng=seed)
    return float(a.mean() - b.mean()), lo, hi


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", invariance_report=invariance_report, mae_ci=mae_ci,
               mae_difference_ci=mae_difference_ci, build_model=build_model, make_dataset=make_dataset)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the experiment

    the small config runs in a minute (and is what `MLFS_CAPSTONE=small`, the default, does);
    `MLFS_CAPSTONE=full` runs the configuration the brief reports (14–18 minutes on a laptop
    CPU). seeds change the model initialization and the data order; the datasets are fixed.
    """)
    return


@app.cell
def _():
    FULL = os.environ.get("MLFS_CAPSTONE", "small") == "full"
    CFG = (dict(train_sizes=(250, 1000, 4000), n_test=1000, epochs=60, seeds=(0, 1, 2), hidden=64) if FULL else
           dict(train_sizes=(100, 300), n_test=200, epochs=20, seeds=(0,), hidden=32))
    train_all = make_dataset(max(CFG["train_sizes"]), seed=1)
    test_set = make_dataset(CFG["n_test"], seed=2)
    _e = np.array([m["energy"] for m in test_set])
    mo.md(f"config: **{'full' if FULL else 'small'}** {CFG} · test energies: mean {_e.mean():.3f}, std {_e.std():.3f}")
    return CFG, test_set, train_all


@app.cell
def _(CFG, test_set, train_all):
    results, models, timings = {}, {}, {}
    _e_test = np.array([m["energy"] for m in test_set])
    for _n in CFG["train_sizes"]:
        _train = train_all[:_n]
        _n_at = np.array([len(m["species"]) for m in _train])
        _mu = sum(m["energy"] for m in _train) / _n_at.sum()
        results[("per-atom mean", _n, 0)] = np.abs(_mu * np.array([len(m["species"]) for m in test_set]) - _e_test)
        for _kind in KINDS:
            for _seed in CFG["seeds"]:
                _t0 = time.perf_counter()
                _m = fit(build_model(_kind, hidden=CFG["hidden"], seed=_seed), _train, epochs=CFG["epochs"], seed=_seed)
                timings[(_kind, _n, _seed)] = time.perf_counter() - _t0
                results[(_kind, _n, _seed)] = evaluate(_m, test_set)
                models[(_kind, _n, _seed)] = _m
    return models, results, timings


@app.cell
def _(CFG, results, timings):
    _n = max(CFG["train_sizes"])
    _rows = []
    for _kind in ("per-atom mean",) + KINDS:
        _seeds = (0,) if _kind == "per-atom mean" else CFG["seeds"]
        _maes = [results[(_kind, _n, s)].mean() for s in _seeds]
        _mae, _lo, _hi = mae_ci(results[(_kind, _n, 0)])
        _t = "" if _kind == "per-atom mean" else f"{np.mean([timings[(_kind, _n, s)] for s in _seeds]):.0f} s"
        _rows.append(f"| {_kind} | {_mae:.4f} [{_lo:.4f}, {_hi:.4f}] | " + ", ".join(f"{v:.4f}" for v in _maes)
                     + f" | {_t} |")
    _d, _dlo, _dhi = mae_difference_ci(results[("gin", _n, 0)], results[("egnn", _n, 0)])
    _d2, _dlo2, _dhi2 = mae_difference_ci(results[("gin_pos", _n, 0)], results[("gin", _n, 0)])
    _vs_base = " · ".join(
        "{}: {:.3f} [{:.3f}, {:.3f}]".format(k, *mae_difference_ci(results[(k, _n, 0)], results[("per-atom mean", _n, 0)]))
        for k in KINDS)
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for _kind in KINDS:
        _ax.loglog(CFG["train_sizes"], [np.mean([results[(_kind, n, s)].mean() for s in CFG["seeds"]])
                                        for n in CFG["train_sizes"]], "o-", label=_kind)
    _ax.loglog(CFG["train_sizes"], [results[("per-atom mean", n, 0)].mean() for n in CFG["train_sizes"]], "k--",
               label="per-atom mean")
    _ax.set(xlabel="training molecules", ylabel="test MAE (energy units)", title="learning curves")
    _ax.set_xticks(CFG["train_sizes"], [str(n) for n in CFG["train_sizes"]])
    _ax.xaxis.set_minor_locator(plt.NullLocator())
    _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    _curve = "| model | " + " | ".join(f"n = {n}" for n in CFG["train_sizes"]) + " |\n|---|" + "---|" * len(
        CFG["train_sizes"]) + "\n" + "\n".join(
        f"| {k} | " + " | ".join(
            f"{np.mean([results[(k, n, s)].mean() for s in ((0,) if k == 'per-atom mean' else CFG['seeds'])]):.4f} ("
            + " / ".join(f"{results[(k, n, s)].mean():.3f}" for s in ((0,) if k == 'per-atom mean' else CFG['seeds'])) + ")"
            for n in CFG["train_sizes"]) + " |" for k in ("per-atom mean",) + KINDS)
    mo.vstack([mo.md(f"n_train = {_n}: test MAE, seed 0 with 95% bootstrap CI over test molecules; all seeds\n\n"
                     "| model | MAE [95% CI] | MAE per seed | train time |\n|---|---|---|---|\n" + "\n".join(_rows)
                     + f"\n\nGIN − EGNN (seed 0, paired): {_d:.3f} [{_dlo:.3f}, {_dhi:.3f}]"
                     + f"\n\ngin_pos − GIN (seed 0, paired): {_d2:.4f} [{_dlo2:.4f}, {_dhi2:.4f}]"
                     + f"\n\nmodel − per-atom mean (seed 0, paired): {_vs_base}"
                     + "\n\nmean test MAE over seeds (per seed in brackets), by training-set size\n\n" + _curve), _fig])
    return


@app.cell
def _(CFG, models, test_set):
    _n = max(CFG["train_sizes"])
    _rows = []
    for _kind in KINDS:
        _r = invariance_report(models[(_kind, _n, 0)], test_set[:100])
        _rows.append(f"| {_kind} | " + " | ".join(f"{_r[k]:.1e}" for k in ("rotation", "reflection", "translation",
                                                                             "permutation")) + " |")
    mo.md("largest change in predicted energy (float64, 100 test molecules, 3 random transforms each)\n\n"
          "| model | rotation | reflection | translation | permutation |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    the reference answers to the brief's questions are in `brief.md`'s "reference results"; a
    learner's write-up goes here: results with intervals and baselines, the invariance audit,
    the learning curves, and the answers to the analysis questions.
    """)
    return


if __name__ == "__main__":
    app.run()
