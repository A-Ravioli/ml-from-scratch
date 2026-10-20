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
    # 20 · capstone: GNNs on molecular energies

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
    raise NotImplementedError("stage1: implement lj_energy")


@app.function
def make_dataset(n_molecules, *, n_atoms=(5, 12), seed=0):
    """stage 1: list of dicts {pos (n, 3) float64 tensor, centred; species (n,) int64; energy float}, by random
    sequential growth: each new atom at 1.0-1.5 sigma from a random existing atom, never closer than 0.95 sigma to any."""
    raise NotImplementedError("stage1: implement make_dataset")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", lj_energy=lj_energy, make_dataset=make_dataset)
    return


@app.function
def collate(molecules, *, cutoff=R_CUT, dtype=torch.float32):
    """stage 2: one disjoint graph: x one-hot species (N, 2), pos (N, 3), edge_index (radius graph within each molecule,
    both directions, sorted per molecule), batch (N,), y energies (B,), num_graphs; floats in `dtype`."""
    raise NotImplementedError("stage2: implement collate")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", collate=collate, make_dataset=make_dataset)
    return


@app.function
def build_model(kind, *, hidden=64, n_layers=3, seed=0):
    """stage 3: an nn.Module for kind in {'gcn', 'gin', 'gin_pos', 'egnn'}; model(collate(mols)) -> (B,) energies."""
    raise NotImplementedError("stage3: implement build_model")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", build_model=build_model, collate=collate, make_dataset=make_dataset)
    return


@app.function
def fit(model, molecules, *, epochs=60, batch_size=32, lr=3e-3, cutoff=R_CUT, seed=0):
    """stage 4: train on molecules (per-atom target normalization, Adam, cosine schedule, grad clipping); returns model."""
    raise NotImplementedError("stage4: implement fit")


@app.function
def evaluate(model, molecules, *, cutoff=R_CUT, batch_size=256):
    """stage 4: per-molecule absolute errors |prediction - energy|, a float64 ndarray (n,)."""
    raise NotImplementedError("stage4: implement evaluate")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", fit=fit, evaluate=evaluate, build_model=build_model,
               make_dataset=make_dataset)
    return


@app.function
def invariance_report(model, molecules, *, n_transforms=3, cutoff=R_CUT, seed=0):
    """stage 5: in float64, the largest |change in predicted energy| under random rotations, reflections,
    translations and atom relabellings: dict with keys 'rotation', 'reflection', 'translation', 'permutation'."""
    raise NotImplementedError("stage5: implement invariance_report")


@app.function
def mae_ci(errors, *, n_boot=2000, alpha=0.05, seed=0):
    """stage 5: (MAE, lo, hi): the mean absolute error and its percentile bootstrap interval (mlfs.stats.bootstrap_ci)."""
    raise NotImplementedError("stage5: implement mae_ci")


@app.function
def mae_difference_ci(errors_a, errors_b, *, n_boot=2000, alpha=0.05, seed=0):
    """stage 5: (MAE_a - MAE_b, lo, hi) with a PAIRED bootstrap over test molecules (both arrays resampled jointly)."""
    raise NotImplementedError("stage5: implement mae_difference_ci")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", invariance_report=invariance_report, mae_ci=mae_ci,
               mae_difference_ci=mae_difference_ci, build_model=build_model, make_dataset=make_dataset)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the experiment

    your cells go here (see the brief's "deliverables"): build the train sizes 250 / 1,000 /
    4,000 from `make_dataset(4000, seed=1)` and the test set `make_dataset(1000, seed=2)`; train
    every model with 3 seeds; report test MAE with bootstrap intervals against the per-atom-mean
    baseline; plot learning curves; run `invariance_report` on each trained model; and compare
    models with paired intervals. keep a small config (a few hundred molecules, a few epochs)
    for checking that the cells run before you start the full one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    1. **results.** a table of test MAE (95% bootstrap CI, seed spread) for the per-atom-mean
       baseline and the four models at 4,000 training molecules, and the learning curves.
    2. **invariance audit.** the four models' `invariance_report`, and what each number means.
    3. **analysis.** answer the brief's questions: what do GCN and GIN actually learn without
       coordinates, why does `gin_pos` lose to the invariant models, where does EGNN's advantage
       come from, which differences are real (paired intervals), and what you'd change next.
    """)
    return


if __name__ == "__main__":
    app.run()
