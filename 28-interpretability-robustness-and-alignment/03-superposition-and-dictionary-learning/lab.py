import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    SPARSITIES = [0.0, 0.7, 0.9, 0.97, 0.99]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # superposition & dictionary learning · lab

    every experiment runs **your** toy model, training loop, feature dimensionality and sparse
    autoencoder. all data is synthetic, which is the point: you know the true features, so you
    can check what the models and the dictionaries found. for each experiment: write down a
    prediction, lock it in, run it, then explain. about half a minute on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: when does a model use superposition?

    the setup of Elhage et al. (2022): 20 features, 5 hidden dimensions, importance
    $0.7^i$ (feature 0 matters most). one `ToySuperposition` per sparsity, each trained for
    3,000 steps (batches of 1,024, Adam, lr $10^{-3}$). a feature counts as **represented** if
    $\|W_i\| > 0.5$; its colour is how much it shares its direction with the others,
    $\sum_{j \ne i} (\hat W_i \cdot W_j)^2$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("5 dimensions, 20 features. how many features will the model represent when features are dense "
                    "(sparsity 0)? and when 99% of the time each feature is zero?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    toys = {}
    for _s in SPARSITIES:
        torch.manual_seed(0)
        _m = ex.ToySuperposition(20, 5, importance=0.7 ** torch.arange(20.0))
        ex.train_toy_model(_m, _s, steps=3000, batch_size=1024, lr=1e-3, generator=torch.Generator().manual_seed(0))
        toys[_s] = _m
    _fig, _axes = plt.subplots(2, len(SPARSITIES), figsize=(12, 4.6), gridspec_kw={"height_ratios": [1, 1.2]})
    _rows = []
    for _k, (_s, _m) in enumerate(toys.items()):
        _W = _m.W.detach()
        _norms = _W.norm(dim=0)
        _unit = _W / _norms.clamp_min(1e-12)
        _share = ((_unit.T @ _W) ** 2).sum(1) - _norms ** 2
        _axes[0, _k].imshow((_W.T @ _W).numpy(), cmap="RdBu", vmin=-1.2, vmax=1.2)
        _axes[0, _k].set(title=f"sparsity {_s}", xticks=[], yticks=[])
        _axes[1, _k].bar(range(20), _norms.numpy(), color=plt.cm.viridis((_share / (_share.max() + 1e-9)).numpy()))
        _axes[1, _k].axhline(0.5, color="gray", lw=0.5, ls=":")
        _axes[1, _k].set(ylim=(0, 1.3), xlabel="feature (by importance)", ylabel="||W_i||" if _k == 0 else None)
        _rows.append((_s, int((_norms > 0.5).sum()), float(ex.superposition_loss(_m, ex.sparse_feature_batch(
            4096, 20, _s, generator=torch.Generator().manual_seed(9))))))
    _axes[0, 0].set_ylabel("W^T W")
    mo.vstack([_fig, mo.md("| sparsity | features represented | loss (fresh batch) |\n|---|---|---|\n"
                           + "\n".join(f"| {s} | {n} | {loss:.4f} |" for s, n, loss in _rows))])
    return (toys,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with dense features, why does the model keep exactly 5 and drop the rest?
    (what does interference cost when every feature is active at once?) why does sparsity
    change the trade-off? in the $W^\top W$ panels, what do the negative off-diagonal entries
    mean, and why are they negative rather than positive?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the geometry of superposition

    your `feature_dimensionality` gives each feature $D_i$, the fraction of a dimension it
    gets: 1 for a dedicated dimension, $\tfrac12$ for an antipodal pair. below: $D_i$ of every
    represented feature of the five models above, and a sixth model with 40 equally important
    features in 10 dimensions at sparsity 0.97 (4,000 steps, lr $3 \times 10^{-3}$).
    """)
    return


@app.cell
def _():
    p2 = ui.predict("would you expect the D values to spread out continuously between 0 and 1, or to cluster? "
                    "if they cluster, at which values?")
    p2
    return (p2,)


@app.cell
def _(p2, toys):
    ui.gate(p2)
    torch.manual_seed(0)
    _uni = ex.ToySuperposition(40, 10)
    ex.train_toy_model(_uni, 0.97, steps=4000, batch_size=1024, lr=3e-3, generator=torch.Generator().manual_seed(0))
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _counts = {}
    for _k, (_s, _m) in enumerate(toys.items()):
        _D = ex.feature_dimensionality(_m.W).numpy()
        _D = _D[_m.W.detach().norm(dim=0).numpy() > 0.5]
        _ax.scatter(np.full(len(_D), _k) + np.linspace(-0.2, 0.2, len(_D)), _D, s=14)
        _counts[_s] = [round(float(v), 2) for v in np.sort(_D)[::-1]]
    _Du = ex.feature_dimensionality(_uni.W).numpy()
    _ax.scatter(np.full(40, len(toys)) + np.linspace(-0.25, 0.25, 40), _Du, s=10, color="k")
    for _y, _lbl in [(1, "1"), (0.5, "1/2"), (0.4, "2/5"), (1 / 3, "1/3"), (0.25, "1/4")]:
        _ax.axhline(_y, color="gray", lw=0.5, ls=":")
        _ax.text(len(toys) + 0.45, _y, _lbl, va="center", fontsize=8)
    _ax.set_xticks(range(len(toys) + 1), [f"s={s}" for s in toys] + ["40 in 10\n(uniform)"])
    _ax.set(ylabel="feature dimensionality D_i", ylim=(0, 1.1))
    mo.vstack([_fig, mo.md("\n".join(f"- sparsity {s}: D = {list(v)}" for s, v in _counts.items())
                           + f"\n- 40 features in 10 dimensions: D between {_Du.min():.3f} and {_Du.max():.3f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the values stick to a few fractions. what geometric arrangement gives each
    of $1$, $\tfrac12$, $\tfrac25$ (on paper: a pentagon in a 2-D subspace) and $\tfrac13$?
    why would gradient descent prefer these symmetric arrangements over arbitrary angles?
    why do all 40 uniform features end up with the same $D$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a sparse autoencoder recovers the features

    the toy model trained at sparsity 0.9 in experiment 1 squeezes 9 features into 5
    dimensions. its hidden activations $h = Wx$ are all an interpretability researcher would
    see. your `SparseAutoencoder` (40 features) is trained on 20,000 of them for 3,000 steps
    at several values of `l1_coeff`; MMCS compares its dictionary with the true feature
    directions (the 9 represented columns of $W$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict("as l1_coeff grows from 1e-4 to 1, what happens to the number of active SAE features per input "
                    "(L0), to the reconstruction, and to how well the dictionary matches the true features?")
    p3
    return (p3,)


@app.cell
def _(p3, toys):
    ui.gate(p3)
    _m = toys[0.9]
    _W = _m.W.detach()
    _true = _W[:, _W.norm(dim=0) > 0.5]
    _x = ex.sparse_feature_batch(20000, 20, 0.9, generator=torch.Generator().manual_seed(3))
    _acts = _x @ _W.T
    _true_l0 = float((_x[:, _W.norm(dim=0) > 0.5] > 0).sum(1).float().mean())
    sae_rows = []
    for _l1 in [1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]:
        torch.manual_seed(0)
        _sae = ex.SparseAutoencoder(5, 40, l1_coeff=_l1)
        ex.train_sae(_sae, _acts, steps=3000, batch_size=256, lr=3e-3, generator=torch.Generator().manual_seed(4))
        _st = ex.sae_stats(_sae, _acts)
        sae_rows.append((_l1, _st["l0"], _st["fvu"], _st["dead"], ex.mmcs(_sae.dictionary(), _true)))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.3))
    _ax.semilogx([r[0] for r in sae_rows], [r[4] for r in sae_rows], "o-", label="MMCS with the true features")
    _ax.semilogx([r[0] for r in sae_rows], [r[2] for r in sae_rows], "s--", label="fraction of variance unexplained")
    _ax.set(xlabel="l1_coeff", ylim=(0, 1.05))
    _ax2 = _ax.twinx()
    _ax2.semilogx([r[0] for r in sae_rows], [r[1] for r in sae_rows], "^:", color="C2", label="L0 (right)")
    _ax2.axhline(_true_l0, color="C2", lw=0.8, ls="-.", label="true L0 (right)")
    _ax2.set_ylabel("active features per input", color="C2")
    _h1, _lb1 = _ax.get_legend_handles_labels()
    _h2, _lb2 = _ax2.get_legend_handles_labels()
    _ax.legend(_h1 + _h2, _lb1 + _lb2, fontsize=7, loc="center left")
    mo.vstack([_fig, mo.md(f"true active features per input: **{_true_l0:.2f}**\n\n"
                           "| l1_coeff | L0 | FVU | dead | MMCS |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {a:g} | {b:.2f} | {c:.4f} | {d:.2f} | {e:.3f} |" for a, b, c, d, e in sae_rows))])
    return


@app.cell
def _():
    l1_pick = mo.ui.dropdown(["0.0001", "0.001", "0.01", "0.03", "0.1", "0.3", "1.0"], value="0.03",
                             label="l1_coeff for one SAE, in detail")
    l1_pick
    return (l1_pick,)


@app.cell
def _(l1_pick, p3, toys):
    ui.gate(p3)
    _W = toys[0.9].W.detach()
    _true = _W[:, _W.norm(dim=0) > 0.5]
    _acts = ex.sparse_feature_batch(20000, 20, 0.9, generator=torch.Generator().manual_seed(3)) @ _W.T
    torch.manual_seed(0)
    _sae = ex.SparseAutoencoder(5, 40, l1_coeff=float(l1_pick.value))
    ex.train_sae(_sae, _acts, steps=3000, batch_size=256, lr=3e-3, generator=torch.Generator().manual_seed(4))
    _D = _sae.dictionary().detach()
    _cos = (_true / _true.norm(dim=0)).T @ _D
    _st = ex.sae_stats(_sae, _acts)
    _fig, _ax = plt.subplots(figsize=(8, 2.6))
    _ax.imshow(_cos.numpy(), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    _ax.set(xlabel="SAE dictionary feature", ylabel="true feature", title=f"cosine, l1_coeff = {l1_pick.value}")
    mo.vstack([_fig, mo.md(f"L0 {_st['l0']:.2f}, FVU {_st['fvu']:.4f}, dead {_st['dead']:.2f}, MMCS "
                           f"{ex.mmcs(_D, _true):.3f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at small `l1_coeff` the reconstruction is nearly perfect. why doesn't
    that make it the best dictionary? (5 dimensions, 40 features: how many different ways
    can you reconstruct a 5-D vector exactly?) what does the sparsity penalty buy, and what
    does too much of it cost? where would you set it if you didn't know the true features?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: feature splitting

    a world with structure: 4 **parent** concepts (orthogonal directions in 16 dimensions),
    each with 4 **children**, variants that point mostly along the parent (cosine 0.86) plus
    a private twist. each input contains a few active children (each fires with probability
    0.05). SAEs of width 4, 8, 16 and 32 (`l1_coeff` 0.1, 3,000 steps) are trained on 20,000
    inputs; MMCS measures how well each dictionary matches the parents and the children.
    """)
    return


@app.cell
def _():
    p4 = ui.predict("an SAE with only 4 features sees data made of 16 child features. what will its 4 features "
                    "look like? what happens to them as the SAE gets wider?")
    p4
    return (p4,)


@app.cell
def _(p4):
    ui.gate(p4)
    _g = torch.Generator().manual_seed(0)
    _Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=_g))
    _parents = _Q[:, :4]
    _kids = []
    for _p in range(4):
        _priv = _Q[:, 4 + 3 * _p:7 + 3 * _p]
        for _u in [_priv[:, 0], _priv[:, 1], _priv[:, 2], -_priv.sum(1) / 3 ** 0.5]:
            _v = _parents[:, _p] + 0.6 * _u
            _kids.append(_v / _v.norm())
    _children = torch.stack(_kids, 1)
    _fires = (torch.rand(20000, 16, generator=_g) < 0.05).float() * (0.5 + 0.5 * torch.rand(20000, 16, generator=_g))
    _acts = _fires @ _children.T
    split_rows, _dicts = [], {}
    for _w in [4, 8, 16, 32]:
        torch.manual_seed(0)
        _sae = ex.SparseAutoencoder(16, _w, l1_coeff=0.1)
        ex.train_sae(_sae, _acts, steps=3000, batch_size=256, lr=3e-3, generator=torch.Generator().manual_seed(1))
        _dicts[_w] = _sae.dictionary().detach()
        _st = ex.sae_stats(_sae, _acts)
        split_rows.append((_w, ex.mmcs(_dicts[_w], _parents), ex.mmcs(_dicts[_w], _children), _st["fvu"], _st["dead"]))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    _axes[0].plot([r[0] for r in split_rows], [r[1] for r in split_rows], "o-", label="parents")
    _axes[0].plot([r[0] for r in split_rows], [r[2] for r in split_rows], "s-", label="children")
    _axes[0].set_xscale("log", base=2)
    _axes[0].xaxis.set_minor_locator(plt.NullLocator())
    _axes[0].set(xticks=[4, 8, 16, 32], xticklabels=["4", "8", "16", "32"], xlabel="SAE width",
                 ylabel="MMCS", ylim=(0.8, 1.01))
    _axes[0].legend(fontsize=8)
    _cos = (_dicts[4].T @ _dicts[16]).numpy()
    _order = np.argsort(_cos.argmax(0) * 10 - _cos.max(0))
    _axes[1].imshow(_cos[:, _order], cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    _axes[1].set(xlabel="width-16 SAE features (sorted)", ylabel="width-4 SAE features", yticks=range(4),
                 title="cosine between the two dictionaries")
    mo.vstack([_fig, mo.md("| width | MMCS parents | MMCS children | FVU | dead |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {w} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.2f} |" for w, a, b, c, d in split_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the narrow SAE learns the parents, which never occur on their own in the
    data. why is that the best use of 4 features? when the SAE widens, what happens to each
    parent feature (look at the cosine grid)? if a real SAE has a feature for "French text",
    what does this suggest about asking whether it's "the" right feature?
    """)
    return


if __name__ == "__main__":
    app.run()
