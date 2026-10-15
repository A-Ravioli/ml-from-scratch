import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def edges_str(G, names=None):
    """a readable edge list for a graph matrix: 'a -> b' directed, 'a - b' undirected."""
    names = names or [str(i) for i in range(len(G))]
    out = []
    for i, j in itertools.combinations(range(len(G)), 2):
        if G[i, j] and G[j, i]:
            out.append(f"{names[i]} – {names[j]}")
        elif G[i, j]:
            out.append(f"{names[i]} → {names[j]}")
        elif G[j, i]:
            out.append(f"{names[j]} → {names[i]}")
    return ", ".join(out) if out else "(no edges)"


@app.function
def sample_linear(D, W, n, rng, roots=None):
    """linear SEM on DAG D (p, p) with weights W: x_v = sum_u W[u, v] x_u + N(0, 1) noise; `roots` overrides columns."""
    p = len(D)
    X = np.zeros((n, p))
    done = set()
    while len(done) < p:
        for v in range(p):
            if v in done or not all(u in done for u in np.flatnonzero(D[:, v])):
                continue
            if roots is not None and v in roots:
                X[:, v] = roots[v]
            else:
                X[:, v] = X @ W[:, v] + rng.normal(size=n)
            done.add(v)
    return X


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # causal discovery · lab

    every experiment here runs **your** PC algorithm and IDA code from `exercise.py`. the
    first two experiments ask what data can and can't reveal about a graph; the third is
    the block's **project**: recover a causal graph from semi-synthetic data built on real
    covariates, and estimate a causal effect from what you recovered. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards. the whole lab takes under a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what the data can tell apart

    four three-variable worlds, all linear-Gaussian with strong edges, 5,000 samples each:

    | world | true graph |
    |---|---|
    | chain | $A \to B \to C$ |
    | reversed chain | $A \leftarrow B \leftarrow C$ |
    | fork | $A \leftarrow B \to C$ |
    | collider | $A \to B \leftarrow C$ |

    your `pc` (α = 0.01) gets only the data.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for each of the four worlds, which edges will PC orient? can it tell the chain from the fork? "
        "the collider from the others?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _worlds = {"chain": [(0, 1), (1, 2)], "reversed chain": [(1, 0), (2, 1)],
               "fork": [(1, 0), (1, 2)], "collider": [(0, 1), (2, 1)]}
    rows1 = []
    for _k, (_name, _edges) in enumerate(_worlds.items()):
        _D = np.zeros((3, 3), dtype=int)
        for _a, _b in _edges:
            _D[_a, _b] = 1
        _X = sample_linear(_D, 0.8 * _D, 5000, np.random.default_rng(_k))
        _G = ex.pc(_X, alpha=0.01)
        rows1.append((_name, edges_str(_D, list("ABC")), edges_str(_G, list("ABC")),
                      edges_str(ex.dag_to_cpdag(_D), list("ABC"))))
    mo.md("| world | true DAG | PC found | true CPDAG |\n|---|---|---|---|\n"
          + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows1))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which conditional independences does each world imply? why do the
    first three produce *exactly the same* set, and what makes the collider different?
    what extra information (an experiment, time order, assumptions about the noise) could
    distinguish a chain from a fork?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how much data does PC need?

    30 random DAGs on 8 variables (each pair joined with probability 0.3), linear-Gaussian
    with edge weights of size 0.5–1 and random signs. the same 30 graphs are sampled at
    five sample sizes and analysed at two significance levels. the structural Hamming
    distance (SHD) counts the pairs where PC's output differs from the **true CPDAG** (the
    best any method can do from observational data); it's split into missing edges, extra
    edges and wrongly oriented ones. the last row, "n = ∞", runs your `pc_skeleton` and
    `orient` with a perfect test computed from the true covariance matrix.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 10,000 samples, how close to the true CPDAG will PC get (average SHD)? does the error keep falling as n "
        "grows? which kind of error dominates at small n: missing edges, extra edges, or orientations?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def pop_partial(S, i, j, c):
        idx = [i, j] + list(c)
        Q = np.linalg.inv(S[np.ix_(idx, idx)])
        return -Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1])

    _rng = np.random.default_rng(7)
    graphs2 = []
    for _ in range(30):
        _order = _rng.permutation(8)
        _D = np.zeros((8, 8), dtype=int)
        for _i, _j in itertools.combinations(range(8), 2):
            if _rng.random() < 0.3:
                _D[_order[_i], _order[_j]] = 1
        _W = _D * _rng.uniform(0.5, 1.0, (8, 8)) * _rng.choice([-1, 1], (8, 8))
        _B = np.linalg.inv(np.eye(8) - _W)
        _S = _B.T @ _B  # the true covariance of the linear SEM
        _lam = min((abs(pop_partial(_S, i, j, c)) for i, j in np.argwhere(np.triu(_D + _D.T))
                    for r in range(7) for c in itertools.combinations([k for k in range(8) if k not in (i, j)], r)),
                   default=1.0)
        graphs2.append((_D, _W, _S, _lam))

    def score(D, G):
        T = np.asarray(ex.dag_to_cpdag(D))
        G = np.asarray(G)
        sk_t, sk_g = (T + T.T) > 0, (G + G.T) > 0
        miss, extra = int(np.triu(sk_t & ~sk_g).sum()), int(np.triu(sk_g & ~sk_t).sum())
        shd = ex.structural_hamming_distance(T, G)
        return shd, miss, extra, shd - miss - extra

    sizes = [100, 300, 1000, 3000, 10000]
    alphas = [0.01, 0.05]
    res2 = {}
    for _n in sizes:
        _data = [sample_linear(D, W, _n, np.random.default_rng(1000 + k)) for k, (D, W, _, _) in enumerate(graphs2)]
        for _a in alphas:
            res2[(_a, _n)] = np.array([score(g[0], ex.pc(X, alpha=_a)) for g, X in zip(graphs2, _data)])
    _pop = []
    for _D, _W, _S, _ in graphs2:
        _adj, _sep = ex.pc_skeleton(8, lambda i, j, c: 1.0 if abs(pop_partial(_S, i, j, c)) < 1e-9 else 0.0)
        _pop.append(score(_D, ex.orient(_adj, _sep)))
    pop2 = np.array(_pop)
    lam2 = np.array([g[3] for g in graphs2])
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _a, _mk in zip(alphas, ["o-", "s-"]):
        _m = [res2[(_a, n)][:, 0].mean() for n in sizes]
        _s = [res2[(_a, n)][:, 0].std(ddof=1) / np.sqrt(30) for n in sizes]
        _ax.errorbar(sizes, _m, yerr=_s, fmt=_mk, capsize=3, label=f"α = {_a}")
    _ax.axhline(pop2[:, 0].mean(), color="k", ls="--", lw=1, label="perfect test (n = ∞)")
    _ax.set(xscale="log", xlabel="sample size n", ylabel="SHD to the true CPDAG (mean ± s.e., 30 graphs)")
    _ax.legend()
    _fig.tight_layout()
    _weak = lam2 < 0.05
    _big = res2[(0.01, 10000)][:, 0]
    _rows = [f"| {a} | {n:,} | {res2[(a, n)][:, 0].mean():.2f} | {res2[(a, n)][:, 1].mean():.2f} "
             f"| {res2[(a, n)][:, 2].mean():.2f} | {res2[(a, n)][:, 3].mean():.2f} |" for a in alphas for n in sizes]
    _rows.append(f"| — | ∞ | {pop2[:, 0].mean():.2f} | {pop2[:, 1].mean():.2f} | {pop2[:, 2].mean():.2f} "
                 f"| {pop2[:, 3].mean():.2f} |")
    mo.vstack([_fig, mo.md(
        f"the 30 graphs have {np.mean([np.triu(g[0] + g[0].T).sum() for g in graphs2]):.1f} edges on average.\n\n"
        "| α | n | SHD | missing | extra | wrong orientation |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\n**near-unfaithful graphs.** for each graph, λ = the smallest |partial correlation| between two "
        f"adjacent variables over every conditioning set. {_weak.sum()} of 30 graphs have λ < 0.05; at n = 10,000 "
        f"(α = 0.01) their mean SHD is {_big[_weak].mean() if _weak.any() else float('nan'):.2f}, against "
        f"{_big[~_weak].mean():.2f} for the other {(~_weak).sum()}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a *missing* edge cause *orientation* errors elsewhere (think
    about unshielded triples and Meek's rules)? the perfect test gets every graph right,
    so where do the errors that remain at $n = 10{,}000$ come from? what does λ measure,
    and why does a graph with a tiny λ need an enormous sample?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3 (project): discover, then estimate

    a semi-synthetic world built on **real** covariates from `mlfs.datasets.california_housing`:
    median income (`inc`), housing age (`age`) and log rooms per household (`rooms`), for
    5,000 districts. each covariate column is shuffled independently, which keeps its real,
    skewed distribution but makes the three independent root causes, so the true graph is
    known exactly. on top of them, a simulated treatment `T` (say, a transit subsidy),
    a mediator `M`, an outcome `Y` and a variable `C` that both `T` and `Y` affect:

    $$T = 0.8\,\text{inc} - 0.6\,\text{age} + \varepsilon,\quad M = 0.9\,T + \varepsilon,\quad Y = 0.4M + 0.6T + 0.4\,\text{inc} + 0.5\,\text{rooms} + \varepsilon,\quad C = 0.5T + 0.3Y + \varepsilon.$$

    the total effect of `T` on `Y` is $0.6 + 0.9 \times 0.4 = 0.96$. the project: (1) run your
    `pc` and compare with the true CPDAG; (2) read the possible parent sets of `T` off the
    learned graph and estimate the effect with your `ida_effects`; (3) compare with the
    naive regression of `Y` on `T` and with "regress `Y` on `T` and everything else". then
    repeat on 20 fresh draws to see how stable each step is. the checkbox keeps the real
    covariates' dependence on each other instead (then the true graph among them is
    unknown, and SHD is computed only for pairs involving a simulated variable).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "will PC orient every edge into and out of T? how close will the IDA estimate be to 0.96, and how far off are "
        "'Y on T' and 'Y on T and everything else'?"
    )
    p3
    return (p3,)


@app.cell
def _():
    keep_real = mo.ui.checkbox(value=False, label="keep the real dependence among inc, age and rooms")
    keep_real
    return (keep_real,)


@app.cell
def _(keep_real, p3):
    ui.gate(p3)
    names3 = ["inc", "age", "rooms", "T", "M", "Y", "C"]
    D3 = np.zeros((7, 7), dtype=int)
    W3 = np.zeros((7, 7))
    for (_a, _b), _w in {(0, 3): 0.8, (1, 3): -0.6, (3, 4): 0.9, (4, 5): 0.4, (3, 5): 0.6, (0, 5): 0.4,
                         (2, 5): 0.5, (3, 6): 0.5, (5, 6): 0.3}.items():
        D3[_a, _b], W3[_a, _b] = 1, _w
    truth3 = 0.6 + 0.9 * 0.4
    _Xc, _, _ = datasets.california_housing()
    _cov = np.column_stack([_Xc[:, 7], _Xc[:, 2], np.log(np.abs(_Xc[:, 3] / np.maximum(np.abs(_Xc[:, 6]), 1e-9)) + 1e-9)])
    covs3 = (_cov - _cov.mean(0)) / _cov.std(0)
    true_cp3 = np.asarray(ex.dag_to_cpdag(D3))
    sim = [[i, j] for i, j in itertools.combinations(range(7), 2) if j >= 3]  # pairs involving a simulated variable

    def draw(seed):
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(covs3), min(5000, len(covs3)), replace=False)
        C = covs3[rows].copy()
        if not keep_real.value:
            for k in range(3):
                C[:, k] = rng.permutation(C[:, k])
        return sample_linear(D3, W3, len(rows), rng, roots={k: C[:, k] for k in range(3)})

    def analyse(X):
        G = np.asarray(ex.pc(X, alpha=0.01))
        shd = sum((G[i, j], G[j, i]) != (true_cp3[i, j], true_cp3[j, i]) for i, j in sim) if keep_real.value \
            else ex.structural_hamming_distance(true_cp3, G)
        ida = ex.ida_effects(X, 3, 5, G)
        A = np.column_stack([np.ones(len(X)), X[:, 3]])
        naive = np.linalg.lstsq(A, X[:, 5], rcond=None)[0][1]
        A_all = np.column_stack([np.ones(len(X)), X[:, 3], np.delete(X, [3, 5], axis=1)])
        everything = np.linalg.lstsq(A_all, X[:, 5], rcond=None)[0][1]
        return G, shd, ida, naive, everything

    _X = draw(0)
    G3, shd3, ida3, naive3, all3 = analyse(_X)
    _sets = ex.possible_parent_sets(G3, 3)
    reps3 = [analyse(draw(100 + _s)) for _s in range(20)]
    _shds = np.array([r[1] for r in reps3])
    _ida_mid = np.array([np.mean(r[2]) for r in reps3])
    _ida_n = np.array([len(r[2]) for r in reps3])
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _vals = [np.array([r[3] for r in reps3]), np.array([r[4] for r in reps3]), np.concatenate([r[2] for r in reps3])]
    _bins = np.linspace(min(v.min() for v in _vals) - 0.02, max(v.max() for v in _vals) + 0.02, 60)
    for _v, _lab in zip(_vals, ["naive: Y on T", "Y on T and everything else", "IDA (every possible parent set)"]):
        _ax.hist(_v, bins=_bins, alpha=0.6, label=_lab)
    _ax.axvline(truth3, color="k", ls="--", label=f"truth {truth3:.2f}")
    _ax.set(xlabel="estimated effect of T on Y", ylabel="count (20 draws)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(
        f"**one draw (n = {len(_X):,}).** true CPDAG: {edges_str(true_cp3, names3)}\n\n"
        f"PC found: {edges_str(G3, names3)}\n\n"
        f"SHD {shd3}. possible parent sets of T: {[tuple(names3[k] for k in s) for s in _sets]}; "
        f"IDA estimates {np.round(ida3, 3).tolist()} (truth {truth3:.2f}); naive {naive3:.3f}; "
        f"everything else {all3:.3f}.\n\n"
        f"**20 fresh draws.** SHD = 0 in {np.mean(_shds == 0):.0%} (mean SHD {_shds.mean():.2f}); "
        f"T's parent set identified uniquely in {np.mean(_ida_n == 1):.0%}; IDA estimate (mean over sets) "
        f"{_ida_mid.mean():.3f} ± {_ida_mid.std(ddof=1):.3f}; naive {np.mean([r[3] for r in reps3]):.3f}; "
        f"everything else {np.mean([r[4] for r in reps3]):.3f}."), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which v-structures let PC orient the edges at `T`, and which Meek rule
    orients `T → M` and `T → C`? name the two mistakes in "regress on everything else"
    (which variables in it are descendants of `T`, and what does each do to the estimate)?
    now tick the checkbox: in the bad draws, which separating set did PC record for
    `inc` and `C`, and which false v-structure does that create at `T`? compute
    $\rho(\text{inc}, C \mid Y)$ with and without the real covariate correlations (hint:
    experiment 2's λ). what does IDA report then, and is the truth inside its set?
    """)
    return


if __name__ == "__main__":
    app.run()
