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

    from mlfs import checks, datasets
    from mlfs.safety import run_with_cache  # yours, from topic 02
    from mlfs.torchnn import get_device  # yours, from ch 15 · 07
    from mlfs.transformer import GPT  # yours, from ch 18 · 04

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # small CPU config (offline, tests, verify, no GPU, or MLFS_CAPSTONE=small) vs full config (cuda or mps).
    # the full config took about 12 minutes on an Apple M2 GPU: see brief.md.
    OFFLINE = datasets.is_offline()
    FULL = (not OFFLINE and str(get_device()) != "cpu" and os.environ.get("MLFS_CAPSTONE", "full") != "small")
    DEVICE = str(get_device()) if FULL else "cpu"
    if FULL:
        CONFIG = dict(p=59, train_frac=0.3, d_model=128, n_heads=4, d_mlp=512, steps=20_000, lr=1e-3,
                      weight_decay=1.0, log_every=100, checkpoint_every=500, seeds=[0, 1, 2, 3, 4])
    else:
        CONFIG = dict(p=31, train_frac=0.6, d_model=64, n_heads=4, d_mlp=256, steps=4_000, lr=1e-3,
                      weight_decay=1.0, log_every=50, checkpoint_every=250, seeds=[0])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 28 · capstone: reverse-engineer a circuit (reference solution)

    read [`brief.md`](brief.md) first. you'll train a one-layer transformer to add numbers
    modulo a prime $p$, watch it memorise its training set and then, thousands of steps
    later, suddenly generalise (**grokking**), and then open it up to find out *how* it adds.
    the claim to test is Nanda et al. (2023)'s: the network represents numbers as rotations
    by a few key frequencies, and adds them with trigonometric identities. the functions are
    the measuring instruments; the runs and the write-up below them are the capstone.

    | stage | you build |
    |---|---|
    | 1 | `modular_addition_data`: every pair $(a, b)$, split into train and test |
    | 2 | `make_model`, `answer_logits`, `train_grokking`: the model and a full-batch AdamW loop that logs everything |
    | 3 | `fourier_basis`, `embedding_frequency_norms`, `key_frequencies`: which frequencies the embedding uses |
    | 4 | `logit_cosine_fit`, `progress_measures`: how much of the output is $\sum_k \alpha_k\cos(w_k(a+b-c))$, and hidden progress |
    | 5 | `neuron_activations`, `neuron_frequency_fit`: what the MLP neurons compute |

    when the checks are green, scroll down: train, analyse, then write up.
    """)
    return


@app.function
def modular_addition_data(p, *, train_frac=0.3, seed=0):
    """stage 1: all p^2 inputs [a, b, p] (grid order a * p + b) with labels (a + b) % p, split by a seeded permutation."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x = np.stack([a.ravel(), b.ravel(), np.full(p * p, p)], axis=1)
    y = (a.ravel() + b.ravel()) % p
    perm = np.random.default_rng(seed).permutation(p * p)
    n_train = int(round(train_frac * p * p))
    tr, te = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    t = lambda v: torch.as_tensor(v, dtype=torch.int64)  # noqa: E731
    return {"p": p, "all_x": t(x), "all_y": t(y), "train_idx": t(tr), "test_idx": t(te),
            "train_x": t(x[tr]), "train_y": t(y[tr]), "test_x": t(x[te]), "test_y": t(y[te])}


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", modular_addition_data=modular_addition_data)
    return


@app.function
def make_model(p, *, d_model=128, n_heads=4, d_mlp=512, seed=0):
    """stage 2: a one-layer GPT over p + 1 tokens (the numbers and '='), block size 3, untied embedding and head."""
    torch.manual_seed(seed)
    return GPT(p + 1, block_size=3, n_layers=1, n_heads=n_heads, d_model=d_model, d_ff=d_mlp, tie_weights=False)


@app.function
def answer_logits(model, x):
    """stage 2: (n, p) logits at the last position ('='), restricted to the p number tokens."""
    return model(x)[0][:, -1, :model.vocab_size - 1]


@app.function
def train_grokking(model, data, *, steps, lr=1e-3, weight_decay=1.0, log_every=100, checkpoint_every=None,
                   device="cpu"):
    """stage 2: full-batch AdamW (betas 0.9, 0.98) on the training pairs; history every log_every updates (+ checkpoints)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    xtr, ytr = data["train_x"].to(device), data["train_y"].to(device)
    xte, yte = data["test_x"].to(device), data["test_y"].to(device)
    hist = {k: [] for k in ("step", "train_loss", "test_loss", "train_acc", "test_acc", "weight_norm")}
    checkpoints = []
    for t in range(steps + 1):
        if t % log_every == 0 or t == steps:
            model.eval()
            with torch.no_grad():
                ltr, lte = answer_logits(model, xtr), answer_logits(model, xte)
                hist["step"].append(t)
                hist["train_loss"].append(F.cross_entropy(ltr, ytr).item())
                hist["test_loss"].append(F.cross_entropy(lte, yte).item())
                hist["train_acc"].append((ltr.argmax(1) == ytr).float().mean().item())
                hist["test_acc"].append((lte.argmax(1) == yte).float().mean().item())
                hist["weight_norm"].append(math.sqrt(sum(p.pow(2).sum().item() for p in model.parameters())))
        if checkpoint_every and (t % checkpoint_every == 0 or t == steps):
            checkpoints.append((t, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        if t == steps:
            break
        model.train()
        loss = F.cross_entropy(answer_logits(model, xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    if checkpoint_every:
        hist["checkpoints"] = checkpoints
    return hist


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", make_model=make_model, answer_logits=answer_logits,
               train_grokking=train_grokking, modular_addition_data=modular_addition_data)
    return


@app.function
def fourier_basis(p):
    """stage 3: (p, p) orthonormal rows: constant, then cos(2 pi k x / p), sin(2 pi k x / p) for k = 1 .. (p - 1) / 2."""
    if p < 3 or p % 2 == 0:
        raise ValueError(f"p must be an odd number >= 3, got {p}")
    x = np.arange(p)
    rows = [np.full(p, 1 / math.sqrt(p))]
    for k in range(1, (p - 1) // 2 + 1):
        rows.append(np.cos(2 * math.pi * k * x / p) * math.sqrt(2 / p))
        rows.append(np.sin(2 * math.pi * k * x / p) * math.sqrt(2 / p))
    return np.stack(rows)


@app.function
def embedding_frequency_norms(W_E, p):
    """stage 3: (K,) fraction of the embedding's non-constant energy at each frequency k = 1..K, K = (p - 1) / 2."""
    W = np.asarray(W_E, dtype=np.float64)[:p]
    C = fourier_basis(p) @ W
    energy = (C[1::2] ** 2).sum(1) + (C[2::2] ** 2).sum(1)
    return energy / energy.sum()


@app.function
def key_frequencies(fractions, *, frac=0.9):
    """stage 3: the fewest frequencies (1-based, sorted) whose fractions add up to at least frac."""
    f = np.asarray(fractions, dtype=np.float64)
    order = np.argsort(-f, kind="stable")
    n = int(np.searchsorted(np.cumsum(f[order]), frac - 1e-12) + 1)
    return sorted(int(k) + 1 for k in order[:min(n, len(f))])


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", fourier_basis=fourier_basis,
               embedding_frequency_norms=embedding_frequency_norms, key_frequencies=key_frequencies)
    return


@app.function
def logit_cosine_fit(full_logits, p, freqs):
    """stage 4: least-squares alpha_k in logits[a,b,c] ~ offset(a,b) + sum_k alpha_k cos(w_k (a+b-c)); returns (alphas, FVE)."""
    L = np.asarray(full_logits, dtype=np.float64).reshape(p, p, p)
    L = L - L.mean(axis=2, keepdims=True)  # per-input offsets never matter for the softmax
    a, b, c = np.meshgrid(np.arange(p), np.arange(p), np.arange(p), indexing="ij")
    alphas, explained = [], 0.0
    for k in freqs:
        phi = np.cos(2 * math.pi * k * (a + b - c) / p)
        alpha = float((L * phi).sum() / (p ** 3 / 2))  # the cosines are orthogonal, each with squared norm p^3 / 2
        alphas.append(alpha)
        explained += alpha ** 2 * p ** 3 / 2
    total = float((L ** 2).sum())
    return np.array(alphas), (explained / total if total > 0 else 0.0)


@app.function
def progress_measures(full_logits, labels, train_idx, p, freqs):
    """stage 4: {'restricted': CE of the constant + key (a+b)-frequency part on all inputs, 'excluded': CE without the key part on train}."""
    L = np.asarray(full_logits, dtype=np.float64)  # (p*p, p), rows in grid order a * p + b
    a, b = np.divmod(np.arange(p * p), p)
    key = np.zeros_like(L)
    for k in freqs:
        for fn in (np.cos, np.sin):
            psi = fn(2 * math.pi * k * (a + b) / p)  # orthogonal over the grid, squared norm p^2 / 2
            key += np.outer(psi, psi @ L) / (p * p / 2)
    restricted = L.mean(axis=0, keepdims=True) + key
    excluded = L - key
    y = np.asarray(labels)
    tr = np.asarray(train_idx)

    def ce(logits, target):
        z = logits - logits.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(target)), target]))

    return {"restricted": ce(restricted, y), "excluded": ce(excluded[tr], y[tr])}


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", logit_cosine_fit=logit_cosine_fit, progress_measures=progress_measures,
               modular_addition_data=modular_addition_data)
    return


@app.function
def neuron_activations(model, x):
    """stage 5: (n, d_mlp) ndarray of the MLP activations (after the nonlinearity) at the last position, via run_with_cache."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, cache = run_with_cache(model, x, names=["blocks.0.mlp.act"])
    finally:
        model.train(was)
    return cache["blocks.0.mlp.act"][:, -1].float().cpu().numpy()


@app.function
def neuron_frequency_fit(acts, p):
    """stage 5: per neuron, the frequency k whose 8 terms (cos, sin of w_k a, w_k b, w_k(a+b), w_k(a-b)) explain most variance; (best_k, FVE)."""
    A = np.asarray(acts, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    a, b = np.divmod(np.arange(p * p), p)
    total = (A ** 2).sum(0)
    best_k = np.zeros(A.shape[1], dtype=int)
    best = np.zeros(A.shape[1])
    for k in range(1, (p - 1) // 2 + 1):
        w = 2 * math.pi * k / p
        explained = np.zeros(A.shape[1])
        for arg in (a, b, a + b, a - b):
            for fn in (np.cos, np.sin):
                psi = fn(w * arg)
                explained += (psi @ A) ** 2 / (p * p / 2)
        fve = np.where(total > 0, explained / np.where(total > 0, total, 1.0), 0.0)
        better = fve > best
        best_k[better], best[better] = k, fve[better]
    return best_k, best


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", neuron_activations=neuron_activations,
               neuron_frequency_fit=neuron_frequency_fit, make_model=make_model,
               modular_addition_data=modular_addition_data)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## the runs

    config: **{"full" if FULL else "small"}** on **{DEVICE}** (offline: {OFFLINE}): $p = {CONFIG["p"]}$,
    {CONFIG["train_frac"]:.0%} of the {CONFIG["p"] ** 2:,} pairs for training, width {CONFIG["d_model"]},
    {CONFIG["n_heads"]} heads, {CONFIG["d_mlp"]} MLP neurons, {CONFIG["steps"]:,} full-batch AdamW steps
    (lr {CONFIG["lr"]}, weight decay {CONFIG["weight_decay"]}), seeds {CONFIG["seeds"]}. the full config takes
    about 12 minutes on an Apple-silicon GPU; the small one (used by the tests, `verify.py` and CPU-only
    machines) about half a minute, and it groks too.
    """)
    return


@app.cell
def _():
    runs = {}
    _t0 = time.perf_counter()
    for _seed in CONFIG["seeds"]:
        _data = modular_addition_data(CONFIG["p"], train_frac=CONFIG["train_frac"], seed=_seed)
        _model = make_model(CONFIG["p"], d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"], d_mlp=CONFIG["d_mlp"],
                            seed=_seed)
        _hist = train_grokking(_model, _data, steps=CONFIG["steps"], lr=CONFIG["lr"],
                               weight_decay=CONFIG["weight_decay"], log_every=CONFIG["log_every"],
                               checkpoint_every=CONFIG["checkpoint_every"], device=DEVICE)
        _model.to("cpu")

        def _first(key, _h=_hist):
            hits = [s for s, v in zip(_h["step"], _h[key]) if v >= 0.99]
            return hits[0] if hits else None

        # AdamW with strong weight decay has occasional loss spikes ("slingshots") that recover within a few hundred
        # steps. if the run ends inside one, analyse the latest checkpoint that still generalised, and say so.
        _acc = dict(zip(_hist["step"], _hist["test_acc"]))
        _good = [t for t, _ in _hist["checkpoints"] if _acc.get(t, 0.0) >= 0.99]
        _at = CONFIG["steps"] if _hist["test_acc"][-1] >= 0.99 or not _good else _good[-1]
        _analysed = copy.deepcopy(_model)
        _analysed.load_state_dict(dict(_hist["checkpoints"])[_at])
        _analysed.eval()
        runs[_seed] = {"data": _data, "model": _model, "hist": _hist, "memorised": _first("train_acc"),
                       "grokked": _first("test_acc"), "analysed": _analysed, "analysed_at": _at}
    t_train = time.perf_counter() - _t0
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for _seed, _r in runs.items():
        _h = _r["hist"]
        _axes[0].plot(_h["step"], _h["train_loss"], color="C0", alpha=0.6, label="train" if _seed == 0 else None)
        _axes[0].plot(_h["step"], _h["test_loss"], color="C3", alpha=0.6, label="test" if _seed == 0 else None)
        _axes[1].plot(_h["step"], _h["test_acc"], alpha=0.7, label=f"seed {_seed}")
        _axes[2].plot(_h["step"], _h["weight_norm"], alpha=0.7)
    _axes[0].set(yscale="log", xlabel="step", ylabel="cross-entropy", title="loss (every seed)")
    _axes[0].legend()
    _axes[1].set(xlabel="step", ylabel="test accuracy", title="generalisation")
    _axes[1].legend(fontsize=7)
    _axes[2].set(xlabel="step", ylabel="||theta||", title="total weight norm")
    mo.vstack([mo.md(f"trained {len(runs)} model(s) in {t_train / 60:.1f} min."), _fig,
               mo.md("| seed | train acc >= 0.99 at step | test acc >= 0.99 at step | final test acc | analysed at step |\n"
                     "|---|---|---|---|---|\n"
                     + "\n".join(f"| {s} | {r['memorised']} | {r['grokked']} | {r['hist']['test_acc'][-1]:.3f} | "
                                 f"{r['analysed_at']} |" for s, r in runs.items()))])
    return (runs,)


@app.cell
def _(runs):
    analysis = {}
    for _seed, _r in runs.items():
        _p = CONFIG["p"]
        _W_E = _r["analysed"].tok_emb.weight.detach().numpy()
        _fr = embedding_frequency_norms(_W_E, _p)
        _keys = key_frequencies(_fr)
        with torch.no_grad():
            _logits = answer_logits(_r["analysed"], _r["data"]["all_x"]).numpy()
        _alphas, _fve = logit_cosine_fit(_logits, _p, _keys)
        _acts = neuron_activations(_r["analysed"], _r["data"]["all_x"])
        _alive = _acts.std(0) > 1e-6
        _bk, _nfve = neuron_frequency_fit(_acts[:, _alive], _p)
        _var = _acts[:, _alive].var(0)
        analysis[_seed] = {"fractions": _fr, "keys": _keys, "alphas": _alphas, "logit_fve": _fve,
                           "neuron_k": _bk, "neuron_fve": _nfve,
                           "neurons_explained": float(np.mean((_nfve > 0.8) & np.isin(_bk, _keys))),
                           "neurons_at_key": float(np.mean(np.isin(_bk, _keys))), "top5": float(np.sort(_fr)[::-1][:5].sum()),
                           "weighted_fve": float(np.sum(_nfve * _var) / np.sum(_var))}
    _a0 = analysis[CONFIG["seeds"][0]]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.3))
    _axes[0].bar(np.arange(1, len(_a0["fractions"]) + 1), _a0["fractions"])
    _axes[0].set(xlabel="frequency k", ylabel="fraction of embedding energy", title=f"seed {CONFIG['seeds'][0]}: embedding spectrum")
    _axes[1].hist(_a0["neuron_fve"], bins=30)
    _axes[1].set(xlabel="variance explained by one frequency", ylabel="neurons", title="MLP neurons")
    _r0 = runs[CONFIG["seeds"][0]]
    _p = CONFIG["p"]
    with torch.no_grad():
        _L0 = answer_logits(_r0["analysed"], _r0["data"]["all_x"]).numpy().reshape(_p, _p, _p)
    _axes[2].imshow(_L0[:, :, 0], cmap="RdBu")
    _axes[2].set(xlabel="b", ylabel="a", title="logit of c = 0 over all (a, b)")
    mo.vstack([_fig, mo.md("| seed | key frequencies (90% of energy) | top-5 energy | logit FVE by their cosines | "
                           "neurons whose best frequency is key | ... with FVE > 0.8 | median neuron FVE | "
                           "variance-weighted neuron FVE |\n|---|---|---|---|---|---|---|---|\n"
                           + "\n".join(f"| {s} | {a['keys']} | {a['top5']:.2f} | {a['logit_fve']:.3f} | "
                                       f"{a['neurons_at_key']:.2f} | {a['neurons_explained']:.2f} | "
                                       f"{np.median(a['neuron_fve']):.2f} | {a['weighted_fve']:.2f} |"
                                       for s, a in analysis.items()))])
    return (analysis,)


@app.cell
def _(analysis, runs):
    _seed = CONFIG["seeds"][0]
    _r, _a = runs[_seed], analysis[_seed]
    _probe = copy.deepcopy(_r["model"])
    progress = {"step": [], "restricted": [], "excluded": []}
    for _t, _sd in _r["hist"]["checkpoints"]:
        _probe.load_state_dict(_sd)
        _probe.eval()
        with torch.no_grad():
            _L = answer_logits(_probe, _r["data"]["all_x"]).numpy()
        _pm = progress_measures(_L, _r["data"]["all_y"].numpy(), _r["data"]["train_idx"].numpy(), CONFIG["p"], _a["keys"])
        progress["step"].append(_t)
        progress["restricted"].append(_pm["restricted"])
        progress["excluded"].append(_pm["excluded"])
    _h = _r["hist"]
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot(_h["step"], _h["train_loss"], color="C0", alpha=0.5, label="train loss")
    _ax.plot(_h["step"], _h["test_loss"], color="C3", alpha=0.5, label="test loss")
    _ax.plot(progress["step"], progress["restricted"], "o-", color="C2", ms=3, label="restricted loss (key freqs only)")
    _ax.plot(progress["step"], progress["excluded"], "s-", color="C1", ms=3, label="excluded loss (key freqs removed)")
    _ax.set(yscale="log", xlabel="step", ylabel="cross-entropy", title=f"seed {_seed}: progress measures")
    _ax.legend(fontsize=7)
    _test = dict(zip(_h["step"], _h["test_loss"]))
    _show = [i for i, t in enumerate(progress["step"]) if t <= 3 * (runs[_seed]["grokked"] or 0) or i == len(progress["step"]) - 1]
    mo.vstack([_fig, mo.md("| step | test loss | restricted loss | excluded loss |\n|---|---|---|---|\n"
                           + "\n".join(f"| {progress['step'][i]} | {_test[progress['step'][i]]:.4f} | "
                                       f"{progress['restricted'][i]:.4f} | {progress['excluded'][i]:.4f} |"
                                       for i in _show))])
    return (progress,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer each prompt in a few sentences, with the numbers from your runs (and their spread
    over seeds) next to every claim. grade yourself with [`rubric.md`](rubric.md).

    1. **did it grok?** how many steps did memorisation take, and how many generalisation, on
       each seed? what happened to the weight norm in between, and why would weight decay
       matter for the switch?
    2. **what representation did it learn?** which key frequencies does the embedding use, how
       much of its energy do they hold, and do different seeds pick the same ones?
    3. **is the algorithm the one Nanda et al. describe?** what fraction of the (centred)
       logits do $\sum_k \alpha_k \cos(w_k(a+b-c))$ explain? what fraction of neurons are
       explained by a single key frequency? write down, in your own words and equations, how
       the network turns $(a, b)$ into $\cos(w(a+b-c))$.
    4. **was the generalisation sudden?** compare the restricted and excluded losses with
       the test loss over training. when did the circuit start forming, relative to the jump
       in test accuracy?
    5. **limitations.** what does this analysis *not* show (consider: one architecture, one
       $p$, correlational versus causal evidence)? propose one intervention (an ablation or
       a patch with your topic 02 tools) that would test a claim you made, and run it if you
       can.
    """)
    return


if __name__ == "__main__":
    app.run()
