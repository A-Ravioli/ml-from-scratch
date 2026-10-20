import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.deeprl import CartPole, RolloutCollector, compute_gae  # yours, from topics 01 and 03

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small networks: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # trust regions & PPO · lab

    every experiment here runs **your** PPO code on your `CartPole`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: does the clip keep you in the trust region?

    we train PPO on CartPole for 20,000 steps, then freeze one fresh batch (8 environments ×
    128 steps) with its GAE advantages. starting each time from the same policy, we run your
    `ppo_update` on that batch for 1 to 40 epochs (minibatches of 256, Adam lr
    $3\cdot10^{-4}$), with the clip ($\varepsilon = 0.2$) and without it
    ($\varepsilon = 10^9$, the plain surrogate). after each update we measure, on the batch,
    the exact mean KL divergence from the old policy, the largest ratio
    $\pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$, and the fraction of ratios outside
    $[0.8, 1.2]$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with the clip, after 40 epochs on one batch, what's the largest ratio in the batch: about 1.2, about 2, or "
        "much larger? how does the KL compare with the unclipped run?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _model, _ = ex.train_ppo(CartPole, 20_000, lr=1e-3, seed=0)
    torch.manual_seed(1)
    _col = RolloutCollector([CartPole() for _ in range(8)], _model, gamma=0.99, seed=1)
    _b = _col.collect(128)
    _adv, _ret = compute_gae(_b["rewards"], _b["values"], _b["dones"], _b["last_value"], gamma=0.99, lam=0.95)
    batch = {"obs": _b["obs"].reshape(-1, 4), "actions": _b["actions"].reshape(-1),
             "log_probs": _b["log_probs"].reshape(-1), "values": _b["values"].reshape(-1),
             "advantages": _adv.reshape(-1), "returns": _ret.reshape(-1)}
    with torch.no_grad():
        _old_dist, _ = _model(batch["obs"])
    _epochs = [1, 2, 5, 10, 20, 40]
    _res = {}
    for _name, _eps in [("clip ε = 0.2", 0.2), ("no clip", 1e9)]:
        _res[_name] = []
        for _E in _epochs:
            _m = copy.deepcopy(_model)
            ex.ppo_update(_m, torch.optim.Adam(_m.parameters(), lr=3e-4, eps=1e-5), batch, n_epochs=_E,
                          minibatch_size=256, clip_eps=_eps, seed=0)
            with torch.no_grad():
                _new_dist, _ = _m(batch["obs"])
                _kl = torch.distributions.kl_divergence(_old_dist, _new_dist).mean().item()
                _r = (_new_dist.log_prob(batch["actions"]) - batch["log_probs"]).exp()
            _res[_name].append((_kl, _r.max().item(), ((_r - 1).abs() > 0.2).float().mean().item()))
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for _name, _v in _res.items():
        _v = np.array(_v)
        for _i, _ax in enumerate(_axes):
            _ax.plot(_epochs, _v[:, _i], "o-", label=_name)
    _axes[0].set(xscale="log", yscale="log", xlabel="epochs on one batch", ylabel="mean KL(old ‖ new)")
    _axes[1].set(xscale="log", xlabel="epochs on one batch", ylabel="largest ratio in the batch")
    _axes[1].axhline(1.2, color="k", ls=":", lw=1)
    _axes[2].set(xscale="log", xlabel="epochs on one batch", ylabel="fraction of ratios outside [0.8, 1.2]")
    for _ax in _axes:
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {E} | " + " | ".join(f"{_res[n][i][0]:.4f} / {_res[n][i][1]:.2f} / {_res[n][i][2]:.2f}"
                                     for n in _res) + " |" for i, E in enumerate(_epochs)]
    mo.vstack([_fig, mo.md("| epochs | clip ε = 0.2: KL / max ratio / outside | no clip: KL / max ratio / outside |\n"
                           "|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the clip zeroes the gradient of a sample once its ratio leaves
    $[1 - \varepsilon, 1 + \varepsilon]$ in the helpful direction. so how can any ratio end
    up outside that range, and keep drifting further as the epochs go on? (think about what
    else moves a sample's probability: the *other* samples' gradients, through shared network
    weights; and Adam's momentum.) by how much does the clip slow the drift compared with the
    plain surrogate? what would actually bound the KL, and what does that say about calling
    PPO a trust-region method?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: implementation details that matter (on CartPole)

    your `train_ppo` on CartPole for 60,000 steps, three seeds per variant, removing one
    detail at a time: advantage normalization, gradient clipping, learning-rate annealing,
    sample reuse (1 epoch per batch instead of 20), and the clip itself. the settings reuse
    each batch aggressively (8 environments × 128 steps, **20 epochs**, minibatches of 256,
    Adam lr $3\cdot10^{-3}$), which is fast when it works and is the regime the clip was
    designed for. (with the gentler 10 epochs at lr $10^{-3}$, try it, almost nothing but
    the epochs matters.)
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which removal hurts most on CartPole at this budget? which ones make no measurable difference?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _base = dict(n_envs=8, n_steps=128, lr=3e-3, n_epochs=20, minibatch_size=256)
    _variants = {
        "PPO (all details)": {},
        "no advantage normalization": dict(norm_adv=False),
        "no gradient clipping": dict(max_grad_norm=None),
        "no lr annealing": dict(anneal_lr=False),
        "1 epoch per batch": dict(n_epochs=1),
        "no clipping (ε = 10⁹)": dict(clip_eps=1e9),
    }
    _rows = []
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    for _i, (_name, _kw) in enumerate(_variants.items()):
        _hs = [ex.train_ppo(CartPole, 60_000, seed=s, **{**_base, **_kw})[1] for s in range(3)]
        _final = np.array([np.mean(h["episode_returns"][-20:]) for h in _hs])
        _avg = np.array([np.mean(h["episode_returns"]) for h in _hs])
        _ax.scatter(_avg, [_i] * 3, color="C0", s=20)
        _ax.errorbar(_avg.mean(), _i, xerr=_avg.std(ddof=1) / np.sqrt(3), fmt="k|", ms=14, capsize=4)
        _rows.append(f"| {_name} | {_final.mean():.0f} ± {_final.std(ddof=1) / np.sqrt(3):.0f} | "
                     f"{_avg.mean():.0f} ± {_avg.std(ddof=1) / np.sqrt(3):.0f} | {', '.join(f'{x:.0f}' for x in _avg)} |")
    _ax.set_yticks(range(len(_variants)), list(_variants))
    _ax.invert_yaxis()
    _ax.set(xlabel="mean episode return over training (dots: seeds; bars: mean ± s.e.)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| variant | last 20 episodes | mean over training | per seed |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which differences are larger than the seed-to-seed spread? why does
    removing the clip hurt here, although experiment 1 showed that the clip doesn't keep the
    ratios inside $[0.8, 1.2]$ (what does it still do to the *rate* of drift)? why is
    sample reuse worth so much? gradient clipping and lr annealing make no visible
    difference on CartPole: think about the reward scale, the network size and the budget,
    and say what you'd change to make them matter.
    """)
    return


if __name__ == "__main__":
    app.run()
