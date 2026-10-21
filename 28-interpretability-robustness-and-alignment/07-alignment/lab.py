import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.rl import value_iteration  # yours, from ch 23 · 03

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    BETAS = np.geomspace(30, 0.01, 40)
    NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]


@app.function
def shaped_reward(P, R_true, bonus_pos, bonus, gamma):
    """R_true + gamma Phi(s') - Phi(s) with Phi = bonus at the bonus cell (potential-based shaping, Ng et al. 1999)."""
    phi = np.zeros(P.shape[0])
    phi[bonus_pos] = bonus
    nxt = P.argmax(-1)
    return R_true + gamma * phi[nxt] - phi[:, None]


@app.function
def planner(kind):
    """an agent that, given a corridor layout, plans with one reward specification and returns its policy."""
    def plan(layout):
        P, Rp, Rt = ex.corridor_mdp(layout["L"], layout["b"], layout["B"])
        R = {"proxy": Rp, "true": Rt, "shaped": shaped_reward(P, Rt, layout["b"], layout["B"], layout["gamma"])}[kind]
        return P, value_iteration(P, R, layout["gamma"])[0]
    return plan


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # alignment · lab

    every experiment here is exact and runs in seconds, so you can turn the knobs. the first
    uses your `kl_regularized_policy` and `best_of_n_distribution` on a synthetic population
    of 5,000 "responses" with a known true quality and three different proxies for it. the
    second and third use your corridor MDP, `gaming_gap` and `eval_suite`. for each
    experiment: write down a prediction, lock it in, run it, then explain.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: optimising a proxy, harder and harder

    5,000 responses with true quality $u \sim \mathcal{N}(0, 1)$ and reference probabilities
    from a softmax of Gaussian noise. three proxies a reward model might learn:
    **noisy** ($u$ + Gaussian error of the same size), **heavy-tailed** ($u$ + an error from
    a Student-$t$ with 2 degrees of freedom, scaled by 0.3: usually small, occasionally huge),
    and **spurious** ($0.5u$ + a feature unrelated to quality, like length). the optimiser is
    your closed-form KL-regularised policy as $\beta$ falls from 30 to 0.01 (lines), and
    best-of-$n$ from 1 to 1,024 (dots), both plotted against their KL from the reference,
    averaged over 20 random populations (the band shows how much one population can differ).
    """)
    return


@app.cell
def _():
    p1 = ui.predict("as the optimiser pushes harder on each proxy (more KL), what happens to the TRUE quality? "
                    "which proxy will be hurt most by optimisation, and how?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _makers = {"noisy": lambda rng, u: u + rng.normal(size=u.size),
               "heavy-tailed": lambda rng, u: u + 0.3 * rng.standard_t(2.0, size=u.size),
               "spurious": lambda rng, u: 0.5 * u + rng.normal(size=u.size)}
    goodhart = {k: {"kl": [], "true": [], "proxy": [], "bon_kl": [], "bon_true": []} for k in _makers}
    _oracle = []
    for _seed in range(20):  # 20 random populations of 5,000 responses
        _rng = np.random.default_rng(_seed)
        _u = _rng.normal(size=5000)
        _ref = np.exp(_rng.normal(size=5000))
        _ref /= _ref.sum()
        for _name, _make in _makers.items():
            _r = _make(_rng, _u)
            _st = [ex.policy_stats(ex.kl_regularized_policy(_ref, _r, b), _ref, _r, _u) for b in BETAS]
            _bon = [ex.policy_stats(ex.best_of_n_distribution(_ref, _r, n), _ref, _r, _u) for n in NS]
            for _key in ("kl", "true", "proxy"):
                goodhart[_name][_key].append([s[_key] for s in _st])
            goodhart[_name]["bon_kl"].append([s["kl"] for s in _bon])
            goodhart[_name]["bon_true"].append([s["true"] for s in _bon])
        _oracle.append([[_o["kl"], _o["true"]] for _o in (ex.policy_stats(ex.kl_regularized_policy(_ref, _u, b), _ref,
                                                                           _u, _u) for b in BETAS)])
    goodhart = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in goodhart.items()}
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for _ax, (_name, _g) in zip(_axes, goodhart.items()):
        _kl = _g["kl"].mean(0)
        _ax.plot(np.mean(_oracle, 0)[:, 0], np.mean(_oracle, 0)[:, 1], ":", color="gray", label="optimising u itself")
        _ax.fill_between(_kl, np.quantile(_g["true"], 0.1, 0), np.quantile(_g["true"], 0.9, 0), color="C0", alpha=0.2,
                         label="true: 10-90% of populations")
        _ax.plot(_kl, _g["true"].mean(0), "-", color="C0", label="true, KL-regularised (mean)")
        _ax.plot(_g["bon_kl"].mean(0), _g["bon_true"].mean(0), "o", color="C0", ms=4, label="true, best-of-n (mean)")
        _ax.plot(_kl, _g["proxy"].mean(0), "--", color="C3", label="proxy (mean)")
        _ax.set(title=f"{_name} proxy", xlabel="KL from reference (nats)", xlim=(0, 9), ylim=(-1.5, 6))
    _axes[0].set_ylabel("expected value")
    _axes[0].legend(fontsize=6, loc="upper left")
    _rows = []
    for _name, _g in goodhart.items():
        _m = _g["true"].mean(0)
        _k = int(np.argmax(_m))
        _rows.append(f"| {_name} | {_m[_k]:.2f} at KL {_g['kl'].mean(0)[_k]:.1f} | {_m[-1]:.2f} "
                     f"[{np.quantile(_g['true'][:, -1], 0.1):.2f}, {np.quantile(_g['true'][:, -1], 0.9):.2f}] | "
                     f"{_g['proxy'].mean(0)[-1]:.2f} | {_g['bon_true'].mean(0).max():.2f} at n = "
                     f"{NS[int(np.argmax(_g['bon_true'].mean(0)))]} | {_g['bon_true'].mean(0)[-1]:.2f} |")
    mo.vstack([_fig, mo.md("| proxy | best mean true quality | mean true at beta = 0.01 [10%, 90%] | mean proxy at beta "
                           "= 0.01 | best mean true, best-of-n | mean true, best-of-1024 |\n|---|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the proxy goes up in every panel. why does the true quality keep (slowly)
    improving for the noisy proxy, but go nowhere (and become a lottery) for the heavy-tailed
    one? which responses does a strong optimiser select when a few of them have enormous
    proxy errors? best-of-$n$ does much better than the KL-regularised policy on the
    heavy-tailed proxy, up to a point: what does best-of-$n$ use from the proxy, and what
    does $e^{r/\beta}$ use? if you could only see the red curves, when would you stop?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: when does an agent game its reward?

    a corridor of 10 cells, a bonus cell at position 2 that pays the proxy reward $B$ every
    time it's entered, and a goal worth 10 at the end. your `gaming_gap` plans with the
    proxy and measures the regret under the true reward, over a grid of bonus sizes and
    discount factors. the dashed curve is the threshold you derive in the lesson,
    $B^\star(\gamma) = G\gamma^{L-1-b}(1-\gamma^2)/\gamma^2$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("a patient agent (gamma near 1) or an impatient one (gamma = 0.5): which games a small bonus? "
                    "is there a discount factor at which gaming is hardest to trigger?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _L, _b, _G = 10, 2, 10.0
    _gammas = np.linspace(0.3, 0.99, 36)
    _bonuses = np.geomspace(0.05, 20, 40)
    regret = np.zeros((len(_bonuses), len(_gammas)))
    for _i, _B in enumerate(_bonuses):
        _P, _Rp, _Rt = ex.corridor_mdp(_L, _b, _B, goal_reward=_G)
        for _j, _g in enumerate(_gammas):
            _out = ex.gaming_gap(_P, _Rp, _Rt, _g)
            regret[_i, _j] = _out["regret"] / max(_out["optimal_value"], 1e-12)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _im = _ax.pcolormesh(_gammas, _bonuses, regret, cmap="Reds", vmin=0, vmax=1, shading="auto")
    _thr = _G * _gammas ** (_L - 1 - _b) * (1 - _gammas ** 2) / _gammas ** 2
    _ax.plot(_gammas, np.clip(_thr, _bonuses[0], _bonuses[-1]), "k--", lw=1.5, label="threshold from the lesson")
    _ax.set(yscale="log", xlabel="discount gamma", ylabel="bonus B", title="fraction of the true value lost",
            ylim=(_bonuses[0], _bonuses[-1]), xlim=(_gammas[0], _gammas[-1]))
    _ax.legend(fontsize=7, loc="upper right")
    _fig.colorbar(_im)
    _k = int(np.argmax(_thr))
    mo.vstack([_fig, mo.md(f"the threshold is highest, **B = {_thr[_k]:.2f}**, at gamma = {_gammas[_k]:.2f}; at "
                           f"gamma = 0.99 a bonus of {_thr[-1]:.3f} (1/{_G / _thr[-1]:.0f} of the goal) is enough "
                           f"to make the agent loop forever, and at gamma = 0.3 one of {_thr[0]:.3f}.")])
    return


@app.cell
def _():
    bonus_pick = mo.ui.slider(0.0, 3.0, value=1.0, step=0.05, label="bonus B")
    gamma_pick = mo.ui.slider(0.3, 0.99, value=0.9, step=0.01, label="discount gamma")
    mo.hstack([bonus_pick, gamma_pick])
    return bonus_pick, gamma_pick


@app.cell
def _(bonus_pick, gamma_pick, p2):
    ui.gate(p2)
    _P, _Rp, _Rt = ex.corridor_mdp(10, 2, bonus_pick.value)
    _out = ex.gaming_gap(_P, _Rp, _Rt, gamma_pick.value)
    _path = ex.rollout_states(_P, _out["policy"], 0, 14)
    _thr = 10 * gamma_pick.value ** 7 * (1 - gamma_pick.value ** 2) / gamma_pick.value ** 2
    mo.md(f"B = {bonus_pick.value:.2f}, gamma = {gamma_pick.value:.2f} (threshold {_thr:.3f}): the proxy-optimal agent "
          f"visits `{_path}`; true value {_out['true_value']:.3f} of a possible {_out['optimal_value']:.3f}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do both very impatient and very patient agents game a small bonus,
    for different reasons? (what does each one compare?) the bonus cell is two steps from the
    start and seven from the goal: how would moving it change the picture? what does this
    say about checking a reward function by asking "does it give more reward for what I
    want?"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: an eval catches what the training reward hides

    200 random corridors (length 5–12, bonus cell anywhere inside, bonus 0.1–3, $\gamma = 0.9$),
    each one a test case: plan, run the policy for $2L$ steps with your `rollout_states`, and
    pass if the agent reaches the goal. three agents plan with three reward specifications:
    the **proxy** (bonus every entry), the **true** reward, and the proxy rewritten as
    **potential-based shaping** (the bonus is a potential $\Phi$, paid as $\gamma\Phi(s') -
    \Phi(s)$: you earn it on the way in and pay it back on the way out). your `eval_suite`
    splits the cases by bonus size and gives every pass rate an interval; `compare_agents`
    gives paired differences.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("what fraction of corridors will the proxy agent fail? will the shaped agent, which is paid the "
                    "same bonus for reaching the cell, game it too?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(0)
    _layouts = []
    for _ in range(200):
        _L = int(_rng.integers(5, 13))
        _layouts.append({"L": _L, "b": int(_rng.integers(1, _L - 1)), "B": float(_rng.uniform(0.1, 3.0)), "gamma": 0.9})

    def _case(layout):
        def check(agent):
            P, pi = agent(layout)
            return layout["L"] - 1 in ex.rollout_states(P, pi, 0, 2 * layout["L"])
        return ("bonus < 1" if layout["B"] < 1 else "bonus >= 1", check)

    _cases = [_case(l) for l in _layouts]
    _agents = {k: planner(k) for k in ("proxy", "true", "shaped")}
    eval_out = {k: ex.eval_suite(a, _cases, rng=0) for k, a in _agents.items()}
    _diffs = {k: ex.compare_agents(_agents["true"], _agents[k], _cases, rng=0) for k in ("proxy", "shaped")}
    _cats = ["bonus < 1", "bonus >= 1", "overall"]
    mo.vstack([
        mo.md("| agent | " + " | ".join(_cats) + " |\n|---|---|---|---|\n"
              + "\n".join(f"| {k} | " + " | ".join(f"{v[c][0]:.2f} [{v[c][1]:.2f}, {v[c][2]:.2f}]" for c in _cats)
                          + " |" for k, v in eval_out.items())),
        mo.md("paired difference true − agent over the same 200 cases: "
              + "; ".join(f"{k}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" for k, d in _diffs.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the proxy agent collects more proxy reward than the true agent in every
    corridor it fails. why would its training curves look *better*? why can't the shaped
    agent ever profit from looping (sum $\gamma\Phi(s') - \Phi(s)$ around a loop), even
    though it's paid the same bonus for entering the cell? what would a real eval need that
    this one has for free (a known true objective, every case checkable)?
    """)
    return


if __name__ == "__main__":
    app.run()
