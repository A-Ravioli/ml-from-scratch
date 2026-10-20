import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.rl import Box, Env  # yours, from ch 23 · topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small networks: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # continuous control · lab

    every experiment here runs **your** `train_sac`. for each one: write down a prediction,
    lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what the temperature does to a policy

    a one-step task with two equally good actions, $a = \pm0.5$, each worth 1, and a valley
    (worth almost nothing) between them; the observation is constant. your `train_sac` learns
    it for 1,500 steps (500 of them random warm-up) at four fixed temperatures. then we draw
    5,000 actions from each trained actor and plot their histogram against the reward.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for alpha = 0, 0.05, 0.2 and 1: where does the policy put its actions? which alpha earns the most reward per "
        "action, and does any of them cover both peaks?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    class TwoPeaks(Env):
        """a one-step task: reward exp(-((a - 0.5) / 0.15)^2) + exp(-((a + 0.5) / 0.15)^2) for a in [-1, 1]."""

        def __init__(self):
            self.observation_space, self.action_space = Box(0.0, 1.0, (1,)), Box(-1.0, 1.0, (1,))

        def reset(self, *, seed=None):
            super().reset(seed=seed)
            return np.array([0.5]), {}

        def step(self, action):
            a = float(np.asarray(action).reshape(-1)[0])
            r = np.exp(-(((a - 0.5) / 0.15) ** 2)) + np.exp(-(((a + 0.5) / 0.15) ** 2))
            return np.array([0.5]), float(r), True, False, {}

    _alphas = [0.0, 0.05, 0.2, 1.0]
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.4))
    _grid = np.linspace(-1, 1, 400)
    _reward = np.exp(-(((_grid - 0.5) / 0.15) ** 2)) + np.exp(-(((_grid + 0.5) / 0.15) ** 2))
    _rows = []
    for _ax, _alpha in zip(_axes, _alphas):
        _actor, _ = ex.train_sac(TwoPeaks(), 1500, learning_starts=500, alpha=_alpha, hidden_sizes=(32, 32),
                                 batch_size=128, lr=3e-3, tau=0.05, seed=0)
        torch.manual_seed(0)
        with torch.no_grad():
            _a, _lp = _actor(torch.full((5000, 1), 0.5))
        _a = _a[:, 0].numpy()
        _r = np.exp(-(((_a - 0.5) / 0.15) ** 2)) + np.exp(-(((_a + 0.5) / 0.15) ** 2))
        _ax.hist(_a, bins=60, range=(-1, 1), density=True, alpha=0.6, label="policy's actions")
        _ax.plot(_grid, 2 * _reward, "k", lw=1, label="reward (×2)")
        _ax.set(title=f"α = {_alpha}", xlabel="action", ylabel="density" if _alpha == 0.0 else None)
        _rows.append(f"| {_alpha} | {_r.mean():.3f} | {np.mean(_a > 0):.2f} | {-_lp.mean().item():.2f} |")
    _axes[0].legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| α | mean reward per action | fraction of actions > 0 | entropy estimate (−mean log π) |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does α = 0 commit to one peak, and what decided which one? from
    α = 0.2 on, the policy is almost uniform on $(-1, 1)$ (whose entropy is
    $\log 2 \approx 0.69$): why is that the best this actor can do once the entropy bonus is
    worth more than the reward it gives up? the Boltzmann policy $\pi \propto e^{Q/\alpha}$
    for α = 0.2 would put almost all its mass on the two peaks ($e^{1/0.2} \approx 150$ times
    more density there than in the valley): why can't your actor represent it, and what kind
    of policy could?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the temperature on the pendulum

    your `train_sac` on your `Pendulum` for 6,000 steps (30 episodes; the first 1,000 steps
    are random warm-up), with hidden layers of 64, batches of 128, Adam lr $10^{-3}$ and
    $\tau = 0.01$, at fixed temperatures α = 0, 0.2, 1 and 5, and with the automatic
    temperature (target entropy $-1$), two seeds each. a good swing-up policy earns about
    $-150$ per episode; a random one about $-1200$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which temperatures solve the pendulum in 6,000 steps? where does the automatic temperature end up?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _settings = [0.0, 0.2, 1.0, 5.0, "auto"]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _rows = []
    for _alpha in _settings:
        _runs = [ex.train_sac(ex.Pendulum(), 6000, hidden_sizes=(64, 64), batch_size=128, lr=1e-3, tau=0.01,
                              learning_starts=1000, alpha=_alpha, seed=s)[1] for s in range(2)]
        _curve = np.mean([h["episode_returns"] for h in _runs], axis=0)
        _axes[0].plot(np.arange(1, len(_curve) + 1), _curve, label=f"α = {_alpha}")
        _final = np.array([np.mean(h["episode_returns"][-5:]) for h in _runs])
        _a_end = f"{np.mean([h['alpha'][-1] for h in _runs]):.3f}"
        if _alpha == "auto":
            for _s, _h in enumerate(_runs):
                _axes[1].plot(_h["alpha"], label=f"seed {_s}")
        _rows.append(f"| {_alpha} | {', '.join(f'{x:.0f}' for x in _final)} | {_a_end} |")
    _axes[0].axvline(5, color="0.6", ls=":", lw=1)
    _axes[0].set(xlabel="episode (200 steps; warm-up ends after episode 5)", ylabel="return (mean of 2 seeds)")
    _axes[0].legend(fontsize=8)
    _axes[1].set(xlabel="episode after learning starts", ylabel="automatic α", yscale="log")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| α | return over the last 5 episodes, per seed | α at the end |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the pendulum is solved even with α = 0: where does the exploration come
    from, and why is this task forgiving? what goes wrong at α = 5 (compare the entropy bonus
    $-\alpha\log\pi$ with the size of the rewards, about $-16$ to 0 per step)? the automatic
    temperature is still falling after 6,000 steps: what decides where it's heading, and why
    would you prefer it on a new task whose reward scale you don't know?
    """)
    return


if __name__ == "__main__":
    app.run()
