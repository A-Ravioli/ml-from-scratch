import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui
    from mlfs.research import mean_ci  # yours, from topic 03
    from mlfs.torchnn import MLP, evaluate, fit, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST from the cache, or a small synthetic stand-in when offline
    X_all = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(np.float32) / 255.0
    y_all = _d["y_train"]
    X_test = _d["X_test"].reshape(len(_d["X_test"]), -1).astype(np.float32) / 255.0
    y_test = _d["y_test"]
    MU, SD = float(X_all.mean()), float(X_all.std())
    N_TRAIN = min(5_000, len(X_all) // 2)


@app.function
def mnist_run(seed, *, smoothing=0.0):
    """test accuracy (%) of a 784 -> 128 -> 10 MLP, 3 epochs of SGD on N_TRAIN images, with label smoothing."""
    idx = np.random.default_rng(seed).choice(len(X_all), N_TRAIN, replace=False)
    torch.manual_seed(seed)
    model = MLP(784, [128], 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    def loss(logits, y):
        return F.cross_entropy(logits, y, label_smoothing=smoothing)

    fit(model, loss, opt, make_loader((X_all[idx] - MU) / SD, y_all[idx], batch_size=64, seed=seed), epochs=3)
    return 100 * evaluate(model, make_loader((X_test - MU) / SD, y_test, batch_size=2000, shuffle=False))["accuracy"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # finding questions · lab

    every experiment here uses **your** proposal tools from `exercise.py`. the first sizes a
    real question with a pilot on MNIST; the second asks what a laptop can afford; the third
    shows why the most promising pilot result is usually an overestimate. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the lab takes about 15 seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how many seeds does a small question need?

    a question in the style of idea 1: *does label smoothing (0.1) raise the test accuracy of a
    small MLP trained for 3 epochs on 5,000 MNIST images?* before planning the real
    experiment, a pilot: 8 paired seeds (each seed fixes the training subset, the
    initialization and the data order for both arms). the pilot's job is not to answer the
    question but to measure the noise: the standard deviation of the paired differences, which
    your `seeds_for_power` turns into a plan.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the paired differences of the pilot have some sd. to detect a true effect of 0.2 accuracy points with 80% "
        "power, how many paired seeds will you need? what's the power of a 3-seed comparison?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _base = np.array([mnist_run(s) for s in range(8)])
    _ls = np.array([mnist_run(s, smoothing=0.1) for s in range(8)])
    pilot_d = _ls - _base
    # floor at 0.05 points: offline, the synthetic stand-in is so easy that every run can score the same
    pilot_sd = max(float(pilot_d.std(ddof=1)), 0.05)
    _m, _lo, _hi = mean_ci(pilot_d)
    _effects = (0.1, 0.2, 0.3, 0.5, 1.0)
    _rows = []
    for _e in _effects:
        try:
            _need = ex.seeds_for_power(pilot_sd, _e)
        except ValueError:
            _need = "> 1000"
        _rows.append(f"| {_e:g} | {_need} | {ex.paired_power(pilot_sd, _e, 3):.2f} | {ex.paired_power(pilot_sd, _e, 5):.2f} |")
    _ns = np.arange(2, 81)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _e in (0.2, 0.5, 1.0):
        _ax.plot(_ns, [ex.paired_power(pilot_sd, _e, int(n)) for n in _ns], label=f"true effect {_e:g} points")
    _ax.axhline(0.8, color="k", ls="--", lw=0.8)
    _ax.axvline(3, color="gray", ls=":", lw=0.8)
    _ax.set(xlabel="paired seeds", ylabel="power (one-sided, α = 0.05)",
            title=f"power of a paired t test, sd of differences {pilot_sd:.2f} points (pilot)")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"pilot: baseline {_base.mean():.2f}%, label smoothing {_ls.mean():.2f}%; paired difference "
              f"{_m:+.2f} points (95% CI [{_lo:+.2f}, {_hi:+.2f}]), sd of the differences **{pilot_sd:.2f}** points."),
        mo.md("| true effect (points) | seeds for 80% power | power with 3 seeds | with 5 |\n|---|---|---|---|\n"
              + "\n".join(_rows)),
        _fig,
    ])
    return (pilot_sd,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the number of seeds grow so fast as the effect you care about
    shrinks? what does a "no significant difference" from a 3-seed comparison tell you about
    an effect of 0.2 points? what does the pilot's own interval say about the question, and why
    is it a bad idea to choose `min_effect` from it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what can a laptop afford?

    we measure this machine's matrix-multiply throughput (float32, whatever thread count
    torch runs with here), then ask your `largest_model` for the biggest compute-optimal
    language model (Chinchilla's 20 tokens per parameter) it could train in an hour, a day and
    a week at 40% of that throughput, and your `run_hours` how long two famous models would
    take. the numbers are for *this* machine; yours will differ.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with a week of this laptop, how many parameters does the largest compute-optimal language model have? and "
        "how long would GPT-2 small (124M parameters, at 20 tokens per parameter) take?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _n = 512
    _a, _b = torch.randn(_n, _n), torch.randn(_n, _n)
    for _ in range(3):
        _a @ _b  # warm up
    _times = []
    for _ in range(5):
        _t = time.perf_counter()
        for _ in range(10):
            _a @ _b
        _times.append((time.perf_counter() - _t) / 10)
    flops = 2 * _n ** 3 / min(_times)
    _rows = []
    for _label, _h in (("an hour", 1.0), ("a day", 24.0), ("a week", 168.0)):
        _p, _tok = ex.largest_model(_h, flops_per_second=flops)
        _rows.append(f"| {_label} | {_p / 1e6:.1f}M | {_tok / 1e6:.0f}M |")
    _gpt2 = ex.run_hours(124e6, 20 * 124e6, flops_per_second=flops)
    _chin = ex.run_hours(70e9, 1.4e12, flops_per_second=flops)
    mo.vstack([
        mo.md(f"measured matmul throughput: **{flops / 1e9:.0f} GFLOP/s** (float32, {torch.get_num_threads()} "
              f"thread{'s' if torch.get_num_threads() > 1 else ''}); at 40% utilization, "
              f"{0.4 * flops / 1e9:.0f} GFLOP/s."),
        mo.md("| budget | largest compute-optimal model | tokens |\n|---|---|---|\n" + "\n".join(_rows)),
        mo.md(f"GPT-2 small (124M parameters, 2.5B tokens): **{_gpt2:,.0f} hours** ({_gpt2 / 24 / 7:.0f} weeks). "
              f"Chinchilla (70B parameters, 1.4T tokens): {_chin / 24 / 365:,.0f} years."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what kind of research question is a good fit for this budget, and which kind
    isn't? the 6ND rule counts only the model's matrix multiplies: what does it leave out for a
    real training run, and in which direction does that push the hours?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the most promising pilot

    you have 20 candidate ideas and pilot each with **one** paired seed. most ideas do little:
    their true effects are drawn from a normal distribution with mean 0 and sd 0.2 points, and
    one paired run has a noise sd of 0.5 points (a little less than the 0.65 experiment 1 measured). you pick
    the idea with the best pilot and follow it up properly with 20 paired seeds. we simulate
    2,000 such research projects.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the winning pilot looked like an improvement of about how many points? what is its true effect, on "
        "average? in how many of the 2,000 projects does the 20-seed follow-up confirm it (95% interval above 0)?"
    )
    p3
    return (p3,)


@app.cell
def _():
    n_ideas = mo.ui.slider(2, 50, value=20, label="ideas piloted")
    n_ideas
    return (n_ideas,)


@app.cell
def _(n_ideas, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(0)
    _pilot, _truth, _follow, _pw, _confirmed = [], [], [], [], 0
    for _ in range(2000):
        _true = _rng.normal(0.0, 0.2, n_ideas.value)
        _obs = _true + _rng.normal(0.0, 0.5, n_ideas.value)
        _best = int(np.argmax(_obs))
        _d = _true[_best] + _rng.normal(0.0, 0.5, 20)  # the follow-up: 20 fresh paired seeds
        _m, _lo, _hi = mean_ci(_d)
        _pilot.append(_obs[_best])
        _truth.append(_true[_best])
        _follow.append(_m)
        _confirmed += _lo > 0
        # "a 95% interval above 0" is a one-sided test at alpha = 0.025
        _pw.append(ex.paired_power(0.5, _true[_best], 20, alpha=0.025))
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _bins = np.linspace(-1, 2.5, 50)
    _ax.hist(_pilot, bins=_bins, alpha=0.6, label="the winner's pilot estimate (1 seed)")
    _ax.hist(_truth, bins=_bins, alpha=0.6, label="the winner's true effect")
    _ax.hist(_follow, bins=_bins, alpha=0.6, label="the follow-up estimate (20 seeds)")
    _ax.set(xlabel="effect (accuracy points)", ylabel="projects (of 2,000)",
            title=f"pick the best of {n_ideas.value} one-seed pilots")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"winner's pilot estimate: **{np.mean(_pilot):+.2f}** points on average; its true effect: "
        f"**{np.mean(_truth):+.2f}**; the follow-up estimate: {np.mean(_follow):+.2f}. the follow-up's 95% interval "
        f"lies above 0 in **{_confirmed / 20:.0f}%** of projects; your `paired_power`, averaged over the winners' "
        f"true effects, predicts {100 * np.mean(_pw):.0f}%.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where did the winner's pilot advantage come from? why does the follow-up
    estimate land near the true effect rather than near the pilot? set the slider to 2 and to
    50 ideas: how does the overestimate change, and what does that mean for a lab that tries
    many ideas and publishes the best one?
    """)
    return


if __name__ == "__main__":
    app.run()
