import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    # a thesis plan with three-point estimates in weeks: (optimistic, likely, pessimistic)
    THESIS = {
        "read & annotate": {"estimate": (2, 3, 6)},
        "proposal": {"estimate": (1, 1, 2), "after": ["read & annotate"]},
        "reproduce baseline": {"estimate": (2, 3, 8), "after": ["proposal"]},
        "pilot": {"estimate": (1, 1, 3), "after": ["proposal"]},
        "main experiments": {"estimate": (3, 4, 10), "after": ["reproduce baseline", "pilot"]},
        "ablations": {"estimate": (1, 2, 5), "after": ["main experiments"]},
        "second dataset": {"estimate": (1, 2, 6), "after": ["main experiments"]},
        "figures": {"estimate": (1, 1, 2), "after": ["ablations", "second dataset"]},
        "write paper": {"estimate": (2, 3, 6), "after": ["figures"]},
        "revise": {"estimate": (1, 1, 3), "after": ["write paper"]},
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # thesis project · lab

    every experiment here uses **your** planner from `exercise.py` on a realistic thesis plan
    of ten tasks (below), each with an optimistic, likely and pessimistic duration in weeks.
    the estimates are invented but typical; replace them with your own. for each experiment:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the lab takes a few seconds.
    """)
    return


@app.cell
def _():
    _plan = ex.critical_path({k: {"weeks": v["estimate"][1], "after": v.get("after", [])} for k, v in THESIS.items()})
    mo.md("| task | after | optimistic | likely | pessimistic | slack (likely plan) |\n|---|---|---|---|---|---|\n"
          + "\n".join(f"| {k} | {', '.join(v.get('after', [])) or '–'} | {v['estimate'][0]} | {v['estimate'][1]} | "
                      f"{v['estimate'][2]} | {_plan['slack'][k]:g} |" for k, v in THESIS.items())
          + f"\n\nwith every task at its likely duration the thesis takes **{_plan['total']:g} weeks**; the critical "
          f"path is {' → '.join(_plan['path'])}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the plan everyone writes

    you set your deadline to the plan's length with every task at its *likely* duration, the
    number in the middle. your `schedule_risk` draws each task's duration from a triangular
    distribution over (optimistic, likely, pessimistic), 20,000 times, and propagates them
    through the task graph.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what's the probability of finishing by the 'likely' total? by how many weeks should you pad the "
        "deadline to be 80% sure?"
    )
    p1
    return (p1,)


@app.cell
def _():
    pad = mo.ui.slider(0, 12, value=0, step=1, label="weeks of padding on the likely total")
    pad
    return (pad,)


@app.cell
def _(p1, pad):
    ui.gate(p1)
    _likely = ex.critical_path({k: {"weeks": v["estimate"][1], "after": v.get("after", [])}
                                for k, v in THESIS.items()})["total"]
    risk = ex.schedule_risk(THESIS, _likely + pad.value, n_sims=20000, rng=0)
    _rng = np.random.default_rng(0)  # the same draws schedule_risk made with rng=0
    _durations = {k: _rng.triangular(*v["estimate"], size=20000) for k, v in THESIS.items()}
    _makespan, _ = ex.simulate_makespan(THESIS, _durations)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.hist(_makespan, bins=60, color="tab:blue", alpha=0.7)
    _ax.axvline(_likely + pad.value, color="tab:green", lw=3, label=f"your deadline: {_likely + pad.value:g} weeks")
    _ax.axvline(_likely, color="k", ls="--", label=f"likely plan: {_likely:g} weeks")
    _ax.axvline(risk["quantiles"][0.8], color="tab:red", label=f"80% sure: {risk['quantiles'][0.8]:.1f} weeks")
    _ax.set(xlabel="weeks to finish the thesis", ylabel="simulations (of 20,000)", title="Monte Carlo finish dates")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"P(done by {_likely + pad.value:g} weeks) = **{risk['p_on_time']:.3g}**. median {risk['quantiles'][0.5]:.1f}, "
        f"80% {risk['quantiles'][0.8]:.1f}, 95% {risk['quantiles'][0.95]:.1f} weeks.\n\n"
        "how often each task is on the critical path: "
        + ", ".join(f"{k} {100 * c:.0f}%" for k, c in sorted(risk["criticality"].items(), key=lambda kv: -kv[1])))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** every task finishes by its likely duration at least some of the time, so
    why is the whole plan so rarely on time? which property of the estimates (look at the
    gaps between likely and pessimistic) does the damage? which tasks would you watch most
    closely, according to the criticality numbers, and does that match the likely plan's
    critical path?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: work in parallel, finish later?

    a common way to "save time" is to run things in parallel: four experiments at once instead
    of one after another. here $k$ independent tasks, each estimated at (2, 3, 6) weeks, run
    in parallel, and the project is done when the last one is. the likely plan says 3 weeks,
    whatever $k$ is.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 1, 2, 4 and 8 parallel tasks, how does the 80%-sure finish date change? and P(done by the "
        "likely 3 weeks)?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rows = []
    _ks = [1, 2, 4, 8]
    _q80 = []
    for _k in _ks:
        _tasks = {f"task {i + 1}": {"estimate": (2, 3, 6)} for i in range(_k)}
        _r = ex.schedule_risk(_tasks, 3.0, n_sims=20000, rng=_k)
        _q80.append(_r["quantiles"][0.8])
        _rows.append(f"| {_k} | {_r['likely_total']:g} | {_r['p_on_time']:.3g} | {_r['quantiles'][0.5]:.2f} | "
                     f"{_r['quantiles'][0.8]:.2f} |")
    mo.md("| parallel tasks | likely plan (weeks) | P(done by 3) | median | 80% sure |\n|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does adding parallel tasks push the finish date later even though no
    task got longer? (this is called *merge bias*.) what does it mean for a thesis plan that
    waits for "all experiments" before writing starts?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: calibrating on your own history

    three-point estimates are still *your* estimates, and people are reliably optimistic about
    their own tasks: in Buehler, Griffin & Ross (1994), students writing an honours thesis
    predicted on average about 34 days and took about 56. so log every finished task as
    (estimate, actual) and let `forecast` resample those ratios. here the history is
    **synthetic**: ratios drawn from a log-normal with a median of 1.6, roughly the ratio in
    that study. the plan uses the single likely estimates.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with a history of 15 logged tasks, what does the forecast put as the 80%-sure finish date for the "
        "18-week plan? how much does it change with 5 logged tasks instead of 15?"
    )
    p3
    return (p3,)


@app.cell
def _():
    n_hist = mo.ui.slider(3, 40, value=15, label="logged tasks in your history")
    n_hist
    return (n_hist,)


@app.cell
def _(n_hist, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _est = _rng.choice([0.5, 1.0, 2.0, 3.0], size=40)
    _ratio = np.exp(_rng.normal(np.log(1.6), 0.45, size=40))
    _history = list(zip(_est[:n_hist.value], (_est * _ratio)[:n_hist.value]))
    _plan = {k: {"estimate": float(v["estimate"][1]), "after": v.get("after", [])} for k, v in THESIS.items()}
    _f = ex.forecast(_plan, _history, 18.0, n_sims=20000, rng=0)
    _again = [ex.forecast(_plan, list(zip(_est2, _est2 * np.exp(r.normal(np.log(1.6), 0.45, n_hist.value)))), 18.0,
                          n_sims=4000, rng=1)["quantiles"][0.8]
              for r in [np.random.default_rng(100 + j) for j in range(20)]
              for _est2 in [r.choice([0.5, 1.0, 2.0, 3.0], size=n_hist.value)]]
    mo.md(f"plan (likely estimates): **{_f['naive_total']:g} weeks**. history of {n_hist.value} tasks, median "
          f"ratio {_f['median_ratio']:.2f}: P(done by 18 weeks) = **{_f['p_on_time']:.3g}**; median "
          f"{_f['quantiles'][0.5]:.1f}, 80% sure **{_f['quantiles'][0.8]:.1f}**, 95% {_f['quantiles'][0.95]:.1f} "
          f"weeks.\n\nacross 20 other synthetic histories of the same size, the 80% date ranges from "
          f"{min(_again):.1f} to {max(_again):.1f} weeks.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the calibrated forecast land so far beyond the plan, even though
    the plan used your own likely estimates? why does the forecast become more stable with a
    longer history, and what's the risk of calibrating on a history from a different kind of
    work (coursework instead of research)?
    """)
    return


if __name__ == "__main__":
    app.run()
