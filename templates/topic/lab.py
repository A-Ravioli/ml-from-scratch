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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # __TITLE__ · lab

    <!-- TEMPLATE: 2–4 experiments that use the learner's own code. at least one result
    should surprise most learners. -->

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how fast does the sample mean settle down?
    """)
    return


@app.cell
def _():
    p1 = ui.predict("if you 4× the sample size, how much smaller does the error of the mean get?")
    p1
    return (p1,)


@app.cell
def _():
    max_n = mo.ui.slider(100, 100_000, value=10_000, step=100, label="largest sample size")
    max_n
    return (max_n,)


@app.cell
def _(max_n, p1):
    ui.gate(p1)
    # names starting with _ stay local to this cell, so later experiments can reuse them
    _rng = np.random.default_rng(0)
    _ns = np.unique(np.geomspace(10, max_n.value, 30).astype(int))
    _errors = [np.mean([abs(ex.mean(_rng.normal(size=n))) for _ in range(50)]) for n in _ns]
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.loglog(_ns, _errors, "o-", label="mean |error| over 50 trials")
    _ax.loglog(_ns, _errors[0] * np.sqrt(_ns[0] / _ns), "--", label=r"$\propto 1/\sqrt{n}$")
    _ax.set(xlabel="sample size n", ylabel="error")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what slope do you see on the log-log plot, and why?
    """)
    return


if __name__ == "__main__":
    app.run()
