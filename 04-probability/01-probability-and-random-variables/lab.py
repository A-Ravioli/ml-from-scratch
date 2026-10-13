import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # probability & random variables · lab

    every experiment here runs **your** code from `exercise.py`, and checks it against a
    simulation where that helps. for each one: write down a prediction, lock it in, run it,
    then explain what happened. compare your explanations with `solutions/lab_notes.md`
    afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the base-rate fallacy

    a test is 90% sensitive, and 9% of healthy people test positive anyway. the slider sets
    how common the disease is. we compute $P(\text{sick} \mid +)$ exactly with your `bayes`,
    and also simulate a population of 200,000 people and count.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at a prevalence of 1%, what fraction of positive tests are real? at 0.1%? "
        "how common must the disease be before a positive test means 'more likely sick than not'?"
    )
    p1
    return (p1,)


@app.cell
def _():
    prevalence = mo.ui.slider(steps=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                              value=0.01, label="prevalence P(sick)", show_value=True)
    prevalence
    return (prevalence,)


@app.function
def screening_posterior(prev, sens=0.9, fpr=0.09):
    """P(sick | +) via your bayes."""
    prior = ex.Discrete({"sick": prev, "healthy": 1 - prev})

    def lik(h):
        p = sens if h == "sick" else fpr
        return ex.Discrete({"+": p, "-": 1 - p})
    return ex.bayes(prior, lik, "+").pmf("sick")


@app.cell
def _(p1, prevalence):
    ui.gate(p1)
    _prev = prevalence.value
    _exact = screening_posterior(_prev)
    _rng = np.random.default_rng(0)
    _n = 200_000
    _sick = _rng.random(_n) < _prev
    _pos = np.where(_sick, _rng.random(_n) < 0.9, _rng.random(_n) < 0.09)
    _sim = _sick[_pos].mean()
    _grid = np.logspace(-4, np.log10(0.6), 80)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.semilogx(_grid, [screening_posterior(_g) for _g in _grid], label="P(sick | +), your bayes")
    _ax.axhline(0.9, color="gray", ls=":", label="P(+ | sick) = 0.9 (the tempting answer)")
    _ax.axhline(0.5, color="gray", lw=0.6)
    _ax.plot([_prev], [_exact], "o", color="C3")
    _ax.set(xlabel="prevalence P(sick)", ylabel="probability", ylim=(0, 1))
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"prevalence **{_prev:g}**: your `bayes` says P(sick | +) = **{_exact:.4f}**. in a simulated "
        f"population of {_n:,}, {_pos.sum():,} tested positive and {_sick[_pos].sum():,} of them were "
        f"sick: **{_sim:.4f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** use natural frequencies (out of 1000 people) to explain the number at 1%
    prevalence. then find the prevalence where the curve crosses 0.5, and explain it with the
    odds form of Bayes' rule (the likelihood ratio here is 0.9 / 0.09 = 10).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Monty Hall, two hosts

    you pick door 0. the host opens door 2, which shows a goat, and offers you door 1.

    - the **informed host** knows where the car is and always opens a goat door that isn't yours.
    - the **ignorant host** opens door 1 or 2 at random. this time it happened to show a goat.

    we compute your chance of winning by switching both exactly (your `bayes`) and by playing
    50,000 games of each, keeping only the games where the host opened door 2 and showed a goat.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "P(win by switching) with the informed host? with the ignorant host? the doors and the goat "
        "you see are identical in both cases."
    )
    p2
    return (p2,)


@app.function
def monty_exact():
    prior = ex.Discrete({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})

    def informed(car):  # observation: (door opened, what it showed)
        if car == 0:
            return ex.Discrete({(1, "goat"): 0.5, (2, "goat"): 0.5})
        return ex.Discrete({(3 - car, "goat"): 1.0})

    def ignorant(car):
        return ex.Discrete({(d, "car" if d == car else "goat"): 0.5 for d in (1, 2)})
    obs = (2, "goat")
    return ex.bayes(prior, informed, obs).pmf(1), ex.bayes(prior, ignorant, obs).pmf(1)


@app.function
def monty_simulate(n_games, informed, rng):
    car = rng.integers(0, 3, size=n_games)
    if informed:
        # car behind 0: open 1 or 2 at random; otherwise open the other goat door
        opened = np.where(car == 0, 1 + rng.integers(0, 2, size=n_games), 3 - car)
    else:
        opened = 1 + rng.integers(0, 2, size=n_games)
    keep = (opened == 2) & (car != 2)  # host opened door 2 and it showed a goat
    return float(np.mean(car[keep] == 1)), int(keep.sum())


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _ex_inf, _ex_ign = monty_exact()
    _sim_inf, _n_inf = monty_simulate(50_000, True, _rng)
    _sim_ign, _n_ign = monty_simulate(50_000, False, _rng)
    mo.md(f"""
    | host | P(win by switching), your `bayes` | simulated | games kept |
    |---|---|---|---|
    | informed | {_ex_inf:.4f} | {_sim_inf:.4f} | {_n_inf:,} of 50,000 |
    | ignorant | {_ex_ign:.4f} | {_sim_ign:.4f} | {_n_ign:,} of 50,000 |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** you saw the same thing in both cases: door 2 opened, a goat. why do the
    answers differ? write out the three likelihoods $P(\text{observation} \mid \text{car})$
    for each host. then explain why the ignorant host's games are thrown away more often.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: adding up dice

    your `add` convolves two distributions exactly. here we add up $k$ fair dice and compare
    the exact distribution of the total with a bell curve that has the same mean and variance.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how many dice do you need before the total's distribution looks like a bell curve? "
        "what is P(total of 10 dice = 35), the single most likely value?"
    )
    p3
    return (p3,)


@app.cell
def _():
    n_dice = mo.ui.slider(1, 12, value=3, label="number of dice k", show_value=True)
    n_dice
    return (n_dice,)


@app.cell
def _(n_dice, p3):
    ui.gate(p3)
    _die = ex.Discrete({_f: 1 / 6 for _f in range(1, 7)})
    _total = _die
    for _ in range(n_dice.value - 1):
        _total = ex.add(_total, _die)
    _xs = np.array(sorted(_total.probs))
    _ps = np.array([_total.pmf(_x) for _x in _xs])
    _mu = float(np.sum(_xs * _ps))
    _var = float(np.sum((_xs - _mu) ** 2 * _ps))
    _gauss = np.exp(-(_xs - _mu) ** 2 / (2 * _var)) / np.sqrt(2 * np.pi * _var)
    _ten = _die
    for _ in range(9):
        _ten = ex.add(_ten, _die)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.bar(_xs, _ps, width=0.8, alpha=0.6, label=f"exact, {n_dice.value} dice (your add)")
    _ax.plot(_xs, _gauss, "k.-", lw=1, label="normal curve, same mean and variance")
    _ax.set(xlabel="total", ylabel="probability")
    _ax.legend(fontsize=8, loc="upper left")
    mo.vstack([_fig, mo.md(
        f"largest gap between the exact pmf and the curve: **{np.abs(_ps - _gauss).max():.4f}** "
        f"(the pmf's peak is {_ps.max():.4f}). P(total of 10 dice = 35) = **{_ten.pmf(35):.4f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the mean of $k$ dice is $3.5k$ and the variance is $35k/12$ (topic 03 shows
    why). why does the distribution become bell-shaped so quickly, even though one die is flat?
    and why is the *single most likely* total of 10 dice so unlikely, when it is the most likely?
    """)
    return


if __name__ == "__main__":
    app.run()
