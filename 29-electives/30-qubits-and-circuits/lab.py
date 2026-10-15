import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # qubits & circuits · lab

    every experiment here runs **your** simulator from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about ten seconds.
    """)
    return


@app.cell
def _():
    def bell_pair():
        """(|00> + |11>)/sqrt2, made by H on qubit 0 then CNOT(0, 1)."""
        return ex.run_circuit([(ex.gate("H"), [0]), (ex.gate("CNOT"), [0, 1])], 2)

    def measure_along(state, q, angle, rng):
        """measure qubit q along the direction `angle` in the x-z plane (0 = Z, pi/2 = X); returns (+1/-1, post-state)."""
        rotated = ex.apply_gate(state, ex.rot("y", -angle), [q])
        bit, post = ex.measure_qubit(rotated, q, rng)
        return 1 - 2 * bit, ex.apply_gate(post, ex.rot("y", angle), [q])

    return bell_pair, measure_along


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: measuring an entangled pair

    Alice holds qubit 0 of a Bell pair $(|00\rangle + |11\rangle)/\sqrt2$, Bob holds qubit 1.
    Alice measures first, then Bob measures in the $Z$ basis. we repeat it 4,000 times, once
    with Alice measuring in the $Z$ basis and once in the $X$ basis (she applies $H$ before
    measuring), and tabulate Bob's results.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "when Alice measures Z and sees 1, what's the chance Bob sees 1? when she switches to the X basis, does "
        "the fraction of 1s Bob sees *overall* change? can Bob tell which basis Alice used?"
    )
    p1
    return (p1,)


@app.cell
def _(bell_pair, measure_along, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _rows = []
    _shots = 4000
    for _basis, _angle in (("Z", 0.0), ("X", np.pi / 2)):
        _a = np.empty(_shots, int)
        _b = np.empty(_shots, int)
        for _i in range(_shots):
            _sa, _post = measure_along(bell_pair(), 0, _angle, _rng)
            _bb, _ = ex.measure_qubit(_post, 1, _rng)
            _a[_i], _b[_i] = (1 - _sa) // 2, _bb
        _se = lambda p, m: np.sqrt(p * (1 - p) / max(m, 1))  # noqa: E731
        _p, _p0, _p1 = _b.mean(), _b[_a == 0].mean(), _b[_a == 1].mean()
        _rows.append(f"| {_basis} | {_p:.3f} ± {_se(_p, _shots):.3f} | {_p0:.3f} ± {_se(_p0, (_a == 0).sum()):.3f} | "
                     f"{_p1:.3f} ± {_se(_p1, (_a == 1).sum()):.3f} |")
    _counts = ex.sample_counts(bell_pair(), 4000, rng=1)
    mo.md("measuring both qubits of a Bell pair 4,000 times: `" + str(_counts) + "`\n\n"
          "| Alice's basis | P(Bob = 1) | P(Bob = 1 \\| Alice = 0) | P(Bob = 1 \\| Alice = 1) |\n|---|---|---|---|\n"
          + "\n".join(_rows) + "\n\n(± one standard error)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the $Z$ row, Bob's result is perfectly predicted by Alice's. in the $X$
    row it isn't predicted at all. yet Bob's *own* column, P(Bob = 1), is $\frac12$ in both rows.
    write down Bob's reduced density matrix after Alice measures (in either basis, averaged over
    her outcome) and show it's $I/2$ both times. why does that mean entanglement can't be used to
    send a message faster than light, even though the collapse looks instantaneous?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the CHSH game

    Alice measures along angle $a \in \{a_0, a_1\}$ and Bob along $b \in \{b_0, b_1\}$ (directions in
    the $x$–$z$ plane), each getting $\pm1$. the CHSH quantity is

    $$S = E(a_0, b_0) + E(a_0, b_1) + E(a_1, b_0) - E(a_1, b_1),$$

    where $E$ is the average product of their outcomes. any **local hidden-variable** model (each
    qubit carries pre-agreed answers for every possible question) gives $|S| \le 2$; we check that
    by trying all 16 deterministic answer sheets. then we estimate $S$ for the Bell pair with
    $a_0 = 0$, $a_1 = \pi/2$, $b_0 = \pi/4 + \delta$, $b_1 = -\pi/4 + \delta$, from 2,000 shots per
    setting.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the best classical strategy reaches |S| = 2. what will the Bell pair reach at delta = 0: below 2, "
        "exactly 2, or more? how far can you rotate Bob's angles (delta) and still beat 2?"
    )
    p2
    return (p2,)


@app.cell
def _():
    delta = mo.ui.slider(-90, 90, value=0, step=5, label="δ: rotate both of Bob's angles (degrees)")
    delta
    return (delta,)


@app.cell
def _(bell_pair, delta, p2):
    ui.gate(p2)

    def _classical_best():
        best = 0
        for a0, a1, b0, b1 in itertools.product((-1, 1), repeat=4):
            best = max(best, abs(a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1))
        return best

    def _E_sampled(a, b, shots, rng):
        psi = ex.apply_gate(ex.apply_gate(bell_pair(), ex.rot("y", -a), [0]), ex.rot("y", -b), [1])
        counts = ex.sample_counts(psi, shots, rng)
        prod = sum(c * (1 if k in ("00", "11") else -1) for k, c in counts.items()) / shots
        return prod

    def _S(d, shots, rng):
        a0, a1, b0, b1 = 0.0, np.pi / 2, np.pi / 4 + d, -np.pi / 4 + d
        Es = [_E_sampled(a, b, shots, rng) for a, b in ((a0, b0), (a0, b1), (a1, b0), (a1, b1))]
        se = np.sqrt(sum((1 - e ** 2) / shots for e in Es))
        return Es[0] + Es[1] + Es[2] - Es[3], se

    _rng = np.random.default_rng(2)
    _ds = np.radians(np.arange(-90, 91, 10))
    _vals = np.array([_S(d, 2000, _rng) for d in _ds])
    _exact = 2 * np.sqrt(2) * np.cos(_ds)
    _s, _se = _S(np.radians(delta.value), 2000, _rng)
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.errorbar(np.degrees(_ds), _vals[:, 0], yerr=2 * _vals[:, 1], fmt="o", ms=4, label="Bell pair, 2,000 shots/setting (±2 s.e.)")
    _ax.plot(np.degrees(_ds), _exact, lw=1, label=r"exact: $2\sqrt{2}\cos\delta$")
    _ax.axhspan(-2, 2, color="0.85", label="reachable by any local hidden-variable model")
    _ax.axvline(delta.value, color="0.4", ls=":", lw=1)
    _ax.set(xlabel="δ (degrees)", ylabel="S", ylim=(-3.1, 3.1))
    _ax.legend(fontsize=8, loc="lower center")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"best of all 16 classical answer sheets: |S| = **{_classical_best()}** · "
                           f"Bell pair at δ = {delta.value}°: S = **{_s:.3f} ± {_se:.3f}** "
                           f"(exact {2 * np.sqrt(2) * np.cos(np.radians(delta.value)):.3f})")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** show that $E(a, b) = \cos(a - b)$ for this Bell pair, and that the four
    angles above give $S = 2\sqrt2$ (Tsirelson's bound: no quantum state does better). why can't
    a table of pre-agreed answers do better than 2? the Bell pair beats 2 for every $|\delta|$ below
    what angle? what assumption about the world does this rule out, and what does it *not* rule out?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how entangled is a random state?

    random circuits on 8 qubits: each layer applies a random rotation $R_z R_y R_z$ to every qubit,
    then CNOTs between neighbours (alternating even and odd pairs). we measure the entanglement
    entropy of qubits 0–3 (half the system) as the depth grows, over 20 random circuits.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the most entanglement half of 8 qubits can have is 4 bits. after many random layers, where does the "
        "entropy settle: around 1 bit, around 2, close to 4?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _n, _depths, _trials = 8, list(range(0, 31)), 20
    _rng = np.random.default_rng(3)
    _S = np.zeros((_trials, len(_depths)))
    for _t in range(_trials):
        _psi = ex.ket("0" * _n)
        for _d in range(max(_depths) + 1):
            _S[_t, _d] = ex.entanglement_entropy(_psi, range(_n // 2))
            for _q in range(_n):
                _a, _b, _c = _rng.uniform(0, 2 * np.pi, 3)
                _psi = ex.apply_gate(_psi, ex.rot("z", _c) @ ex.rot("y", _b) @ ex.rot("z", _a), [_q])
            for _q in range(_d % 2, _n - 1, 2):
                _psi = ex.apply_gate(_psi, ex.gate("CNOT"), [_q, _q + 1])
    _page = _n / 2 - 1 / (2 * np.log(2))
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _m, _sd = _S.mean(0), _S.std(0, ddof=1)
    _ax.plot(_depths, _m, "o-", label="mean over 20 random circuits")
    _ax.fill_between(_depths, _m - _sd, _m + _sd, alpha=0.25, label="± 1 s.d.")
    _ax.axhline(_n / 2, color="k", lw=1, ls="--", label="maximum: 4 bits")
    _ax.axhline(_page, color="C3", lw=1, ls=":", label=f"Page's average for a random state: {_page:.2f}")
    _ax.set(xlabel="layers", ylabel="entropy of qubits 0–3 (bits)", ylim=(0, 4.3))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"after 30 layers: **{_m[-1]:.2f} ± {_sd[-1]:.2f} bits** (mean ± s.d. over circuits)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the entropy climb in steps, only every other layer (which layers
    put a CNOT across the cut between qubits 3 and 4)? why can one CNOT add at most one bit? why
    does it level off short of 4 bits, at Page's value? the
    random states here are nearly maximally entangled: what would that mean for the expectation
    value of a local observable like $Z_0$, sampled over random circuits? (topic 31 is about
    exactly this.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: the price of a qubit

    the time to apply one Hadamard to qubit 0, and the memory of the state, as $n$ grows
    (best of 5 runs per $n$).
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "your laptop has 16 GB of memory. what's the largest number of qubits whose state vector fits? how "
        "about a 50-qubit state?"
    )
    p4
    return (p4,)


@app.cell
def _(p4):
    ui.gate(p4)
    _ns = list(range(4, 21, 2))
    _times = []
    for _n in _ns:
        _psi = ex.ket("0" * _n)
        _best = np.inf
        for _ in range(5):
            _t0 = time.perf_counter()
            ex.apply_gate(_psi, ex.gate("H"), [0])
            _best = min(_best, time.perf_counter() - _t0)
        _times.append(_best)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.semilogy(_ns, np.array(_times) * 1e3, "o-", label="time for one gate (ms)")
    _ax.semilogy(_ns, 16 * 2.0 ** np.array(_ns) / 1e6, "s--", label="state vector size (MB)")
    _ax.set(xlabel="qubits n")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {n} | {16 * 2.0 ** n / 2 ** 30:,.3g} GiB |" for n in (20, 30, 40, 50))
    mo.vstack([_fig, mo.md("| qubits | complex128 state vector |\n|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for small $n$ the time is flat; for large $n$ every added qubit doubles it.
    why the two regimes? what does the table say about simulating a 50-qubit device exactly, and
    which kinds of circuits can still be simulated at that size (think about how much
    entanglement experiment 3 built up)?
    """)
    return


if __name__ == "__main__":
    app.run()
