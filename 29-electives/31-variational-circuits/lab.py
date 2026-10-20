import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.datasets import make_circles
    from mlfs.kernels import SVC  # yours, from ch 09

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1), and through it
    # your topic 30 simulator as ex.qsim
    ex = notebooks.load_sibling("exercise")
    qsim = ex.qsim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # variational circuits · lab

    every experiment here runs **your** code from `exercise.py` on your topic 30 simulator. for
    each one: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about a
    minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    ex.qsim_load.callout()  # shown only if your topic 30 simulator isn't finished
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: gradients from shots

    on hardware you never get $\langle Z_0\rangle$ exactly, only an estimate from a finite number of
    shots. we take a fixed 2-qubit, 2-layer model and one parameter, and estimate
    $\partial\langle Z_0\rangle/\partial\theta$ from 1,000 shots per circuit, 300 times, with
    central finite differences at several step sizes $h$ and with the parameter-shift rule. the
    error is measured against the exact gradient.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with 1,000 shots per circuit, which finite-difference step h gives the most accurate gradient: 1, 0.1, "
        "0.01 or 0.001? how does the parameter-shift rule compare with the best h?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _theta = _rng.uniform(0, 2 * np.pi, size=(2, 2, 2))
    _x = np.array([0.7, 2.1])
    _idx = (0, 1, 0)
    _shots, _reps = 1000, 300

    def _f_exact(t):
        return ex.expectation(t, _x)

    def _f_shots(t, rng):
        psi = qsim.run_circuit(ex.model_ops(t, _x), 2)
        return ex.estimate_pauli(psi, "ZI", _shots, rng)

    def _shifted(h):
        tp, tm = _theta.copy(), _theta.copy()
        tp[_idx] += h
        tm[_idx] -= h
        return tp, tm

    _exact = ex.parameter_shift(_f_exact, _theta)[_idx]
    _hs = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    _rmse_fd, _rmse_fd_exact = [], []
    for _h in _hs:
        _tp, _tm = _shifted(_h)
        _est = [(_f_shots(_tp, _rng) - _f_shots(_tm, _rng)) / (2 * _h) for _ in range(_reps)]
        _rmse_fd.append(np.sqrt(np.mean((np.array(_est) - _exact) ** 2)))
        _rmse_fd_exact.append(abs((_f_exact(_tp) - _f_exact(_tm)) / (2 * _h) - _exact))
    _tp, _tm = _shifted(np.pi / 2)
    _ps = np.array([(_f_shots(_tp, _rng) - _f_shots(_tm, _rng)) / 2 for _ in range(_reps)])
    _rmse_ps = np.sqrt(np.mean((_ps - _exact) ** 2))
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.loglog(_hs, _rmse_fd, "o-", label=f"finite differences, {_shots} shots")
    _ax.loglog(_hs, np.maximum(_rmse_fd_exact, 1e-17), "s--", label="finite differences, exact expectations")
    _ax.axhline(_rmse_ps, color="C3", label=f"parameter shift, {_shots} shots")
    _ax.set(xlabel="finite-difference step h", ylabel="RMS error of the gradient", ylim=(1e-9, 50))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _best = int(np.argmin(_rmse_fd))
    mo.vstack([_fig, mo.md(
        f"exact gradient {_exact:.4f} · parameter shift: RMS error **{_rmse_ps:.4f}** · best finite difference: "
        f"h = {_hs[_best]}, RMS error **{_rmse_fd[_best]:.4f}** · at h = 0.001: **{_rmse_fd[-1]:.1f}** "
        f"({_reps} repetitions each)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with exact expectations, smaller $h$ is better (until floating point
    bites). with shots, the error *grows* as $h$ shrinks: derive its standard deviation in
    terms of $h$ and the number of shots. why is the finite-difference error at $h = 1$ not
    small either? what does the parameter-shift rule do that no choice of $h$ can?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a variational classifier

    two concentric circles (`make_circles`), rescaled to $[0, \pi]^2$: 40 training and 160 test
    points. we train a 2-qubit, 3-layer classifier with your `train_classifier` (40 Adam steps,
    lr 0.1), with the data encoded once and with re-uploading before every layer, over 3
    initializations each. the baseline is your ch 09 `SVC` with an RBF kernel ($C = 10$,
    $\gamma = 1$) on the same points.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what test accuracy will the circuit reach with the data encoded once, and with re-uploading? will "
        "either match the RBF SVM?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _X, _y = make_circles(200, noise=0.1, rng=0)
    _perm = np.random.default_rng(0).permutation(len(_X))
    _X, _y = _X[_perm], 2 * _y[_perm] - 1
    _lo, _hi = _X.min(0), _X.max(0)
    _X = (_X - _lo) / (_hi - _lo) * np.pi
    _Xtr, _ytr, _Xte, _yte = _X[:40], _y[:40], _X[40:], _y[40:]
    _svc_acc = np.mean(SVC(C=10, gamma=1.0, rng=0).fit(_Xtr, _ytr).predict(_Xte) == _yte)
    _results = {}
    for _re in (False, True):
        _runs = []
        for _seed in range(3):
            _theta, _losses = ex.train_classifier(_Xtr, _ytr, layers=3, steps=40, lr=0.1, reupload=_re, rng=_seed)
            _runs.append((np.mean(ex.classify(_theta, _Xte, reupload=_re) == _yte), _theta, _losses))
        _results[_re] = _runs
    _g = np.linspace(0, np.pi, 30)
    _G = np.array([[a, b] for b in _g for a in _g])
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for _ax, _re in zip(_axes[:2], (False, True)):
        _acc, _theta, _ = max(_results[_re], key=lambda r: r[0])
        _Z = np.array([ex.expectation(_theta, p, reupload=_re) for p in _G]).reshape(30, 30)
        _ax.contourf(_g, _g, _Z, levels=np.linspace(-1, 1, 11), cmap="RdBu", alpha=0.6)
        _ax.contour(_g, _g, _Z, levels=[0], colors="k", linewidths=1)
        _ax.scatter(*_Xtr.T, c=_ytr, cmap="RdBu", edgecolors="k", s=25)
        _ax.set(title=f"{'re-uploading' if _re else 'encoded once'} (best seed, test acc {_acc:.2f})",
                aspect="equal", xticks=[], yticks=[])
    for _re in (False, True):
        for _i, (_, _, _l) in enumerate(_results[_re]):
            _axes[2].plot(_l, color="C1" if _re else "C0", label=("re-uploading" if _re else "encoded once") if _i == 0 else None)
    _axes[2].set(xlabel="Adam step", ylabel="training MSE")
    _axes[2].legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _re in (False, True):
        _a = np.array([r[0] for r in _results[_re]])
        _mse = np.array([r[2][-1] for r in _results[_re]])
        _rows.append(f"| circuit, {'re-uploading' if _re else 'encoded once'} | {_mse.mean():.3f} ± "
                     f"{_mse.std(ddof=1) / np.sqrt(len(_mse)):.3f} | {_a.mean():.3f} ± "
                     f"{_a.std(ddof=1) / np.sqrt(len(_a)):.3f} | {', '.join(f'{v:.2f}' for v in _a)} |")
    _rows.append(f"| RBF SVM (ch 09) | | {_svc_acc:.3f} | |")
    mo.vstack([_fig, mo.md("| model | final training MSE | test accuracy (mean ± s.e. over 3 inits) | per init |\n"
                           "|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with the data encoded once, each feature enters only through $\cos x_q$ and
    $\sin x_q$: write $f(x)$ as a sum of products $r_a(x_1)r_b(x_2)$ with $r \in \{1, \cos, \sin\}$.
    why can't the circuit make the bowl-shaped function the circles need, even though
    $\sin x_1 + \sin x_2$ would do it (hint: what must $W^\dagger Z_0W$ square to)? what does
    re-uploading change? and what does the SVM's accuracy say about whether the circuit is a
    good model for this problem?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: barren plateaus

    for random parameters $\theta \sim U(0, 2\pi)$, the distribution of one gradient component
    $\partial\langle O\rangle/\partial\theta_{0,0,0}$ (your `gradient_samples`, 200 random circuits per
    point). three settings, for 2 to 10 qubits:

    - **global, shallow**: $O = Z^{\otimes n}$, 2 layers;
    - **local, shallow**: $O = Z_0Z_1$, 2 layers;
    - **local, deep**: $O = Z_0Z_1$, $2n$ layers.

    the error bars are bootstrap 95% intervals for the variance.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the gradient's mean is 0 by symmetry. how does its variance change from 2 to 10 qubits in each of the "
        "three settings: constant, slowly falling, or exponentially falling?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _ns = [2, 4, 6, 8, 10]
    _settings = {
        "global, shallow (2 layers)": (lambda n: "Z" * n, lambda n: 2),
        "local Z0Z1, shallow (2 layers)": (lambda n: "ZZ" + "I" * (n - 2), lambda n: 2),
        "local Z0Z1, deep (2n layers)": (lambda n: "ZZ" + "I" * (n - 2), lambda n: 2 * n),
    }
    _boot = np.random.default_rng(1)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for _k, (_label, (_obs, _layers)) in enumerate(_settings.items()):
        _v, _lo, _hi = [], [], []
        for _n in _ns:
            _g = ex.gradient_samples(_n, _layers(_n), 200, observable=_obs(_n), rng=100 * _k + _n)
            _bs = [np.var(_boot.choice(_g, len(_g))) for _ in range(500)]
            _v.append(np.var(_g))
            _lo.append(np.percentile(_bs, 2.5))
            _hi.append(np.percentile(_bs, 97.5))
        _v = np.array(_v)
        _ax.errorbar(_ns, _v, yerr=[_v - np.array(_lo), np.array(_hi) - _v], fmt="o-", capsize=3, label=_label)
        _rows.append(f"| {_label} | " + " | ".join(f"{x:.1e}" for x in _v) + " |")
    _ns_a = np.array(_ns)
    _ax.semilogy(_ns_a, 0.4 * 2.0 ** -_ns_a, "k:", label=r"$\propto 2^{-n}$")
    _ax.set(xlabel="qubits n", ylabel="Var[∂⟨O⟩/∂θ]", yscale="log")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| setting | " + " | ".join(f"n = {n}" for n in _ns) + " |\n|---|"
                           + "---|" * len(_ns) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in which settings does the variance fall like $2^{-n}$? the mean is 0, so
    by Chebyshev, how many shots would you need at $n = 50$ just to tell the sign of a typical
    gradient? why does the local observable on a shallow circuit escape, and why does making
    the same circuit deep bring the plateau back (think about how far a gate's influence
    spreads per layer, and topic 30's experiment 3)? what does the escape cost you: is a
    shallow, local model still able to represent anything a classical computer can't?
    """)
    return


if __name__ == "__main__":
    app.run()
