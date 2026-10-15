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
    # convex sets & functions · lab

    every experiment here runs **your** checker from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: which ML losses are convex?

    nine losses, each as a function of its **parameters** on the same 30-point synthetic
    dataset. we sample 40 random parameter vectors (standard deviation 2) and run your
    `check_convexity` (Jensen on all 780 pairs, $t \in \{0.25, 0.5, 0.75\}$) and your
    `min_hessian_eigenvalue` at the same points.

    | loss | model | parameters |
    |---|---|---|
    | squared error | $w^\top x$ | $w \in \mathbb{R}^2$ |
    | absolute error | $w^\top x$ | $w$ |
    | logistic (cross-entropy) | $\sigma(w^\top x)$ | $w$ |
    | hinge | $w^\top x$ | $w$ |
    | softmax cross-entropy | $\operatorname{softmax}(W^\top x)$, 3 classes | $W \in \mathbb{R}^{2\times3}$ |
    | squared error **on a sigmoid** | $\sigma(w^\top x)$ | $w$ |
    | squared error, deep linear | $a\,b\,x_1$ | $(a, b)$ |
    | squared error, one tanh unit | $v\tanh(w^\top x)$ | $(w, v)$ |
    | matrix factorization | $M \approx uv^\top$ | $u, v \in \mathbb{R}^2$ |
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which of the nine losses will your checker find to be NOT convex? in particular: squared error "
        "on a sigmoid output, and the 'deep linear' model a·b·x, which is linear in its input."
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(30, 2))
    _w_true = np.array([1.5, -1.0])
    _y01 = (_X @ _w_true + 0.5 * _rng.normal(size=30) > 0).astype(float)
    _ypm = 2 * _y01 - 1
    _yreg = _X @ _w_true + 0.3 * _rng.normal(size=30)
    _ycls = np.argmax(_X @ np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]) + 0.3 * _rng.normal(size=(30, 3)), axis=1)
    _M = np.outer([1.0, 2.0], [0.5, -1.0]) + 0.1 * _rng.normal(size=(2, 2))

    def _sig(z):
        return 0.5 * (1 + np.tanh(0.5 * z))

    def _softmax_ce(th):
        Z = _X @ th.reshape(2, 3)
        Z = Z - Z.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(Z).sum(axis=1)) - Z[np.arange(30), _ycls]))

    _losses = {
        "squared error (linear)": (lambda w: float(np.mean((_X @ w - _yreg) ** 2)), 2, True),
        "absolute error (linear)": (lambda w: float(np.mean(np.abs(_X @ w - _yreg))), 2, False),
        "logistic cross-entropy": (lambda w: float(np.mean(np.logaddexp(0, _X @ w) - _y01 * (_X @ w))), 2, True),
        "hinge": (lambda w: float(np.mean(np.maximum(0, 1 - _ypm * (_X @ w)))), 2, False),
        "softmax cross-entropy": (_softmax_ce, 6, True),
        "squared error on a sigmoid": (lambda w: float(np.mean((_sig(_X @ w) - _y01) ** 2)), 2, True),
        "deep linear (a·b·x)": (lambda th: float(np.mean((th[0] * th[1] * _X[:, 0] - _yreg) ** 2)), 2, True),
        "one tanh unit": (lambda th: float(np.mean((th[2] * np.tanh(_X @ th[:2]) - _yreg) ** 2)), 3, True),
        "matrix factorization": (lambda th: float(np.sum((_M - np.outer(th[:2], th[2:])) ** 2)), 4, True),
    }
    _rows = []
    _gaps = {}
    for _name, (_f, _d, _smooth) in _losses.items():
        _pts = 2.0 * np.random.default_rng(1).normal(size=(40, _d))
        _ok, _wit = ex.check_convexity(_f, _pts)
        _gap, _ = ex.min_jensen_gap(_f, _pts)
        _lam, _ = ex.min_hessian_eigenvalue(_f, _pts)
        _gaps[_name] = _gap
        _hess = f"{_lam:.3g}" if _smooth else f"{_lam:.3g} (kinks: unreliable)"
        _verdict = "no violation found" if _ok else "**NOT convex**"
        _rows.append(f"| {_name} | {_verdict} | {_gap:.3g} | {_hess} |")
    _fig, _ax = plt.subplots(figsize=(8.5, 3.6))
    _names = list(_gaps)
    _vals = np.array([_gaps[n] for n in _names])
    _ax.barh(_names, _vals, color=["C3" if v < -1e-9 else "C0" for v in _vals])
    _ax.axvline(0, color="k", lw=0.8)
    _ax.set_xscale("symlog", linthresh=1e-6)
    _ax.set_xticks([-100, -1, -1e-2, -1e-4, 0, 1e-4, 1e-2])
    _ax.set(xlabel="most negative Jensen gap (symlog: |gap| < 1e-6 is drawn linearly)",
            title="red: the graph rises above some chord (not convex)")
    _ax.invert_yaxis()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| loss | your verdict | min Jensen gap | min Hessian eigenvalue |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** sort the nine losses using the rules from idea 4 (nonnegative sums,
    affine precomposition, pointwise max, composition). which rule fails for the sigmoid
    with squared error, and why does the same sigmoid with *cross-entropy* stay convex? for
    the deep linear model and matrix factorization, find two parameter vectors with the same
    loss whose midpoint is worse (hint: flip signs). why are the Hessian values for the
    absolute-error and hinge losses not to be trusted?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a needle in a haystack

    $f(x) = x^2 + h\,e^{-(x - 0.37)^2/(2s^2)}$ with a tiny bump ($h = 0.01$, $s = 0.02$). the
    parabola has curvature 2, but at the top of the bump the curvature is $2 - h/s^2 = -23$:
    the function is nonconvex on an interval of width about $0.04$ out of the sampling range
    $[-2, 2]$. for each number of random sample points $n$ we run your `check_convexity`
    (Jensen) and your `min_hessian_eigenvalue` (Hessian test) on 50 random samples, and
    count how often each one catches the bump.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with n = 50 uniform sample points on [-2, 2], how often does the Jensen test catch the bump? "
        "and the Hessian test? which one needs fewer points, and why?"
    )
    p2
    return (p2,)


@app.cell
def _():
    width = mo.ui.slider(0.005, 0.08, value=0.02, step=0.005, label="bump width s (height h = 25 s²: f'' = -23 at the top)")
    width
    return (width,)


@app.cell
def _(p2, width):
    ui.gate(p2)
    _s = width.value
    _h = 25 * _s**2

    def _f(v):
        return float(v[0] * v[0] + _h * np.exp(-((v[0] - 0.37) ** 2) / (2 * _s * _s)))

    _ns = [10, 25, 50, 100, 200]
    _jen, _hes = [], []
    for _n in _ns:
        _cj = _ch = 0
        for _seed in range(50):
            _pts = np.random.default_rng(100 + _seed).uniform(-2, 2, size=(_n, 1))
            _ok, _ = ex.check_convexity(_f, _pts)
            _cj += not _ok
            _lam, _ = ex.min_hessian_eigenvalue(_f, _pts, h=1e-3 * _s)
            _ch += _lam < -1e-6
        _jen.append(_cj / 50)
        _hes.append(_ch / 50)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    _xs = np.linspace(0.37 - 6 * _s, 0.37 + 6 * _s, 400)
    _a1.plot(_xs, [_f(np.array([x])) for x in _xs])
    _a1.set(title="f near the bump", xlabel="x")
    _a2.plot(_ns, _jen, "o-", label="Jensen (check_convexity)")
    _a2.plot(_ns, _hes, "s-", label="Hessian (min eigenvalue)")
    _a2.set(xscale="log", xlabel="sample points n", ylabel="fraction of 50 runs that catch it",
            ylim=(-0.05, 1.05), title=f"detection rate, bump width {_s}")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| n | Jensen catches it | Hessian catches it |\n|---|---|---|\n"
        + "\n".join(f"| {n} | {j:.0%} | {h:.0%} |" for n, j, h in zip(_ns, _jen, _hes)))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the nonconvex region is about $4s$ wide. what's the chance that a single
    uniform point lands in it, and what does the Hessian test need? what does a Jensen
    violation need: one point in the region, or a chord whose *ends* are both close to it?
    why does that make the Jensen test so much slower to fire? move the slider: how does the
    width change things? what does this tell you about the sentence "my checker found no
    violation, so the loss is convex"?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: convex sublevel sets, nonconvex function

    two functions on $\mathbb{R}^2$:

    - the **Welsch** robust loss $f(x) = 1 - e^{-\|x\|^2}$, which is used to downweight outliers;
    - the **$\ell_{1/2}$ penalty** $g(x) = \sqrt{|x_1|} + \sqrt{|x_2|}$, which is used to encourage sparsity.

    for each, we test the sublevel set $\{x : f(x) \le c\}$ with your
    `set_convexity_violation` (on 300 points from a box) and the function itself with your
    `check_convexity`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a function whose sublevel sets {f <= c} are convex for every c: is it convex? "
        "predict the verdicts for the Welsch loss and the l_1/2 penalty."
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _welsch = lambda v: float(1 - np.exp(-(v @ v)))
    _lhalf = lambda v: float(np.sqrt(abs(v[0])) + np.sqrt(abs(v[1])))
    _pts = np.random.default_rng(3).uniform(-2.0, 2.0, size=(300, 2))
    _rows = []
    for _name, _f, _c in (("Welsch 1 − exp(−‖x‖²)", _welsch, 0.8), ("ℓ½: √|x₁| + √|x₂|", _lhalf, 1.5)):
        _set = ex.set_convexity_violation(lambda x, f=_f, c=_c: f(x) <= c, _pts)
        _fn, _wit = ex.check_convexity(_f, _pts[:60])
        _set_txt = "no violation found" if _set is None else f"**NOT convex** (pair {_set[:2]}, t = {_set[2]})"
        _fn_txt = "no violation found" if _fn else "**NOT convex**"
        _rows.append(f"| {_name} | {_c} | {_set_txt} | {_fn_txt} |")
    _g = np.linspace(-2, 2, 201)
    _GX, _GY = np.meshgrid(_g, _g)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4))
    for _ax, (_name, _f) in zip(_axes, (("Welsch", _welsch), ("ℓ½ penalty", _lhalf))):
        _Z = np.array([[_f(np.array([a, b])) for a in _g] for b in _g])
        _cs = _ax.contour(_GX, _GY, _Z, levels=8)
        _ax.clabel(_cs, fontsize=7)
        _ax.set_aspect("equal")
        _ax.set_title(f"sublevel sets of {_name}")
    _fig.tight_layout()
    _r = np.linspace(0, 2, 200)
    mo.vstack([_fig, mo.md(
        "| function | level c | sublevel set {f ≤ c} | the function |\n|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nalong a ray, the Welsch loss is $1 - e^{{-r^2}}$, with second derivative $(2 - 4r^2)e^{{-r^2}}$: "
        f"negative for $r > {1 / np.sqrt(2):.3f}$.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** every sublevel set of the Welsch loss is a disk. why doesn't that make
    it convex? (look at it along a ray from the origin.) such functions are called
    **quasiconvex**: what does Jensen's inequality get replaced by? what do the contours of
    the $\ell_{1/2}$ penalty look like, and how does their shape explain why it produces
    sparser solutions than $\ell_1$? which direction of the implication "convex function ⇒
    convex sublevel sets" holds, and why?
    """)
    return


if __name__ == "__main__":
    app.run()
