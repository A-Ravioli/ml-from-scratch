import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def best_time(fn, *args, budget=0.2):
    """Smallest wall-clock time of fn(*args) over repeated runs within about `budget` seconds."""
    best, spent, runs = np.inf, 0.0, 0
    while spent < budget or runs < 3:
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        best, spent, runs = min(best, dt), spent + dt, runs + 1
        if dt > budget:
            break
    return best


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # python for numerical work · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    the question for the whole lab: your `dot` and your `Vector` are correct. **are they
    fast?** timings depend on your machine, so focus on ratios and trends, not exact numbers.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: your dot product against NumPy's

    we time three ways of computing the same dot product of two length-$n$ vectors, for $n$
    from 3 to a million: your `dot` on two Python lists, your `Vector`'s `@`, and NumPy's
    `np.dot` on two NumPy arrays (topic 02 explains what those are; for now, a black box).
    the inputs are built before the clock starts, so only the dot product is timed.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for n = 1,000,000, how many times faster is np.dot than your dot? and for a "
        "3-element vector, which of the three is fastest?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n_max = mo.ui.dropdown(options={"10,000": 10_000, "100,000": 100_000, "1,000,000": 1_000_000},
                           value="1,000,000", label="largest n")
    n_max
    return (n_max,)


@app.cell
def _(n_max, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _ns = [n for n in (3, 10, 100, 1_000, 10_000, 100_000, 1_000_000) if n <= n_max.value]
    _t = {"your dot (lists)": [], "your Vector @": [], "np.dot (arrays)": []}
    for _n in _ns:
        _a, _b = _rng.normal(size=_n), _rng.normal(size=_n)
        _la, _lb = _a.tolist(), _b.tolist()
        _va, _vb = ex.Vector(_la), ex.Vector(_lb)
        _budget = 0.05 if _n < 10_000 else 0.2
        _t["your dot (lists)"].append(best_time(ex.dot, _la, _lb, budget=_budget))
        _t["your Vector @"].append(best_time(lambda u, v: u @ v, _va, _vb, budget=_budget))
        _t["np.dot (arrays)"].append(best_time(np.dot, _a, _b, budget=_budget))
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for _name, _times in _t.items():
        _ax1.loglog(_ns, np.array(_times) * 1e6, "o-", label=_name)
        _ax2.semilogx(_ns, np.array(_times) / np.array(_ns) * 1e9, "o-", label=_name)
    _ax1.set(xlabel="n", ylabel="time (µs)", title="time for one dot product")
    _ax2.set(xlabel="n", ylabel="ns per element", title="cost per element", yscale="log")
    _ax1.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {n:,} | {a * 1e6:,.2f} | {v * 1e6:,.2f} | {d * 1e6:,.2f} | {a / d:,.1f}× |"
        for n, a, v, d in zip(_ns, *_t.values()))
    mo.vstack([_fig, mo.md(
        "| n | your dot (µs) | Vector @ (µs) | np.dot (µs) | your dot ÷ np.dot |\n"
        "|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** look at the right-hand plot. why is the cost per element of your `dot`
    roughly constant, while NumPy's falls steeply (and creeps up again at a million)? what is NumPy paying for
    at $n = 3$ that your function isn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what is a Python number, really?

    a NumPy array of a million floats and a Python list of the same million floats hold the
    same values. we measure how much memory each one uses, using `sys.getsizeof` (the size of
    one object in bytes) and `ndarray.nbytes`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "a float64 value needs 8 bytes. how many bytes per number does a Python list of a "
        "million floats use, counting the floats themselves? and a NumPy array?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _n = 1_000_000
    _arr = np.random.default_rng(1).normal(size=_n)
    _lst = _arr.tolist()
    _list_object = sys.getsizeof(_lst)                 # the list itself: one pointer per slot
    _floats = sum(sys.getsizeof(x) for x in _lst)      # every float is a separate object
    _ids = np.array([id(x) for x in _lst[:1000]])
    _gaps = np.diff(_ids)
    mo.md(f"""
    | | bytes | bytes per number |
    |---|---|---|
    | one Python float, `sys.getsizeof(1.0)` | {sys.getsizeof(1.0)} | |
    | the list object (its array of pointers) | {_list_object:,} | {_list_object / _n:.1f} |
    | the {_n:,} float objects it points to | {_floats:,} | {_floats / _n:.1f} |
    | **list total** | **{_list_object + _floats:,}** | **{(_list_object + _floats) / _n:.1f}** |
    | **NumPy array data**, `arr.nbytes` | **{_arr.nbytes:,}** | **{_arr.nbytes / _n:.1f}** |

    where do the first 1,000 floats of the list live? the gap between consecutive float
    objects' memory addresses (`id(x)` in CPython) is {int(np.median(_gaps))} bytes in the
    median case, but ranges from {_gaps.min():,} to {_gaps.max():,}. a NumPy array's numbers
    sit exactly 8 bytes apart, one after another.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a Python float is 24 bytes, but its value is only 8 of them. what do you
    think the other 16 bytes store? (hint: every Python object must know its own type, and
    when it can be deleted.) what does a loop like your `dot` have to do with those extra
    bytes on every single iteration?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: just use NumPy arrays, then?

    if NumPy stores numbers compactly, maybe your Python loop gets faster if you hand it NumPy
    arrays instead of lists. we time your `dot` (a Python `for` loop) on two lists, and then
    on the same data as two NumPy arrays, along with the built-in `sum` on each.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "your dot on two NumPy arrays of 100,000 floats, compared with your dot on two lists: "
        "faster, about the same, or slower? by how much?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _n = 100_000
    _a = np.random.default_rng(2).normal(size=_n)
    _b = np.random.default_rng(3).normal(size=_n)
    _la, _lb = _a.tolist(), _b.tolist()
    _rows = [
        ("your dot, two lists", best_time(ex.dot, _la, _lb)),
        ("your dot, two NumPy arrays", best_time(ex.dot, _a, _b)),
        ("built-in sum, a list", best_time(sum, _la)),
        ("built-in sum, a NumPy array", best_time(sum, _a)),
        ("np.sum, a NumPy array", best_time(np.sum, _a)),
    ]
    _base = _rows[0][1]
    _x, _y = _a[0], _la[0]
    mo.md("| what | time (ms) | relative to your dot on lists |\n|---|---|---|\n"
          + "\n".join(f"| {name} | {t * 1e3:.3f} | {t / _base:.2f}× |" for name, t in _rows)
          + f"\n\none element of the array, `a[0]`, has type `{type(_x).__name__}`; one element "
          f"of the list has type `{type(_y).__name__}`.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what has to happen every time a Python `for` loop takes the next element
    out of a NumPy array? why does that make things slower, not faster? what's the lesson for
    how to use NumPy (topic 02 is all about this)?
    """)
    return


if __name__ == "__main__":
    app.run()
