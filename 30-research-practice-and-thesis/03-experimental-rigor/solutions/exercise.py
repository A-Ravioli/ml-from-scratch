import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections.abc import Mapping

    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks, lesson
    from mlfs.stats import bootstrap_ci, permutation_test  # yours, from ch 05
    from mlfs.utils import as_rng  # yours, from ch 00 · 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # experimental rigor · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. these five functions become `mlfs.research`, which every later topic
    in this chapter, the thesis capstone and your own projects use.

    | stage | you build |
    |---|---|
    | 1 | `run_seeds`: one experiment, many seeds, results collected per metric → `mlfs.research` |
    | 2 | `mean_ci`: a mean with a $t$ or bootstrap interval → `mlfs.research` |
    | 3 | `compare_runs`: a difference between two methods with an interval and a p-value → `mlfs.research` |
    | 4 | `ablation_table`: a Markdown ablation table with intervals and paired differences → `mlfs.research` |
    | 5 | `expected_best`: the expected best score of $k$ tuning trials (budget-aware baselines) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def run_seeds(fn, seeds=(0, 1, 2, 3, 4)):
    """stage 1: call fn(seed) once per seed, in order; scalars -> float ndarray (n_seeds,), dicts -> dict key -> ndarray."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("run_seeds needs at least one seed")
    results = [fn(s) for s in seeds]
    if all(isinstance(r, Mapping) for r in results):
        keys = list(results[0])
        for s, r in zip(seeds, results):
            if set(r) != set(keys):
                raise ValueError(f"seed {s} returned keys {sorted(r)}, seed {seeds[0]} returned {sorted(keys)}")
        out = {}
        for k in keys:
            vals = [r[k] for r in results]
            if not all(np.ndim(v) == 0 for v in vals):
                raise ValueError(f"metric {k!r} isn't a scalar for every seed")
            out[k] = np.array(vals, dtype=float)
        return out
    if any(isinstance(r, Mapping) for r in results):
        raise ValueError("fn returned a dict for some seeds and a scalar for others")
    if not all(np.ndim(r) == 0 for r in results):
        raise ValueError("fn must return a scalar or a dict of scalars for every seed")
    return np.array(results, dtype=float)


@app.cell(hide_code=True)
def _():
    checks.run(1, run_seeds=run_seeds)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mean_ci(values, *, alpha=0.05, method="t", rng=None):
    """stage 2: (mean, lo, hi): a Student-t (n - 1 dof) or bootstrap (mlfs.stats.bootstrap_ci) interval for the mean."""
    v = np.asarray(values, dtype=float).ravel()
    if len(v) < 2:
        raise ValueError(f"an interval needs at least 2 values (seeds), got {len(v)}")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite: a nan or inf is a crashed run; report it, don't average it")
    m = float(v.mean())
    if method == "t":
        half = float(sps.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        return m, m - half, m + half
    if method == "bootstrap":
        lo, hi = bootstrap_ci(v, np.mean, alpha=alpha, rng=rng)
        return m, float(lo), float(hi)
    raise ValueError(f"method must be 't' or 'bootstrap', got {method!r}")


@app.cell(hide_code=True)
def _():
    checks.run(2, mean_ci=mean_ci)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def compare_runs(a, b, *, paired=True, alpha=0.05, n_permutations=10000, rng=None):
    """stage 3: dict(diff, ci, p_value, n_a, n_b, paired) for a vs b: bootstrap CI and permutation-test p."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"each side needs at least 2 runs, got {len(a)} and {len(b)}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("runs must be finite: a nan or inf is a crashed run; report it, don't drop it")
    rng = as_rng(rng)
    if paired:
        if len(a) != len(b):
            raise ValueError(f"paired=True needs one value per seed on both sides, got {len(a)} and {len(b)}")
        d = a - b
        diff = float(d.mean())
        ci = bootstrap_ci(d, np.mean, alpha=alpha, rng=rng)
    else:
        diff = float(a.mean() - b.mean())
        boot_b = np.random.default_rng(rng.integers(2**63))

        def _stat(a_star):
            # a fresh, independent resample of b for every resample of a: a two-sample bootstrap
            return a_star.mean() - b[boot_b.integers(0, len(b), len(b))].mean()

        ci = bootstrap_ci(a, _stat, alpha=alpha, rng=rng)
    _, p = permutation_test(a, b, n_permutations=n_permutations, paired=paired, rng=rng)
    return {"diff": diff, "ci": (float(ci[0]), float(ci[1])), "p_value": float(p),
            "n_a": len(a), "n_b": len(b), "paired": bool(paired)}


@app.cell(hide_code=True)
def _():
    checks.run(3, compare_runs=compare_runs)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ablation_table(results, *, baseline=None, alpha=0.05, higher_is_better=True, fmt="{:.3f}"):
    """stage 4: a Markdown table: mean ± t-CI per config, n, and (with a baseline) the paired difference ± CI; best row bold."""
    if not results:
        raise ValueError("results is empty")
    if baseline is not None and baseline not in results:
        raise ValueError(f"baseline {baseline!r} isn't one of the configs {list(results)}")
    level = f"{100 * (1 - alpha):g}%"
    rows, means = [], []
    base = None if baseline is None else np.asarray(results[baseline], dtype=float).ravel()
    for name, vals in results.items():
        v = np.asarray(vals, dtype=float).ravel()
        m, lo, hi = mean_ci(v, alpha=alpha)
        cells = [str(name), f"{fmt.format(m)} ± {fmt.format((hi - lo) / 2)}", str(len(v))]
        if base is not None:
            if name == baseline:
                cells.append("(baseline)")
            else:
                if len(v) != len(base):
                    raise ValueError(f"{name!r} has {len(v)} runs, the baseline {len(base)}: an ablation table "
                                     "compares runs paired by seed")
                dm, dlo, dhi = mean_ci(v - base, alpha=alpha)
                s = fmt.format(dm)
                cells.append(f"{s if s.startswith('-') else '+' + s} ± {fmt.format((dhi - dlo) / 2)}")
        rows.append(cells)
        means.append(m)
    best = max(means) if higher_is_better else min(means)
    header = ["config", f"mean ± {level} CI", "n"]
    if base is not None:
        header.append(f"Δ vs {baseline} ± {level} CI")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for cells, m in zip(rows, means):
        if m == best:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@app.cell(hide_code=True)
def _():
    checks.run(4, ablation_table=ablation_table)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def expected_best(scores, k, *, higher_is_better=True):
    """stage 5: expected best of k trials drawn with replacement from the observed trial scores (Dodge et al., 2019)."""
    v = np.asarray(scores, dtype=float).ravel()
    if len(v) == 0:
        raise ValueError("no trial scores")
    if not np.all(np.isfinite(v)):
        raise ValueError("trial scores must be finite")
    ks = np.asarray(k)
    if not np.issubdtype(ks.dtype, np.integer) or np.any(ks < 1):
        raise ValueError(f"k must be a positive integer (or an array of them), got {k!r}")
    v = np.sort(v) if higher_is_better else np.sort(v)[::-1]
    n = len(v)
    i = np.arange(1, n + 1)
    kk = ks.reshape(-1, 1).astype(float)
    w = (i / n) ** kk - ((i - 1) / n) ** kk  # P(the best of k draws is the i-th smallest)
    out = w @ v
    return float(out[0]) if ks.ndim == 0 else out.reshape(ks.shape)


@app.cell(hide_code=True)
def _():
    checks.run(5, expected_best=expected_best)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.research import run_seeds, mean_ci, compare_runs, ablation_table` gives every
    experiment you run (your thesis included) seeds, intervals and paired comparisons in four
    lines.
    """)
    return


if __name__ == "__main__":
    app.run()
