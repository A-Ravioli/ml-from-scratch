import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks, lesson

    # the reproducibility checklist the report renders: (id, item). adapted from the ML reproducibility
    # checklist of Pineau et al. (2021); answer each one in your report.
    CHECKLIST = [
        ("claims", "every claim tested is stated with the paper's evidence and a success criterion fixed before the runs"),
        ("gaps", "every detail the paper leaves unspecified is listed with the choice made and why"),
        ("scale", "the scale-down is argued: what is preserved, what is shrunk, and why the claim should survive"),
        ("code", "the code, configurations and the command for each figure and table are available"),
        ("data", "the data, its version, preprocessing and splits are described or provided"),
        ("seeds", "the number of seeds per configuration and how seeds were chosen are stated"),
        ("stats", "every reported number has an interval, and the method used to compute it is stated"),
        ("compute", "hardware, wall-clock time per run and the total compute are reported"),
        ("versions", "library versions (and the authors' code version, if consulted) are recorded"),
        ("deviations", "every deviation from the paper, and every change made after reading the authors' code, is listed"),
    ]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # reproducing papers · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make the analysis half of the reproduction protocol: a
    verdict for a reproduced number, a verdict for a reproduced comparison, how many seeds you
    need for a verdict to be possible, and a report generator.

    | stage | you build |
    |---|---|
    | 1 | `equivalence_verdict`: does my number match the paper's within a margin? |
    | 2 | `effect_verdict`: does the paper's comparison (A beats B by Δ) hold? |
    | 3 | `seeds_needed`: how many seeds before "reproduced" is even possible? |
    | 4 | `reproduction_report`: the claims, verdicts, gaps and checklist, as Markdown |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def equivalence_verdict(runs, paper_value, *, margin, alpha=0.05):
    """stage 1: dict(mean, ci, n, verdict): the (1 - 2 alpha) t interval of runs against paper_value ± margin."""
    v = np.asarray(runs, dtype=float).ravel()
    if len(v) < 2:
        raise ValueError(f"a verdict needs at least 2 runs, got {len(v)}")
    if not np.all(np.isfinite(v)):
        raise ValueError("runs must be finite: a nan is a crashed run; report it")
    if not margin > 0:
        raise ValueError(f"margin must be positive, got {margin}")
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    m = float(v.mean())
    half = float(sps.t.ppf(1 - alpha, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    lo, hi = m - half, m + half
    band = (paper_value - margin, paper_value + margin)
    if band[0] <= lo and hi <= band[1]:
        verdict = "reproduced"
    elif hi < band[0] or lo > band[1]:
        verdict = "not reproduced"
    else:
        verdict = "inconclusive"
    return {"mean": m, "ci": (lo, hi), "n": len(v), "verdict": verdict}


@app.cell(hide_code=True)
def _():
    checks.run(1, equivalence_verdict=equivalence_verdict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def effect_verdict(runs, baseline_runs, paper_effect, *, margin, alpha=0.05, paired=True):
    """stage 2: dict(diff, ci, n, verdict) for the comparison runs - baseline against paper_effect ± margin."""
    a = np.asarray(runs, dtype=float).ravel()
    b = np.asarray(baseline_runs, dtype=float).ravel()
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"each side needs at least 2 runs, got {len(a)} and {len(b)}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("runs must be finite: a nan is a crashed run; report it")
    if not margin > 0:
        raise ValueError(f"margin must be positive, got {margin}")
    if paper_effect != 0 and margin >= abs(paper_effect):
        raise ValueError(f"a margin ({margin}) as large as the paper's effect ({paper_effect}) can't tell the "
                         "claim apart from no effect at all")
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if paired:
        if len(a) != len(b):
            raise ValueError(f"paired=True needs one run per seed on both sides, got {len(a)} and {len(b)}")
        d = a - b
        diff = float(d.mean())
        se, df = d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1
        n = len(d)
    else:
        diff = float(a.mean() - b.mean())
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else len(a) + len(b) - 2
        n = min(len(a), len(b))
    half = float(sps.t.ppf(1 - alpha, df) * se)
    lo, hi = diff - half, diff + half
    band = (paper_effect - margin, paper_effect + margin)
    direction = (paper_effect > 0 and lo > 0) or (paper_effect < 0 and hi < 0)
    disjoint = hi < band[0] or lo > band[1]
    if band[0] <= lo and hi <= band[1]:
        verdict = "reproduced"
    elif direction and disjoint:
        verdict = "partly reproduced"
    elif direction:
        verdict = "direction reproduced, size inconclusive"
    elif disjoint:
        verdict = "not reproduced"
    else:
        verdict = "inconclusive"
    return {"diff": diff, "ci": (lo, hi), "n": n, "verdict": verdict}


@app.cell(hide_code=True)
def _():
    checks.run(2, effect_verdict=effect_verdict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def seeds_needed(sd, margin, *, alpha=0.05, power=0.8, max_seeds=1000):
    """stage 3: smallest n >= 2 with (t_{1-alpha, n-1} + z_{(1+power)/2}) sd / sqrt(n) <= margin."""
    if not sd > 0 or not margin > 0:
        raise ValueError(f"sd and margin must be positive, got {sd} and {margin}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}")
    z = sps.norm.ppf((1 + power) / 2)
    for n in range(2, max_seeds + 1):
        if (sps.t.ppf(1 - alpha, n - 1) + z) * sd / np.sqrt(n) <= margin:
            return n
    raise ValueError(f"more than {max_seeds} seeds needed: the margin is too small for this much seed noise")


@app.cell(hide_code=True)
def _():
    checks.run(3, seeds_needed=seeds_needed)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def reproduction_report(title, claims, results, *, gaps=(), checklist=None, alpha=0.05):
    """stage 4: (markdown, verdicts): summary table, claims, gaps and checklist; verdicts maps claim id -> verdict."""
    ids = [c["id"] for c in claims]
    if len(set(ids)) != len(ids):
        raise ValueError(f"claim ids must be unique, got {ids}")
    level = f"{100 * (1 - 2 * alpha):g}%"
    rows, sections, verdicts = [], [], {}
    for c in claims:
        if c["id"] not in results:
            raise ValueError(f"no results for claim {c['id']!r}")
        kind, r = c.get("kind", "value"), results[c["id"]]
        if kind == "value":
            out = equivalence_verdict(r, c["paper"], margin=c["margin"], alpha=alpha)
            paper = f"{c['paper']:.3g} ± {c['margin']:.3g}"
            ours = f"{out['mean']:.3g} [{out['ci'][0]:.3g}, {out['ci'][1]:.3g}]"
            what = "our mean"
        elif kind == "effect":
            ours_runs, base_runs = r
            out = effect_verdict(ours_runs, base_runs, c["paper"], margin=c["margin"], alpha=alpha,
                                 paired=c.get("paired", True))
            paper = f"Δ = {c['paper']:.3g} ± {c['margin']:.3g}"
            ours = f"Δ = {out['diff']:.3g} [{out['ci'][0]:.3g}, {out['ci'][1]:.3g}]"
            what = "our difference"
        else:
            raise ValueError(f"claim {c['id']!r}: kind must be 'value' or 'effect', got {kind!r}")
        verdicts[c["id"]] = out["verdict"]
        rows.append(f"| {c['id']} | {c['metric']} | {paper} | {ours} | {out['n']} | **{out['verdict']}** |")
        sections.append(f"### {c['id']}: {c['statement']}\n\nevidence in the paper: {c['evidence']}. criterion: "
                        f"reproduced if the {level} interval for {what} lies within {paper}.")
    gap_rows = []
    for g in gaps:
        if not all(isinstance(g.get(f), str) and g[f].strip() for f in ("detail", "choice", "reason")):
            raise ValueError(f"every gap needs a non-empty detail, choice and reason, got {g!r}")
        gap_rows.append(f"| {g['detail']} | {g['choice']} | {g['reason']} |")
    answers = dict(checklist or {})
    known = {k for k, _ in CHECKLIST}
    if set(answers) - known:
        raise ValueError(f"unknown checklist items: {sorted(set(answers) - known)}")
    items = []
    for k, text in CHECKLIST:
        a = answers.get(k)
        if isinstance(a, str) and a.strip():
            items.append(f"- [x] {text}: {a.strip()}")
        else:
            items.append(f"- [ ] {text}: **unanswered**")
    n_open = sum(1 for k, _ in CHECKLIST if not (isinstance(answers.get(k), str) and answers[k].strip()))
    md = "\n".join([
        f"# {title}", "",
        "## summary", "",
        f"| claim | metric | paper | ours (mean, {level} CI) | n | verdict |",
        "|---|---|---|---|---|---|",
        *rows, "",
        "## claims", "",
        "\n\n".join(sections), "",
        "## gaps", "",
        *(["| detail the paper doesn't specify | our choice | why |", "|---|---|---|", *gap_rows]
          if gap_rows else ["none listed."]), "",
        "## reproducibility checklist", "",
        f"{len(CHECKLIST) - n_open} of {len(CHECKLIST)} items answered.", "",
        *items,
    ])
    return md + "\n", verdicts


@app.cell(hide_code=True)
def _():
    checks.run(4, reproduction_report=reproduction_report)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. for R1, R2 and R3, write
    your claims table as a list of claim dicts *before* the runs, and let
    `reproduction_report` render the summary once they're done.
    """)
    return


if __name__ == "__main__":
    app.run()
