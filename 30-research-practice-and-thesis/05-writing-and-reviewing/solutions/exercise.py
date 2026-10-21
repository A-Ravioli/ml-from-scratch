import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re

    import marimo as mo
    import numpy as np
    from matplotlib.container import BarContainer, ErrorbarContainer

    from mlfs import checks, lesson

    # colormaps whose brightness isn't monotone, so they invent boundaries (Borland & Taylor, 2007)
    RAINBOW = {"jet", "rainbow", "gist_rainbow", "hsv", "nipy_spectral", "gist_ncar", "turbo"}
    # where in a paper a review points: "figure 3", "Table 2", "sec. 4.1", "eq. (5)", "appendix B", "§3", "line 120"
    LOCATION = re.compile(
        r"(?:\b(?i:figure|fig|table|tab|section|sec|equation|eq|appendix|app|algorithm|alg|theorem|thm|lemma|line)"
        r"\b\.?|§)\s*\(?[0-9A-Z]"
    )
    # the review's criteria, in the order score_review reports them
    CRITERIA = ("summary", "strengths", "weaknesses_specific", "weaknesses_actionable", "questions", "scores_valid",
                "consistent")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # writing & reviewing · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make a pre-submission toolkit: a figure linter, a LaTeX
    checker, a review-quality scorer, and a way to measure how much two reviewing committees
    agree.

    | stage | you build |
    |---|---|
    | 1 | `lint_figure`: labels, font sizes, legends, colormaps and truncated bars in a matplotlib figure |
    | 2 | `check_latex`: labels, references, citations, environments, floats and the abstract of a LaTeX draft |
    | 3 | `score_review`: does a peer review meet the rubric? |
    | 4 | `committee_agreement`: how often would two committees make the same decisions? |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def lint_figure(fig, *, min_fontsize=8.0):
    """stage 1: list of problems 'axes i: rule: message' (rules: xlabel, ylabel, fontsize, legend, colormap, bars)."""
    problems = []
    for i, ax in enumerate(fig.axes):
        if ax.get_label() == "<colorbar>":
            continue
        data_lines = [ln for ln in ax.lines if ln.get_transform() is ax.transData
                      and not ln.get_label().startswith("_nolegend")]
        plotted = bool(data_lines or ax.collections or ax.containers or ax.patches)
        if plotted:
            if not ax.get_xlabel().strip():
                problems.append(f"axes {i}: xlabel: missing x-axis label")
            if not ax.get_ylabel().strip():
                problems.append(f"axes {i}: ylabel: missing y-axis label")
        texts = [t for t in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts) if t.get_text().strip()]
        texts += [*ax.xaxis.get_ticklabels(), *ax.yaxis.get_ticklabels()]
        if ax.get_legend() is not None:
            texts += list(ax.get_legend().get_texts())
        small = [t.get_fontsize() for t in texts if t.get_fontsize() < min_fontsize]
        if small:
            problems.append(f"axes {i}: fontsize: text at {min(small):g} pt, below {min_fontsize:g} pt")
        handles, _ = ax.get_legend_handles_labels()
        unlabeled = [ln for ln in data_lines if ln.get_label().startswith("_")]
        unlabeled += [c for c in ax.containers if isinstance(c, ErrorbarContainer) and c.get_label().startswith("_")]
        if (len(handles) >= 2 or len(unlabeled) >= 2) and ax.get_legend() is None:
            problems.append(f"axes {i}: legend: {max(len(handles), len(unlabeled))} series and no legend")
        mappables = list(ax.images) + [c for c in ax.collections if c.get_array() is not None]
        for name in sorted({m.get_cmap().name for m in mappables} & RAINBOW):
            problems.append(f"axes {i}: colormap: '{name}' isn't perceptually uniform; use viridis or cividis")
        for c in ax.containers:
            if isinstance(c, BarContainer) and len(c.patches):
                lim = ax.get_ylim() if c.orientation == "vertical" else ax.get_xlim()
                lo, hi = min(lim), max(lim)
                if lo > 0 or hi < 0:
                    problems.append(f"axes {i}: bars: the value axis runs from {lo:g} to {hi:g}, without 0, so bar "
                                    "lengths mislead")
                    break
    return problems


@app.cell(hide_code=True)
def _():
    checks.run(1, lint_figure=lint_figure)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def check_latex(source, *, bib_keys=None, max_abstract_words=250):
    """stage 2: list of problems 'kind: message' (kinds: env, label, ref, cite, float, abstract) in a LaTeX source."""
    text = re.sub(r"(?<!\\)%.*", "", source)
    problems = []
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", text):
        kind, name = m.groups()
        if kind == "begin":
            stack.append(name)
        elif not stack:
            problems.append(f"env: \\end{{{name}}} without a \\begin")
        elif stack[-1] != name:
            problems.append(f"env: \\end{{{name}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
    for name in stack:
        problems.append(f"env: \\begin{{{name}}} is never closed")
    labels = re.findall(r"\\label\{([^}]*)\}", text)
    for lab in sorted({x for x in labels if labels.count(x) > 1}):
        problems.append(f"label: '{lab}' is defined {labels.count(lab)} times")
    refs = set()
    for m in re.finditer(r"\\(?:ref|eqref|autoref|pageref|cref|Cref)\{([^}]*)\}", text):
        refs.update(k.strip() for k in m.group(1).split(",") if k.strip())
    for r in sorted(refs - set(labels)):
        problems.append(f"ref: '{r}' is referenced but never defined")
    if bib_keys is not None:
        cites = set()
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", text):
            cites.update(k.strip() for k in m.group(1).split(",") if k.strip())
        for c in sorted(cites - set(bib_keys)):
            problems.append(f"cite: '{c}' isn't in the bibliography")
    counts = {}
    for m in re.finditer(r"\\begin\{(figure|table)(\*?)\}(.*?)\\end\{\1\2\}", text, flags=re.S):
        env, body = m.group(1), m.group(3)
        counts[env] = counts.get(env, 0) + 1
        what = f"{env} {counts[env]}"
        cap = body.find("\\caption")
        lab = re.search(r"\\label\{([^}]*)\}", body)
        if cap < 0:
            problems.append(f"float: {what} has no \\caption")
        if lab is None:
            problems.append(f"float: {what} has no \\label")
        else:
            if cap >= 0 and lab.start() < cap:
                problems.append(f"float: {what}'s \\label comes before its \\caption (it will get the wrong number)")
            if lab.group(1) not in refs:
                problems.append(f"float: {what} ('{lab.group(1)}') is never referenced in the text")
    for m in re.finditer(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S):
        n = len(re.sub(r"\\[A-Za-z]+\*?", " ", m.group(1)).split())
        if n > max_abstract_words:
            problems.append(f"abstract: {n} words, more than {max_abstract_words}")
    return problems


@app.cell(hide_code=True)
def _():
    checks.run(2, check_latex=check_latex)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def score_review(review, *, abstract=None):
    """stage 3: dict criterion -> bool for each of CRITERIA, plus 'score' (how many hold)."""
    def words(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    def grams(s, n=4):
        w = words(s)
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

    def str_list(v):
        return isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x.strip() for x in v)

    out = {}
    summary = review.get("summary")
    ok = isinstance(summary, str) and len(summary.split()) >= 30
    if ok and abstract is not None:
        g = grams(summary)
        ok = not g or len(g & grams(abstract)) / len(g) <= 0.25
    out["summary"] = ok
    out["strengths"] = str_list(review.get("strengths"))
    weak = review.get("weaknesses")
    weak_ok = isinstance(weak, list) and len(weak) > 0 and all(isinstance(w, dict) for w in weak)
    out["weaknesses_specific"] = weak_ok and all(
        isinstance(w.get("text"), str) and w["text"].strip() and isinstance(w.get("location"), str)
        and bool(LOCATION.search(w["location"])) for w in weak)
    out["weaknesses_actionable"] = weak_ok and all(
        isinstance(w.get("suggestion"), str) and w["suggestion"].strip()
        for w in weak if w.get("severity") == "major")
    out["questions"] = str_list(review.get("questions"))
    sc = review.get("scores")
    ranges = {"soundness": (1, 4), "presentation": (1, 4), "contribution": (1, 4), "overall": (1, 10),
              "confidence": (1, 5)}
    out["scores_valid"] = isinstance(sc, dict) and all(
        isinstance(sc.get(k), int) and not isinstance(sc.get(k), bool) and lo <= sc[k] <= hi
        for k, (lo, hi) in ranges.items())
    if out["scores_valid"]:
        n_major = sum(1 for w in weak if isinstance(w, dict) and w.get("severity") == "major") if weak_ok else 0
        out["consistent"] = not (sc["overall"] >= 7 and n_major >= 2) and not (sc["overall"] <= 3 and n_major == 0)
    else:
        out["consistent"] = False
    out = {k: bool(out[k]) for k in CRITERIA}
    out["score"] = sum(out.values())
    return out


@app.cell(hide_code=True)
def _():
    checks.run(3, score_review=score_review)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def committee_agreement(scores_a, scores_b, accept_rate):
    """stage 4: dict(k, agree, flip_rate, kappa) when two committees each accept their top round(accept_rate n) papers."""
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"both committees must score the same papers, got {len(a)} and {len(b)} scores")
    n = len(a)
    k = int(round(accept_rate * n))
    if not 0 < k < n:
        raise ValueError(f"accept_rate {accept_rate} accepts {k} of {n} papers; it must accept some but not all")
    acc_a = np.zeros(n, dtype=bool)
    acc_b = np.zeros(n, dtype=bool)
    acc_a[np.argsort(-a, kind="stable")[:k]] = True
    acc_b[np.argsort(-b, kind="stable")[:k]] = True
    p_o = float(np.mean(acc_a == acc_b))
    r = k / n
    p_e = r * r + (1 - r) * (1 - r)
    return {"k": k, "agree": p_o, "flip_rate": float(np.mean(~acc_b[acc_a])), "kappa": (p_o - p_e) / (1 - p_e)}


@app.cell(hide_code=True)
def _():
    checks.run(4, committee_agreement=committee_agreement)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then run `lint_figure` on
    every figure of your thesis and `check_latex` on its source before anyone else reads it.
    """)
    return


if __name__ == "__main__":
    app.run()
