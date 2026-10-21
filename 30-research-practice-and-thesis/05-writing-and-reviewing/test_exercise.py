"""Checks for 30 · 05 writing & reviewing.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. lint_figure on small matplotlib
figures, one flaw at a time (labels, font sizes, legends, colormaps, truncated bars), and on figures that must
pass (reference lines, error bars, images, colorbars); check_latex on a clean draft and on one bug at a time;
score_review criterion by criterion; committee_agreement on hand examples, ties, rounding and large random
committees.
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

RULE = re.compile(r"^axes (\d+): (xlabel|ylabel|fontsize|legend|colormap|bars): ")


def _lint(nb, fig, **kwargs):
    try:
        got = nb.lint_figure(fig, **kwargs)
    finally:
        plt.close(fig)
    assert isinstance(got, list), f"lint_figure returns a list of strings, got {got!r}"
    for p in got:
        assert RULE.match(p), f"each problem must start with 'axes i: rule: ', got {p!r}"
    return {(int(RULE.match(p).group(1)), RULE.match(p).group(2)) for p in got}


def _good():
    fig, ax = plt.subplots()
    x = np.arange(5)
    ax.plot(x, x, label="a")
    ax.plot(x, 2 * x, label="b")
    ax.set(xlabel="epoch", ylabel="loss")
    ax.legend()
    return fig, ax


# ---------------------------------------------------------------- stage 1: lint_figure


def test_stage1_good_figure_passes(nb):
    fig, ax = _good()
    ax.axhline(1.0, color="k")  # a reference line isn't a data series
    ax.errorbar([1, 2], [3, 4], yerr=[0.5, 0.5], fmt="none")  # error bars aren't series either
    assert _lint(nb, fig) == set(), "a labelled figure with a legend and default fonts has no problems"


def test_stage1_labels(nb):
    fig, ax = _good()
    ax.set_xlabel("")
    assert _lint(nb, fig) == {(0, "xlabel")}, "a missing x-axis label is 'axes 0: xlabel: ...'"
    fig, axes = plt.subplots(1, 2)
    for a in axes:
        a.plot([0, 1], [0, 1])
        a.set_xlabel("x")
    axes[0].set_ylabel("y")
    assert _lint(nb, fig) == {(1, "ylabel")}, "problems name the axes by its index in fig.axes"
    fig, ax = plt.subplots()
    ax.imshow(np.random.default_rng(0).random((4, 4)))
    assert _lint(nb, fig) == set(), "an image-only axes (a heatmap, a digit) doesn't need axis labels"


def test_stage1_font_sizes(nb):
    fig, ax = _good()
    ax.tick_params(labelsize=6)
    assert _lint(nb, fig) == {(0, "fontsize")}, "tick labels at 6 pt are below the default minimum of 8 pt"
    fig, ax = _good()
    ax.tick_params(labelsize=6)
    assert _lint(nb, fig, min_fontsize=5) == set(), "min_fontsize sets the threshold"
    for setter in (lambda a: a.set_title("t", fontsize=7), lambda a: a.legend(fontsize=6),
                   lambda a: a.text(1, 1, "note", fontsize=5), lambda a: a.set_ylabel("loss", fontsize=7)):
        fig, ax = _good()
        setter(ax)
        assert _lint(nb, fig) == {(0, "fontsize")}, "titles, legends, annotations and axis labels count too"
    fig, ax = _good()
    ax.set_title("", fontsize=4)
    assert _lint(nb, fig) == set(), "an empty title isn't text a reader has to read"


def test_stage1_legends(nb):
    fig, ax = _good()
    ax.get_legend().remove()
    assert _lint(nb, fig) == {(0, "legend")}, "two labelled series and no legend"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.plot([0, 1], [1, 0])
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, fig) == {(0, "legend")}, "two unlabelled lines can't be told apart either"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="only one")
    ax.axvline(0.5)
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, fig) == set(), "one series (plus a reference line) needs no legend"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.axhline(0.5)
    ax.axvline(0.5)
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, fig) == set(), (
        "axhline/axvline draw reference lines (not in data coordinates: their transform isn't ax.transData); "
        "they aren't series"
    )
    fig, ax = plt.subplots()
    ax.errorbar([0, 1], [0, 1], yerr=[0.1, 0.1], capsize=3)
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, fig) == set(), "one errorbar series is one series (its parts are labelled '_nolegend_')"
    ax = plt.subplots()[1]
    ax.errorbar([0, 1], [0, 1], yerr=[0.1, 0.1])
    ax.errorbar([0, 1], [1, 0], yerr=[0.1, 0.1])
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, ax.figure) == {(0, "legend")}, "two unlabelled error-bar series can't be told apart either"
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2], label="a")
    ax.bar([0, 1], [0.5, 0.5], label="b")
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, fig) == {(0, "legend")}, "two labelled bar series need a legend too"


def test_stage1_colormaps(nb):
    rng = np.random.default_rng(0)
    for cmap in ("jet", "rainbow", "hsv", "turbo"):
        fig, ax = plt.subplots()
        im = ax.imshow(rng.random((4, 4)), cmap=cmap)
        fig.colorbar(im)
        assert _lint(nb, fig) == {(0, "colormap")}, (
            f"an image with the {cmap!r} colormap is flagged once, on its own axes; the colorbar axes is skipped"
        )
    fig, ax = plt.subplots()
    ax.scatter([0, 1], [0, 1], c=[0.0, 1.0], cmap="nipy_spectral")
    ax.set(xlabel="x", ylabel="y")
    assert _lint(nb, fig) == {(0, "colormap")}, "a colour-mapped scatter counts too"
    fig, ax = plt.subplots()
    ax.imshow(rng.random((4, 4)), cmap="viridis")
    assert _lint(nb, fig) == set()


def test_stage1_truncated_bars(nb):
    fig, ax = plt.subplots()
    ax.bar(["baseline", "ours"], [93.0, 93.5])
    ax.set(xlabel="method", ylabel="accuracy (%)", ylim=(92.5, 94))
    assert _lint(nb, fig) == {(0, "bars")}, "bars on an axis starting at 92.5 exaggerate a 0.5-point gap"
    fig, ax = plt.subplots()
    ax.bar(["baseline", "ours"], [93.0, 93.5])
    ax.set(xlabel="method", ylabel="accuracy (%)")
    assert _lint(nb, fig) == set(), "bars from 0 are fine"
    fig, ax = plt.subplots()
    ax.barh([0, 1], [3.0, 4.0])
    ax.set(xlabel="value", ylabel="item", xlim=(2, 5))
    assert _lint(nb, fig) == {(0, "bars")}, "horizontal bars: check the x-axis"
    fig, ax = plt.subplots()
    ax.bar([0, 1], [-3.0, -1.0])
    ax.set(xlabel="method", ylabel="change (points)", ylim=(-4, -0.5))
    assert _lint(nb, fig) == {(0, "bars")}, "negative bars on an axis that stops below 0 are truncated too"
    fig, ax = plt.subplots()
    ax.bar([0, 1], [-3.0, -1.0])
    ax.set(xlabel="method", ylabel="change (points)")
    assert _lint(nb, fig) == set(), "negative bars that reach 0 are fine"
    fig, ax = plt.subplots()
    ax.hist(np.random.default_rng(1).normal(size=200))
    ax.set(xlabel="value", ylabel="count")
    assert _lint(nb, fig) == set(), "a histogram's bars start at 0"


# ---------------------------------------------------------------- stage 2: check_latex

def _latex(nb, *args, **kwargs):
    return bounded(nb.check_latex, *args, what="check_latex", hint="every scanning loop must reach the end.", **kwargs)


DOC = r"""
\begin{document}
\begin{abstract}
We show that residual connections fix the degradation problem.
\end{abstract}
\section{Introduction}\label{sec:intro}
As figure~\ref{fig:curves} and table~\ref{tab:main} show, deeper residual networks train better \citep{he2016, ioffe2015}.
50\% of runs % a comment with \ref{fig:ghost} and \cite{ghost}
converge \cite[p.~3]{he2016}. see \cref{eq:res}.
\begin{equation}\label{eq:res} y = F(x) + x \end{equation}
\begin{figure}
  \centering
  \caption{Training curves.}\label{fig:curves}
\end{figure}
\begin{table*}
  \caption{Main results.}
  \label{tab:main}
\end{table*}
\end{document}
"""
BIB = ["he2016", "ioffe2015"]


def _kinds(problems):
    assert isinstance(problems, list) and all(isinstance(p, str) for p in problems), "return a list of strings"
    for p in problems:
        assert re.match(r"^(env|label|ref|cite|float|abstract): ", p), f"each problem starts with its kind, got {p!r}"
    return [p.split(":")[0] for p in problems]


def test_stage2_clean_draft_passes(nb):
    got = _latex(nb, DOC, bib_keys=BIB)
    assert got == [], (
        f"this draft is clean (the \\ref and \\cite inside the comment don't count, and 50\\% isn't a comment), "
        f"got {got}"
    )


def test_stage2_references_labels_and_citations(nb):
    got = _latex(nb, DOC.replace(r"\ref{fig:curves}", r"\ref{fig:curve}"), bib_keys=BIB)
    assert sorted(_kinds(got)) == ["float", "ref"] and any("fig:curve'" in p for p in got), (
        f"a typo in a \\ref is an undefined reference ('fig:curve'), and the figure is now never referenced: {got}"
    )
    got = _latex(nb, DOC.replace(r"\label{sec:intro}", r"\label{eq:res}"), bib_keys=BIB)
    assert _kinds(got) == ["label"] and "eq:res" in got[0], f"a label defined twice: {got}"
    got = _latex(nb, DOC.replace("ioffe2015}", "ioffe2015, goodfellow2016}"), bib_keys=BIB)
    assert _kinds(got) == ["cite"] and "goodfellow2016" in got[0], f"a cited key missing from the bib: {got}"
    assert _latex(nb, DOC.replace("ioffe2015}", "ioffe2015, goodfellow2016}")) == [], (
        "without bib_keys, citations aren't checked"
    )
    got = _latex(nb, DOC.replace(r"\cref{eq:res}", r"\cref{eq:res,eq:nope}"), bib_keys=BIB)
    assert _kinds(got) == ["ref"] and "'eq:nope'" in got[0], "\\cref can list several labels; report each one"
    got = _latex(nb, DOC.replace("converge", r"converge (and 10\% see \ref{fig:typo})"), bib_keys=BIB)
    assert _kinds(got) == ["ref"] and "fig:typo" in got[0], (
        "\\% is a literal percent sign, not a comment: the \\ref after it on the same line still counts"
    )
    got = _latex(nb, DOC.replace(r"\cite[p.~3]{he2016}", r"\cite[p.~3]{he2061}"), bib_keys=BIB)
    assert _kinds(got) == ["cite"] and "he2061" in got[0], "\\cite[p.~3]{key}: the optional argument isn't a key"


def test_stage2_environments(nb):
    got = _latex(nb, DOC.replace(r"\end{equation}", r"\end{equaton}"), bib_keys=BIB)
    assert _kinds(got) == ["env"], (
        f"\\end{{equaton}} doesn't close \\begin{{equation}}: one problem, and the mismatched environment is then "
        f"closed anyway, so nothing after it is reported. got {got}"
    )
    got = _latex(nb, DOC.replace(r"\end{document}", ""), bib_keys=BIB)
    assert _kinds(got) == ["env"] and "document" in got[0], f"\\begin{{document}} is never closed: {got}"
    got = _latex(nb, DOC + "\n\\end{itemize}\n", bib_keys=BIB)
    assert _kinds(got) == ["env"] and "itemize" in got[0], f"a stray \\end{{itemize}}: {got}"


def test_stage2_floats(nb):
    got = _latex(nb, DOC.replace(r"\caption{Training curves.}", ""), bib_keys=BIB)
    assert _kinds(got) == ["float"] and "caption" in got[0], f"a figure without a caption: {got}"
    got = _latex(nb, DOC.replace("\\caption{Main results.}\n  \\label{tab:main}",
                                     "\\label{tab:main}\n  \\caption{Main results.}"), bib_keys=BIB)
    assert _kinds(got) == ["float"] and "before" in got[0], (
        f"a \\label before its \\caption gets the section's number, not the table's: flag it. got {got}"
    )
    got = _latex(nb, DOC.replace(r"\label{fig:curves}", ""), bib_keys=BIB)
    assert sorted(_kinds(got)) == ["float", "ref"], (
        f"a figure without a label (and the now-undefined reference to it): {got}"
    )
    got = _latex(nb, DOC.replace(r"table~\ref{tab:main}", "the table"), bib_keys=BIB)
    assert _kinds(got) == ["float"] and "tab:main" in got[0], (
        f"a table (here table*) that the text never references: {got}"
    )


def test_stage2_abstract_length(nb):
    long = DOC.replace("We show that", " ".join(["word"] * 260))
    got = _latex(nb, long, bib_keys=BIB)
    assert _kinds(got) == ["abstract"], f"an abstract of about 270 words is over the default 250: {got}"
    assert _latex(nb, long, bib_keys=BIB, max_abstract_words=300) == [], "max_abstract_words sets the limit"
    assert _latex(nb, DOC.replace("We show that", r"\textbf{We} \emph{show} that"), bib_keys=BIB) == []
    spaced = DOC.replace("We show that", " ".join(["word"] * 230 + [r"\quad"] * 20))
    assert _latex(nb, spaced, bib_keys=BIB) == [], (
        "about 236 words plus 20 \\quad commands: commands aren't words, so this abstract is under 250"
    )


# ---------------------------------------------------------------- stage 3: score_review

ABSTRACT = ("Deeper neural networks are more difficult to train. We present a residual learning framework to ease the "
            "training of networks that are substantially deeper than those used previously.")


def _review():
    return {
        "summary": "the paper argues that very deep plain convolutional networks are hard to optimize, and proposes "
                   "adding identity shortcuts so each block learns a correction to its input. experiments on image "
                   "classification show the shortcut versions train to lower error at every depth tested.",
        "strengths": ["a simple idea with a clear motivation", "experiments at several depths"],
        "weaknesses": [
            {"text": "the degradation claim rests on one run per depth", "location": "figure 1",
             "suggestion": "report the mean and spread over at least 5 seeds", "severity": "major"},
            {"text": "the projection shortcut variants are hard to compare", "location": "table 3",
             "severity": "minor"},
        ],
        "questions": ["does the effect survive without batch normalization?"],
        "scores": {"soundness": 3, "presentation": 3, "contribution": 4, "overall": 7, "confidence": 4},
    }


def _check(nb, review, **kwargs):
    got = nb.score_review(review, **kwargs)
    crit = ["summary", "strengths", "weaknesses_specific", "weaknesses_actionable", "questions", "scores_valid",
            "consistent"]
    assert set(got) == set(crit) | {"score"}, f"return one bool per criterion plus 'score', got keys {sorted(got)}"
    assert got["score"] == sum(bool(got[c]) for c in crit), "score counts the criteria that hold"
    return {c for c in crit if not got[c]}


def test_stage3_good_review(nb):
    r = _review()
    snapshot = repr(r)
    assert _check(nb, r, abstract=ABSTRACT) == set(), "this review meets every criterion"
    assert nb.score_review(r)["score"] == 7
    assert repr(r) == snapshot, "score_review must not modify the review"


def test_stage3_each_criterion(nb):
    def with_(**changes):
        r = _review()
        for k, v in changes.items():
            r[k] = v
        return r

    cases = [
        ("summary", with_(summary=" ".join(["word"] * 29))),
        ("strengths", with_(strengths=[])),
        ("strengths", with_(strengths=[" "])),
        ("questions", with_(questions=[])),
        ("scores_valid", with_(scores={**_review()["scores"], "overall": 11})),
        ("scores_valid", with_(scores={**_review()["scores"], "soundness": 0})),
        ("scores_valid", with_(scores={**_review()["scores"], "confidence": True})),
    ]
    for crit, r in cases:
        got = _check(nb, r)
        expected = {crit} | ({"consistent"} if crit == "scores_valid" else set())
        assert got == expected, f"this review should fail only {sorted(expected)}, got {sorted(got)}"
    r = _review()
    r["weaknesses"][1]["location"] = "throughout the paper"
    assert _check(nb, r) == {"weaknesses_specific"}, "every weakness needs a location like 'table 3' or 'line 120'"
    r = _review()
    r["weaknesses"][0].pop("suggestion")
    assert _check(nb, r) == {"weaknesses_actionable"}, "a MAJOR weakness needs a suggestion (minor ones needn't)"
    r = _review()
    r["weaknesses"] = []
    assert _check(nb, r) == {"weaknesses_specific", "weaknesses_actionable"}, "a review with no weaknesses fails both"


def test_stage3_copied_summary(nb):
    r = _review()
    r["summary"] = ABSTRACT + " the results look good and the paper is clear and well written overall."
    assert _check(nb, r, abstract=ABSTRACT) == {"summary"}, (
        "a summary that mostly repeats the abstract's 4-grams isn't in the reviewer's own words"
    )
    assert _check(nb, r) == set(), "without an abstract, the copy check can't run"


def test_stage3_copy_threshold(nb):
    words = ABSTRACT.split()
    own = [f"note{i}" for i in range(30)]
    r = _review()
    r["summary"] = " ".join(words[:15] + own[:25])  # 12 of its 37 4-grams come from the abstract: 32%
    assert _check(nb, r, abstract=ABSTRACT) == {"summary"}, "32% of the 4-grams copied is over the 25% limit"
    r["summary"] = " ".join(words[:10] + own)  # 7 of 37: 19%
    assert _check(nb, r, abstract=ABSTRACT) == set(), "19% of the 4-grams copied is under the 25% limit"


def test_stage3_consistency(nb):
    r = _review()
    r["weaknesses"][1]["severity"] = "major"
    r["weaknesses"][1]["suggestion"] = "add a table with matched parameter counts"
    assert _check(nb, r) == {"consistent"}, "an overall score of 7 with two major weaknesses is inconsistent"
    r["scores"]["overall"] = 6
    assert _check(nb, r) == set(), "6 with two major weaknesses is fine"
    r = _review()
    r["weaknesses"][0]["severity"] = "minor"
    r["scores"]["overall"] = 3
    assert _check(nb, r) == {"consistent"}, "rejecting (3 or below) with no major weakness is inconsistent"
    r["scores"]["overall"] = 4
    assert _check(nb, r) == set()


# ---------------------------------------------------------------- stage 4: committee_agreement


def test_stage4_worked_example(nb):
    got = nb.committee_agreement([5, 4, 3, 2, 1, 0], [4, 5, 0, 1, 2, 3], 1 / 3)
    assert got["k"] == 2 and np.isclose(got["agree"], 1.0) and np.isclose(got["flip_rate"], 0.0) and np.isclose(
        got["kappa"], 1.0), f"both committees accept papers 0 and 1: agree 1, flip_rate 0, kappa 1; got {got}"
    got = nb.committee_agreement([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5], 1 / 3)
    assert np.isclose(got["agree"], 1 / 3) and np.isclose(got["flip_rate"], 1.0) and np.isclose(got["kappa"], -0.5), (
        "opposite rankings: they agree only on rejecting papers 2 and 3 (agree 1/3); every paper A accepts, B rejects "
        "(flip_rate 1); kappa = (1/3 - 5/9) / (1 - 5/9) = -0.5. got " + str(got)
    )


def test_stage4_ties_and_rounding(nb):
    got = nb.committee_agreement([1, 1, 1, 1], [0, 0, 1, 1], 0.5)
    assert np.isclose(got["flip_rate"], 1.0), (
        "ties are broken by order (a stable sort): A accepts papers 0 and 1, B accepts 2 and 3"
    )
    assert nb.committee_agreement(np.arange(10), np.arange(10), 0.26)["k"] == 3, "k = round(0.26 × 10) = 3"


def test_stage4_random_committees(nb):
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    same = nb.committee_agreement(a, a.copy(), 0.25)
    assert np.isclose(same["kappa"], 1.0) and same["flip_rate"] == 0.0
    ind = nb.committee_agreement(a, rng.normal(size=20000), 0.25)
    assert abs(ind["kappa"]) < 0.03 and abs(ind["flip_rate"] - 0.75) < 0.02, (
        f"independent committees agree only by chance: kappa ≈ 0 and 75% of A's accepted papers are rejected by B "
        f"when 25% are accepted. got {ind}"
    )


def test_stage4_refusals(nb):
    with pytest.raises(ValueError):
        nb.committee_agreement([1, 2, 3], [1, 2], 0.5)
    for rate in (0.0, 1.0, 0.04):
        with pytest.raises(ValueError):
            nb.committee_agreement(np.arange(10), np.arange(10), rate)
