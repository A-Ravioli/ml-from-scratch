import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re

    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks, lesson
    from mlfs.dltheory import compute_optimal  # yours, from ch 19 · topic 05

    # wording that makes a hypothesis hard to refute: activities instead of predictions, and hedges
    VAGUE = re.compile(
        r"\b(explore|investigate|study|understand(ing)?|shed light|look into|examine|analy[sz]e"
        r"|may|might|could|possibly|potentially|in some cases|interesting)\b",
        re.IGNORECASE,
    )
    # the proposal's sections, after the Heilmeier catechism
    SECTIONS = ("title", "goal", "state_of_the_art", "approach", "impact")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # finding questions · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make a proposal checker: is the hypothesis falsifiable,
    are there enough seeds to detect the effect you care about, does the compute fit, and is
    the proposal complete and short?

    | stage | you build |
    |---|---|
    | 1 | `check_hypothesis`: a linter for falsifiable hypotheses |
    | 2 | `paired_power`, `seeds_for_power`: the power of a paired $t$ test, and the seeds it needs |
    | 3 | `run_hours`, `largest_model`: compute in hours, and the biggest compute-optimal model you can afford |
    | 4 | `budget_plan`: does the whole experiment plan fit the budget, with a safety factor? |
    | 5 | `check_proposal`: the two-page proposal, checked end to end |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def check_hypothesis(h):
    """stage 1: list of problems ('field: message') with a hypothesis dict; [] means it's falsifiable as stated."""
    problems = []
    for field in ("statement", "metric", "method", "baseline", "setting", "refuted_if"):
        if not isinstance(h.get(field), str) or not h[field].strip():
            problems.append(f"{field}: missing or empty")
    if isinstance(h.get("statement"), str):
        vague = sorted({m.group(0).lower() for m in VAGUE.finditer(h["statement"])})
        if vague:
            problems.append(f"statement: vague wording {vague}: say what will happen, not what you'll look at")
    if (isinstance(h.get("method"), str) and isinstance(h.get("baseline"), str)
            and h["method"].strip() and h["method"].strip().lower() == h["baseline"].strip().lower()):
        problems.append("baseline: must differ from the method")
    if h.get("direction") not in ("greater", "less"):
        problems.append(f"direction: must be 'greater' or 'less', got {h.get('direction')!r}")
    eff = h.get("min_effect")
    if isinstance(eff, bool) or not isinstance(eff, (int, float)) or not np.isfinite(eff) or eff <= 0:
        problems.append(f"min_effect: must be a positive number (the smallest effect that matters), got {eff!r}")
    seeds = h.get("n_seeds")
    if isinstance(seeds, bool) or not isinstance(seeds, (int, np.integer)) or seeds < 2:
        problems.append(f"n_seeds: must be an integer >= 2, got {seeds!r}")
    return problems


@app.cell(hide_code=True)
def _():
    checks.run(1, check_hypothesis=check_hypothesis)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def paired_power(sd_diff, effect, n, *, alpha=0.05, alternative="greater"):
    """stage 2: power of a paired t test with n pairs to detect a true mean difference effect (noncentral t)."""
    if n < 2 or not sd_diff > 0:
        raise ValueError(f"need n >= 2 and sd_diff > 0, got n={n}, sd_diff={sd_diff}")
    df, nc = n - 1, effect * np.sqrt(n) / sd_diff
    if alternative == "greater":
        return float(sps.nct.sf(sps.t.ppf(1 - alpha, df), df, nc))
    if alternative == "less":
        return float(sps.nct.cdf(-sps.t.ppf(1 - alpha, df), df, nc))
    if alternative == "two-sided":
        c = sps.t.ppf(1 - alpha / 2, df)
        return float(sps.nct.sf(c, df, nc) + sps.nct.cdf(-c, df, nc))
    raise ValueError(f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}")


@app.function
def seeds_for_power(sd_diff, min_effect, *, alpha=0.05, power=0.8, alternative="greater", max_seeds=1000):
    """stage 2: smallest n >= 2 with paired_power(sd_diff, min_effect, n) >= power; ValueError past max_seeds."""
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}")
    for n in range(2, max_seeds + 1):
        if paired_power(sd_diff, min_effect, n, alpha=alpha, alternative=alternative) >= power:
            return n
    raise ValueError(f"more than {max_seeds} seeds needed to detect {min_effect} with sd {sd_diff}")


@app.cell(hide_code=True)
def _():
    checks.run(2, paired_power=paired_power, seeds_for_power=seeds_for_power)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def run_hours(n_params, n_tokens, *, flops_per_second, utilization=0.4):
    """stage 3: hours to train n_params on n_tokens at 6 N D FLOPs, achieving utilization × flops_per_second."""
    if not (n_params > 0 and n_tokens > 0 and flops_per_second > 0 and 0 < utilization <= 1):
        raise ValueError("n_params, n_tokens and flops_per_second must be positive, utilization in (0, 1]")
    return float(6.0 * n_params * n_tokens / (flops_per_second * utilization) / 3600.0)


@app.function
def largest_model(hours, *, flops_per_second, utilization=0.4, tokens_per_param=20.0):
    """stage 3: (n_params, n_tokens) compute-optimal for the FLOPs in `hours`, via mlfs.dltheory.compute_optimal."""
    if not (hours > 0 and flops_per_second > 0 and 0 < utilization <= 1):
        raise ValueError("hours and flops_per_second must be positive, utilization in (0, 1]")
    n, d = compute_optimal(hours * 3600.0 * flops_per_second * utilization, tokens_per_param=tokens_per_param)
    return float(n), float(d)


@app.cell(hide_code=True)
def _():
    checks.run(3, run_hours=run_hours, largest_model=largest_model)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def budget_plan(experiments, budget_hours, *, safety=2.0):
    """stage 4: dict(rows, total, with_safety, budget, fits): runs and hours per experiment, total × safety vs budget."""
    for x, what in ((budget_hours, "budget_hours"), (safety, "safety")):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"{what} must be a number, got {x!r}")
    if not budget_hours > 0 or not safety >= 1:
        raise ValueError(f"budget_hours must be positive and safety >= 1, got {budget_hours}, {safety}")
    if not isinstance(experiments, (list, tuple)) or not all(isinstance(e, dict) for e in experiments):
        raise ValueError("experiments must be a list of dicts")
    names = [e.get("name") for e in experiments]
    if not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"every experiment needs a non-empty string name, got {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"experiment names must be unique, got {names}")
    rows = []
    for e in experiments:
        configs, seeds, h = e.get("configs"), e.get("seeds"), e.get("hours_per_run")
        if not (isinstance(configs, (int, np.integer)) and configs >= 1 and isinstance(seeds, (int, np.integer))
                and seeds >= 1 and isinstance(h, (int, float)) and h > 0):
            raise ValueError(f"experiment {e.get('name')!r} needs integer configs >= 1, seeds >= 1 and "
                             "hours_per_run > 0")
        runs = int(configs) * int(seeds)
        rows.append({"name": e["name"], "runs": runs, "hours": runs * float(h)})
    total = float(sum(r["hours"] for r in rows))
    return {"rows": rows, "total": total, "with_safety": total * safety, "budget": float(budget_hours),
            "fits": total * safety <= budget_hours}


@app.cell(hide_code=True)
def _():
    checks.run(4, budget_plan=budget_plan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def check_proposal(proposal, *, max_words=1000, alpha=0.05, power=0.8, safety=2.0):
    """stage 5: list of problems with a proposal: sections, length, hypothesis, power and budget; [] means ready."""
    problems = []
    for field in SECTIONS:
        if not isinstance(proposal.get(field), str) or not proposal[field].strip():
            problems.append(f"{field}: missing or empty")
    for field in ("risks", "checkpoints"):
        v = proposal.get(field)
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            problems.append(f"{field}: needs a non-empty list of non-empty strings")
    weeks = proposal.get("timeline_weeks")
    if isinstance(weeks, bool) or not isinstance(weeks, (int, float)) or not weeks > 0:
        problems.append(f"timeline_weeks: must be a positive number, got {weeks!r}")
    words = 0
    for field in SECTIONS:
        if isinstance(proposal.get(field), str):
            words += len(proposal[field].split())
    for field in ("risks", "checkpoints"):
        if isinstance(proposal.get(field), list):
            words += sum(len(x.split()) for x in proposal[field] if isinstance(x, str))
    if words > max_words:
        problems.append(f"length: {words} words, more than {max_words}")
    h = proposal.get("hypothesis")
    if not isinstance(h, dict):
        problems.append("hypothesis: missing")
        h_problems = ["missing"]
    else:
        h_problems = check_hypothesis(h)
        problems += [f"hypothesis.{p}" for p in h_problems]
    sd = proposal.get("pilot_sd")
    if isinstance(sd, bool) or not isinstance(sd, (int, float)) or not sd > 0:
        problems.append(f"pilot_sd: must be a positive number (from a pilot run), got {sd!r}")
    elif not h_problems:
        try:
            need = seeds_for_power(sd, h["min_effect"], alpha=alpha, power=power)
        except ValueError:
            need = None
        if need is None or need > h["n_seeds"]:
            problems.append(f"power: detecting {h['min_effect']} with pilot sd {sd} at {power:.0%} power needs "
                            f"{need if need else 'more than 1000'} seeds, the plan has {h['n_seeds']}")
    exps, budget = proposal.get("experiments"), proposal.get("budget_hours")
    try:
        plan = budget_plan(exps, budget, safety=safety)
    except (ValueError, TypeError) as err:
        problems.append(f"budget: {err}")
    else:
        if not plan["fits"]:
            problems.append(f"budget: {plan['total']:g} h × safety {safety:g} = {plan['with_safety']:g} h, more "
                            f"than the {budget:g} h available")
    return problems


@app.cell(hide_code=True)
def _():
    checks.run(5, check_proposal=check_proposal)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then write your own
    two-page proposal as a dict (or load it from a file) and get `check_proposal` to return
    `[]` before you show it to anyone.
    """)
    return


if __name__ == "__main__":
    app.run()
