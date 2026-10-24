import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # thesis project · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make a planner for a project whose tasks you can't
    estimate exactly: the critical path of a plan, the probability of finishing it by a
    deadline, and a forecast calibrated on how wrong your own past estimates were.

    | stage | you build |
    |---|---|
    | 1 | `topo_order`, `critical_path`: earliest finishes, slack and the critical path of a task graph |
    | 2 | `simulate_makespan`, `schedule_risk`: Monte Carlo finish dates from three-point estimates |
    | 3 | `forecast`: a schedule forecast resampled from your own estimate-to-actual history |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def topo_order(tasks):
    """stage 1: task names, each after all the tasks it comes after (ties alphabetical); ValueError on cycles."""
    deps = {}
    for name, t in tasks.items():
        after = set(t.get("after", []))
        unknown = sorted(after - set(tasks))
        if unknown:
            raise ValueError(f"task {name!r} comes after unknown tasks {unknown}")
        deps[name] = after
    waiting = {k: len(v) for k, v in deps.items()}
    unlocks = {k: [] for k in tasks}
    for k, v in deps.items():
        for d in v:
            unlocks[d].append(k)
    ready = [k for k, c in waiting.items() if c == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(k)
        for nxt in unlocks[k]:
            waiting[nxt] -= 1
            if waiting[nxt] == 0:
                heapq.heappush(ready, nxt)
    if len(order) < len(tasks):
        raise ValueError(f"the plan has a cycle among {sorted(set(tasks) - set(order))}")
    return order


@app.function
def critical_path(tasks):
    """stage 1: dict(start, finish, slack, total, path) for tasks {name: {'weeks', 'after'}} started as early as possible."""
    order = topo_order(tasks)
    for name in order:
        w = tasks[name].get("weeks")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not w > 0:
            raise ValueError(f"task {name!r} needs a positive duration in weeks, got {w!r}")
    start, finish = {}, {}
    for name in order:
        start[name] = max((finish[d] for d in set(tasks[name].get("after", []))), default=0.0)
        finish[name] = start[name] + float(tasks[name]["weeks"])
    total = max(finish.values(), default=0.0)
    succ = {k: [] for k in tasks}
    for k in tasks:
        for d in set(tasks[k].get("after", [])):
            succ[d].append(k)
    latest = {}
    for name in reversed(order):
        latest[name] = min((latest[s] - tasks[s]["weeks"] for s in succ[name]), default=total)
    slack = {k: latest[k] - finish[k] for k in order}
    path = []
    if order:
        cur = min((k for k in order if np.isclose(finish[k], total)))
        while cur is not None:
            path.append(cur)
            prev = [d for d in set(tasks[cur].get("after", [])) if np.isclose(finish[d], start[cur])]
            cur = min(prev) if prev else None
        path.reverse()
    return {"start": {k: start[k] for k in order}, "finish": {k: finish[k] for k in order}, "slack": slack,
            "total": total, "path": path}


@app.cell(hide_code=True)
def _():
    checks.run(1, topo_order=topo_order, critical_path=critical_path)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def simulate_makespan(tasks, durations):
    """stage 2: (makespan (n_sims,), critical {name: bool (n_sims,)}) for sampled durations {name: (n_sims,)}."""
    order = topo_order(tasks)
    if set(durations) != set(tasks):
        raise ValueError("durations must have exactly one array per task")
    dur = {k: np.asarray(durations[k], dtype=float) for k in order}
    n = len(dur[order[0]]) if order else 0
    if any(v.shape != (n,) for v in dur.values()):
        raise ValueError("every task's durations must be a 1-D array of the same length")
    finish = {}
    for k in order:
        start = np.zeros(n)
        for d in set(tasks[k].get("after", [])):
            start = np.maximum(start, finish[d])
        finish[k] = start + dur[k]
    makespan = np.max(np.stack([finish[k] for k in order]), axis=0) if order else np.zeros(0)
    succ = {k: [] for k in tasks}
    for k in tasks:
        for d in set(tasks[k].get("after", [])):
            succ[d].append(k)
    latest = {}
    for k in reversed(order):
        latest[k] = makespan.copy()
        for s in succ[k]:
            latest[k] = np.minimum(latest[k], latest[s] - dur[s])
    tol = 1e-9 * np.maximum(1.0, makespan)
    critical = {k: np.abs(latest[k] - finish[k]) <= tol for k in order}
    return makespan, critical


@app.function
def schedule_risk(tasks, deadline, *, n_sims=10000, rng=None):
    """stage 2: dict(p_on_time, quantiles, criticality, likely_total) from triangular (o, m, p) estimates."""
    rng = as_rng(rng)
    order = list(tasks)
    durations = {}
    for name in order:
        est = tasks[name].get("estimate")
        if not (isinstance(est, (tuple, list)) and len(est) == 3):
            raise ValueError(f"task {name!r} needs an (optimistic, likely, pessimistic) estimate")
        o, m, p = (float(x) for x in est)
        if not 0 < o <= m <= p:
            raise ValueError(f"task {name!r}: need 0 < optimistic <= likely <= pessimistic, got {est}")
        durations[name] = np.full(n_sims, o) if o == p else rng.triangular(o, m, p, size=n_sims)
    makespan, critical = simulate_makespan(tasks, durations)
    likely = critical_path({k: {"weeks": float(tasks[k]["estimate"][1]), "after": tasks[k].get("after", [])}
                            for k in order})["total"]
    return {"p_on_time": float(np.mean(makespan <= deadline)),
            "quantiles": {q: float(np.quantile(makespan, q)) for q in (0.5, 0.8, 0.95)},
            "criticality": {k: float(np.mean(critical[k])) for k in order},
            "likely_total": float(likely)}


@app.cell(hide_code=True)
def _():
    checks.run(2, simulate_makespan=simulate_makespan, schedule_risk=schedule_risk)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def forecast(tasks, history, deadline, *, n_sims=10000, rng=None):
    """stage 3: dict(p_on_time, quantiles, naive_total, median_ratio): each estimate × a ratio resampled from history."""
    rng = as_rng(rng)
    pairs = [(float(e), float(a)) for e, a in history]
    if not pairs:
        raise ValueError("history is empty: log some (estimate, actual) pairs first")
    if any(not (e > 0 and a > 0) for e, a in pairs):
        raise ValueError("estimates and actuals in the history must be positive")
    ratios = np.array([a / e for e, a in pairs])
    order = list(tasks)
    durations = {}
    for name in order:
        est = tasks[name].get("estimate")
        if isinstance(est, bool) or not isinstance(est, (int, float)) or not est > 0:
            raise ValueError(f"task {name!r} needs a positive single-number estimate, got {est!r}")
        durations[name] = est * rng.choice(ratios, size=n_sims, replace=True)
    makespan, _ = simulate_makespan(tasks, durations)
    naive = critical_path({k: {"weeks": float(tasks[k]["estimate"]), "after": tasks[k].get("after", [])}
                           for k in order})["total"]
    return {"p_on_time": float(np.mean(makespan <= deadline)),
            "quantiles": {q: float(np.quantile(makespan, q)) for q in (0.5, 0.8, 0.95)},
            "naive_total": float(naive), "median_ratio": float(np.median(ratios))}


@app.cell(hide_code=True)
def _():
    checks.run(3, forecast=forecast)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then write your thesis plan
    as a task graph with three-point estimates, and keep a log of (estimate, actual) pairs as
    you finish tasks: after a month, `forecast` tells you when you'll really be done.
    """)
    return


if __name__ == "__main__":
    app.run()
