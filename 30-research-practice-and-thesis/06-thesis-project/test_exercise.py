"""Checks for 30 · 06 thesis project.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. topo_order and critical_path on a
hand-built thesis plan (starts, finishes, slack, the critical path and its tie-breaking), cycles without hanging,
and refusals; simulate_makespan on hand-picked durations (makespan and which tasks are critical in each
simulation); schedule_risk against the triangular distribution's exact CDF and quantiles, parallel and serial
plans, criticality and its rng/n_sims handling; forecast against hand-computable resampling cases.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

PLAN = {
    "read": {"weeks": 4},
    "reproduce": {"weeks": 6, "after": ["read"]},
    "pilot": {"weeks": 2, "after": ["read"]},
    "experiments": {"weeks": 8, "after": ["reproduce", "pilot"]},
    "write": {"weeks": 4, "after": ["experiments"]},
    "figures": {"weeks": 1, "after": ["experiments"]},
}


def _cp(nb, *args, **kwargs):
    return bounded(nb.critical_path, *args, what="critical_path", **kwargs)


def _risk(nb, *args, **kwargs):
    return bounded(nb.schedule_risk, *args, what="schedule_risk", hint="vectorize over simulations.", **kwargs)


def _fc(nb, *args, **kwargs):
    return bounded(nb.forecast, *args, what="forecast", hint="vectorize over simulations.", **kwargs)


def _sim(nb, *args, **kwargs):
    return bounded(nb.simulate_makespan, *args, what="simulate_makespan", hint="stop when no task is ready.",
                   **kwargs)


def _order(nb, tasks):
    return bounded(nb.topo_order, tasks, what="topo_order",
                   hint="stop when no task is ready; a cycle means some tasks never become ready.")


# ---------------------------------------------------------------- stage 1: topo_order, critical_path


def test_stage1_topo_order(nb):
    got = _order(nb, PLAN)
    assert got == ["read", "pilot", "reproduce", "experiments", "figures", "write"], (
        f"every task after the tasks it comes after, the alphabetically first ready task at each step: expected "
        f"read, pilot, reproduce, experiments, figures, write; got {got}"
    )
    assert _order(nb, {"c": {}, "a": {}, "d": {}, "b": {}}) == ["a", "b", "c", "d"], (
        "independent tasks come out alphabetically (use a heap or sort the ready set every time)"
    )
    rng = np.random.default_rng(0)
    for trial in range(10):
        n = int(rng.integers(5, 30))
        names = [f"t{i:02d}" for i in rng.permutation(n)]
        tasks = {names[i]: {"after": [names[j] for j in rng.choice(i, size=min(i, int(rng.integers(0, 3))),
                                                                   replace=False)] if i else []}
                 for i in range(n)}
        got = _order(nb, tasks)
        done = set()
        for k in got:
            ready = sorted(t for t in tasks if t not in done and all(d in done for d in tasks[t]["after"]))
            assert ready and k == ready[0], f"take the alphabetically first ready task: {ready[:1]}, got {k!r}"
            done.add(k)
    assert _order(nb, {"b": {"weeks": 1}, "a": {"weeks": 1, "after": ["b", "b"]}}) == ["b", "a"], (
        "a repeated dependency counts once"
    )
    with pytest.raises(ValueError) as err:
        _order(nb, {"x": {"weeks": 1, "after": ["z"]}, "y": {"weeks": 1, "after": ["x"]},
                    "z": {"weeks": 1, "after": ["y"]}, "w": {"weeks": 1}})
    assert all(k in str(err.value) for k in "xyz"), f"name the tasks in the cycle, got {err.value}"
    with pytest.raises(ValueError):
        _order(nb, {"a": {"weeks": 1, "after": ["ghost"]}})


def test_stage1_worked_example(nb):
    got = _cp(nb, PLAN)
    assert set(got) >= {"start", "finish", "slack", "total", "path"}, f"missing keys, got {sorted(got)}"
    assert got["start"] == {"read": 0, "reproduce": 4, "pilot": 4, "experiments": 10, "write": 18, "figures": 18}, (
        f"each task starts when the last task it comes after finishes: got {got['start']}"
    )
    assert got["finish"] == {"read": 4, "reproduce": 10, "pilot": 6, "experiments": 18, "write": 22, "figures": 19}
    assert got["total"] == 22, f"the plan takes 22 weeks, got {got['total']}"
    assert got["slack"] == {"read": 0, "reproduce": 0, "pilot": 4, "experiments": 0, "write": 0, "figures": 3}, (
        f"slack = latest finish - earliest finish: the pilot can slip 4 weeks (it only has to finish by week 10), "
        f"the figures 3; got {got['slack']}"
    )
    assert got["path"] == ["read", "reproduce", "experiments", "write"], (
        f"the critical path, from the first task to the last: got {got['path']}"
    )


def test_stage1_ties_and_duplicates(nb):
    got = _cp(nb, {"b": {"weeks": 2}, "a": {"weeks": 2}, "c": {"weeks": 1, "after": ["a", "b"]}})
    assert got["path"] == ["a", "c"] and got["slack"]["b"] == 0, (
        f"when two predecessors are both critical, follow the alphabetically first: expected ['a', 'c'], got "
        f"{got['path']}"
    )
    got = _cp(nb, {"y": {"weeks": 3}, "x": {"weeks": 3}})
    assert got["path"] == ["x"] and got["total"] == 3, "two tasks ending the plan together: the path ends at 'x'"
    got = _cp(nb, {"a": {"weeks": 1.5}, "b": {"weeks": 2.25, "after": ["a", "a"]}})
    assert np.isclose(got["total"], 3.75) and got["path"] == ["a", "b"], "fractional weeks and repeated deps"


def test_stage1_refusals(nb):
    for weeks in (0, -1, True, "2", None):
        task = {"weeks": weeks} if weeks is not None else {}
        with pytest.raises(ValueError):
            _cp(nb, {"a": {"weeks": 1}, "b": {**task, "after": ["a"]}})
    with pytest.raises(ValueError):
        bounded(nb.critical_path, {"a": {"weeks": 1, "after": ["b"]}, "b": {"weeks": 1, "after": ["a"]}},
                what="critical_path")


# ---------------------------------------------------------------- stage 2: simulate_makespan, schedule_risk

FORK = {"a": {"after": []}, "b": {}, "c": {"after": ["a", "b"]}}


def test_stage2_simulate_makespan_by_hand(nb):
    makespan, critical = _sim(nb, FORK, {"a": [3.0, 1.0, 2.0], "b": [1.0, 3.0, 1.0], "c": [1.0, 1.0, 5.0]})
    assert np.allclose(makespan, [4.0, 4.0, 7.0]), f"makespans are max(a, b) + c per simulation, got {makespan}"
    assert list(np.asarray(critical["a"])) == [True, False, True], "a is critical when it's the longer branch"
    assert list(np.asarray(critical["b"])) == [False, True, False]
    assert list(np.asarray(critical["c"])) == [True, True, True], "the final task is always critical"
    with pytest.raises(ValueError):
        _sim(nb, FORK, {"a": [1.0], "b": [1.0]})
    with pytest.raises(ValueError):
        _sim(nb, FORK, {"a": [1.0, 2.0], "b": [1.0], "c": [1.0, 1.0]})


def test_stage2_triangular_exact_values(nb):
    got = _risk(nb, {"t": {"estimate": (1, 2, 4)}}, 2.0, n_sims=40000, rng=0)
    assert abs(got["p_on_time"] - 1 / 3) < 0.012, (
        f"for a triangular(1, 2, 4) task, P(done by the likely value 2) = (2 - 1) / (4 - 1) = 1/3, got "
        f"{got['p_on_time']:.3f}. the likely value is the mode, and the right tail makes the mean later."
    )
    assert abs(got["quantiles"][0.5] - (4 - np.sqrt(3))) < 0.03, (
        f"its median is 4 - √3 = 2.268 weeks, got {got['quantiles'][0.5]:.3f}"
    )
    assert set(got["quantiles"]) == {0.5, 0.8, 0.95} and got["quantiles"][0.5] < got["quantiles"][0.8] < \
        got["quantiles"][0.95]
    assert got["likely_total"] == 2.0 and got["criticality"] == {"t": 1.0}


def test_stage2_parallel_and_serial(nb):
    par = _risk(nb, {"a": {"estimate": (1, 2, 4)}, "b": {"estimate": (1, 2, 4)}}, 2.0, n_sims=40000, rng=1)
    assert abs(par["p_on_time"] - 1 / 9) < 0.01, (
        f"two independent parallel tasks both done by 2: (1/3)² = 1/9, got {par['p_on_time']:.3f}"
    )
    assert abs(par["criticality"]["a"] - 0.5) < 0.02, "each of two identical parallel tasks is critical half the time"
    ser = _risk(nb, {"a": {"estimate": (1, 2, 4)}, "b": {"estimate": (1, 2, 4), "after": ["a"]}}, 4.0,
                           n_sims=40000, rng=2)
    rng = np.random.default_rng(99)
    ref = np.mean(rng.triangular(1, 2, 4, 400000) + rng.triangular(1, 2, 4, 400000) <= 4.0)
    assert abs(ser["p_on_time"] - ref) < 0.012 and ser["likely_total"] == 4.0, (
        f"two tasks in a row, deadline = the sum of the likely values: P ≈ {ref:.3f}, got {ser['p_on_time']:.3f}"
    )
    lop = _risk(nb, {"a": {"estimate": (1, 2, 3)}, "b": {"estimate": (5, 6, 7)}}, 10.0, n_sims=2000, rng=3)
    assert lop["criticality"] == {"a": 0.0, "b": 1.0}, f"b always takes longer than a, got {lop['criticality']}"


def test_stage2_draw_order_and_quantiles(nb):
    tasks = {"z": {"estimate": (1, 2, 6)}, "a": {"estimate": (1, 3, 4), "after": ["z"]}, "m": {"estimate": (2, 2, 5)}}
    got = _risk(nb, tasks, 5.0, n_sims=1000, rng=11)
    rng = np.random.default_rng(11)
    dz, da, dm = (rng.triangular(o, m, p, size=1000) for o, m, p in ((1, 2, 6), (1, 3, 4), (2, 2, 5)))
    makespan = np.maximum(dz + da, dm)
    assert np.isclose(got["p_on_time"], np.mean(makespan <= 5.0)), (
        "draw the durations in the order of `tasks` (z, a, m here), not sorted or topological order: the same seed "
        "must give everyone the same simulations"
    )
    want = {q: float(np.quantile(makespan, q)) for q in (0.5, 0.8, 0.95)}
    assert all(np.isclose(got["quantiles"][q], want[q]) for q in want), (
        f"quantiles are np.quantile's defaults (linear interpolation): expected {want}, got {got['quantiles']}"
    )


def test_stage2_fixed_estimates_rng_and_n_sims(nb):
    tasks = {"a": {"estimate": (2, 2, 2)}, "b": {"estimate": (3, 3, 3), "after": ["a"]}}
    got = _risk(nb, tasks, 5.0, n_sims=50, rng=0)
    assert got["p_on_time"] == 1.0 and all(np.isclose(v, 5.0) for v in got["quantiles"].values()), (
        f"optimistic = likely = pessimistic means a fixed duration: 5 weeks every time, got {got}"
    )
    assert _risk(nb, tasks, 4.99, n_sims=50, rng=0)["p_on_time"] == 0.0
    risky = {"a": {"estimate": (1, 2, 6)}, "b": {"estimate": (1, 3, 5), "after": ["a"]}}
    r1 = _risk(nb, risky, 5.0, n_sims=500, rng=7)
    r2 = _risk(nb, risky, 5.0, n_sims=500, rng=np.random.default_rng(7))
    assert r1 == r2, "the same seed (as an int or a Generator) must give the same result"
    p = _risk(nb, risky, 5.0, n_sims=10, rng=0)["p_on_time"]
    assert np.isclose(p * 10, round(p * 10)), f"with n_sims=10, p_on_time is a multiple of 0.1, got {p}"
    for bad in ((2, 1, 3), (0, 1, 2), (1, 2), (1, 3, 2)):
        with pytest.raises(ValueError):
            _risk(nb, {"a": {"estimate": bad}}, 3.0, n_sims=10, rng=0)


# ---------------------------------------------------------------- stage 3: forecast

CHAIN = {"a": {"estimate": 1.0}, "b": {"estimate": 1.0, "after": ["a"]}}


def test_stage3_constant_ratios(nb):
    tasks = {"read": {"estimate": 4.0}, "write": {"estimate": 2.0, "after": ["read"]},
             "figs": {"estimate": 1.0, "after": ["read"]}}
    got = _fc(nb, tasks, [(1.0, 1.0), (3.0, 3.0)], 6.0, n_sims=200, rng=0)
    assert set(got) >= {"p_on_time", "quantiles", "naive_total", "median_ratio"}, f"missing keys: {sorted(got)}"
    assert got["naive_total"] == 6.0 and got["median_ratio"] == 1.0 and got["p_on_time"] == 1.0, (
        f"a perfectly calibrated history (actual = estimate) forecasts the plan itself, got {got}"
    )
    got = _fc(nb, tasks, [(2.0, 4.0), (1.0, 2.0)], 6.0, n_sims=200, rng=0)
    assert got["p_on_time"] == 0.0 and all(np.isclose(v, 12.0) for v in got["quantiles"].values()), (
        f"if every past task took twice its estimate, the forecast is twice the plan: 12 weeks, got {got}"
    )
    assert got["median_ratio"] == 2.0


def test_stage3_resampling(nb):
    got = _fc(nb, {"t": {"estimate": 4.0}}, [(1.0, 1.0), (1.0, 2.0)], 6.0, n_sims=40000, rng=1)
    assert abs(got["p_on_time"] - 0.5) < 0.012, (
        f"ratios 1 and 2, each drawn with probability 1/2: the 4-week task takes 4 or 8 weeks, so P(≤ 6) = 0.5; "
        f"got {got['p_on_time']:.3f}"
    )
    got = _fc(nb, CHAIN, [(1.0, 1.0), (1.0, 3.0)], 2.0, n_sims=40000, rng=2)
    assert abs(got["p_on_time"] - 0.25) < 0.012, (
        "two tasks in a row each draw their own ratio: both must draw 1 to finish by 2 weeks, P = 1/4; got "
        f"{got['p_on_time']:.3f}. draw a separate ratio for each task in each simulation."
    )
    assert np.isclose(got["median_ratio"], 2.0), "the median of ratios (1, 3) is 2"
    a = _fc(nb, CHAIN, [(1.0, 1.5), (2.0, 1.0), (1.0, 4.0)], 3.0, n_sims=300, rng=5)
    b = _fc(nb, CHAIN, [(1.0, 1.5), (2.0, 1.0), (1.0, 4.0)], 3.0, n_sims=300, rng=np.random.default_rng(5))
    assert a == b, "the same seed must give the same forecast"
    assert np.isclose(a["median_ratio"], 1.5), "ratios (1.5, 0.5, 4): the median is 1.5 (not the mean, 2)"


def test_stage3_refusals(nb):
    with pytest.raises(ValueError):
        _fc(nb, CHAIN, [], 3.0, n_sims=10, rng=0)
    with pytest.raises(ValueError):
        _fc(nb, CHAIN, [(1.0, 0.0)], 3.0, n_sims=10, rng=0)
    with pytest.raises(ValueError):
        _fc(nb, CHAIN, [(0.0, 1.0)], 3.0, n_sims=10, rng=0)
    with pytest.raises(ValueError):
        _fc(nb, {"a": {"estimate": (1, 2, 3)}}, [(1.0, 1.0)], 3.0, n_sims=10, rng=0)
