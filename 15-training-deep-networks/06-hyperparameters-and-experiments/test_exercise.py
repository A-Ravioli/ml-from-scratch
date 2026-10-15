"""Checks for exercise.py (hyperparameters & experiments).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The searches are run on
cheap deterministic objectives whose answers are known, and checked exactly: which configs are
sampled (from the generator, in order), which budgets are spent (Li et al.'s bracket arithmetic),
which configs are promoted, and what is returned. Every search runs under `mlfs.testing.bounded`
with a counting objective, so a loop that never ends fails instead of hanging. The experiment log is
checked in memory and through a JSON-lines file in a temporary directory.
"""
import json
import math
from collections import Counter

import numpy as np
import pytest

from mlfs.testing import bounded


def _sample(rng):
    return {"x": float(rng.uniform(-5, 5))}


def _objective(max_calls=5000):
    """loss (x - 1)^2 + 1/budget: more budget is always better, and the order of configs never changes."""
    calls = []

    def evaluate(config, budget):
        calls.append((config["x"], budget))
        if len(calls) > max_calls:
            raise AssertionError(f"evaluate was called more than {max_calls} times: is a loop missing its stopping rule?")
        return (config["x"] - 1.0) ** 2 + 1.0 / budget

    return evaluate, calls


def _run(fn, *args, **kwargs):
    # a lambda, because the searches have their own `budget` keyword
    return bounded(lambda: fn(*args, **kwargs), budget=3_000_000, what=fn.__name__,
                   hint="every rung shrinks; stop when it can't.")


# ---------------------------------------------------------------- stage 1: random search


def test_stage1_samples_in_order_and_keeps_the_best(nb):
    evaluate, calls = _objective()
    best, loss, history = _run(nb.random_search, _sample, evaluate, n_trials=10, budget=4, rng=0)
    r = np.random.default_rng(0)
    want = [r.uniform(-5, 5) for _ in range(10)]
    assert [c["x"] for c, _, _ in history] == pytest.approx(want), (
        "draw the configs with sample_config(rng), one per trial, in order, from ONE generator as_rng(rng)")
    assert all(b == 4 for _, b, _ in history) and len(calls) == 10, "every trial is evaluated once, at `budget`"
    i = int(np.argmin([(x - 1) ** 2 for x in want]))
    assert best == history[i][0] and math.isclose(loss, (want[i] - 1) ** 2 + 0.25), (
        f"the best config is the one with the lowest loss ({want[i]:.3f}, closest to 1); return (config, loss, history)")
    assert all(len(h) == 3 for h in history), "history entries are (config, budget, loss) tuples"


def test_stage1_framework_agnostic_losses(nb):
    best, loss, history = _run(nb.random_search, lambda rng: {"a": int(rng.integers(0, 3))},
                               lambda c, b: np.float32(c["a"]), n_trials=6, rng=1)
    assert isinstance(loss, float), "losses are stored as Python floats (float(evaluate(...)))"
    assert best["a"] == min(c["a"] for c, _, _ in history)


# ---------------------------------------------------------------- stage 2: successive halving


def test_stage2_run_bracket_promotes_the_best(nb):
    configs = [{"x": x} for x in [3.0, 1.1, -2.0, 0.5, 4.0, 1.0, 2.0, -1.0, 0.0]]
    evaluate, calls = _objective()
    history = []
    last = _run(nb.run_bracket, configs, evaluate, n_rungs=3, min_budget=2, eta=3, history=history)
    budgets = [b for _, b in calls]
    assert budgets == [2] * 9 + [6] * 3 + [18], (
        f"rung i runs at min_budget * eta**i; 9 configs, eta = 3: 9 at budget 2, the best 3 at 6, the best 1 at 18; "
        f"got the budgets {budgets}")
    assert [x for x, b in calls if b == 6] == [1.0, 1.1, 0.5], (
        "the survivors of a rung are its best floor(n / eta) configs, best first: (x - 1)^2 is smallest for 1.0, 1.1, 0.5")
    assert last == [({"x": 1.0}, pytest.approx(1 / 18))], "run_bracket returns the last rung as [(config, loss)]"
    assert len(history) == 13 and history[-1] == ({"x": 1.0}, 18, pytest.approx(1 / 18)), (
        "every evaluation is appended to `history` as (config, budget, loss)")


def test_stage2_successive_halving_budgets(nb):
    evaluate, calls = _objective()
    best, loss, history = _run(nb.successive_halving, _sample, evaluate, n_configs=27, min_budget=1, eta=3, rng=0)
    counts = Counter(b for _, b in calls)
    assert counts == {1: 27, 3: 9, 9: 3, 27: 1}, (
        f"27 configs, eta = 3: rungs of 27, 9, 3 and 1 configs at budgets 1, 3, 9, 27 (until one is left); got {dict(counts)}")
    r = np.random.default_rng(0)
    xs = [r.uniform(-5, 5) for _ in range(27)]
    winner = min(xs, key=lambda x: (x - 1) ** 2)
    assert math.isclose(best["x"], winner) and math.isclose(loss, (winner - 1) ** 2 + 1 / 27), (
        "the best config is the survivor of the last rung, with its loss at the largest budget")
    assert len(history) == 40
    late = lambda c, b: (c["x"] - 1.0) ** 2 + b / 100.0  # more budget looks worse: early losses are all lower
    best2, loss2, hist2 = _run(nb.successive_halving, _sample, late, n_configs=27, min_budget=1, eta=3, rng=0)
    last = [(c, l) for c, b, l in hist2 if b == 27]
    assert len(last) == 1 and best2 is last[0][0] and loss2 == last[0][1], (
        "successive halving returns the survivor of the LAST rung with its loss at the largest budget, not the lowest "
        "loss anywhere in the history (losses at different budgets aren't comparable)")


def test_stage2_uneven_counts(nb):
    evaluate, calls = _objective()
    _run(nb.successive_halving, _sample, evaluate, n_configs=10, min_budget=2, eta=3, rng=1)
    assert Counter(b for _, b in calls) == {2: 10, 6: 3, 18: 1}, (
        "10 configs, eta = 3: keep floor(10/3) = 3, then floor(3/3) = 1; budgets 2, 6, 18")
    evaluate, calls = _objective()
    _run(nb.successive_halving, _sample, evaluate, n_configs=2, min_budget=1, eta=3, rng=1)
    assert Counter(b for _, b in calls) == {1: 2}, "2 configs and eta = 3: floor(2/3) = 0, so one rung only"


# ---------------------------------------------------------------- stage 3: hyperband


def test_stage3_bracket_arithmetic(nb):
    evaluate, calls = _objective()
    best, loss, history = _run(nb.hyperband, _sample, evaluate, max_budget=81, eta=3, rng=0)
    counts = Counter(b for _, b in calls)
    want = {1: 81, 3: 27 + 34, 9: 9 + 11 + 15, 27: 3 + 3 + 5 + 8, 81: 1 + 1 + 1 + 2 + 5}
    assert counts == want, (
        f"max_budget = 81, eta = 3: s_max = 4 and five brackets s = 4..0 with n = ceil(5 * 3^s / (s + 1)) = 81, 34, "
        f"15, 8, 5 configs starting at budgets 1, 3, 9, 27, 81, each run as successive halving up to 81. "
        f"evaluations per budget: want {want}, got {dict(counts)}")
    assert len(calls) == 206 and math.isclose(sum(b for _, b in calls), 1902), "206 evaluations, 1902 budget units in all"
    finals = [(c, l) for c, b, l in history if b == 81]
    assert loss == min(l for _, l in finals) and best in [c for c, _ in finals], (
        "hyperband returns the best config evaluated at max_budget (losses at different budgets aren't comparable)")


def test_stage3_sampling_order_and_small_budgets(nb):
    evaluate, calls = _objective()
    _run(nb.hyperband, _sample, evaluate, max_budget=9, eta=3, rng=5)
    r = np.random.default_rng(5)
    first_bracket = [r.uniform(-5, 5) for _ in range(9)]
    assert [x for x, b in calls[:9]] == pytest.approx(first_bracket), (
        "brackets run from the most aggressive (s = s_max, the most configs) to s = 0; each bracket samples its configs "
        "from the same generator when it starts")
    assert Counter(b for _, b in calls) == {1: 9, 3: 3 + 5, 9: 1 + 1 + 3}, (
        "max_budget = 9: brackets (9 configs from 1), (5 from 3), (3 from 9)")
    evaluate, calls = _objective()
    _run(nb.hyperband, _sample, evaluate, max_budget=10, eta=3, rng=0)
    assert max(b for _, b in calls) == pytest.approx(10) and min(b for _, b in calls) == pytest.approx(10 / 9), (
        "when max_budget isn't a power of eta, rungs start at max_budget / eta^s (10/9 here) and still end at max_budget")


def test_stage3_compares_only_full_budget_losses(nb):
    def overfits(config, budget):  # more budget makes every config look WORSE here
        return (config["x"] - 1.0) ** 2 + budget / 10.0

    best, loss, history = _run(nb.hyperband, _sample, overfits, max_budget=9, eta=3, rng=3)
    full = [(c, l) for c, b, l in history if b == 9]
    assert loss == min(l for _, l in full) and any(best is c for c, _ in full), (
        "hyperband's answer must be the best loss among the evaluations at max_budget; a loss measured at a smaller "
        f"budget isn't comparable (here early losses are all lower). got {loss:.3f}, the best full-budget loss is "
        f"{min(l for _, l in full):.3f}")


def test_stage3_finds_the_optimum_cheaply(nb):
    evaluate, calls = _objective()
    best, loss, _ = _run(nb.hyperband, _sample, evaluate, max_budget=27, eta=3, rng=2)
    spent = sum(b for _, b in calls)
    n_configs = len({x for x, _ in calls})
    closest = min({x for x, _ in calls}, key=lambda x: abs(x - 1))
    assert best["x"] == closest, (
        f"on this objective the ranking never changes with the budget, so hyperband must return the config closest to 1 "
        f"of all {n_configs} it tried ({closest:.3f}); got {best['x']:.3f}")
    assert spent < n_configs * 27 / 3, (
        f"hyperband tried {n_configs} configs for {spent:.0f} budget units: far less than evaluating them all at the "
        "full budget would cost")


# ---------------------------------------------------------------- stage 4: ExperimentLog


def test_stage4_log_best_group(nb):
    log = nb.ExperimentLog()
    cfg = {"lr": 0.1}
    rec = log.log(cfg, {"val_acc": 0.9, "loss": 0.3}, seed=0)
    cfg["lr"] = 99  # mutating the caller's dict afterwards must not change the record
    assert rec == {"id": 0, "config": {"lr": 0.1}, "seed": 0, "metrics": {"val_acc": 0.9, "loss": 0.3}}, (
        f"a record is {{'id', 'config', 'seed', 'metrics'}} with COPIES of the dicts; got {rec}")
    log.log({"lr": 0.1}, {"val_acc": 0.8, "loss": 0.5}, seed=1)
    log.log({"lr": 0.01}, {"val_acc": 0.95, "loss": 0.2}, seed=0)
    assert [r["id"] for r in log.runs] == [0, 1, 2]
    assert log.best("loss")["id"] == 2 and log.best("val_acc", mode="max")["id"] == 2
    assert log.best("val_acc")["id"] == 1, "mode='min' (the default) picks the lowest value"
    groups = log.group("val_acc")
    assert [g["config"] for g in groups] == [{"lr": 0.1}, {"lr": 0.01}], "one group per distinct config, first-seen order"
    g = groups[0]
    assert g["n"] == 2 and math.isclose(g["mean"], 0.85) and math.isclose(g["std"], np.std([0.9, 0.8], ddof=1)), (
        f"mean and the sample std (ddof=1) across the runs of a config; got {g}")
    assert groups[1]["std"] == 0.0, "a single run has std 0.0"
    with pytest.raises(ValueError):
        log.best("loss", mode="median")


def test_stage4_group_ignores_key_order(nb):
    log = nb.ExperimentLog()
    log.log({"a": 1, "b": 2}, {"m": 1.0})
    log.log({"b": 2, "a": 1}, {"m": 3.0})
    groups = log.group("m")
    assert len(groups) == 1 and groups[0]["n"] == 2, "{'a': 1, 'b': 2} and {'b': 2, 'a': 1} are the same config"


def test_stage4_jsonl_file(nb, tmp_path):
    path = tmp_path / "runs.jsonl"
    log = nb.ExperimentLog(path)
    log.log({"lr": 0.1, "width": 64}, {"loss": np.float32(0.25)}, seed=3)
    log.log({"lr": 0.2, "width": 64}, {"loss": 0.5}, seed=4)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["config"] == {"lr": 0.1, "width": 64}, (
        "with a path, every logged record is appended to the file as one JSON line")
    assert json.loads(lines[0])["metrics"]["loss"] == 0.25, "metrics are stored as plain floats (json can't save np.float32)"
    again = nb.ExperimentLog(path)
    assert again.runs == log.runs, "opening an existing file loads its runs"
    again.log({"lr": 0.3}, {"loss": 0.1})
    assert again.runs[-1]["id"] == 2 and len(path.read_text().strip().splitlines()) == 3, (
        "a reopened log continues the ids and appends to the same file")
