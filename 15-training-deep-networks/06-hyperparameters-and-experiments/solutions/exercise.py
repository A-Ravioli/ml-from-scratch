import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import math
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hyperparameters & experiments · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. every topic so far added knobs: init,
    normalization, dropout, weight decay, learning rate, schedule, batch size. this topic
    builds the machinery for turning them systematically: random search, successive halving
    and hyperband (which spend most of the compute on the promising configurations), and an
    experiment log that keeps every run, so that the comparison at the end is honest. the
    searches are framework-agnostic and join **`mlfs.train`**.

    | stage | you build |
    |---|---|
    | 1 | `random_search`: sample, evaluate, keep the best |
    | 2 | `run_bracket` and `successive_halving`: evaluate many cheaply, promote the best |
    | 3 | `hyperband`: successive halving at several aggressiveness levels |
    | 4 | `ExperimentLog`: record every run, find the best, and average over seeds |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def random_search(sample_config, evaluate, *, n_trials=20, budget=1, rng=None):
    """stage 1: n_trials configs from sample_config(rng), each scored by evaluate(config, budget); (best_config, best_loss, history)."""
    rng = as_rng(rng)
    history = []
    for _ in range(n_trials):
        config = sample_config(rng)
        history.append((config, budget, float(evaluate(config, budget))))
    best = min(history, key=lambda h: h[2]) if history else (None, None, math.inf)
    return best[0], best[2], history


@app.cell(hide_code=True)
def _():
    checks.run(1, random_search=random_search)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def rung_budget(min_budget, eta, i):
    """the budget of rung i, min_budget * eta**i, as an int when it is a whole number (given)."""
    b = min_budget * eta ** i
    return int(round(b)) if abs(b - round(b)) < 1e-9 else b


@app.function
def run_bracket(configs, evaluate, *, n_rungs, min_budget, eta, history):
    """stage 2: rung i scores the survivors at rung_budget(min_budget, eta, i), keeping the best floor(n/eta) (>= 1) for the next; returns the last rung's [(config, loss)]."""
    survivors = list(configs)
    scored = []
    for i in range(n_rungs):
        budget = rung_budget(min_budget, eta, i)
        scored = []
        for config in survivors:
            loss = float(evaluate(config, budget))
            history.append((config, budget, loss))
            scored.append((config, loss))
        if i < n_rungs - 1:
            keep = max(len(scored) // eta, 1)
            survivors = [c for c, _ in sorted(scored, key=lambda s: s[1])[:keep]]
    return scored


@app.function
def successive_halving(sample_config, evaluate, *, n_configs=27, min_budget=1, eta=3, rng=None):
    """stage 2: n_configs samples, then rungs at min_budget * eta**i keeping the best 1/eta until one is left; (best_config, best_loss, history)."""
    rng = as_rng(rng)
    configs = [sample_config(rng) for _ in range(n_configs)]
    n_rungs, m = 1, n_configs
    while m // eta >= 1:
        m //= eta
        n_rungs += 1
    history = []
    last = run_bracket(configs, evaluate, n_rungs=n_rungs, min_budget=min_budget, eta=eta, history=history)
    best_config, best_loss = min(last, key=lambda s: s[1])
    return best_config, best_loss, history


@app.cell(hide_code=True)
def _():
    checks.run(2, run_bracket=run_bracket, successive_halving=successive_halving)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def hyperband(sample_config, evaluate, *, max_budget=81, eta=3, rng=None):
    """stage 3: brackets s = s_max..0 (s_max = floor(log_eta max_budget)), each n = ceil((s_max+1) eta^s / (s+1)) configs from max_budget / eta^s; best at max_budget."""
    rng = as_rng(rng)
    s_max = 0
    while eta ** (s_max + 1) <= max_budget:
        s_max += 1
    history, finalists = [], []
    for s in range(s_max, -1, -1):
        n = math.ceil((s_max + 1) * eta ** s / (s + 1))
        configs = [sample_config(rng) for _ in range(n)]
        finalists += run_bracket(configs, evaluate, n_rungs=s + 1, min_budget=max_budget / eta ** s, eta=eta,
                                 history=history)
    best_config, best_loss = min(finalists, key=lambda f: f[1])
    return best_config, best_loss, history


@app.cell(hide_code=True)
def _():
    checks.run(3, hyperband=hyperband)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class ExperimentLog:
    """an append-only record of runs (config, seed, metrics), optionally mirrored to a JSON-lines file."""

    def __init__(self, path=None):
        """stage 4: self.runs = []; with a path, load the runs already in that file (one JSON object per line)."""
        self.path = None if path is None else Path(path)
        self.runs = []
        if self.path is not None and self.path.exists():
            with self.path.open() as f:
                self.runs = [json.loads(line) for line in f if line.strip()]

    def log(self, config, metrics, *, seed=None):
        """stage 4: append {'id', 'config', 'seed', 'metrics'} (plain dict copies), append it to the file too; return the record."""
        record = {"id": len(self.runs), "config": dict(config), "seed": seed,
                  "metrics": {k: float(v) for k, v in metrics.items()}}
        self.runs.append(record)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def best(self, metric, *, mode="min"):
        """stage 4: the record with the lowest (mode='min') or highest ('max') value of metric (the first among ties)."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        sign = 1.0 if mode == "min" else -1.0
        return min(self.runs, key=lambda r: sign * r["metrics"][metric])

    def group(self, metric):
        """stage 4: one dict(config, n, mean, std) per distinct config (first-appearance order); std over seeds with ddof=1 (0.0 if n = 1)."""
        groups = {}
        for r in self.runs:
            key = json.dumps(r["config"], sort_keys=True)
            groups.setdefault(key, (r["config"], []))[1].append(r["metrics"][metric])
        out = []
        for config, values in groups.values():
            v = np.asarray(values, dtype=float)
            out.append({"config": config, "n": len(v), "mean": float(v.mean()),
                        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0})
        return out


@app.cell(hide_code=True)
def _():
    checks.run(4, ExperimentLog=ExperimentLog)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.train import hyperband` gives you these searches; `evaluate(config, budget)`
    can train anything, an `mlfs.nn` network or (from topic 07) a PyTorch one.
    """)
    return


if __name__ == "__main__":
    app.run()
