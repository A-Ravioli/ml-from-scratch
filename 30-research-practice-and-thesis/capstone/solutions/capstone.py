import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import hashlib
    import json
    import re
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, datasets
    from mlfs.research import ablation_table, mean_ci  # yours, from topic 03
    from mlfs.torchnn import MLP, evaluate, fit, make_loader  # yours, from ch 15 · topic 07

    # the paper's sections, in order; make_project writes them as '## ' headings
    PAPER_SECTIONS = ("abstract", "introduction", "related work", "method", "experiments", "limitations",
                      "conclusion", "references")
    # the files make_project creates, relative to the project root
    SKELETON = ("README.md", "proposal.md", "plan.json", "log.md", "paper/paper.md", "configs/.gitkeep",
                "runs/.gitkeep", "figures/.gitkeep")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 30 · capstone: the thesis (reference solution)

    read [`brief.md`](brief.md) first. the thesis is original work, written up as a paper,
    and optionally submitted to a workshop; its quality is judged with [`rubric.md`](rubric.md),
    not by tests. what the tests check is the **tooling** that keeps a thesis honest and
    reproducible: a project skeleton, stable configuration hashes, a run registry that every
    result goes through, tables built from it, and a report whose every number is filled in
    from the registry instead of typed by hand.

    | stage | you build |
    |---|---|
    | 1 | `make_project`: the thesis skeleton (README, proposal, plan, log, paper), never overwriting |
    | 2 | `config_hash`: a stable 12-character id for an experiment configuration |
    | 3 | `RunRegistry`: an append-only log of runs (config, seed, metrics) in a JSONL file |
    | 4 | `results_table`: a paired ablation table straight from the registry |
    | 5 | `render_report`: fill `{{mean:…}}`, `{{diff:…}}`, `{{n:…}}` and `{{table:…}}` from the registry |

    below the stages, the reference solution runs a small thesis-shaped experiment end to end
    with these tools; the write-up prompts are at the end.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 1: a project skeleton

    every thesis needs the same few files from day one: a README that says how to reproduce
    everything, the proposal (topic 04), the plan (topic 06), a weekly log, and the paper. the
    skeleton makes the habit free, and it must never destroy work: running it twice changes
    nothing.
    """)
    return


@app.function
def make_project(root, title, *, question, author=""):
    """stage 1: create SKELETON under root (dirs as needed), skipping existing files; return created paths, sorted."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("the project needs a title")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("the project needs a research question")
    root = Path(root)
    readme = (f"# {title}\n\n**question.** {question}\n\n" + (f"**author.** {author}\n\n" if author else "")
              + "## reproduce\n\nevery number in `paper/paper.md` is rendered from `runs/registry.jsonl`; list the "
                "commands that regenerate the runs and the paper here.\n\n"
              + "## layout\n\n- `proposal.md`: the two-page proposal (topic 04)\n- `plan.json`: the task graph "
                "(topic 06)\n- `log.md`: the weekly log\n- `configs/`, `runs/`, `figures/`: configurations, the run "
                "registry, generated figures\n- `paper/paper.md`: the paper, with `{{...}}` placeholders\n")
    proposal = (f"# proposal: {title}\n\n"
                + "".join(f"## {h}\n\n" for h in ("goal", "state of the art", "approach", "impact", "risks",
                                                   "checkpoints", "hypothesis", "budget")))
    plan = json.dumps({"read & annotate": {"estimate": [2, 3, 6], "after": []},
                       "proposal": {"estimate": [1, 1, 2], "after": ["read & annotate"]}}, indent=2) + "\n"
    log = f"# research log: {title}\n\n## week 1\n\n- done:\n- learned:\n- finished tasks (estimate, actual):\n- next:\n"
    paper = f"# {title}\n\n" + "".join(f"## {s}\n\n" for s in PAPER_SECTIONS)
    contents = {"README.md": readme, "proposal.md": proposal, "plan.json": plan, "log.md": log,
                "paper/paper.md": paper, "configs/.gitkeep": "", "runs/.gitkeep": "", "figures/.gitkeep": ""}
    created = []
    for rel in SKELETON:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        path.write_text(contents[rel])
        created.append(rel)
    return sorted(created)


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", make_project=make_project)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 2: configuration hashes

    a run is identified by its configuration and its seed. the configuration's id has to be
    *stable*: the same settings, written in any key order, must hash the same, and any change
    must change it. canonical JSON (sorted keys, no spaces) through SHA-256 does both.
    """)
    return


@app.function
def config_hash(config):
    """stage 2: first 12 hex digits of sha256 of the canonical JSON of config (sorted keys, compact, NumPy -> Python)."""
    def plain(x):
        if isinstance(x, dict):
            return {str(k): plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        if isinstance(x, np.ndarray):
            return [plain(v) for v in x.tolist()]
        if isinstance(x, np.generic):
            return x.item()
        return x

    text = json.dumps(plain(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", config_hash=config_hash)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 3: the run registry

    every run your thesis reports is a line in `runs/registry.jsonl`: its configuration, its
    hash, its seed, its metrics and a name. appending (never editing) keeps the history; one
    line per run keeps it diffable and greppable; refusing a second run of the same
    configuration and seed keeps you from silently overwriting a result you didn't like.
    """)
    return


@app.class_definition
class RunRegistry:
    """stage 3: an append-only JSONL log of runs: log(config, seed, metrics), runs(...), values(name, metric)."""

    def __init__(self, path):
        """stage 3: open the registry at path, loading any records already in the file."""
        self.path = Path(path)
        self.records = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        """stage 3: the number of runs in the registry."""
        return len(self.records)

    def log(self, config, seed, metrics, *, name=None, notes=""):
        """stage 3: append one run; ValueError for a repeated (config, seed), a bad seed or non-finite metrics."""
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics must be a non-empty dict of name -> number")
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v):
                raise ValueError(f"metric {k!r} must be a finite number, got {v!r}")
            clean[str(k)] = float(v)
        h = config_hash(config)
        if any(r["config_hash"] == h and r["seed"] == int(seed) for r in self.records):
            raise ValueError(f"config {h} with seed {seed} is already logged; a rerun needs a new seed (or delete "
                             "the line by hand, and say so in the log)")
        record = {"id": f"{h}-{int(seed)}", "config_hash": h, "name": name if name is not None else h,
                  "config": json.loads(json.dumps(config, default=lambda x: x.tolist() if hasattr(x, "tolist")
                                                   else str(x))),
                  "seed": int(seed), "metrics": clean, "notes": notes}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        return record

    def runs(self, name=None, config_hash=None):
        """stage 3: the records matching name and/or config_hash, in logging order."""
        return [r for r in self.records
                if (name is None or r["name"] == name) and (config_hash is None or r["config_hash"] == config_hash)]

    def values(self, name, metric):
        """stage 3: dict seed -> metric value for the runs called name, sorted by seed."""
        rs = self.runs(name=name)
        if not rs:
            raise ValueError(f"no runs named {name!r}")
        if len({r["config_hash"] for r in rs}) > 1:
            raise ValueError(f"the name {name!r} is used for {len({r['config_hash'] for r in rs})} different "
                             "configurations; give each its own name")
        missing = sorted(r["seed"] for r in rs if metric not in r["metrics"])
        if missing:
            raise ValueError(f"runs named {name!r} with seeds {missing} have no metric {metric!r}")
        return {r["seed"]: r["metrics"][metric] for r in sorted(rs, key=lambda r: r["seed"])}


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", RunRegistry=RunRegistry, config_hash=config_hash)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 4: tables from the registry

    an ablation table must compare runs seed by seed. runs crash and get rerun, so the
    configurations don't always share every seed: the table pairs on the seeds they *all*
    have, and says which runs it left out.
    """)
    return


@app.function
def results_table(registry, metric, names, *, baseline=None, alpha=0.05, higher_is_better=True, fmt="{:.3f}"):
    """stage 4: mlfs.research.ablation_table over the seeds every named config shares, plus a note on left-out runs."""
    names = list(names)
    vals = {n: registry.values(n, metric) for n in names}
    shared = sorted(set.intersection(*(set(v) for v in vals.values())))
    if len(shared) < 2:
        raise ValueError(f"the configs {names} share {len(shared)} seeds; a paired table needs at least 2")
    table = ablation_table({n: [vals[n][s] for s in shared] for n in names}, baseline=baseline, alpha=alpha,
                           higher_is_better=higher_is_better, fmt=fmt)
    left = [(n, sorted(set(vals[n]) - set(shared))) for n in names]
    left = [(n, s) for n, s in left if s]
    if left:
        table += (f"\n\n*paired on {len(shared)} shared seeds; left out: "
                  + "; ".join(f"{n} seeds {', '.join(map(str, s))}" for n, s in left) + "*")
    return table


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", results_table=results_table, RunRegistry=RunRegistry)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 5: a report rendered from the registry

    numbers typed into a paper drift: a run gets repeated, a table gets updated, the abstract
    doesn't. so the paper never contains a typed result. it contains placeholders, and
    `render_report` fills them from the registry every time you build the paper.
    """)
    return


@app.function
def render_report(template, registry, *, alpha=0.05, fmt=".3f"):
    """stage 5: replace every {{mean:..}}, {{n:..}}, {{diff:..}}, {{table:..}} placeholder with numbers from registry."""
    def fill(m):
        kind, args = m.group(1), m.group(2).split(":")
        if kind == "mean" and len(args) == 2:
            v = list(registry.values(args[0], args[1]).values())
            mu, lo, hi = mean_ci(v, alpha=alpha)
            return f"{mu:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"
        if kind == "n" and len(args) == 2:
            return str(len(registry.values(args[0], args[1])))
        if kind == "diff" and len(args) == 3:
            a, b = registry.values(args[0], args[2]), registry.values(args[1], args[2])
            shared = sorted(set(a) & set(b))
            if len(shared) < 2:
                raise ValueError(f"{args[0]!r} and {args[1]!r} share {len(shared)} seeds; a difference needs 2")
            mu, lo, hi = mean_ci([a[s] - b[s] for s in shared], alpha=alpha)
            return f"{mu:+{fmt}} [{lo:+{fmt}}, {hi:+{fmt}}]"
        if kind == "table" and len(args) == 2:
            names = [n.strip() for n in args[1].split(",") if n.strip()]
            return results_table(registry, args[0], names, baseline=names[0], alpha=alpha, fmt="{:" + fmt + "}")
        raise ValueError(f"bad placeholder {m.group(0)!r}: use {{{{mean:name:metric}}}}, {{{{n:name:metric}}}}, "
                         "{{diff:name:baseline:metric}} or {{table:metric:name1,name2,...}}")

    return re.sub(r"\{\{(\w+):([^{}]*)\}\}", fill, template)


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", render_report=render_report, RunRegistry=RunRegistry,
               results_table=results_table)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## a thesis-shaped experiment, end to end

    the brief's worked example, small enough to run here: *does label smoothing help a small
    MLP trained briefly on MNIST, and how does the effect depend on the smoothing strength?*
    (topic 04's pilot suggested a large effect at 0.1.) the pipeline is the one your thesis
    uses: a skeleton in a scratch directory, every run logged in the registry, a table and a
    paragraph rendered from it. offline (synthetic data) it runs a small configuration.
    """)
    return


@app.cell
def _():
    offline = datasets.is_offline()
    CONFIG = ({"seeds": list(range(3)), "epochs": 1, "n_train": 600} if offline
              else {"seeds": list(range(10)), "epochs": 3, "n_train": 5000})
    _d = datasets.mnist()
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(np.float32) / 255.0
    _Xt = _d["X_test"].reshape(len(_d["X_test"]), -1).astype(np.float32) / 255.0
    _mu, _sd = float(_X.mean()), float(_X.std())

    def run_config(cfg, seed):
        idx = np.random.default_rng(seed).choice(len(_X), cfg["n_train"], replace=False)
        torch.manual_seed(seed)
        model = MLP(784, [cfg["width"]], 10)
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

        def loss(logits, y):
            return F.cross_entropy(logits, y, label_smoothing=cfg["smoothing"])

        fit(model, loss, opt, make_loader((_X[idx] - _mu) / _sd, _d["y_train"][idx], batch_size=64, seed=seed),
            epochs=cfg["epochs"])
        res = evaluate(model, make_loader((_Xt - _mu) / _sd, _d["y_test"], batch_size=2000, shuffle=False), loss)
        return {"test_acc": 100 * res["accuracy"], "test_loss": res["loss"]}

    # a scratch project, deleted at the end of this cell; the registry keeps its records in memory
    with tempfile.TemporaryDirectory() as _tmp:
        project = Path(_tmp) / "label-smoothing-thesis"
        created = make_project(project, "label smoothing for small, briefly trained networks",
                               question="does label smoothing raise the test accuracy of a small MLP trained for a "
                                        "few epochs, and how does the effect depend on its strength?")
        registry = RunRegistry(project / "runs" / "registry.jsonl")
        for _name, _s in (("no smoothing", 0.0), ("smoothing 0.05", 0.05), ("smoothing 0.1", 0.1),
                          ("smoothing 0.2", 0.2)):
            _cfg = {"width": 128, "lr": 0.05, "smoothing": _s, "epochs": CONFIG["epochs"],
                    "n_train": CONFIG["n_train"]}
            for _seed in CONFIG["seeds"]:
                registry.log(_cfg, _seed, run_config(_cfg, _seed), name=_name)
    mo.md(f"created {len(created)} files under a scratch directory; logged {len(registry)} runs "
          f"({len(CONFIG['seeds'])} seeds × 4 configurations, {CONFIG['epochs']} epoch(s) on {CONFIG['n_train']:,} "
          f"images{', synthetic data' if offline else ''}).")
    return (registry,)


@app.cell
def _(registry):
    _paper = "\n".join([
        "**results.** without label smoothing the MLP reaches {{mean:no smoothing:test_acc}} % test accuracy",
        "(mean and 95% t interval over {{n:no smoothing:test_acc}} seeds). smoothing of 0.1 changes it by",
        "{{diff:smoothing 0.1:no smoothing:test_acc}} points, paired by seed.",
        "",
        "{{table:test_acc:no smoothing,smoothing 0.05,smoothing 0.1,smoothing 0.2}}",
    ])
    mo.md(render_report(_paper, registry, fmt=".2f"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    the thesis itself lives in your project directory, not in this notebook: the paper in
    `paper/paper.md` (or LaTeX, checked with topic 05's `check_latex`), rendered with your
    `render_report`. use this section for the short process report the brief asks for:

    1. **the question and its origin.** which gap (a reproduction discrepancy, an anomaly, a
       stated limitation) led to it, and the hypothesis as you pre-registered it.
    2. **the plan against reality.** your topic 06 forecast at the start, the actual timeline,
       and what you cut. your (estimate, actual) log, summarized.
    3. **the evidence.** the main table and figure, rendered from the registry; how many
       seeds, why that many (topic 04's power calculation), and which claims survived.
    4. **what didn't work.** the experiments that failed or were inconclusive, and what you
       learned from them.
    5. **reproducibility.** the commands that rebuild every number, and the reproducibility
       checklist from topic 02, answered for your own work.

    grade it with [`rubric.md`](rubric.md).
    """)
    return


if __name__ == "__main__":
    app.run()
