import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import hashlib
    import json
    import re
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from mlfs import checks
    from mlfs.research import ablation_table, mean_ci  # yours, from topic 03

    # the paper's sections, in order; make_project writes them as '## ' headings
    PAPER_SECTIONS = ("abstract", "introduction", "related work", "method", "experiments", "limitations",
                      "conclusion", "references")
    # the files make_project creates, relative to the project root
    SKELETON = ("README.md", "proposal.md", "plan.json", "log.md", "paper/paper.md", "configs/.gitkeep",
                "runs/.gitkeep", "figures/.gitkeep")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 30 · capstone: the thesis

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

    below the stages there's a cell for your own project, and the write-up prompts are at the
    end.
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
    raise NotImplementedError("stage1: implement make_project")


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
    raise NotImplementedError("stage2: implement config_hash")


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
        raise NotImplementedError("stage3: implement RunRegistry.__init__")

    def __len__(self):
        """stage 3: the number of runs in the registry."""
        raise NotImplementedError("stage3: implement RunRegistry.__len__")

    def log(self, config, seed, metrics, *, name=None, notes=""):
        """stage 3: append one run; ValueError for a repeated (config, seed), a bad seed or non-finite metrics."""
        raise NotImplementedError("stage3: implement RunRegistry.log")

    def runs(self, name=None, config_hash=None):
        """stage 3: the records matching name and/or config_hash, in logging order."""
        raise NotImplementedError("stage3: implement RunRegistry.runs")

    def values(self, name, metric):
        """stage 3: dict seed -> metric value for the runs called name, sorted by seed."""
        raise NotImplementedError("stage3: implement RunRegistry.values")


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
    raise NotImplementedError("stage4: implement results_table")


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
    raise NotImplementedError("stage5: implement render_report")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", render_report=render_report, RunRegistry=RunRegistry,
               results_table=results_table)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## your thesis project

    when stages 1–5 are green, point the cell below at your own project directory (outside
    this repository). the reference solution (`solutions/capstone.py`) runs a small
    thesis-shaped experiment end to end with these tools, the brief's worked example: read it
    for the pattern.
    """)
    return


@app.cell
def _():
    my_project = None  # e.g. Path("~/thesis").expanduser(), outside this repository
    if my_project is None:
        _out = mo.md("*set `my_project` to your project directory once stages 1–5 are green.*")
    else:
        _created = make_project(my_project, "your title", question="your research question")
        _registry = RunRegistry(my_project / "runs" / "registry.jsonl")
        # log runs with _registry.log(config, seed, {"test_acc": ...}, name="baseline"), then:
        _out = mo.md(render_report((my_project / "paper" / "paper.md").read_text(), _registry))
    _out
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
