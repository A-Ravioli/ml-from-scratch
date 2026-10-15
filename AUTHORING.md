# authoring guide: the canonical chapter model

every chapter in this repo is built from the same template. this file is the template's spec. `CURRICULUM.md` says *what* each chapter teaches; this file says *how* every chapter teaches it.

the one-sentence version: **you never read more than a few pages without writing code that a check verifies, and you never finish a topic without running an experiment that surprises you.**

---

## 1. design principles

1. **build it to understand it.** every concept ends in something the learner implements: a function, a class, a model, a plot. if a concept can't be built, it gets a derivation or a simulation instead, never just prose.
2. **predict before you see.** before any result is shown, the learner writes down a guess. lab notebooks enforce this: experiments won't run until a prediction is locked in.
3. **small verified steps.** exercises are split into *stages*, each with its own checks. checks run live in the notebook as you type, so a learner is never more than ~30 minutes from green.
4. **your code compounds.** what you build in chapter N is imported in chapter N+1 (see §6, the `mlfs` library). your autograd engine trains your CNN; your bootstrap CIs grade your GAN.
5. **scaffolding fades.** early chapters hand you signatures, docstrings, and hints. late chapters hand you a spec and checks. the research chapter hands you a question (see §5).
6. **theory and practice check each other.** every lab compares something derived on paper against something measured in code.
7. **honest evaluation from day one.** from chapter 07 on, every reported number comes with a baseline and an uncertainty estimate.

## 2. two formats: markdown to read, marimo to do

| format | used for | why |
|---|---|---|
| **markdown** (`.md`) | lessons, hints, lab notes, chapter READMEs, checkpoints, capstone briefs & rubrics | readable anywhere (GitHub, editors), easy to review and diff, renders LaTeX |
| **[marimo](https://marimo.io) notebooks** (`.py`) | exercises, labs, capstones, and their reference solutions | reactive: change a function and every cell that depends on it re-runs, including its checks. stored as plain Python, so they diff cleanly, import as modules, and run headless in CI |

marimo notebooks are saved as `.py` files, but learners never treat them as scripts: they open them with `marimo edit`. the only plain Python files in a topic are its `test_*.py` checks, which learners don't edit.

the two formats work together: exercise notebooks pull the matching section of `lesson.md` in next to each stage (`lesson.section("idea 1")`), so the lesson text lives in one place but you read and build in one window.

---

## 3. the learning loop (one topic)

every topic runs the same six-step loop. steps interleave rather than happen in sequence: read a section, build a stage, read the next section.

| step | where | what the learner does | ~time |
|---|---|---|---|
| **1. predict** | `lesson.md` → *warm-up* | answers 3–5 questions cold, writes guesses down | 5% |
| **2. learn** | `lesson.md`, shown section by section in `exercise.py` | intuition → formalism → worked example | 20% |
| **3. derive** | `lesson.md` → *on paper* | 2–4 pen-and-paper derivations the code depends on | 15% |
| **4. build** | `exercise.py` notebook | implements stage by stage; live checks under each stage go green | 35% |
| **5. experiment** | `lab.py` notebook | uses *their own implementation*; locks in a prediction, runs, explains | 20% |
| **6. reflect** | `lesson.md` → *check your understanding* | answers socratic questions, logs what surprised them | 5% |

a chapter wraps its topics with:

- **capstone**: one larger project combining every topic in the chapter on a real (or realistic) dataset.
- **checkpoint**: a self-test (concept questions, by-hand derivations, and a *blind rebuild* of the core algorithm with no notes). passing a checkpoint is also how a learner tests out of a chapter.

---

## 4. chapter anatomy

```
NN-chapter-name/
├── README.md                    # chapter map (§5.1)
├── checkpoint.md                # chapter self-test + blind rebuild (§5.9)
├── 01-topic-name/
│   ├── lesson.md                # theory, with build/predict callouts (§5.2)
│   ├── exercise.py              # marimo: staged build + live checks (§5.3)
│   ├── test_exercise.py         # the checks, grouped by stage (§5.4)
│   ├── hints.md                 # progressive hints per stage (§5.5)
│   ├── lab.py                   # marimo: experiments on your own code (§5.6)
│   └── solutions/
│       ├── exercise.py          # marimo: reference implementation, same API
│       └── lab_notes.md         # what each experiment shows, and why
├── 02-topic-name/
│   └── …
└── capstone/
    ├── brief.md                 # task, data, deliverables, thresholds (§5.8)
    ├── rubric.md                # how to self-assess the write-up
    ├── capstone.py              # marimo: starter notebook + write-up
    ├── test_capstone.py         # functional checks + metric thresholds
    └── solutions/capstone.py
```

chapter and topic directories are numbered (`NN-slug`) in teaching order. `python scripts/scaffold.py plan NN` creates all of this for a chapter, with names and plan text taken from `CURRICULUM.md`.

---

## 5. file specs

### 5.1 chapter `README.md`

hook paragraph · what you'll build (and which `mlfs` modules) · prerequisites · time & scaffolding level · newly unlocked library components · topic table · capstone summary · checkpoint link · notation table · connections · 3–6 core references. the scaffold pre-fills the goal, topic table, and capstone from the plan.

### 5.2 `lesson.md`

target length is 1,500–4,000 words. math goes in `$…$` / `$$…$$` (renders on GitHub and in marimo). use these `##` sections in this order (the audit checks for them):

1. **why this matters**: a concrete problem or failure, before any definitions
2. **warm-up: predict first**: 3–5 cold questions, answered later in the lesson
3. **prerequisites check**: quick self-test items linked to the topics that teach them
4. **core ideas**: `### idea N` subsections, each going intuition → formal statement → derivation
5. **worked example**: one example fully by hand; the learner's code must reproduce it
6. **on paper**: 2–4 derivations, solutions in `<details>`
7. **experiment**: what the lab investigates (no spoilers)
8. **check your understanding**: 5–10 socratic questions, answers in `<details>`
9. **common bugs**: the 3–6 mistakes everyone makes, with symptoms
10. **going deeper**: papers (author, year, title, link) and textbook sections
11. **connections**: ← builds on / → used in

callouts:
- `> **▶ build: stage N.** …`: right after the idea that stage implements, one per stage.
- `> **🔮 predict:** …`: at least two per lesson.
- no section runs longer than ~800 words without a callout.

each `### idea N` subsection is what the exercise notebook shows next to stage N, so keep it self-contained.

### 5.3 `exercise.py` (marimo)

```python
with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")          # the lesson text for this stage, rendered inline
    return


@app.function
def mean(x: np.ndarray) -> float:
    """stage 1: the arithmetic mean of a 1-D array `x` (shape (n,), n >= 1)."""
    raise NotImplementedError("stage1: implement mean(x)")


@app.cell(hide_code=True)
def _():
    checks.run(1, mean=mean)          # live checks: re-run whenever `mean` changes
    return
```

conventions:
- **the public API lives in `@app.function` / `@app.class_definition` cells.** marimo makes these importable as plain module attributes, which is how tests, labs, and later chapters (`mlfs`) use them. they may only depend on names from the setup cell and other top-level definitions.
- each stage follows the same pattern: a lesson section cell → one or more definition cells → a check cell `checks.run(N, name=name, …)` that passes in every definition the stage's checks use.
- stub bodies are `raise NotImplementedError("stageN: …")`, never `pass`. the live check shows that message as "not started".
- the first cell is a markdown table of the stages. the last cell points to the lab.
- the API (names, signatures, return types) is fixed once published. tests and `solutions/exercise.py` depend on it.
- docstrings say *what*, with shapes and dtypes. *how* goes in `hints.md`. how much the docstrings give away follows the scaffolding level (§7).
- imports in the setup cell are limited to NumPy, the standard library, earlier `mlfs` modules, and whatever the framework policy (§9) allows.
- cell-local scratch variables start with `_` (marimo keeps them private to the cell).

`solutions/exercise.py` is the same notebook with every stub implemented.

### 5.4 `test_exercise.py`

```python
def test_stage2_worked_example(nb):
    got = nb.variance(np.array([1.0, 2.0, 3.0, 6.0]))
    assert np.isclose(got, 3.5), (
        f"variance of (1, 2, 3, 6) should be 3.5, got {got}. "
        "if you got 4.67, you divided by n-1 instead of n."
    )
```

rules:
- **every test takes exactly one argument, `nb`**, and uses only `nb.<name>` to reach the code under test. under pytest, `nb` is the notebook module. inside the notebook, it's a namespace of whatever `checks.run` passed in.
- **name tests `test_stageN_*`** so `pytest -k stageN` and the live checks can find them.
- **failure messages teach.** say what was expected, what came back, and the most likely cause. learners see these messages in the notebook.
- **fast and deterministic.** under ~10 s per topic on a laptop CPU, with fixed seeds. anything slower gets `@pytest.mark.slow` (pytest runs it; live checks skip it).
- **no network.** use small synthetic data.
- test kinds: *oracle* (compare with NumPy/SciPy/PyTorch), *property* (invariants), *gradient checks* (finite differences), *behavioral* ("reaches loss < X in N steps"), *hand-example* (the lesson's worked example).

### 5.5 `hints.md`

one `## stage N` section per stage, with 1–3 hints of increasing strength, each in its own `<details>` block.

### 5.6 `lab.py` (marimo)

```python
with app.setup:
    from mlfs import notebooks, ui
    ex = notebooks.load_sibling("exercise")   # the learner's own code


@app.cell
def _():
    p1 = ui.predict("if you 4× the sample size, how much smaller does the error get?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)            # waits until a prediction is locked in (skipped when headless)
    ...                    # the experiment, using ex.mean etc.
```

- 2–4 experiments. each one has a `ui.predict` cell, a gated experiment cell, and a markdown "📝 **explain:**" prompt.
- use marimo UI elements (`mo.ui.slider`, `mo.ui.dropdown`, …) so learners can turn knobs and watch results update.
- at least one result should surprise most learners.
- headless (`python lab.py`, which `verify.py` runs against the reference solution) it must finish without errors in a few minutes.
- `solutions/lab_notes.md` gives, for each experiment, what people usually predict, what actually happens, and why.
- **building on an earlier topic that isn't an `mlfs` module** (elective blocks): load it with
  `prev = notebooks.load_topic("30-qubits-and-circuits", key="qsim", smoke=lambda m: m.run_circuit(...))`.
  It gives the reference under pytest/verify (or `MLFS_REFERENCE=qsim`), and the learner's notebook otherwise. If the
  smoke test hits an unfinished stage, it falls back to the reference. Show `prev.callout()` so the learner sees when
  that happened, and use `prev.module`. Exercises whose own checks depend on the earlier topic use the same call in
  their setup cell, so an unfinished earlier topic never shows up as "not started" in this topic's checks.

### 5.7 `checks` in practice

the same `test_stageN_*` functions run in three places:

| where | command | runs against |
|---|---|---|
| in the notebook | automatic (check cells) | the live definitions passed to `checks.run` |
| terminal | `pytest path/to/test_exercise.py -k stage1` | `exercise.py` on disk |
| reference / CI | `MLFS_SOLUTIONS=1 pytest …` or `scripts/verify.py` | `solutions/exercise.py` |

### 5.8 capstone

`brief.md` covers the goal, the dataset (downloaded once and cached under `~/.cache/mlfs/`), deliverables, metric thresholds, and stretch goals. `capstone.py` is a marimo notebook with the required `@app.function` definitions, check cells (`checks.run(1, _tests="test_capstone.py", …)`), and a write-up section. `test_capstone.py` holds functional checks plus minimum metrics, named `test_stageN_*` like topic tests. `rubric.md` grades the write-up on correctness, results versus baselines (with uncertainty), analysis and ablations, and clarity.

### 5.9 `checkpoint.md`

1. **concept check**: 10–15 questions with answers in `<details>`.
2. **by hand**: 2–3 derivations or calculations, no computer.
3. **blind rebuild**: reimplement the chapter's core algorithm in 60 minutes with no notes, checked by that topic's tests.
4. **readiness**: move on at ≥80% on 1–2 plus a passing 3. the same checkpoint works as a placement test.

---

## 6. `mlfs`: the learner's cumulative library

`mlfs/registry.toml` maps module names to the topics that build them. a module can merge several topics:

```toml
[modules]
tensor = "14-neural-networks-from-scratch/03-a-tensor-autograd-engine"
linalg = ["02-linear-algebra/03-solving-linear-systems", "02-linear-algebra/04-orthogonality-and-least-squares"]
```

`from mlfs.linalg import lu_solve` then imports the functions and classes those topics' exercise notebooks define. which copy you get:

| context | default | override |
|---|---|---|
| notebooks / your own scripts | **your** `exercise.py` | `MLFS_REFERENCE=tensor,optim` (per module) or `MLFS_REFERENCE=*` |
| pytest / `verify.py` | **reference** solutions for every *other* module, so one bug can't break later chapters | `MLFS_REFERENCE=none` to test against your own earlier code |
| `MLFS_SOLUTIONS=1` | every notebook resolves to its reference solution | |

**contracts.** every module's public interface is frozen in `mlfs/contracts.py` (explained in `CONTRACTS.md`): names, parameter names, order, and defaults. topics that own a module must implement it exactly. later chapters may use only what the contract lists. `tests/test_contracts.py` checks reference solutions against it. changing a contract is an orchestrator decision, not a chapter decision. a notebook may define extra public helpers beyond its contract (marimo cells can't share `_private` names), but later chapters must not rely on them, and they must not clash with other names in the same module.

register a module with `python scripts/scaffold.py topic NN MM --provides <name>` (the registry for the whole course is already filled in). names can't collide with `mlfs`'s own submodules (`notebooks`, `registry`, `checks`, `lesson`, `ui`, `pytest_plugin`, `datasets`, `contracts`). `CURRICULUM.md` lists which topic owns each module.

`mlfs` itself provides the runtime: `notebooks` (load notebooks as modules), `checks` (live checks), `lesson` (render lesson sections), `ui` (predict gates), `datasets` (see §8), `contracts`, and a pytest plugin (the `nb` fixture).

---

## 7. scaffolding levels

| level | used in | exercise gives you | hints | capstone starter |
|---|---|---|---|---|
| **L1 guided** | part 0 | signatures, detailed docstrings, step comments inside functions | 3 levels | most of the code; you fill in the core |
| **L2 supported** | parts I–II (early) | signatures + docstrings with shapes | 3 levels | data loading + evaluation |
| **L3 independent** | parts II (late)–III | signatures + one-line docstrings | 2 levels | data loading only |
| **L4 spec-driven** | parts IV–VI | the spec (in the lesson) + checks; you design internal structure | 1 level | none: brief + checks |
| **L5 open** | part VII | a research question | none | none |

## 8. data policy

- tests never download anything; they use small synthetic data generated deterministically.
- labs and capstones load data through `mlfs.datasets` (`mnist`, `cifar10`, `tiny_shakespeare`, `sms_spam`, `olivetti_faces`, `california_housing`), which downloads once and caches under `$MLFS_DATA` (default `~/.cache/mlfs/`).
- with `MLFS_OFFLINE=1` (set by `verify.py` and CI), every loader returns a small deterministic synthetic stand-in with the same structure and dtypes, so labs must still run (and say something sensible) on it.
- need another dataset? request it; don't download ad hoc from a notebook.

## 9. framework & compute policy

- **chapters 00–14: NumPy only.** you build your own autograd in chapter 14.
- **chapter 15 onward:** PyTorch is allowed, under the **build-before-you-use rule**: you may use a library component only after you've implemented it yourself in an earlier topic (`nn.Conv2d` after 16.01, `nn.MultiheadAttention` after 18.01, and so on). each chapter README lists what is newly unlocked.
- **the rule covers NumPy/SciPy too.** once a learner has built something (say `mlfs.linalg.svd`), later chapters may call the library equivalent (`np.linalg.svd`) where speed matters. reference implementations are written for clarity, not speed (the pilot's Jacobi SVD takes seconds on a 280×4096 matrix). use the learner's `mlfs` version when *reusing their code* is the point; use the library version for heavy lifting in labs and capstones.
- **compute:** everything through chapter 21 runs on a laptop CPU. chapters 22 and 24–28 benefit from a GPU (a free Colab T4 is enough for the default configs). parts of chapter 26 (Triton) need an NVIDIA GPU. every capstone has a small config that runs on CPU.

## 10. voice & style

- second person, direct, curious. explain *why* before *what*.
- give intuition before formalism, and a picture or small numeric example before a theorem.
- one notation table per chapter (in its README); stay consistent with it.
- cite with author and year (*Vaswani et al., 2017*) and a stable link (arXiv, DOI, or publisher).
- don't hide difficulty. if something is hard, say so and say why.

## 11. definition of done (per topic)

`python scripts/audit.py` and `python scripts/verify.py` check everything marked †.

- [ ] † all seven topic files exist; no `TEMPLATE` markers left
- [ ] † `lesson.md` has every §5.2 section, a `▶ build: stage N` callout per stage, and at least two `🔮 predict` callouts
- [ ] † `exercise.py` stubs raise `NotImplementedError("stageN: …")`, and every stage has a `checks.run(N, …)` cell
- [ ] † every stub stage has `test_stageN_*` tests; stages are numbered contiguously from 1
- [ ] † the reference solution passes every check; the untouched stubs fail every stage
- [ ] † `solutions/exercise.py` has no placeholders
- [ ] † `hints.md` has a section per stage; `lab.py` has at least one `ui.predict` gate and runs headless against the reference
- [ ] † every notebook passes `marimo check --strict`
- [ ] tests run in under 10 s, with fixed seeds, no network, and failure messages that teach
- [ ] if the topic owns an `mlfs` module, it's registered
- [ ] the lab's surprising result is actually surprising (try it on someone)

## 12. tooling

| command | does |
|---|---|
| `python scripts/scaffold.py plan NN` | creates chapter NN and every topic in its plan, pre-filled from `CURRICULUM.md` (never overwrites; `--force` to override) |
| `python scripts/scaffold.py chapter NN` / `topic NN MM [--provides name]` | one chapter skeleton / one topic (optionally registering an `mlfs` module) |
| `python scripts/stubify.py <topic dir>...` | regenerates `exercise.py` from `solutions/exercise.py`: every function or method whose docstring starts with `stage N:` gets a `NotImplementedError` stub |
| `python scripts/audit.py [--chapter NN] [--marimo] [--strict] [--json]` | plan coverage, structure, placeholders, stage/test/hint/callout consistency, registry, `marimo check` |
| `python scripts/verify.py [--chapter NN \| --topic PATH] [--fast] [-j N]` | for each topic and capstone: solutions pass, stubs fail every stage, labs run headless |
| `pytest tests` | tests for the tooling itself |

CI (`.github/workflows/ci.yml`) runs the tooling tests, `audit.py --marimo`, and `verify.py --fast` on every push and PR. templates live in `templates/`. they include a small working mean/variance example so the tooling can be tested end to end, and the audit reports scaffolded files as "still templates" until their `TEMPLATE` markers are gone.
