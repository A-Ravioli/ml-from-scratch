# ml-from-scratch

a hands-on machine learning curriculum that takes you from high school math to research. you learn every idea by building it: from matrix decompositions, to your own autograd engine, to a GPT, a diffusion model, and an RLHF pipeline, all checked by tests as you go.

> **status:** rebuilding from scratch. the full plan is in [`CURRICULUM.md`](CURRICULUM.md) and the chapter format is in [`AUTHORING.md`](AUTHORING.md). chapters appear as they're written; run `python scripts/audit.py` to see progress.

## how it works

every topic follows the same loop:

1. **predict**: answer a few questions cold, before reading anything
2. **learn**: read the lesson (`lesson.md`), one idea at a time
3. **derive**: work through the key derivations on paper
4. **build**: implement each idea in an interactive [marimo](https://marimo.io) notebook (`exercise.py`). checks under every stage turn green as soon as your code is right
5. **experiment**: run experiments on *your own* implementation (`lab.py`). each one makes you lock in a prediction before it runs
6. **reflect**: answer socratic questions about what you saw

every chapter ends with a **capstone project** and a **checkpoint** (concept questions plus rebuilding the core algorithm from memory). passing a checkpoint also lets you skip that chapter.

what you build adds up to a library, `mlfs`. later chapters import your earlier code: your optimizers train your models, your autograd engine trains your networks, your bootstrap CIs grade your experiments.

## curriculum

31 chapters in 8 parts, about 184 topics and 29 capstones, roughly 2,500 hours (about four years at ~13 hours a week). see [`CURRICULUM.md`](CURRICULUM.md) for every topic, what you build in it, and what its lab investigates.

| part | chapters |
|---|---|
| **0 · onramp** | 00 python & numpy for ML · 01 math onramp |
| **I · mathematical foundations** | 02 linear algebra · 03 calculus & analysis · 04 probability · 05 statistics & information theory · 06 optimization |
| **II · classical ML** | 07 the ML workflow · 08 linear models · 09 nonparametric & kernel methods · 10 trees & ensembles · 11 unsupervised learning · 12 probabilistic & bayesian ML · 13 learning theory |
| **III · deep learning** | 14 neural networks from scratch · 15 training deep networks · 16 convolutional networks · 17 sequence models · 18 transformers · 19 deep learning theory · 20 geometric deep learning |
| **IV · generative models** | 21 autoregressive models, VAEs, GANs & flows · 22 energy-based models, diffusion & flow matching |
| **V · reinforcement learning** | 23 RL foundations · 24 deep RL |
| **VI · modern & frontier ML** | 25 self-supervised & multimodal · 26 ML systems & scale · 27 large language models · 28 interpretability, robustness & alignment |
| **VII · research** | 29 electives (pick two) · 30 research practice & thesis |

paper reproductions come after chapters 18, 22, and 24. chapter 30 is an original research project.

## setup

you need python 3.11+. with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.12
```

```bash
uv pip install -e ".[all]"
```

or with pip, inside a virtual environment:

```bash
pip install -e ".[all]"
```

the editable install (`-e`) matters: it's how `mlfs` finds your notebooks. `".[all]"` adds PyTorch and the comparison libraries used from part III on. a plain `-e .` is enough for parts 0–II.

## working through a topic

```bash
cd 02-linear-algebra/03-solving-linear-systems
```

read `lesson.md`, then open the exercise notebook:

```bash
marimo edit exercise.py
```

the notebook shows each part of the lesson next to the code you write, with live checks under every stage. you can also run the checks from a terminal:

```bash
pytest test_exercise.py -k stage1
```

when every stage is green, run the experiments:

```bash
marimo edit lab.py
```

stuck? `hints.md` has progressive hints for every stage. reference solutions are in `solutions/`, but try on your own first.

if your code from an earlier chapter is broken, you can keep going by switching that module to the reference solution:

```bash
MLFS_REFERENCE=optim marimo edit exercise.py
```

## repository layout

```
CURRICULUM.md        the full plan: every chapter and topic
AUTHORING.md         the canonical chapter model (for contributors)
NN-chapter/          a chapter: README, checkpoint, topics, capstone
  MM-topic/          lesson.md, exercise.py, test_exercise.py, hints.md, lab.py, solutions/
mlfs/                the runtime: notebook loading, live checks, lesson rendering, registry
templates/           what scripts/scaffold.py copies into new chapters and topics
scripts/             scaffold.py, audit.py, verify.py
tests/               tests for the tooling
```

## contributing

the curriculum is a living document. contributions are welcome: new topics, clearer explanations, better experiments, bug fixes. every chapter follows [`AUTHORING.md`](AUTHORING.md). the workflow:

```bash
python scripts/scaffold.py plan 14
```

that creates chapter 14 and all its topics from the plan. replace the template content, then check your work:

```bash
python scripts/audit.py --chapter 14 --marimo
```

```bash
python scripts/verify.py --chapter 14
```

`audit.py` checks structure and consistency. `verify.py` checks that reference solutions pass, untouched stubs fail every stage, and labs run. CI runs both on every PR.

## license

MIT. learn freely, build amazing things.
