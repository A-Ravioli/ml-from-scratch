"""Checks for the chapter 30 capstone: the thesis tooling.

There are no accuracy thresholds: the thesis is original work, judged with rubric.md. These checks cover the tooling
every thesis result goes through. make_project creates exactly the skeleton and never overwrites; config_hash is
canonical, order-free and type-faithful; RunRegistry appends, reloads, refuses duplicates and bad metrics, and
groups runs by name; results_table pairs on shared seeds and matches mlfs.research.ablation_table; render_report
fills every placeholder kind from the registry with the right intervals and refuses malformed ones. Every test
works in its own temporary directory.
"""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlfs.research import ablation_table, mean_ci
from mlfs.testing import bounded

SKELETON = ["README.md", "configs/.gitkeep", "figures/.gitkeep", "log.md", "paper/paper.md", "plan.json",
            "proposal.md", "runs/.gitkeep"]


# ---------------------------------------------------------------- stage 1: make_project


def test_stage1_creates_the_skeleton(nb):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "theses" / "mine"
        got = nb.make_project(root, "Label smoothing at small scale", question="does it help?", author="A. Learner")
        assert got == SKELETON, f"make_project should create exactly {SKELETON} (sorted, relative paths), got {got}"
        readme = (root / "README.md").read_text()
        assert readme.splitlines()[0] == "# Label smoothing at small scale", "the README starts with '# title'"
        assert "does it help?" in readme and "A. Learner" in readme, "the README states the question and the author"
        paper = (root / "paper" / "paper.md").read_text()
        heads = [line[3:] for line in paper.splitlines() if line.startswith("## ")]
        assert heads == ["abstract", "introduction", "related work", "method", "experiments", "limitations",
                         "conclusion", "references"], f"the paper template's '## ' sections, in order; got {heads}"
        assert isinstance(json.loads((root / "plan.json").read_text()), dict), "plan.json holds a JSON task graph"
        for d in ("configs", "runs", "figures"):
            assert (root / d).is_dir()


def test_stage1_never_overwrites(nb):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("my own notes\n")
        (root / "paper").mkdir()
        (root / "paper" / "paper.md").write_text("draft 3\n")
        got = nb.make_project(root, "t", question="q")
        assert got == [p for p in SKELETON if p not in ("README.md", "paper/paper.md")], (
            f"existing files are skipped and left out of the returned list, got {got}"
        )
        assert (root / "README.md").read_text() == "my own notes\n", "make_project must never overwrite a file"
        assert (root / "paper" / "paper.md").read_text() == "draft 3\n"
        before = {p: (root / p).read_text() for p in SKELETON}
        assert nb.make_project(root, "t", question="q") == [], "a second run creates nothing"
        assert {p: (root / p).read_text() for p in SKELETON} == before, "and changes nothing"


def test_stage1_author_and_refusals(nb):
    with tempfile.TemporaryDirectory() as tmp:
        nb.make_project(Path(tmp) / "a", "title", question="q?")
        assert "author" not in (Path(tmp) / "a" / "README.md").read_text().lower(), (
            "without an author, the README has no author line"
        )
        for title, question in (("", "q"), ("  ", "q"), ("t", ""), ("t", " ")):
            with pytest.raises(ValueError):
                nb.make_project(Path(tmp) / "b", title, question=question)
        assert not (Path(tmp) / "b").exists() or not any((Path(tmp) / "b").rglob("*.md")), (
            "refuse before writing anything"
        )


# ---------------------------------------------------------------- stage 2: config_hash


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def test_stage2_canonical_json(nb):
    got = nb.config_hash({"width": 128, "lr": 0.1})
    want = _sha('{"lr":0.1,"width":128}')
    assert got == want, (
        f"hash the canonical JSON (sorted keys, separators (',', ':')) with sha256 and keep 12 hex digits: expected "
        f"{want}, got {got}"
    )
    nested = {"model": {"width": 64, "depth": 2}, "opt": {"name": "sgd", "lr": 0.05}, "tags": ["a", "b"]}
    reordered = {"tags": ["a", "b"], "opt": {"lr": 0.05, "name": "sgd"}, "model": {"depth": 2, "width": 64}}
    assert nb.config_hash(nested) == nb.config_hash(reordered), "key order, at any depth, must not matter"
    assert nb.config_hash({"name": "café"}) == _sha('{"name":"café"}'), "keep non-ASCII text as is (ensure_ascii=False)"


def test_stage2_types(nb):
    base = {"lr": 0.1, "width": 128, "sizes": [1, 2]}
    assert nb.config_hash({**base, "sizes": (1, 2)}) == nb.config_hash(base), "a tuple hashes like a list"
    assert nb.config_hash({**base, "lr": np.float64(0.1), "width": np.int64(128)}) == nb.config_hash(base), (
        "NumPy scalars hash like the Python numbers they hold"
    )
    assert nb.config_hash({**base, "sizes": np.array([1, 2])}) == nb.config_hash(base), "arrays hash like lists"
    assert nb.config_hash({**base, "width": 128.0}) != nb.config_hash(base), "1 and 1.0 are different configs in JSON"
    assert nb.config_hash({"flag": True}) != nb.config_hash({"flag": 1}), "True and 1 differ"
    assert nb.config_hash({**base, "lr": 0.1000001}) != nb.config_hash(base), "any change changes the hash"
    with pytest.raises(ValueError):
        nb.config_hash({"lr": float("nan")})
    with pytest.raises(TypeError):
        nb.config_hash({"fn": object()})


# ---------------------------------------------------------------- stage 3: RunRegistry

CFG_A = {"width": 128, "smoothing": 0.0}
CFG_B = {"width": 128, "smoothing": 0.1}


def test_stage3_log_and_reload(nb):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs" / "registry.jsonl"
        reg = nb.RunRegistry(path)
        assert len(reg) == 0
        rec = reg.log(CFG_A, 0, {"acc": 0.91, "loss": 0.3}, name="base", notes="first run")
        h = nb.config_hash(CFG_A)
        assert rec["config_hash"] == h and rec["id"] == f"{h}-0" and rec["seed"] == 0 and rec["name"] == "base", (
            f"a record carries config_hash, id = '<hash>-<seed>', seed and name; got {rec}"
        )
        assert rec["config"] == CFG_A and rec["metrics"] == {"acc": 0.91, "loss": 0.3} and rec["notes"] == "first run"
        reg.log(CFG_A, 1, {"acc": 0.92, "loss": 0.29}, name="base")
        reg.log(CFG_B, 0, {"acc": 0.93, "loss": 0.35})
        lines = path.read_text().splitlines()
        assert len(lines) == 3 and all(json.loads(line) for line in lines), "one JSON object per line, appended"
        again = nb.RunRegistry(path)
        assert len(again) == 3 and again.runs() == reg.runs(), "a new RunRegistry on the same file sees every run"
        assert again.runs(name=nb.config_hash(CFG_B))[0]["seed"] == 0, "without a name, the name is the config hash"


def test_stage3_refusals(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = nb.RunRegistry(Path(tmp) / "r.jsonl")
        reg.log(CFG_A, 0, {"acc": 0.9}, name="base")
        with pytest.raises(ValueError):
            reg.log(dict(reversed(list(CFG_A.items()))), 0, {"acc": 0.95}, name="base")  # same config, same seed
        for seed in (True, 1.0, "1"):
            with pytest.raises(ValueError):
                reg.log(CFG_A, seed, {"acc": 0.9})
        for metrics in ({}, {"acc": float("nan")}, {"acc": float("inf")}, {"acc": "0.9"}, {"acc": True}):
            with pytest.raises(ValueError):
                reg.log(CFG_A, 5, metrics)
        assert len(reg) == 1 and len((Path(tmp) / "r.jsonl").read_text().splitlines()) == 1, (
            "refused runs are neither kept nor written"
        )


def test_stage3_queries(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = nb.RunRegistry(Path(tmp) / "r.jsonl")
        for seed in (2, 0, 1):
            reg.log(CFG_A, seed, {"acc": 0.90 + seed / 100}, name="base")
        reg.log(CFG_B, 0, {"acc": 0.95}, name="ls")
        assert [r["seed"] for r in reg.runs(name="base")] == [2, 0, 1], "runs() keeps logging order"
        assert len(reg.runs(config_hash=nb.config_hash(CFG_B))) == 1 and len(reg.runs()) == 4
        assert reg.runs(name="base", config_hash=nb.config_hash(CFG_B)) == [], "both filters apply together"
        got = reg.values("base", "acc")
        assert list(got) == [0, 1, 2] and np.allclose(list(got.values()), [0.90, 0.91, 0.92]), (
            f"values(name, metric) is seed -> value, sorted by seed; got {got}"
        )
        with pytest.raises(ValueError):
            reg.values("nope", "acc")
        with pytest.raises(ValueError):
            reg.values("ls", "loss")
        reg.log({"width": 256, "smoothing": 0.0}, 0, {"acc": 0.9}, name="base")
        with pytest.raises(ValueError):
            reg.values("base", "acc")  # one name, two configurations


# ---------------------------------------------------------------- stage 4: results_table


def _registry(nb, tmp, runs):
    reg = nb.RunRegistry(Path(tmp) / "r.jsonl")
    for name, cfg, seeds, accs in runs:
        for s, a in zip(seeds, accs):
            reg.log(cfg, s, {"acc": a, "loss": 1 - a}, name=name)
    return reg


def test_stage4_matches_ablation_table(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = _registry(nb, tmp, [("base", CFG_A, [0, 1, 2], [0.90, 0.92, 0.91]),
                                  ("ls", CFG_B, [2, 0, 1], [0.89, 0.88, 0.90])])
        got = nb.results_table(reg, "acc", ["base", "ls"], baseline="base")
        want = ablation_table({"base": [0.90, 0.92, 0.91], "ls": [0.88, 0.90, 0.89]}, baseline="base")
        assert got == want, (
            "with every seed shared, results_table is mlfs.research.ablation_table on the values in seed order "
            f"(ls's runs were logged out of order).\nexpected:\n{want}\ngot:\n{got}"
        )
        got = nb.results_table(reg, "loss", ["ls", "base"], higher_is_better=False, alpha=0.1, fmt="{:.2f}")
        want = ablation_table({"ls": [0.12, 0.10, 0.11], "base": [0.10, 0.08, 0.09]}, higher_is_better=False,
                              alpha=0.1, fmt="{:.2f}")
        assert got == want, "metric, order of names, higher_is_better, alpha and fmt are all passed through"


def test_stage4_pairs_on_shared_seeds(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = _registry(nb, tmp, [("base", CFG_A, [0, 1, 2, 3], [0.90, 0.92, 0.91, 0.5]),
                                  ("ls", CFG_B, [0, 1, 2, 5, 6], [0.88, 0.90, 0.89, 0.1, 0.2])])
        got = nb.results_table(reg, "acc", ["base", "ls"], baseline="base")
        want = ablation_table({"base": [0.90, 0.92, 0.91], "ls": [0.88, 0.90, 0.89]}, baseline="base")
        note = "*paired on 3 shared seeds; left out: base seeds 3; ls seeds 5, 6*"
        assert got == want + "\n\n" + note, (
            f"pair on the seeds every config has (0, 1, 2), then note what was left out:\n{note}\ngot:\n{got}"
        )
        reg2 = _registry(nb, Path(tmp) / "x", [("a", CFG_A, [0, 1], [0.9, 0.8]), ("b", CFG_B, [1, 2], [0.7, 0.6])])
        with pytest.raises(ValueError):
            nb.results_table(reg2, "acc", ["a", "b"])  # only one shared seed


# ---------------------------------------------------------------- stage 5: render_report


def _render(nb, *args, **kwargs):
    return bounded(nb.render_report, *args, what="render_report",
                   hint="replace each placeholder once (re.sub with a function), and raise on a bad one.", **kwargs)


def _three(nb, tmp):
    return _registry(nb, tmp, [("base", CFG_A, [0, 1, 2, 3], [0.90, 0.92, 0.91, 0.93]),
                               ("ls", CFG_B, [0, 1, 2, 3], [0.93, 0.93, 0.95, 0.94]),
                               ("big", {"width": 512}, [0, 1, 2], [0.95, 0.96, 0.97])])


def test_stage5_mean_n_and_diff(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = _three(nb, tmp)
        m, lo, hi = mean_ci([0.90, 0.92, 0.91, 0.93])
        d, dlo, dhi = mean_ci(np.array([0.93, 0.93, 0.95, 0.94]) - np.array([0.90, 0.92, 0.91, 0.93]))
        b, blo, bhi = mean_ci(np.array([0.95, 0.96, 0.97]) - np.array([0.90, 0.92, 0.91]))
        template = ("base reaches {{mean:base:acc}} over {{n:base:acc}} seeds; ls changes it by {{diff:ls:base:acc}}; "
                    "big by {{diff:big:base:acc}}. untouched: {not a placeholder} and {{ also not }}.")
        got = _render(nb, template, reg)
        want = (f"base reaches {m:.3f} [{lo:.3f}, {hi:.3f}] over 4 seeds; ls changes it by "
                f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]; big by {b:+.3f} [{blo:+.3f}, {bhi:+.3f}]. untouched: "
                "{not a placeholder} and {{ also not }}.")
        assert got == want, (
            "placeholders become numbers from the registry: mean with its 95% t interval, the number of runs, and "
            f"the paired difference on shared seeds with signs.\nexpected: {want}\ngot:      {got}"
        )


def test_stage5_table_alpha_and_fmt(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = _three(nb, tmp)
        got = _render(nb, "results:\n\n{{table:acc:base,ls}}\n", reg)
        assert got == "results:\n\n" + nb.results_table(reg, "acc", ["base", "ls"], baseline="base") + "\n", (
            "{{table:metric:a,b,...}} is results_table with the first name as the baseline"
        )
        m, lo, hi = mean_ci([0.90, 0.92, 0.91, 0.93], alpha=0.2)
        got = _render(nb, "{{mean:base:acc}}", reg, alpha=0.2, fmt=".1%")
        assert got == f"{m:.1%} [{lo:.1%}, {hi:.1%}]", f"alpha and fmt (a format spec) are used, got {got}"
        got = _render(nb, "{{table:acc:base,big}}", reg, alpha=0.2)
        assert "80% CI" in got and "left out: base seeds 3" in got, "alpha reaches the table, which pairs on seeds"
        got = _render(nb, "{{table:acc:base,ls}}", reg, fmt=".2f")
        assert got == nb.results_table(reg, "acc", ["base", "ls"], baseline="base", fmt="{:.2f}"), (
            "fmt reaches the table too: fmt='.2f' becomes results_table's fmt='{:.2f}'"
        )


def test_stage5_refusals(nb):
    with tempfile.TemporaryDirectory() as tmp:
        reg = _three(nb, tmp)
        for bad in ("{{median:base:acc}}", "{{mean:base}}", "{{diff:ls:acc}}", "{{mean:nobody:acc}}",
                    "{{mean:base:f1}}", "{{n:base:acc:extra}}", "{{n:nobody:acc}}", "{{n:base:f1}}"):
            with pytest.raises(ValueError):
                _render(nb, f"x {bad} y", reg)
        assert _render(nb, "no placeholders here", reg) == "no placeholders here"
