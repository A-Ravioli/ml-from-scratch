"""Tests for the curriculum tooling (scripts/ and mlfs/). Run with `pytest tests`."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _common import parse_curriculum, slugify  # noqa: E402

from mlfs import checks, lesson, notebooks  # noqa: E402


def script(name: str, *args: str, root: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / name), "--root", str(root), *args],
                          capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def scaffolded(tmp_path_factory) -> Path:
    """A throwaway curriculum root with chapter 14, two topics, and one registered module."""
    root = tmp_path_factory.mktemp("curriculum")
    assert script("scaffold.py", "chapter", "14", root=root).returncode == 0
    assert script("scaffold.py", "topic", "14", "2", "--provides", "demo", root=root).returncode == 0
    assert script("scaffold.py", "topic", "14", "3", root=root).returncode == 0
    return root


def topic(root: Path, num: str) -> Path:
    chapter = next(root.glob("14-*"))
    return next(chapter.glob(f"{num}-*"))


def test_slugify():
    assert slugify("MDPs & bellman equations") == "mdps-and-bellman-equations"
    assert slugify("infinite width: NTK & mean field") == "infinite-width-ntk-and-mean-field"


def test_parse_curriculum_matches_plan():
    plan = parse_curriculum()
    assert sorted(plan) == [f"{i:02d}" for i in range(31)]
    ch14 = plan["14"]
    assert ch14.dirname == "14-neural-networks-from-scratch"
    assert [t.num for t in ch14.topics] == ["01", "02", "03", "04", "05", "06"]
    assert ch14.topics[2].title == "a tensor autograd engine"
    assert "mlfs.tensor" in ch14.topics[2].build
    assert ch14.capstone and ch14.meta.startswith("L3")
    assert all(c.topics for n, c in plan.items() if n != "29")


def test_lesson_extract_stops_at_same_level_and_ignores_fences():
    text = "# t\n## a\none\n```\n## not a heading\n```\n### sub\ntwo\n## b\nthree\n"
    got = lesson.extract(text, "a")
    assert got.startswith("## a") and "## not a heading" in got and "### sub" in got
    assert "three" not in got
    assert lesson.extract(text, "missing") is None


def test_scaffold_fills_plan_and_registers(scaffolded):
    t = topic(scaffolded, "02")
    assert t.name == "02-scalar-reverse-mode-autodiff"
    assert "computational graphs, backprop" in (t / "lesson.md").read_text()
    readme = next(scaffolded.glob("14-*")) / "README.md"
    assert "[a tensor autograd engine](03-a-tensor-autograd-engine/)" in readme.read_text()
    registry = (scaffolded / "mlfs" / "registry.toml").read_text()
    assert 'demo = "14-neural-networks-from-scratch/02-scalar-reverse-mode-autodiff"' in registry


def test_scaffold_never_overwrites(scaffolded):
    lesson_md = topic(scaffolded, "02") / "lesson.md"
    lesson_md.write_text("# my real lesson\n")
    out = script("scaffold.py", "topic", "14", "2", root=scaffolded)
    assert "skip" in out.stdout
    assert lesson_md.read_text() == "# my real lesson\n"


def test_registry_import_switches_between_learner_and_reference(scaffolded):
    code = "import numpy as np; from mlfs.demo import variance; print(variance(np.array([1., 2, 3, 6])))"
    base = dict(os.environ, MLFS_ROOT=str(scaffolded))
    base.pop("MLFS_SOLUTIONS", None)
    base.pop("MLFS_REFERENCE", None)
    learner = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=base)
    assert "NotImplementedError: stage2" in learner.stderr
    ref = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(base, MLFS_REFERENCE="demo"))
    assert ref.stdout.strip() == "3.5"


def test_registry_merges_several_topics_into_one_module(scaffolded):
    for num in ("2", "3"):
        assert script("scaffold.py", "topic", "14", num, "--provides", "multi", root=scaffolded).returncode == 0
    code = ("from mlfs import multi; print(len(multi.__notebooks__), sorted(n for n in vars(multi) "
            "if not n.startswith('_')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, MLFS_ROOT=str(scaffolded), MLFS_REFERENCE="*"))
    assert out.stdout.strip() == "2 ['mean', 'variance']", out.stderr


def test_live_checks_report_todo_pass_and_fail(scaffolded):
    t = topic(scaffolded, "02")
    stub = notebooks.load(notebooks.resolve(t, "exercise", solution=False))
    ref = notebooks.load(notebooks.resolve(t, "exercise", solution=True))
    assert {r.status for r in checks.run_stage(t, 1, {"mean": stub.mean})} == {"todo"}
    assert {r.status for r in checks.run_stage(t, 1, {"mean": ref.mean})} == {"pass"}
    wrong = checks.run_stage(t, 2, {"variance": lambda x: float(np.var(x, ddof=1))})
    assert wrong[0].status == "fail" and "n-1" in wrong[0].message
    missing = checks.run_stage(t, 2, {})
    assert missing[0].status == "error" and "checks.run(2" in missing[0].message


def test_audit_reports_templates_and_errors(scaffolded):
    out = script("audit.py", "--chapter", "14", root=scaffolded)
    assert out.returncode == 0, out.stdout
    assert "2/6 planned exist" in out.stdout and "still templates" in out.stdout
    assert script("audit.py", "--chapter", "14", "--strict", root=scaffolded).returncode == 1

    sol = topic(scaffolded, "03") / "solutions" / "exercise.py"
    original = sol.read_text()
    sol.write_text(original.replace("return float(np.sum(x) / len(x))", 'raise NotImplementedError("stage1")'))
    try:
        out = script("audit.py", "--chapter", "14", root=scaffolded)
        assert out.returncode == 1 and "placeholder in reference solution" in out.stdout
    finally:
        sol.write_text(original)


@pytest.mark.slow
def test_verify_passes_clean_scaffold_and_catches_bugs(scaffolded):
    out = script("verify.py", "--chapter", "14", "-j", "4", root=scaffolded)
    assert out.returncode == 0, out.stdout
    assert "3/3 verified" in out.stdout

    ex = topic(scaffolded, "03") / "exercise.py"
    original = ex.read_text()
    ex.write_text(original.replace('raise NotImplementedError("stage2: implement variance(x)")',
                                   "return float(np.var(x))"))
    try:
        rel = str(topic(scaffolded, "03").relative_to(scaffolded))
        out = script("verify.py", "--topic", rel, "--skip-labs", root=scaffolded)
        assert out.returncode == 1 and "stage 2 passes against the untouched" in out.stdout
    finally:
        ex.write_text(original)


def test_marimo_check_accepts_templates():
    files = [str(p) for p in (REPO / "templates").rglob("*.py") if not p.name.startswith("test_")]
    proc = subprocess.run([sys.executable, "-m", "marimo", "check", "--strict", *files],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_contracts_doc_is_up_to_date():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "contracts_doc.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_stubify_turns_staged_definitions_into_stubs():
    from stubify import stubify

    src = (REPO / "templates" / "topic" / "solutions" / "exercise.py").read_text()
    out = stubify(src)
    assert 'raise NotImplementedError("stage1: implement mean")' in out
    assert 'raise NotImplementedError("stage2: implement variance")' in out
    assert "np.sum(x) / len(x)" not in out and "(reference solution)" not in out
    multi = 'def f(x, modified=False):\n    """stage 1 (modified=False) and stage 2 (modified=True): f."""\n    return x\n'
    stub = stubify(multi)
    assert "if modified:" in stub and "stage2: implement f with modified=True" in stub
    assert "stage1: implement f with modified=False" in stub


def test_live_checks_expand_parametrized_tests(tmp_path):
    (tmp_path / "test_exercise.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('a,b', [(1, 2), (3, 4)])\n"
        "@pytest.mark.parametrize('c', [10, pytest.param(20)])\n"
        "def test_stage1_add(nb, a, b, c):\n"
        "    assert nb.add(a, b) == a + b and c in (10, 20)\n")
    results = checks.run_stage(tmp_path, 1, {"add": lambda x, y: x + y})
    assert len(results) == 4 and {r.status for r in results} == {"pass"}, results
    assert "a=1" in results[0].name and "c=" in results[0].name


def test_bounded_stops_runaway_loops_and_restores_tracer():
    import sys

    from mlfs.testing import bounded

    def forever():
        while True:
            pass

    before = sys.gettrace()
    with pytest.raises(AssertionError, match="didn't finish"):
        bounded(forever, budget=10_000, what="forever")
    assert sys.gettrace() is before
    assert bounded(lambda a, b=0: a + b, 2, b=3) == 5


def test_live_checks_accept_dict_parameters(tmp_path):
    (tmp_path / "test_exercise.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('cfg', [{'a': 1}, {'a': 2}])\n"
        "def test_stage1_cfg(nb, cfg):\n"
        "    assert nb.get(cfg) == cfg['a']\n")
    results = checks.run_stage(tmp_path, 1, {"get": lambda c: c["a"]})
    assert [r.status for r in results] == ["pass", "pass"], results


def test_audit_allows_abstract_not_implemented_but_flags_stage_stubs():
    import re

    from audit import PLACEHOLDERS

    assert not PLACEHOLDERS.search('        raise NotImplementedError("subclasses implement fit")')
    assert PLACEHOLDERS.search('    raise NotImplementedError("stage2: implement fit")')
    assert PLACEHOLDERS.search("    # TODO finish")


def test_toy_generators_shapes_and_determinism():
    from mlfs import datasets as d

    X, y = d.make_blobs(100, 4, rng=0)
    assert X.shape == (100, 2) and set(y) == {0, 1, 2, 3}
    assert np.array_equal(d.make_moons(50, rng=1)[0], d.make_moons(50, rng=1)[0])
    Xc, yc = d.make_circles(60, rng=2)
    assert Xc.shape == (60, 2) and yc.sum() == 30
    Xs, t = d.make_swiss_roll(80, rng=3)
    assert Xs.shape == (80, 3) and t.shape == (80,)
    assert d.eight_gaussians(40, rng=4).shape == (40, 2)


def test_load_topic_falls_back_only_for_unfinished_learner_code(scaffolded, monkeypatch):
    import numpy as np

    monkeypatch.setenv("MLFS_ROOT", str(scaffolded))
    monkeypatch.delenv("MLFS_SOLUTIONS", raising=False)
    monkeypatch.delenv("MLFS_REFERENCE", raising=False)
    t3 = topic(scaffolded, "03")

    def smoke(m):
        m.variance(np.array([1.0, 2.0, 3.0, 6.0]))

    learner = notebooks.load_topic(t3.name, key="demo3", smoke=smoke)  # the template's stubs are unfinished
    assert learner.fallback and "stage" in learner.reason
    assert learner.module.variance(np.array([1.0, 2.0, 3.0, 6.0])) == pytest.approx(3.5)
    assert learner.callout() is not None

    no_smoke = notebooks.load_topic(t3, key="demo3")  # no smoke test: the learner's code, as is
    assert not no_smoke.fallback and no_smoke.callout() is None
    with pytest.raises(NotImplementedError):
        no_smoke.module.variance(np.array([1.0, 2.0]))

    monkeypatch.setenv("MLFS_REFERENCE", "demo3")
    ref = notebooks.load_topic(t3.name, key="demo3", smoke=smoke)
    assert not ref.fallback and ref.module.variance(np.array([1.0, 2.0, 3.0, 6.0])) == pytest.approx(3.5)

    def broken(m):
        raise ValueError("wrong answer")

    monkeypatch.delenv("MLFS_REFERENCE")
    with pytest.raises(ValueError):
        notebooks.load_topic(t3.name, key="demo3", smoke=broken)
    with pytest.raises(FileNotFoundError):
        notebooks.load_topic("99-no-such-topic", key="x")
