"""Check reference solutions against the cross-chapter API contracts in mlfs/contracts.py.

A module is checked once any of its owning topics exist: every contracted name that's present
must match its contracted signature (parameter names, kinds, order, defaults). Once *all* the
owning topics exist, every contracted name must be present too.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import pytest

from mlfs import notebooks, registry
from mlfs.contracts import CONTRACTS

ROOT = notebooks.repo_root()


def _params(sig: inspect.Signature) -> list[tuple]:
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def _contract_sig(text: str) -> inspect.Signature:
    scope: dict = {}
    exec(f"def _f{text}: pass", scope)  # noqa: S102 - trusted, repo-authored strings
    return inspect.signature(scope["_f"])


def _same(contract: str, actual) -> str | None:
    want = _params(_contract_sig(contract))
    try:
        got = _params(inspect.signature(actual))
    except (TypeError, ValueError):
        return "has no inspectable signature"
    if want != got:
        fmt = lambda ps: "(" + ", ".join(n + ("" if d is inspect.Parameter.empty else f"={d!r}")
                                          for n, _, d in ps) + ")"
        return f"signature {fmt(got)} != contract {fmt(want)}"
    return None


def _built(topic: str) -> bool:
    return (ROOT / topic / "solutions" / "exercise.py").exists()


def test_registry_matches_contracts():
    reg = registry.modules()
    assert set(reg) == set(CONTRACTS), (
        f"only in registry: {sorted(set(reg) - set(CONTRACTS))}; "
        f"only in contracts: {sorted(set(CONTRACTS) - set(reg))}")
    for name, spec in CONTRACTS.items():
        assert reg[name] == spec["topics"], f"mlfs.{name}: registry topics != contract topics"
        assert name not in registry.RESERVED


def test_contract_signatures_parse():
    for name, spec in CONTRACTS.items():
        for fn, sig in spec.get("functions", {}).items():
            _contract_sig(sig)
        for cls, methods in spec.get("classes", {}).items():
            for m, sig in methods.items():
                _contract_sig(sig)


@pytest.mark.parametrize("module", sorted(CONTRACTS))
def test_reference_matches_contract(module, monkeypatch):
    spec = CONTRACTS[module]
    built = [t for t in spec["topics"] if _built(t)]
    if not built:
        pytest.skip(f"mlfs.{module}: no owning topic written yet")
    complete = len(built) == len(spec["topics"])
    monkeypatch.setenv("MLFS_REFERENCE", "*")
    monkeypatch.setenv("MLFS_ROOT", str(ROOT))
    if not complete:  # load only the topics that exist so far
        full = registry.modules()
        monkeypatch.setattr(registry, "modules", lambda: {**full, module: built})
    monkeypatch.delitem(sys.modules, f"mlfs.{module}", raising=False)
    mod = importlib.import_module(f"mlfs.{module}")

    problems = []
    for fn, sig in spec.get("functions", {}).items():
        if not hasattr(mod, fn):
            if complete:
                problems.append(f"missing function {fn}")
            continue
        if err := _same(sig, getattr(mod, fn)):
            problems.append(f"{fn}: {err}")
    for cls, methods in spec.get("classes", {}).items():
        if not hasattr(mod, cls):
            if complete:
                problems.append(f"missing class {cls}")
            continue
        obj = getattr(mod, cls)
        for m, sig in methods.items():
            if not hasattr(obj, m):
                problems.append(f"{cls}.{m}: missing")
            elif err := _same(sig, getattr(obj, m)):
                problems.append(f"{cls}.{m}: {err}")
    assert not problems, f"mlfs.{module} ({', '.join(built)}):\n  " + "\n  ".join(problems)

