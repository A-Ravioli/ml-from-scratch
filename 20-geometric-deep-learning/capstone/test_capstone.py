"""Checks for capstone.py (GNNs on molecular energies).

the benchmark is checked against an independent Lennard-Jones oracle and the generator's stated
geometry; the batching against a brute-force radius graph; the four models for exact invariances in
float64 (and the coordinate baseline for *not* being rotation-invariant); training for beating the
per-atom-mean baseline on a small synthetic config; the audit and the intervals against their
definitions. training runs one bounded forward pass first, so a runaway loop fails instead of hanging.
"""
import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlfs.testing import bounded  # noqa: E402

SIGMA = np.array([[1.0, 1.1], [1.1, 1.2]])
EPS = np.array([[1.0, math.sqrt(0.6)], [math.sqrt(0.6), 0.6]])
R_CUT = 2.0


def _oracle_energy(pos, species):
    pos, species = np.asarray(pos, dtype=np.float64), np.asarray(species)
    total = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            r = float(np.linalg.norm(pos[i] - pos[j]))
            if r < R_CUT:
                s, e = SIGMA[species[i], species[j]], EPS[species[i], species[j]]
                total += 4 * e * ((s / r) ** 12 - (s / r) ** 6) - 4 * e * ((s / R_CUT) ** 12 - (s / R_CUT) ** 6)
    return total


def _dataset(nb, n, seed, n_atoms=(5, 12)):
    return bounded(nb.make_dataset, n, n_atoms=n_atoms, seed=seed, budget=20_000_000, what="make_dataset",
                   hint="cap the number of placement attempts.")


def _orthogonal(seed, proper=True):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if (np.linalg.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return torch.tensor(q, dtype=torch.float64)


# ---------------------------------------------------------------- stage 1: the benchmark


def test_stage1_energy_matches_oracle(nb):
    pos = np.array([[0.0, 0, 0], [1.12, 0, 0], [0, 1.3, 0.2], [2.5, 0.1, 0.0]])
    species = np.array([0, 1, 1, 0])
    got, want = nb.lj_energy(pos, species), _oracle_energy(pos, species)
    assert isinstance(got, float), f"lj_energy returns a Python float; got {type(got).__name__}"
    assert abs(got - want) < 1e-10, (
        f"truncated, shifted Lennard-Jones energy should be {want:.6f}, got {got:.6f}. pairs with r >= {R_CUT} "
        "contribute 0; each pair within the cutoff contributes V_ab(r) - V_ab(cutoff) with the pair's own sigma and eps")
    assert abs(nb.lj_energy(torch.tensor(pos), torch.tensor(species)) - want) < 1e-10, "tensors work too"
    two = np.array([[0.0, 0, 0], [2 ** (1 / 6), 0, 0]])
    v_cut = 4 * ((1 / R_CUT) ** 12 - (1 / R_CUT) ** 6)
    assert abs(nb.lj_energy(two, [0, 0]) - (-1.0 - v_cut)) < 1e-12, "two A atoms at the minimum 2^(1/6): -eps - V(cutoff)"


def test_stage1_dataset_geometry(nb):
    mols = _dataset(nb, 40, seed=0)
    assert len(mols) == 40 and set(mols[0]) >= {"pos", "species", "energy"}
    sizes = [len(m["species"]) for m in mols]
    assert min(sizes) >= 5 and max(sizes) <= 12 and len(set(sizes)) > 3, f"n_atoms uniform in [5, 12]; got {sizes}"
    for m in mols:
        p, s = m["pos"], m["species"]
        assert p.dtype == torch.float64 and s.dtype == torch.int64 and p.shape == (len(s), 3)
        assert float(p.mean(0).abs().max()) < 1e-10, "positions are centred on their centroid"
        assert set(s.tolist()) <= {0, 1}
        d = torch.cdist(p, p).numpy()
        sig = SIGMA[s.numpy()][:, s.numpy()]
        np.fill_diagonal(d, np.inf)
        assert (d >= 0.95 * sig - 1e-9).all(), "no two atoms closer than 0.95 sigma_ab"
        assert ((d <= 1.5 * sig + 1e-9).any(axis=1)).all(), "every atom sits within 1.5 sigma of another (it grew there)"
        assert abs(m["energy"] - _oracle_energy(p, s)) < 1e-9, "energy = lj_energy of the stored positions"
    e = np.array([m["energy"] for m in mols])
    assert e.std() > 0.5, f"energies should vary between molecules; std {e.std():.3f}"


def test_stage1_deterministic(nb):
    a, b, c = _dataset(nb, 5, seed=3), _dataset(nb, 5, seed=3), _dataset(nb, 5, seed=4)
    assert all(torch.equal(x["pos"], y["pos"]) for x, y in zip(a, b)), "the same seed gives the same molecules"
    assert not all(x["pos"].shape == y["pos"].shape and torch.equal(x["pos"], y["pos"]) for x, y in zip(a, c))
    small = _dataset(nb, 10, seed=0, n_atoms=(3, 4))
    assert all(3 <= len(m["species"]) <= 4 for m in small), "n_atoms sets the range of cluster sizes"


# ---------------------------------------------------------------- stage 2: batching


def test_stage2_collate(nb):
    mols = _dataset(nb, 4, seed=1)
    b = nb.collate(mols)
    n = sum(len(m["species"]) for m in mols)
    assert b["x"].shape == (n, 2) and b["pos"].shape == (n, 3) and b["batch"].shape == (n,), "x (N, 2), pos (N, 3)"
    assert b["x"].dtype == torch.float32 and b["edge_index"].dtype == torch.int64 and b["num_graphs"] == 4
    assert torch.equal(b["x"].argmax(1), torch.cat([m["species"] for m in mols])), "x is the one-hot species"
    assert torch.allclose(b["y"].double(), torch.tensor([m["energy"] for m in mols], dtype=torch.float64), atol=1e-4)
    want, off = set(), 0
    for m in mols:
        p = m["pos"]
        for i in range(len(p)):
            for j in range(len(p)):
                if i != j and float((p[i] - p[j]).norm()) < R_CUT:
                    want.add((j + off, i + off))
        off += len(p)
    got = set(map(tuple, b["edge_index"].T.tolist()))
    assert got == want, (
        f"edge_index must hold every ordered pair of atoms of the same molecule closer than the cutoff ({R_CUT}), with "
        f"node ids offset per molecule; missing {sorted(want - got)[:4]}, extra {sorted(got - want)[:4]}")
    b64 = nb.collate(mols, cutoff=1.5, dtype=torch.float64)
    assert b64["pos"].dtype == torch.float64 and b64["edge_index"].shape[1] < b["edge_index"].shape[1]


# ---------------------------------------------------------------- stage 3: models


def _predict64(nb, model, mols):
    m = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        return m(nb.collate(mols, dtype=torch.float64))


@pytest.mark.parametrize("kind", ["gcn", "gin", "gin_pos", "egnn"])
def test_stage3_models_and_invariances(nb, kind):
    mols = _dataset(nb, 6, seed=2)
    model = nb.build_model(kind, hidden=16, n_layers=2, seed=0)
    out = model(nb.collate(mols))
    assert out.shape == (6,), f"build_model({kind!r}) gives a model mapping a collated batch to (B,) energies; got {tuple(out.shape)}"
    assert sum(p.numel() for p in model.parameters()) > 0
    base = _predict64(nb, model, mols)
    rng = np.random.default_rng(0)
    perms = [torch.tensor(rng.permutation(len(m["species"]))) for m in mols]
    permuted = [dict(m, pos=m["pos"][p], species=m["species"][p]) for m, p in zip(mols, perms)]
    err = float((_predict64(nb, model, permuted) - base).abs().max())
    assert err < 1e-8, f"{kind}: relabelling atoms must not change the predicted energy; change {err:.1e}"
    reordered = _predict64(nb, model, mols[::-1]).flip(0)
    assert float((reordered - base).abs().max()) < 1e-8, f"{kind}: molecules in a batch must not interact"
    rotated = [dict(m, pos=m["pos"] @ _orthogonal(5, proper=False).T + 2.0) for m in mols]
    err = float((_predict64(nb, model, rotated) - base).abs().max())
    if kind == "gin_pos":
        assert err > 1e-3, (
            "the 'gin_pos' baseline feeds raw coordinates to GIN as node features, so turning the molecules must "
            f"change its predictions; change {err:.1e}")
    else:
        assert err < 1e-8, (
            f"{kind}: rotating/reflecting and translating every molecule must not change the predicted energy; "
            f"change {err:.1e}")
    if kind != "gin_pos":
        far = [dict(m, pos=torch.cat([m["pos"], m["pos"] + 50.0]), species=torch.cat([m["species"], m["species"]]))
               for m in mols[:2]]
        doubled = _predict64(nb, model, far)
        assert float((doubled - 2 * base[:2]).abs().max()) < 1e-8, (
            f"{kind}: a 'molecule' made of two far-apart copies (no edges between them) has twice the energy, and the "
            "model must predict exactly twice: sum the per-atom contributions (global_add_pool), don't average them")
    with pytest.raises(ValueError):
        nb.build_model("transformer")


# ---------------------------------------------------------------- stage 4: training


def _baseline_mae(train, test):
    mu = sum(m["energy"] for m in train) / sum(len(m["species"]) for m in train)
    return float(np.mean([abs(mu * len(m["species"]) - m["energy"]) for m in test]))


def test_stage4_fit_and_evaluate(nb):
    ratios = []
    for k, seed in enumerate((10, 20)):  # two independent datasets: a rare hard one can't fail correct code alone
        train, test = _dataset(nb, 200, seed=seed, n_atoms=(4, 7)), _dataset(nb, 150, seed=seed + 1, n_atoms=(4, 7))
        base = _baseline_mae(train, test)
        model = nb.build_model("egnn", hidden=32, n_layers=2, seed=k)
        bounded(model, nb.collate(train[:8]), what="the model's forward pass")
        model = nb.fit(model, train, epochs=30, lr=3e-3, seed=k)
        errs = nb.evaluate(model, test)
        assert isinstance(errs, np.ndarray) and errs.shape == (150,) and (errs >= 0).all(), (
            "evaluate returns one absolute error per test molecule, a numpy array (n,)")
        want = np.array([abs(float(p) - m["energy"]) for p, m in zip(model(nb.collate(test)).detach(), test)])
        assert np.allclose(errs, want, atol=1e-3), "evaluate's errors are |model(collate(mols)) - energy|, in energy units"
        ratios.append(errs.mean() / base)
    assert min(ratios) < 0.85, (
        "an EGNN trained by your fit (200 molecules, 30 epochs) should beat the per-atom-mean baseline by at least 15% on "
        "at least one of two datasets (the reference: 0.55-0.81 of the baseline on 30 datasets, 0.93 on the hardest of 100; an untrained model "
        f"is at about 1); got test MAE / baseline = {', '.join(f'{r:.3f}' for r in ratios)}")


# ---------------------------------------------------------------- stage 5: audit and intervals


def test_stage5_invariance_report(nb):
    mols = _dataset(nb, 5, seed=12)
    keys = {"rotation", "reflection", "translation", "permutation"}
    egnn = nb.build_model("egnn", hidden=16, n_layers=2, seed=1)
    x0 = {k: v.detach().clone() for k, v in egnn.state_dict().items()}  # before any audit runs
    rep = nb.invariance_report(egnn, mols, n_transforms=2, seed=0)
    assert set(rep) == keys and all(isinstance(v, float) for v in rep.values()), f"a dict of floats with keys {keys}"
    assert max(rep.values()) < 1e-8, f"the EGNN is exactly invariant: in float64 every change is ~1e-12; got {rep}"
    pos_model = nb.build_model("gin_pos", hidden=16, n_layers=2, seed=1)
    rep = nb.invariance_report(pos_model, mols, n_transforms=2, seed=0)
    assert rep["rotation"] > 1e-3 and rep["translation"] > 1e-3 and rep["permutation"] < 1e-8, (
        f"GIN on raw coordinates changes under rotations and translations but not relabelling; got {rep}")
    after = egnn.state_dict()
    assert all(after[k].dtype == v.dtype and torch.equal(after[k], v) for k, v in x0.items()), (
        "the audit must not modify the model: work on a float64 COPY (copy.deepcopy(model).double()), since .double() "
        "converts a module in place")

    class SignedVolume(torch.nn.Module):
        """a stub 'model': the signed volume of each molecule's first four atoms (rotation-invariant, mirror-odd)."""

        def forward(self, batch):
            out = []
            for g in range(batch["num_graphs"]):
                p = batch["pos"][batch["batch"] == g][:4]
                out.append(torch.linalg.det(p[1:] - p[:1]))
            return torch.stack(out)

    rep = nb.invariance_report(SignedVolume(), mols, n_transforms=2, seed=0)
    assert rep["rotation"] < 1e-8 and rep["translation"] < 1e-8, f"a signed volume survives rotations and shifts; got {rep}"
    assert rep["reflection"] > 1e-3, (
        f"a signed volume flips sign under a reflection, so the audit's reflection entry must be large; got {rep}. is your "
        "'reflection' really an improper orthogonal matrix (det -1)?")


def test_stage5_intervals(nb):
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(1.0, 0.5, 400))
    mae, lo, hi = nb.mae_ci(a, n_boot=2000, seed=0)
    assert abs(mae - a.mean()) < 1e-12 and lo < mae < hi, "(MAE, lo, hi) with lo < MAE < hi"
    se = a.std() / np.sqrt(len(a))
    assert 0.75 * 3.92 * se < hi - lo < 1.25 * 3.92 * se, (
        f"a 95% bootstrap interval of a mean is about 3.92 standard errors wide ({3.92 * se:.4f}); got {hi - lo:.4f}")
    b = a + 0.05 + rng.normal(0, 0.01, 400)  # model b: consistently a little worse on the same molecules
    d, dlo, dhi = nb.mae_difference_ci(b, a, n_boot=2000, seed=0)
    assert abs(d - (b.mean() - a.mean())) < 1e-12 and 0 < dlo < d < dhi, (
        f"the paired interval for MAE_b - MAE_a must exclude 0 here; got ({dlo:.4f}, {dhi:.4f})")
    assert dhi - dlo < 0.2 * (hi - lo), (
        "pair the errors: resample test molecules jointly for both models. an unpaired interval would be about as wide "
        "as each model's own interval and miss a difference this small")
