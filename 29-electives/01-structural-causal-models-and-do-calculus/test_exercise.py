"""Checks for exercise.py (structural causal models & do-calculus).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The graph
checks compare against an independent oracle written here: d-separation by enumerating
every path and applying the blocking rules one path at a time, on random DAGs. Statistical
checks use large samples and tolerances of 5+ standard errors.
"""
import itertools

import numpy as np

from mlfs.testing import bounded

# ---------------------------------------------------------------- helpers and oracles


def _random_dag(n, p, rng):
    """random DAG on nodes 'v0'..'v{n-1}', listed in a shuffled (non-topological) dict order."""
    names = [f"v{i}" for i in range(n)]
    perm = rng.permutation(n)  # perm gives a causal order
    parents = {v: [] for v in names}
    for a, b in itertools.combinations(range(n), 2):
        if rng.random() < p:
            parents[names[perm[b]]].append(names[perm[a]])
    order = rng.permutation(n)
    return {names[i]: parents[names[i]] for i in order}


def _children(parents):
    return {v: [c for c in parents if v in parents[c]] for v in parents}


def _desc_oracle(parents, v):
    ch, out, frontier = _children(parents), set(), [v]
    while frontier:
        nxt = []
        for u in frontier:
            for c in ch[u]:
                if c not in out:
                    out.add(c)
                    nxt.append(c)
        frontier = nxt
    return out


def _paths(parents, x, y):
    """every simple path x ... y in the skeleton, as a node list."""
    nbrs = {v: set(parents[v]) | set(_children(parents)[v]) for v in parents}
    out, stack = [], [[x]]
    while stack:
        path = stack.pop()
        for w in nbrs[path[-1]]:
            if w in path:
                continue
            if w == y:
                out.append(path + [w])
            else:
                stack.append(path + [w])
    return out


def _blocked(parents, path, zs):
    for a, b, c in zip(path, path[1:], path[2:]):
        collider = a in parents[b] and c in parents[b]
        if collider:
            if b not in zs and not (_desc_oracle(parents, b) & zs):
                return True
        elif b in zs:
            return True
    return False


def _dsep_oracle(parents, x, y, zs):
    return all(_blocked(parents, p, set(zs)) for p in _paths(parents, x, y))


def _backdoor_oracle(parents, x, y, zs):
    zs = set(zs)
    if zs & _desc_oracle(parents, x):
        return False
    back = [p for p in _paths(parents, x, y) if p[1] in parents[x]]  # paths starting with an arrow INTO x
    return all(_blocked(parents, p, zs) for p in back)


def _bern(p):
    """Bernoulli(p) from a N(0, 1) noise draw u: P(u < Phi^-1(p)) = p."""
    from scipy.special import ndtri
    return lambda u: (u < ndtri(np.clip(p, 1e-12, 1 - 1e-12))).astype(float)


# ---------------------------------------------------------------- stage 1: walking a DAG


def test_stage1_topological_order_is_valid(nb):
    rng = np.random.default_rng(0)
    for _ in range(30):
        g = _random_dag(7, 0.4, rng)
        order = bounded(nb.topological_order, g, what="topological_order")
        assert sorted(order) == sorted(g), f"the order must list every node exactly once; got {order} for nodes {list(g)}"
        pos = {v: i for i, v in enumerate(order)}
        for v, pa in g.items():
            for p in pa:
                assert pos[p] < pos[v], (
                    f"{p} is a parent of {v}, so it must come first; got order {order} for graph {g}")


def test_stage1_rejects_cycles_and_unknown_parents(nb):
    for bad, why in [({"a": ["b"], "b": ["a"]}, "a 2-cycle a <-> b"),
                     ({"a": [], "b": ["a", "c"], "c": ["b"]}, "the cycle b -> c -> b"),
                     ({"a": ["a"]}, "a self-loop")]:
        try:
            bounded(nb.topological_order, bad, what="topological_order",
                    hint="with a cycle, no node is ever ready: detect that instead of looping forever.")
        except ValueError:
            continue
        raise AssertionError(f"topological_order should raise ValueError on {why}, graph {bad}")
    try:
        nb.topological_order({"a": ["ghost"]})
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("a parent that isn't a node ('ghost') should raise ValueError")


def test_stage1_ancestors_descendants_hand(nb):
    g = {"Y": ["X", "Z"], "X": ["Z"], "Z": ["W"], "W": [], "C": ["X", "Y"]}
    assert nb.ancestors(g, "Y") == {"X", "Z", "W"}, f"ancestors of Y are X, Z, W; got {nb.ancestors(g, 'Y')}"
    assert nb.ancestors(g, "W") == set(), "W has no parents, so no ancestors"
    assert nb.descendants(g, "Z") == {"X", "Y", "C"}, (
        f"descendants of Z are X, Y and C (via X or Y); got {nb.descendants(g, 'Z')}")
    assert "Z" not in nb.descendants(g, "Z") and "Y" not in nb.ancestors(g, "Y"), "a node is not its own ancestor"


def test_stage1_ancestors_descendants_random(nb):
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = _random_dag(8, 0.35, rng)
        for v in g:
            want_d = _desc_oracle(g, v)
            got_d = set(bounded(nb.descendants, g, v, what="descendants"))
            assert got_d == want_d, f"descendants({v}) in {g}: expected {sorted(want_d)}, got {sorted(got_d)}"
            want_a = {u for u in g if v in _desc_oracle(g, u)}
            got_a = set(bounded(nb.ancestors, g, v, what="ancestors"))
            assert got_a == want_a, f"ancestors({v}) in {g}: expected {sorted(want_a)}, got {sorted(got_a)}"


# ---------------------------------------------------------------- stage 2: the SCM


def _linear_scm(nb):
    # dict order puts Y FIRST: its noise is column 0, X's is column 1
    parents = {"Y": ["X"], "X": []}
    mech = {"X": lambda pa, u: u, "Y": lambda pa, u: 2.0 * pa["X"] + u}
    return nb.SCM(parents, mech)


def test_stage2_attributes(nb):
    m = _linear_scm(nb)
    assert m.nodes == ["Y", "X"], f"nodes keep the dict order of parents; got {m.nodes}"
    assert list(m.order) == ["X", "Y"], f"order is a topological order (X before Y); got {m.order}"
    assert m.parents == {"Y": ["X"], "X": []}, "store the parents dict"


def test_stage2_noise_layout_and_values(nb):
    m = _linear_scm(nb)
    s = m.sample(5, rng=3)
    U = np.random.default_rng(3).normal(size=(5, 2))
    assert set(s) == {"X", "Y"} and all(np.shape(v) == (5,) for v in s.values()), (
        "sample returns a dict node -> array of shape (n,)")
    assert np.allclose(s["X"], U[:, 1]) and np.allclose(s["Y"], 2 * U[:, 1] + U[:, 0]), (
        "draw ALL the noise at once, rng.normal(size=(n, len(nodes))), with column k belonging to nodes[k] "
        "(dict order, NOT topological order). here Y is listed first, so Y's noise is column 0. "
        f"expected X={U[:, 1].round(3).tolist()}, got X={np.round(s['X'], 3).tolist()}")
    assert all(np.asarray(v).dtype == float for v in s.values()), "values are float arrays"


def test_stage2_do_keeps_other_noise(nb):
    m = _linear_scm(nb)
    U = np.random.default_rng(4).normal(size=(6, 2))
    s = m.sample(6, do={"X": 1.5}, rng=4)
    assert np.allclose(s["X"], 1.5), f"do(X = 1.5) forces every X to 1.5; got {np.round(s['X'], 3).tolist()}"
    assert np.allclose(s["Y"], 3.0 + U[:, 0]), (
        "under do(X = 1.5), Y = 2 * 1.5 + U_Y with the SAME noise column as without the intervention "
        "(still draw the full noise matrix, even for the intervened node)")
    xs = np.arange(6, dtype=float)
    s2 = m.sample(6, do={"X": xs}, rng=4)
    assert np.allclose(s2["X"], xs) and np.allclose(s2["Y"], 2 * xs + U[:, 0]), "do also accepts an array of shape (n,)"
    try:
        m.sample(3, do={"Q": 1.0}, rng=0)
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("intervening on a node that doesn't exist ('Q') should raise ValueError")


def test_stage2_mechanism_sees_only_its_parents(nb):
    seen = {}

    def f_c(pa, u):
        seen["C"] = sorted(pa)
        return pa["A"] + pa["B"]

    parents = {"C": ["A", "B"], "A": [], "B": ["A"]}
    mech = {"A": lambda pa, u: u, "B": lambda pa, u: 0 * u + 7.0, "C": f_c}
    s = nb.SCM(parents, mech).sample(4, rng=0)
    assert seen["C"] == ["A", "B"], f"C's mechanism should receive a dict with exactly its parents A, B; got {seen['C']}"
    assert np.allclose(s["B"], 7.0) and np.allclose(s["C"], s["A"] + 7.0), "C = A + B"
    const = nb.SCM({"K": []}, {"K": lambda pa, u: 3.0}).sample(4, rng=0)
    assert np.shape(const["K"]) == (4,) and np.allclose(const["K"], 3.0), (
        "a mechanism that returns a scalar should be broadcast to shape (n,)")


def test_stage2_intervention_differs_from_conditioning(nb):
    # Z -> X, Z -> Y, X -> Y with effect 1; Z makes the X-Y regression slope 1 + 3 * cov / var > 1
    parents = {"Z": [], "X": ["Z"], "Y": ["X", "Z"]}
    mech = {"Z": lambda pa, u: u, "X": lambda pa, u: pa["Z"] + u, "Y": lambda pa, u: pa["X"] + 3 * pa["Z"] + u}
    m = nb.SCM(parents, mech)
    obs = m.sample(100_000, rng=5)
    slope = np.cov(obs["X"], obs["Y"])[0, 1] / np.var(obs["X"], ddof=1)
    y1 = m.sample(100_000, do={"X": 1.0}, rng=6)["Y"].mean()
    y0 = m.sample(100_000, do={"X": 0.0}, rng=6)["Y"].mean()
    assert abs(slope - 2.5) < 0.03, f"the observational slope of Y on X should be 2.5 (confounded); got {slope:.3f}"
    assert abs((y1 - y0) - 1.0) < 1e-8, (
        f"with the same seed, E[Y | do(X=1)] - E[Y | do(X=0)] should be exactly the causal effect 1; got {y1 - y0:.4f}")


def test_stage2_intervene_graph(nb):
    g = {"Y": ["X", "Z"], "X": ["Z"], "Z": []}
    h = nb.intervene_graph(g, ["X"])
    assert h == {"Y": ["X", "Z"], "X": [], "Z": []}, f"do(X) cuts the edges INTO X only; got {h}"
    assert g["X"] == ["Z"], "intervene_graph must not modify its input"
    assert nb.intervene_graph(g, "Y") == {"Y": [], "X": ["Z"], "Z": []}, "a single node name also works"


# ---------------------------------------------------------------- stage 3: d-separation and the backdoor criterion


def test_stage3_three_junctions(nb):
    chain = {"A": [], "B": ["A"], "C": ["B"]}
    fork = {"B": [], "A": ["B"], "C": ["B"]}
    coll = {"A": [], "C": [], "B": ["A", "C"], "D": ["B"]}
    ds = nb.d_separated
    assert not ds(chain, {"A"}, {"C"}, set()) and ds(chain, {"A"}, {"C"}, {"B"}), (
        "a chain A -> B -> C is open, and blocked by conditioning on B")
    assert not ds(fork, {"A"}, {"C"}, set()) and ds(fork, {"A"}, {"C"}, {"B"}), (
        "a fork A <- B -> C is open, and blocked by conditioning on B")
    assert ds(coll, {"A"}, {"C"}, set()), "a collider A -> B <- C is blocked when nothing is conditioned on"
    assert not ds(coll, {"A"}, {"C"}, {"B"}), "conditioning on the collider B opens the path A -> B <- C"
    assert not ds(coll, {"A"}, {"C"}, {"D"}), (
        "conditioning on a DESCENDANT of the collider (D, a child of B) also opens it")


def test_stage3_matches_path_oracle(nb):
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(25):
        g = _random_dag(7, 0.35, rng)
        nodes = list(g)
        for _ in range(12):
            x, y = rng.choice(nodes, 2, replace=False)
            rest = [v for v in nodes if v not in (x, y)]
            zs = set(rng.choice(rest, rng.integers(0, 4), replace=False))
            want = _dsep_oracle(g, x, y, zs)
            got = bounded(nb.d_separated, g, {x}, {y}, zs, what="d_separated")
            assert bool(got) == want, (
                f"d_separated({x}, {y} | {sorted(zs)}) in graph {g}: expected {want}, got {got}. "
                "check the collider rule: a collider is open if it OR ANY DESCENDANT is conditioned on.")
            checked += 1
    assert checked == 300
    for _ in range(25):  # set-valued queries: X and Y d-separated iff every pair (x, y) is
        g = _random_dag(7, 0.3, rng)
        nodes = list(rng.permutation(list(g)))
        xs, ys = set(nodes[:2]), set(nodes[2:2 + rng.integers(1, 3)])
        zs = set(nodes[4:4 + rng.integers(0, 3)])
        want = all(_dsep_oracle(g, x, y, zs) for x in xs for y in ys)
        got = bounded(nb.d_separated, g, xs, ys, zs, what="d_separated")
        assert bool(got) == want, (
            f"d_separated({sorted(xs)}, {sorted(ys)} | {sorted(zs)}) in {g}: expected {want}, got {got}. with SETS, "
            "every node of xs must be separated from every node of ys: start the search from all of xs.")
    assert not nb.d_separated({"A": [], "B": []}, {"A", "B"}, {"B"}), (
        "B is in both xs and ys: X and Y share a node, so they can't be d-separated (return False)")


def test_stage3_backdoor_classics(nb):
    bd = nb.is_backdoor_adjustment
    conf = {"Z": [], "X": ["Z"], "Y": ["X", "Z"]}
    assert bd(conf, "X", "Y", {"Z"}) and not bd(conf, "X", "Y", set()), "with Z -> X, Z -> Y, {Z} is valid and {} is not"
    med = {"X": [], "M": ["X"], "Y": ["M"]}
    assert bd(med, "X", "Y", set()) and not bd(med, "X", "Y", {"M"}), (
        "a mediator X -> M -> Y is a descendant of X: adjusting for it is NOT a valid backdoor adjustment")
    coll = {"X": [], "Y": ["X"], "C": ["X", "Y"]}
    assert not bd(coll, "X", "Y", {"C"}), "C is a collider (and a descendant of X): never adjust for it"
    mbias = {"U1": [], "U2": [], "M": ["U1", "U2"], "X": ["U1"], "Y": ["X", "U2"]}
    assert bd(mbias, "X", "Y", set()), "M-bias: the empty set is valid (the path X <- U1 -> M <- U2 -> Y is blocked at M)"
    assert not bd(mbias, "X", "Y", {"M"}), "M-bias: adjusting for the pre-treatment collider M opens a backdoor path"
    assert bd(mbias, "X", "Y", {"M", "U1"}), "adding U1 to {M} blocks the opened path again"


def test_stage3_backdoor_matches_oracle(nb):
    rng = np.random.default_rng(3)
    for _ in range(25):
        g = _random_dag(7, 0.4, rng)
        nodes = list(g)
        for _ in range(8):
            x, y = rng.choice(nodes, 2, replace=False)
            if y not in _desc_oracle(g, x):
                continue
            rest = [v for v in nodes if v not in (x, y)]
            zs = set(rng.choice(rest, rng.integers(0, 4), replace=False))
            want = _backdoor_oracle(g, x, y, zs)
            got = bounded(nb.is_backdoor_adjustment, g, x, y, zs, what="is_backdoor_adjustment")
            assert bool(got) == want, (
                f"backdoor criterion for {x} -> {y} with Z = {sorted(zs)} in {g}: expected {want}, got {got}. "
                "Z must contain no descendant of x and block every path into x (delete x's OUTGOING edges, "
                "then test d-separation).")


# ---------------------------------------------------------------- stage 4: adjustment formulas


def _kidney():
    # Charig et al. (1986): (treatment, small stone, successes, total). treatment 1 = A (open surgery)
    rows = []
    for t, small, succ, tot in [(1, 1, 81, 87), (1, 0, 192, 263), (0, 1, 234, 270), (0, 0, 55, 80)]:
        rows += [(t, small, 1)] * succ + [(t, small, 0)] * (tot - succ)
    a = np.array(rows, dtype=float)
    return {"T": a[:, 0], "S": a[:, 1], "R": a[:, 2]}


def test_stage4_kidney_stones_worked_example(nb):
    d = _kidney()
    a = nb.backdoor_adjust(d, "T", "R", ["S"], 1)
    b = nb.backdoor_adjust(d, "T", "R", ["S"], 0)
    assert abs(a - 0.83255) < 1e-4 and abs(b - 0.77888) < 1e-4, (
        f"the lesson's worked example: E[R | do(A)] = 0.8325, E[R | do(B)] = 0.7789; got {a:.4f}, {b:.4f}. "
        "weight each stratum's success rate by P(stratum) over ALL patients, not by P(stratum | treatment).")
    a0 = nb.backdoor_adjust(d, "T", "R", [], 1)
    b0 = nb.backdoor_adjust(d, "T", "R", [], 0)
    assert abs(a0 - 0.78) < 1e-9 and abs(b0 - 289 / 350) < 1e-9, (
        f"with no adjustment set the answer is the plain conditional mean E[R | T]: 0.780 and 0.826; got {a0:.4f}, {b0:.4f}")


def test_stage4_backdoor_two_covariates(nb):
    rng = np.random.default_rng(7)
    n = 4000
    z1 = rng.integers(0, 2, n).astype(float)
    z2 = rng.integers(0, 3, n).astype(float)
    x = (rng.random(n) < 0.2 + 0.3 * z1 + 0.1 * z2).astype(float)
    y = x + z1 - z2 + rng.normal(size=n)
    want = 0.0
    for a in (0, 1):
        for b in (0, 1, 2):
            s = (z1 == a) & (z2 == b)
            want += s.mean() * y[s & (x == 1)].mean()
    got = nb.backdoor_adjust({"x": x, "y": y, "z1": z1, "z2": z2}, "x", "y", ["z1", "z2"], 1)
    assert abs(got - want) < 1e-9, (
        f"with two adjustment variables the strata are the 6 combinations of (z1, z2); expected {want:.5f}, got {got:.5f}")


def test_stage4_backdoor_positivity(nb):
    d = {"x": np.array([1, 1, 0, 0, 0.0]), "z": np.array([0, 0, 0, 1, 1.0]), "y": np.arange(5.0)}
    try:
        nb.backdoor_adjust(d, "x", "y", ["z"], 1)
    except ValueError:
        return
    raise AssertionError("no unit with x = 1 has z = 1, so E[y | x=1, z=1] doesn't exist: raise ValueError (positivity)")


def test_stage4_backdoor_recovers_intervention(nb):
    z = {"Z": [], "X": ["Z"], "Y": ["X", "Z"]}
    mech = {"Z": lambda pa, u: _bern(0.4)(u),
            "X": lambda pa, u: _bern(0.15 + 0.7 * pa["Z"])(u),
            "Y": lambda pa, u: 1.0 * pa["X"] - 2.0 * pa["Z"] + 0.5 * u}
    m = nb.SCM(z, mech)
    obs = m.sample(60_000, rng=11)
    est = nb.backdoor_adjust(obs, "X", "Y", ["Z"], 1) - nb.backdoor_adjust(obs, "X", "Y", ["Z"], 0)
    naive = obs["Y"][obs["X"] == 1].mean() - obs["Y"][obs["X"] == 0].mean()
    assert abs(est - 1.0) < 0.05, f"the true effect is 1.0; the backdoor estimate should be close, got {est:.3f}"
    assert naive < 0, f"(sanity) the naive difference should be negative here, got {naive:.3f}"


def test_stage4_frontdoor_hand_table(nb):
    rows = []
    for x, m, ones, tot in [(0, 0, 40, 160), (0, 1, 20, 40), (1, 0, 18, 40), (1, 1, 112, 160)]:
        rows += [(x, m, 1)] * ones + [(x, m, 0)] * (tot - ones)
    a = np.array(rows, dtype=float)
    d = {"x": a[:, 0], "m": a[:, 1], "y": a[:, 2]}
    v1, v0 = nb.frontdoor_adjust(d, "x", "y", "m", 1), nb.frontdoor_adjust(d, "x", "y", "m", 0)
    assert abs(v1 - 0.55) < 1e-9 and abs(v0 - 0.40) < 1e-9, (
        f"the lesson's front-door table: E[y | do(x=1)] = 0.55 and E[y | do(x=0)] = 0.40; got {v1:.4f}, {v0:.4f}. "
        "the inner sum averages E[y | m, x'] over the MARGINAL P(x'), not over P(x' | m).")


def test_stage4_frontdoor_recovers_intervention(nb):
    # U is unobserved and confounds X and Y; X -> M -> Y is the only causal path
    parents = {"U": [], "X": ["U"], "M": ["X"], "Y": ["M", "U"]}
    mech = {"U": lambda pa, u: _bern(0.5)(u),
            "X": lambda pa, u: _bern(0.2 + 0.6 * pa["U"])(u),
            "M": lambda pa, u: _bern(0.1 + 0.7 * pa["X"])(u),
            "Y": lambda pa, u: 2.0 * pa["M"] + 3.0 * pa["U"] + u}
    m = nb.SCM(parents, mech)
    obs = m.sample(80_000, rng=12)
    est = nb.frontdoor_adjust(obs, "X", "Y", "M", 1) - nb.frontdoor_adjust(obs, "X", "Y", "M", 0)
    # truth: E[Y | do(X)] = 2 P(M=1 | X) + 3 E[U], so the effect is 2 * 0.7 = 1.4
    assert abs(est - 1.4) < 0.06, f"the true effect of X on Y is 2 * 0.7 = 1.4; front-door estimate {est:.3f}"
