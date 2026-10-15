"""Checks for exercise.py (probability & random variables).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
from fractions import Fraction

import pytest


def _die(nb, sides=6):
    return nb.Discrete({k: 1 / sides for k in range(1, sides + 1)})


def _same(D, want, tol=1e-9):
    """True if Discrete D has exactly the outcomes in `want` (dict) with matching probabilities."""
    got = {x: p for x, p in D.probs.items() if p > 0}
    want = {x: float(p) for x, p in want.items() if p > 0}
    return set(got) == set(want) and all(abs(got[x] - want[x]) <= tol for x in want)


def _show(D):
    return "{" + ", ".join(f"{x!r}: {p:.4f}" for x, p in D.probs.items()) + "}"


# ---------------------------------------------------------------- stage 1: Discrete


def test_stage1_stores_a_copy_and_drops_zeros(nb):
    raw = {"H": 0.5, "T": 0.5, "edge": 0.0}
    D = nb.Discrete(raw)
    assert isinstance(D.probs, dict), "store the distribution as a dict in self.probs"
    assert set(D.probs) == {"H", "T"}, (
        f"outcomes with probability 0 should be dropped; self.probs has keys {sorted(D.probs)}")
    raw2 = {"H": 0.5, "T": 0.5}
    D2 = nb.Discrete(raw2)
    raw2["H"] = 0.9
    assert D2.probs["H"] == 0.5 and D2.probs is not raw2, (
        "changing the caller's dict changed your distribution. store a NEW dict (e.g. build one in a loop).")


def test_stage1_validates(nb):
    for bad, why in (({"a": 0.5, "b": 0.6}, "sums to 1.1"),
                     ({"a": 1.5, "b": -0.5}, "has a negative probability"),
                     ({"a": 0.3}, "sums to 0.3"),
                     ({"a": float("nan"), "b": 1.0}, "contains nan")):
        with pytest.raises(ValueError):
            nb.Discrete(bad)
            pytest.fail(f"Discrete({bad}) {why}: it should raise ValueError")
    nb.Discrete({"a": 0.1, "b": 0.2, "c": 0.7})  # 0.1 + 0.2 + 0.7 isn't exactly 1 in floats, but is within 1e-9


def test_stage1_pmf_and_prob(nb):
    D = _die(nb)
    assert abs(D.pmf(3) - 1 / 6) < 1e-12, f"P(X = 3) for a fair die is 1/6, got {D.pmf(3)}"
    assert D.pmf(7) == 0.0, f"P(X = 7) for a die is 0 (7 isn't an outcome), got {D.pmf(7)}"
    assert abs(D.prob(lambda x: x % 2 == 0) - 0.5) < 1e-12, "P(even) for a fair die is 1/2 (predicate event)"
    assert abs(D.prob({1, 2, 9}) - 2 / 6) < 1e-12, (
        f"P(X in {{1, 2, 9}}) = 2/6: 9 isn't an outcome, so it adds nothing. got {D.prob({1, 2, 9})}")
    assert abs(D.prob([1, 1, 2]) - 2 / 6) < 1e-12, (
        "an event is a SET of outcomes: listing outcome 1 twice must not count it twice. "
        f"P(X in [1, 1, 2]) should be 2/6, got {D.prob([1, 1, 2])}")
    assert D.prob(set()) == 0.0, "the empty event has probability 0"


def test_stage1_tuple_outcomes_work(nb):
    J = nb.Discrete({(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.5})
    assert abs(J.pmf((1, 0)) - 0.5) < 1e-12, "outcomes can be tuples: J.pmf((1, 0)) should be 0.5"
    assert abs(J.prob(lambda t: t[0] == 0) - 0.5) < 1e-12, "P(first coordinate is 0) should be 0.5"


# ---------------------------------------------------------------- stage 2: functions of RVs, independence, sums


def test_stage2_pushforward_merges_outcomes(nb):
    D = _die(nb)
    S = nb.pushforward(D, lambda x: x % 3)
    assert _same(S, {0: 2 / 6, 1: 2 / 6, 2: 2 / 6}), (
        f"X mod 3 for a fair die: each residue has probability 2/6, got {_show(S)}. outcomes that map to "
        "the same value must have their probabilities ADDED, not overwritten.")
    sq = nb.pushforward(nb.Discrete({-1: 0.2, 1: 0.3, 2: 0.5}), lambda x: x * x)
    assert _same(sq, {1: 0.5, 4: 0.5}), f"X^2 with P(-1)=.2, P(1)=.3, P(2)=.5 is {{1: .5, 4: .5}}, got {_show(sq)}"


def test_stage2_product_is_independent_joint(nb):
    X = nb.Discrete({"a": 0.25, "b": 0.75})
    Y = nb.Discrete({0: 0.6, 1: 0.4})
    J = nb.product(X, Y)
    want = {("a", 0): 0.15, ("a", 1): 0.1, ("b", 0): 0.45, ("b", 1): 0.3}
    assert _same(J, want), f"product(X, Y) should be {want}, got {_show(J)}. outcomes are pairs (x, y)."


def test_stage2_two_dice(nb):
    S = nb.add(_die(nb), _die(nb))
    want = {s: (6 - abs(s - 7)) / 36 for s in range(2, 13)}
    assert _same(S, want), (
        f"the sum of two fair dice has the triangle distribution 1/36, 2/36, ..., 6/36, ..., 1/36 on 2..12. "
        f"got {_show(S)}")


def test_stage2_add_is_a_convolution(nb):
    X = nb.Discrete({0: 0.5, 1: 0.5})
    total = X
    for _ in range(3):
        total = nb.add(total, X)
    want = {0: 1 / 16, 1: 4 / 16, 2: 6 / 16, 3: 4 / 16, 4: 1 / 16}
    assert _same(total, want), (
        f"the sum of 4 independent fair coins (0/1) is Binomial(4, 1/2) = (1, 4, 6, 4, 1)/16, got {_show(total)}")
    Z = nb.add(X, nb.Discrete({10: 1.0}))
    assert _same(Z, {10: 0.5, 11: 0.5}), f"adding a constant 10 just shifts: expected {{10: .5, 11: .5}}, got {_show(Z)}"


# ---------------------------------------------------------------- stage 3: conditioning


def test_stage3_condition_renormalizes(nb):
    D = _die(nb)
    C = nb.condition(D, lambda x: x >= 5)
    assert _same(C, {5: 0.5, 6: 0.5}), (
        f"a fair die given X >= 5 is uniform on {{5, 6}}, got {_show(C)}. keep the outcomes inside the "
        "event and divide by P(event) = 2/6.")
    C2 = nb.condition(D, {2, 3, 4, 10})
    assert _same(C2, {2: 1 / 3, 3: 1 / 3, 4: 1 / 3}), f"conditioning on the set {{2, 3, 4, 10}} gives 1/3 each, got {_show(C2)}"


def test_stage3_impossible_event_raises(nb):
    with pytest.raises(ValueError):
        nb.condition(_die(nb), lambda x: x > 6)
        pytest.fail("conditioning on an impossible event (P = 0) should raise ValueError")


def test_stage3_worked_example_dice(nb):
    S2 = nb.product(_die(nb), _die(nb))
    given = nb.condition(S2, lambda t: t[0] + t[1] == 8)
    p_double = given.prob(lambda t: t[0] == t[1])
    assert abs(p_double - 1 / 5) < 1e-12, (
        f"given that two dice sum to 8, the outcomes (2,6),(3,5),(4,4),(5,3),(6,2) are equally likely, so "
        f"P(doubles | sum 8) = 1/5. got {p_double:.4f}")
    first = nb.condition(S2, lambda t: t[0] == 6)
    p7 = first.prob(lambda t: t[0] + t[1] == 7)
    assert abs(p7 - 1 / 6) < 1e-12, f"P(sum 7 | first die 6) = 1/6, got {p7:.4f}"


def test_stage3_skewed_conditioning(nb):
    X = nb.Discrete({0: 0.7, 1: 0.2, 2: 0.1})
    C = nb.condition(X, lambda x: x > 0)
    assert _same(C, {1: 2 / 3, 2: 1 / 3}), (
        f"P(X | X > 0) with P = (.7, .2, .1) is (2/3, 1/3) on {{1, 2}}: the RATIO 2:1 is kept. got {_show(C)}")


# ---------------------------------------------------------------- stage 4: joints and marginals


_JOINT = {("sun", "dry"): 0.5, ("sun", "wet"): 0.1, ("rain", "dry"): 0.1, ("rain", "wet"): 0.3}


def test_stage4_marginal_single_coordinate(nb):
    J = nb.Discrete(_JOINT)
    W = nb.marginal(J, 0)
    assert _same(W, {"sun": 0.6, "rain": 0.4}), (
        f"marginal(J, 0) sums out the second coordinate: P(sun) = .5 + .1 = .6, got {_show(W)}. "
        "with an int index the outcomes are plain values, not 1-tuples.")
    G = nb.marginal(J, 1)
    assert _same(G, {"dry": 0.6, "wet": 0.4}), f"marginal(J, 1) should be {{dry: .6, wet: .4}}, got {_show(G)}"


def test_stage4_marginal_of_triples(nb):
    J = nb.product(nb.product(nb.Discrete({0: 0.5, 1: 0.5}), nb.Discrete({"x": 0.2, "y": 0.8})),
                   nb.Discrete({True: 0.1, False: 0.9}))
    J = nb.Discrete({(a, b, c): p for ((a, b), c), p in J.probs.items()})  # flatten to triples
    M = nb.marginal(J, (2, 0))
    want = {(True, 0): 0.05, (True, 1): 0.05, (False, 0): 0.45, (False, 1): 0.45}
    assert _same(M, want), (
        f"marginal(J, (2, 0)) keeps coordinates 2 and 0, IN THAT ORDER, as tuples. expected {want}, got {_show(M)}")


def test_stage4_independence(nb):
    assert not nb.is_independent(nb.Discrete(_JOINT)), (
        "weather and wetness are dependent: P(sun, dry) = .5 but P(sun) P(dry) = .36")
    J = nb.product(nb.Discrete({"a": 0.3, "b": 0.7}), nb.Discrete({0: 0.5, 1: 0.25, 2: 0.25}))
    assert nb.is_independent(J), "a product distribution is independent by construction"
    # uncorrelated but dependent: X uniform on {-1, 0, 1}, Y = X^2
    K = nb.Discrete({(-1, 1): 1 / 3, (0, 0): 1 / 3, (1, 1): 1 / 3})
    assert not nb.is_independent(K), (
        "X uniform on {-1, 0, 1} and Y = X^2 are dependent (Y is a function of X): P(X=0, Y=1) = 0 but "
        "P(X=0) P(Y=1) = 2/9 (and P(X=0, Y=0) = 1/3 but P(X=0) P(Y=0) = 1/9). compare P(x, y) with "
        "P(x) P(y) for every x in the support of the first marginal and every y in the second.")


def test_stage4_independence_tolerance(nb):
    J = nb.product(nb.Discrete({0: 0.1, 1: 0.9}), nb.Discrete({0: 0.3, 1: 0.7}))
    almost = dict(J.probs)
    almost[(0, 0)] += 1e-4
    almost[(1, 1)] -= 1e-4
    K = nb.Discrete(almost)
    assert not nb.is_independent(K), "a 1e-4 perturbation is far above the default tol=1e-9: not independent"
    assert nb.is_independent(K, tol=1e-3), "with tol=1e-3 the same perturbation should count as independent"


# ---------------------------------------------------------------- stage 5: chain rule and bayes


def _test_likelihood(nb):
    def lik(h):
        return nb.Discrete({"+": 0.9, "-": 0.1}) if h == "sick" else nb.Discrete({"+": 0.09, "-": 0.91})
    return lik


def test_stage5_chain_rule(nb):
    prior = nb.Discrete({"sick": 0.01, "healthy": 0.99})
    J = nb.chain(prior, _test_likelihood(nb))
    want = {("sick", "+"): 0.009, ("sick", "-"): 0.001, ("healthy", "+"): 0.0891, ("healthy", "-"): 0.9009}
    assert _same(J, want), f"chain(prior, likelihood) should be P(h) P(d | h) on pairs (h, d): {want}, got {_show(J)}"


def test_stage5_worked_example_base_rate(nb):
    prior = nb.Discrete({"sick": 0.01, "healthy": 0.99})
    post = nb.bayes(prior, _test_likelihood(nb), "+")
    want = 0.009 / (0.009 + 0.0891)
    got = post.pmf("sick")
    assert abs(got - want) < 1e-12, (
        f"with 1% prevalence, 90% sensitivity and a 9% false-positive rate, P(sick | +) = "
        f"0.009 / 0.0981 = {want:.4f}, got {got:.4f}. "
        + ("you returned the likelihood P(+ | sick): that's the base-rate fallacy." if abs(got - 0.9) < 1e-9 else
           "posterior = P(+ | h) P(h), normalized over h."))
    assert abs(sum(post.probs.values()) - 1) < 1e-12 and set(post.probs) == {"sick", "healthy"}, (
        "the posterior is a distribution over the HYPOTHESES (sick, healthy), summing to 1")


def test_stage5_monty_hall(nb):
    # you pick door 0; the host, who knows where the car is, opens a goat door among 1 and 2
    def host(car):
        return nb.Discrete({1: 0.5, 2: 0.5}) if car == 0 else nb.Discrete({3 - car: 1.0})
    post = nb.bayes(nb.Discrete({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}), host, 2)
    assert _same(post, {0: 1 / 3, 1: 2 / 3}), (
        f"monty hall: after the host opens door 2, P(car behind 0) = 1/3 and P(car behind 1) = 2/3. "
        f"got {_show(post)}")


def test_stage5_sequential_updates_agree_with_batch(nb):
    """updating on two coin flips one at a time equals updating on the pair at once."""
    biases = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]
    prior = nb.Discrete({float(b): 1 / 3 for b in biases})

    def one_flip(b):
        return nb.Discrete({"H": b, "T": 1 - b})

    def two_flips(b):
        return nb.Discrete({(f1, f2): (b if f1 == "H" else 1 - b) * (b if f2 == "H" else 1 - b)
                            for f1 in "HT" for f2 in "HT"})
    seq = nb.bayes(nb.bayes(prior, one_flip, "H"), one_flip, "H")
    batch = nb.bayes(prior, two_flips, ("H", "H"))
    want = {float(b): float(b * b / sum(c * c for c in biases)) for b in biases}
    assert _same(batch, want), f"P(bias | HH) is proportional to bias^2: expected {want}, got {_show(batch)}"
    assert _same(seq, want), (
        f"updating twice on one flip should give the same posterior as one update on both flips; got {_show(seq)}")


def test_stage5_impossible_observation_raises(nb):
    prior = nb.Discrete({"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        nb.bayes(prior, lambda h: nb.Discrete({"x": 1.0}), "y")
        pytest.fail("an observation with probability 0 under every hypothesis should raise ValueError")
