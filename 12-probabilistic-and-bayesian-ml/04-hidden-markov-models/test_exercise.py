"""Checks for exercise.py (hidden markov models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles: brute
force over all K^T hidden paths for short sequences (the likelihood, every posterior marginal,
the best path, and one exact EM update from expected counts), plus a log-space forward pass for
long sequences. Sampling and learning checks use tolerances of several standard errors, and every
loop runs under mlfs.testing.bounded.
"""
import itertools

import numpy as np
from scipy.special import logsumexp

from mlfs.testing import bounded


def _model(nb, K=3, M=4, seed=0, **kw):
    rng = np.random.default_rng(seed)
    h = nb.HMM(K, M, **kw)
    h.startprob_ = rng.dirichlet(np.ones(K) * 2)
    h.transmat_ = rng.dirichlet(np.ones(K) * 2, size=K)
    h.emissionprob_ = rng.dirichlet(np.ones(M) * 2, size=K)
    return h


def _casino(nb, **kw):
    h = nb.HMM(2, 6, **kw)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.95, 0.05], [0.1, 0.9]])
    h.emissionprob_ = np.vstack([np.full(6, 1 / 6), [0.1] * 5 + [0.5]])
    return h


def _joint(h, obs, path):
    p = h.startprob_[path[0]] * h.emissionprob_[path[0], obs[0]]
    for t in range(1, len(obs)):
        p *= h.transmat_[path[t - 1], path[t]] * h.emissionprob_[path[t], obs[t]]
    return p


def _brute(h, obs):
    K = h.transmat_.shape[0]
    paths = list(itertools.product(range(K), repeat=len(obs)))
    probs = np.array([_joint(h, obs, p) for p in paths])
    return paths, probs


def _log_forward(h, obs):
    la = np.log(h.startprob_) + np.log(h.emissionprob_[:, obs[0]])
    for x in obs[1:]:
        la = logsumexp(la[:, None] + np.log(h.transmat_), axis=0) + np.log(h.emissionprob_[:, x])
    return logsumexp(la)


# ---------------------------------------------------------------- stage 1: sampling


def test_stage1_init_and_shapes(nb):
    h = nb.HMM(3, 4, max_iters=7, tol=1e-3, rng=5)
    assert h.get_params() == {"n_states": 3, "n_symbols": 4, "max_iters": 7, "tol": 1e-3, "rng": 5}, (
        f"__init__ stores its arguments unchanged; get_params() = {h.get_params()}")
    h = _model(nb)
    obs, states = h.sample(50, rng=0)
    assert np.shape(obs) == (50,) and np.shape(states) == (50,), "sample(n) returns (obs (n,), states (n,))"
    assert np.issubdtype(np.asarray(obs).dtype, np.integer) and np.issubdtype(np.asarray(states).dtype, np.integer)
    assert set(np.unique(states)) <= {0, 1, 2} and set(np.unique(obs)) <= {0, 1, 2, 3}
    o2, s2 = h.sample(50, rng=0)
    assert np.array_equal(obs, o2) and np.array_equal(states, s2), "the same rng seed gives the same sample"


def test_stage1_sample_statistics(nb):
    h = _model(nb, seed=1)
    obs, states = h.sample(20000, rng=1)
    counts = np.zeros((3, 3))
    np.add.at(counts, (states[:-1], states[1:]), 1)
    rows = counts.sum(1, keepdims=True)
    se = np.sqrt(h.transmat_ * (1 - h.transmat_) / rows)
    assert np.all(np.abs(counts / rows - h.transmat_) < 5 * se + 1e-3), (
        f"empirical transition frequencies {np.round(counts / rows, 3).tolist()} should match transmat_ "
        f"{np.round(h.transmat_, 3).tolist()}: the next state is drawn from row transmat_[current state]")
    em = np.zeros((3, 4))
    np.add.at(em, (states, obs), 1)
    rows = em.sum(1, keepdims=True)
    se = np.sqrt(h.emissionprob_ * (1 - h.emissionprob_) / rows)
    assert np.all(np.abs(em / rows - h.emissionprob_) < 5 * se + 1e-3), (
        "empirical emission frequencies should match emissionprob_: the symbol is drawn from the row of the CURRENT state")
    firsts = np.array([h.sample(1, rng=s)[1][0] for s in range(3000)])
    freq = np.bincount(firsts, minlength=3) / 3000
    assert np.all(np.abs(freq - h.startprob_) < 5 * np.sqrt(h.startprob_ * (1 - h.startprob_) / 3000) + 1e-3), (
        f"the first state is drawn from startprob_ {np.round(h.startprob_, 3).tolist()}; got {np.round(freq, 3).tolist()}")


# ---------------------------------------------------------------- stage 2: forward-backward


def _tiny(nb):
    """the lesson's worked example."""
    h = nb.HMM(2, 2)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    h.emissionprob_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    return h


def test_stage2_worked_example(nb):
    ll, post = _tiny(nb).forward_backward(np.array([0, 1]))
    assert np.isclose(ll, np.log(0.19)), f"lesson worked example: p(x) = 0.078 + 0.112 = 0.19, log 0.19 = -1.6607; got {ll}"
    want = np.array([[0.1, 0.09], [0.078, 0.112]]) / 0.19
    assert np.allclose(post, want), (
        f"posteriors: t = 1 is alpha_1 * beta_1 = (0.4 * 0.25, 0.15 * 0.6) / 0.19 = (0.526, 0.474), t = 2 is "
        f"(0.411, 0.589); got {np.round(post, 3).tolist()}")


def test_stage2_brute_force(nb):
    for seed in range(3):
        h = _model(nb, seed=10 + seed)
        obs = np.random.default_rng(seed).integers(0, 4, size=6)
        paths, probs = _brute(h, obs)
        ll, post = h.forward_backward(obs)
        assert np.isclose(ll, np.log(probs.sum())), (
            f"log p(obs) should be log of the sum over all {len(paths)} paths = {np.log(probs.sum()):.6f}; got {ll:.6f}")
        want = np.zeros((6, 3))
        for p, w in zip(paths, probs):
            want[np.arange(6), p] += w
        want /= probs.sum()
        assert np.shape(post) == (6, 3) and np.allclose(post, want, atol=1e-10), (
            "posteriors[t, k] = P(z_t = k | obs): the brute-force marginals disagree. alpha_t beta_t, normalized per t")


def test_stage2_long_sequence_no_underflow(nb):
    h = _casino(nb)
    obs, _ = h.sample(5000, rng=3)
    ll, post = h.forward_backward(obs)
    want = _log_forward(h, obs)
    assert np.isfinite(ll) and np.isclose(ll, want, rtol=1e-9), (
        f"5,000 rolls: log p(obs) = {want:.3f} (about -1.7 per roll). got {ll}. unscaled alpha underflows to 0 after a "
        "few hundred steps: normalize alpha_t at every step and add up the logs of the normalizers")
    assert np.allclose(np.sum(post, axis=1), 1) and np.all(np.isfinite(post)), "every row of the posteriors sums to 1"


# ---------------------------------------------------------------- stage 3: viterbi


def test_stage3_worked_example(nb):
    path, lp = _tiny(nb).viterbi(np.array([0, 1]))
    assert list(path) == [1, 1] and np.isclose(lp, np.log(0.084)), (
        f"lesson worked example: delta_2 = (0.072, 0.084), the best path is (1, 1) with p(x, z) = 0.084, even though "
        f"the most probable state at t = 1 alone is 0. got path {list(path)}, log_prob {lp:.4f}")


def test_stage3_brute_force(nb):
    for seed in range(3):
        h = _model(nb, seed=20 + seed)
        obs = np.random.default_rng(100 + seed).integers(0, 4, size=7)
        paths, probs = _brute(h, obs)
        path, lp = h.viterbi(obs)
        best = int(np.argmax(probs))
        got_p = _joint(h, obs, [int(z) for z in path])
        assert np.isclose(got_p, probs[best], rtol=1e-9), (
            f"the most probable path (by brute force over {len(paths)} paths) is {list(paths[best])} with "
            f"p(obs, path) = {probs[best]:.3e}; yours, {[int(z) for z in path]}, has {got_p:.3e}")
        assert np.isclose(lp, np.log(probs[best])), f"log_prob is log p(obs, best path) = {np.log(probs[best]):.6f}; got {lp:.6f}"


def test_stage3_long_and_forbidden(nb):
    h = _casino(nb)
    obs, states = h.sample(3000, rng=4)
    path, lp = h.viterbi(obs)
    logp = np.log(h.startprob_[path[0]]) + np.log(h.emissionprob_[path[0], obs[0]]) + np.sum(
        np.log(h.transmat_[path[:-1], path[1:]]) + np.log(h.emissionprob_[path[1:], obs[1:]]))
    assert np.isclose(lp, logp, rtol=1e-9), (
        f"3,000 steps: log_prob must equal the log joint of the returned path ({logp:.3f}); got {lp}. work in log space")
    assert np.mean(np.asarray(path) == states) > 0.7, "on the casino, Viterbi should recover most of the hidden states"
    g = nb.HMM(3, 2)
    g.startprob_ = np.array([1.0, 0.0, 0.0])
    g.transmat_ = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    g.emissionprob_ = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    path, lp = g.viterbi(np.array([0, 1, 1, 0, 0]))
    assert list(path) == [0, 1, 1, 2, 2] and np.isfinite(lp), (
        f"a left-to-right model with zero transitions (log 0 = -inf must not break anything): expected [0, 1, 1, 2, 2], "
        f"got {list(path)}")


# ---------------------------------------------------------------- stage 4: Baum-Welch


def _em_step_oracle(h, seqs):
    K, M = h.emissionprob_.shape
    start, trans, emit = np.zeros(K), np.zeros((K, K)), np.zeros((K, M))
    for obs in seqs:
        paths, probs = _brute(h, obs)
        probs = probs / probs.sum()
        for p, w in zip(paths, probs):
            start[p[0]] += w
            for t in range(1, len(obs)):
                trans[p[t - 1], p[t]] += w
            for t in range(len(obs)):
                emit[p[t], obs[t]] += w
    return start / start.sum(), trans / trans.sum(1, keepdims=True), emit / emit.sum(1, keepdims=True)


def test_stage4_one_em_step(nb):
    h = _model(nb, K=2, M=3, seed=30, max_iters=1)
    rng = np.random.default_rng(31)
    seqs = [rng.integers(0, 3, size=6), rng.integers(0, 3, size=5)]
    want = _em_step_oracle(h, seqs)
    bounded(h.fit, seqs, what="HMM.fit")
    for name, got, w in zip(["startprob_", "transmat_", "emissionprob_"],
                            [h.startprob_, h.transmat_, h.emissionprob_], want):
        assert np.allclose(got, w, atol=1e-10), (
            f"one Baum-Welch step from the preset parameters (max_iters=1), on two sequences: {name} should be the "
            f"normalized EXPECTED counts {np.round(w, 4).tolist()}; got {np.round(got, 4).tolist()}. fit must start "
            "from the parameters if they're already set, and pool the counts over all sequences")


def test_stage4_monotone_and_score(nb):
    rng = np.random.default_rng(40)
    true = _model(nb, K=3, M=5, seed=41)
    seqs = [true.sample(200, rng=rng)[0] for _ in range(3)]
    h = nb.HMM(3, 5, max_iters=60, tol=1e-9, rng=0)
    bounded(h.fit, seqs, what="HMM.fit", budget=2_000_000, hint="cap the EM loop at max_iters.")
    hist = np.asarray(h.history_)
    assert len(hist) >= 2 and np.all(np.diff(hist) >= -1e-6), (
        f"EM never decreases the log-likelihood: history_ (one value per E-step) should be non-decreasing; got "
        f"differences {np.round(np.diff(hist)[:5], 5).tolist()}...")
    assert h.n_iter_ == len(hist) <= 60, "n_iter_ counts the EM iterations, at most max_iters"
    for a in ("startprob_", "transmat_", "emissionprob_"):
        assert np.allclose(getattr(h, a).sum(-1), 1), f"{a} must stay row-stochastic"
    total = h.score(seqs)
    assert np.isclose(total, sum(h.forward_backward(s)[0] for s in seqs)), "score(list) is the sum of the log-likelihoods"
    assert np.isclose(h.score(seqs[0]), h.forward_backward(seqs[0])[0]), "score(one sequence) works too"
    assert total >= hist[-1] - 1e-6, "the final parameters are at least as good as the last E-step's"
    h2 = nb.HMM(3, 5, max_iters=3, rng=0)
    bounded(h2.fit, seqs, what="HMM.fit")
    assert h2.n_iter_ == 3, f"max_iters=3 means exactly 3 EM iterations here (no convergence yet); n_iter_ = {h2.n_iter_}"
    h3 = nb.HMM(3, 5, max_iters=50, tol=1e10, rng=0)
    bounded(h3.fit, seqs, what="HMM.fit")
    assert h3.n_iter_ == 2, (
        f"with an enormous tol, the second iteration's improvement is already below it: stop after 2 iterations; "
        f"n_iter_ = {h3.n_iter_}. (stop when history_[-1] - history_[-2] < tol)")


def test_stage4_recovers_the_casino(nb):
    truth = _casino(nb)
    rng = np.random.default_rng(50)
    seqs = [truth.sample(1000, rng=rng)[0] for _ in range(2)]
    h = nb.HMM(2, 6, max_iters=60, tol=1e-4)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.9, 0.1], [0.15, 0.85]])
    h.emissionprob_ = np.vstack([np.full(6, 1 / 6) + [0.02, 0, 0, 0, 0, -0.02], [0.15] * 5 + [0.25]])
    bounded(h.fit, seqs, what="HMM.fit", budget=3_000_000)
    p6 = h.emissionprob_[:, 5]
    assert 0.3 < p6[1] < 0.75 and p6[1] - p6[0] > 0.2, (
        f"2,000 rolls, starting near the right answer: Baum-Welch should keep a loaded state (true P(6) = 0.5) and a "
        f"fair one (1/6); got P(6) = {np.round(p6, 3).tolist()}. (2,000 rolls only pin these down to about +-0.1; the check is loose on purpose)")
    assert h.score(seqs) >= truth.score(seqs) + 0.5, (
        f"the maximum-likelihood parameters fit the training data BETTER than the true ones (by several nats on "
        f"2,000 rolls): got {h.score(seqs):.2f} against the truth's {truth.score(seqs):.2f}")
