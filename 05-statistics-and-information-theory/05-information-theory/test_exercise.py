"""Checks for exercise.py (information theory).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook. Information measures are checked against hand
examples, identities and scipy; Huffman codes are checked for prefix-freeness and optimality
(any optimal code passes, since ties can be broken either way).
"""
import heapq
import math

import numpy as np
from scipy import stats

from mlfs.testing import bounded

LN2 = math.log(2)


def _huffman(nb, probs):
    return bounded(nb.huffman_code, probs, what="huffman_code",
                   hint="each merge must POP the two least probable nodes (heapq.heappop), so the heap shrinks by one.")


def _decode(nb, bits, code):
    return bounded(nb.decode, bits, code, what="decode",
                   hint="after emitting a symbol, reset the buffer to '' and keep reading bits.")


def _optimal_length(p):
    """expected length of an optimal prefix code: the sum of all merged weights in Huffman's algorithm."""
    p = list(np.asarray(p, float) / np.sum(p))
    if len(p) == 1:
        return 1.0
    heapq.heapify(p)
    total = 0.0
    while len(p) > 1:
        a, b = heapq.heappop(p), heapq.heappop(p)
        total += a + b
        heapq.heappush(p, a + b)
    return total


def _prefix_free(code):
    return not any(a != b and b.startswith(a) for a in code for b in code)


# ---------------------------------------------------------------- stage 1: entropy, cross-entropy, KL


def test_stage1_entropy_hand_examples(nb):
    assert abs(nb.entropy([0.5, 0.25, 0.25], base=2) - 1.5) < 1e-12, (
        f"H(1/2, 1/4, 1/4) = 1/2 · 1 + 1/4 · 2 + 1/4 · 2 = 1.5 bits, got {nb.entropy([0.5, 0.25, 0.25], base=2)}")
    assert abs(nb.entropy([0.5, 0.25, 0.25]) - 1.5 * LN2) < 1e-12, "without base, entropy is in nats: 1.5 bits = 1.5 ln 2 nats"
    assert abs(nb.entropy(np.ones(8), base=2) - 3.0) < 1e-12, "a uniform distribution over 8 outcomes has 3 bits (weights are normalized)"
    assert abs(nb.entropy([2, 1, 1], base=2) - 1.5) < 1e-12, "normalize the weights first: (2, 1, 1) is (1/2, 1/4, 1/4)"
    h = nb.entropy([1.0, 0.0, 0.0])
    assert h == 0.0, f"a certain outcome has entropy 0; 0 log 0 counts as 0, not nan. got {h}"
    assert isinstance(nb.entropy([0.3, 0.7]), float), "return a Python float"


def test_stage1_matches_scipy(nb):
    rng = np.random.default_rng(0)
    for k in (2, 5, 30):
        p, q = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
        assert abs(nb.entropy(p) - stats.entropy(p)) < 1e-12, "entropy disagrees with scipy.stats.entropy"
        assert abs(nb.kl_divergence(p, q) - stats.entropy(p, q)) < 1e-12, "kl_divergence disagrees with scipy.stats.entropy(p, q)"
        assert abs(nb.cross_entropy(p, q) - (nb.entropy(p) + nb.kl_divergence(p, q))) < 1e-12, (
            "cross-entropy decomposes as H(p, q) = H(p) + KL(p || q)")
        assert abs(nb.kl_divergence(p, q, base=2) - stats.entropy(p, q, base=2)) < 1e-12, "base=2 gives bits"


def test_stage1_kl_properties(nb):
    p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    assert abs(nb.cross_entropy(p, q, base=2) - (-0.5 * math.log2(0.9) - 0.5 * math.log2(0.1))) < 1e-12, (
        "H(p, q) = -sum p log q: for p = (1/2, 1/2), q = (0.9, 0.1) that's 1.737 bits")
    kpq, kqp = nb.kl_divergence(p, q), nb.kl_divergence(q, p)
    assert abs(kpq - 0.5108256237659907) < 1e-12 and abs(kqp - 0.3680642071684971) < 1e-12, (
        f"KL(p || q) = 0.5108 and KL(q || p) = 0.3681 nats: KL is NOT symmetric. got {kpq:.4f} and {kqp:.4f}. "
        "KL(p || q) = sum p log(p / q), with p the FIRST argument.")
    assert nb.kl_divergence(q, q) == 0.0, "KL(q || q) = 0"
    assert nb.kl_divergence([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]) < math.inf, (
        "q > 0 where p = 0 is fine: those terms contribute 0")
    assert nb.kl_divergence([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]) == 0.0, (
        "q = 0 where p = 0 is fine too: only q = 0 where p > 0 makes KL infinite")
    assert nb.kl_divergence([0.4, 0.4, 0.2], [0.5, 0.5, 0.0]) == math.inf, (
        "q = 0 where p > 0: KL is infinite (an event q calls impossible happens)")
    assert nb.cross_entropy([0.4, 0.4, 0.2], [0.5, 0.5, 0.0]) == math.inf, "the cross-entropy is infinite there too"
    try:
        nb.entropy([0.5, -0.1, 0.6])
    except ValueError:
        return
    raise AssertionError("negative probabilities should raise ValueError")


# ---------------------------------------------------------------- stage 2: mutual information


def test_stage2_hand_and_extremes(nb):
    joint = np.array([[0.4, 0.1], [0.1, 0.4]])
    want = 1 + 1 - (-(2 * 0.4 * math.log2(0.4) + 2 * 0.1 * math.log2(0.1)))  # H(X) + H(Y) - H(X, Y)
    got = nb.mutual_information(joint, base=2)
    assert abs(got - want) < 1e-12, (
        f"for the joint table [[0.4, 0.1], [0.1, 0.4]], I(X; Y) = H(X) + H(Y) - H(X, Y) = 1 + 1 - 1.7219 = "
        f"{want:.4f} bits, got {got}")
    px, py = np.array([0.2, 0.3, 0.5]), np.array([0.6, 0.4])
    assert abs(nb.mutual_information(np.outer(px, py))) < 1e-12, "independent variables (joint = outer product) have I = 0"
    assert abs(nb.mutual_information(np.eye(4) / 4, base=2) - 2.0) < 1e-12, (
        "if Y = X, uniform over 4 values, I(X; Y) = H(X) = 2 bits")
    assert abs(nb.mutual_information(joint * 100, base=2) - want) < 1e-12, "a table of counts should be normalized first"


def test_stage2_identities(nb):
    rng = np.random.default_rng(1)
    for shape in ((3, 4), (6, 2), (5, 5)):
        j = rng.dirichlet(np.ones(np.prod(shape))).reshape(shape)
        j[0, 0] = 0.0
        j /= j.sum()
        mi = nb.mutual_information(j)
        h = lambda v: -np.sum(v[v > 0] * np.log(v[v > 0]))
        want = h(j.sum(1)) + h(j.sum(0)) - h(j.ravel())
        assert abs(mi - want) < 1e-12, f"I(X; Y) should equal H(X) + H(Y) - H(X, Y) = {want:.6f}, got {mi:.6f}"
        assert abs(nb.mutual_information(j.T) - mi) < 1e-12, "mutual information is symmetric: I(X; Y) = I(Y; X)"


# ---------------------------------------------------------------- stage 3: Gaussians


def test_stage3_gaussian_entropy(nb):
    assert abs(nb.gaussian_entropy(4.0) - 0.5 * math.log(2 * math.pi * math.e * 4.0)) < 1e-12, (
        "a 1-D Gaussian with variance 4 has entropy (1/2) log(2 pi e 4) nats; a scalar cov is a variance")
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + 0.5 * np.eye(5)
    want = stats.multivariate_normal(np.zeros(5), cov).entropy()
    got = nb.gaussian_entropy(cov)
    assert abs(got - want) < 1e-10, (
        f"entropy of a 5-D Gaussian: scipy says {want:.6f} nats, got {got:.6f}. "
        "(d/2) log(2 pi e) + (1/2) log det cov, with log det = 2 sum log diag(L).")


def _kl_oracle(m0, S0, m1, S1):
    d = len(m0)
    S1i = np.linalg.inv(S1)
    return 0.5 * (np.trace(S1i @ S0) + (m1 - m0) @ S1i @ (m1 - m0) - d
                  + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])


def test_stage3_gaussian_kl(nb):
    got = nb.gaussian_kl(0.0, 1.0, 1.0, 4.0)
    want = math.log(2.0) + (1.0 + 1.0) / 8.0 - 0.5
    assert abs(got - want) < 1e-12, (
        f"KL(N(0, 1) || N(1, 4)) = log(sigma1/sigma0) + (sigma0^2 + (mu0 - mu1)^2) / (2 sigma1^2) - 1/2 = {want:.6f}, "
        f"got {got}")
    rng = np.random.default_rng(3)
    for d in (2, 4):
        A, B = rng.normal(size=(2, d, d))
        S0, S1 = A @ A.T + np.eye(d), B @ B.T + np.eye(d)
        m0, m1 = rng.normal(size=(2, d))
        got, want = nb.gaussian_kl(m0, S0, m1, S1), _kl_oracle(m0, S0, m1, S1)
        assert abs(got - want) < 1e-9, f"{d}-D Gaussian KL should be {want:.6f}, got {got:.6f}"
        back = nb.gaussian_kl(m1, S1, m0, S0)
        assert abs(back - _kl_oracle(m1, S1, m0, S0)) < 1e-9 and abs(back - got) > 1e-3, (
            "the arguments are (mu0, cov0, mu1, cov1) for KL(N0 || N1), and KL is not symmetric")
        assert abs(nb.gaussian_kl(m0, S0, m0, S0)) < 1e-12, "KL of a Gaussian to itself is 0"


# ---------------------------------------------------------------- stage 4: Huffman codes


def test_stage4_dyadic_hand_example(nb):
    p = [0.5, 0.25, 0.125, 0.125]
    code = _huffman(nb, p)
    assert len(code) == 4 and all(set(c) <= {"0", "1"} and c for c in code), (
        f"return one nonempty string of '0'/'1' per symbol, got {code}")
    assert [len(c) for c in code] == [1, 2, 3, 3], (
        f"probabilities (1/2, 1/4, 1/8, 1/8) get codeword lengths (1, 2, 3, 3), got {[len(c) for c in code]}")
    assert _prefix_free(code), f"{code} isn't prefix-free: some codeword is the start of another"
    L = nb.expected_length(p, code)
    assert abs(L - 1.75) < 1e-12, f"the expected length is 1/2 + 2/4 + 3/8 + 3/8 = 1.75 bits = the entropy, got {L}"


def test_stage4_optimal_on_random_distributions(nb):
    rng = np.random.default_rng(4)
    for k in (2, 3, 7, 26, 100):
        for _ in range(3):
            p = rng.dirichlet(np.full(k, 0.5))
            code = _huffman(nb, p)
            assert len(code) == k and _prefix_free(code), f"k = {k}: the code must have {k} prefix-free codewords"
            kraft = sum(2.0 ** -len(c) for c in code)
            assert abs(kraft - 1) < 1e-12, f"a Huffman code is complete: sum 2^-len = 1 (Kraft), got {kraft}"
            L, opt = nb.expected_length(p, code), _optimal_length(p)
            assert abs(L - opt) < 1e-12, (
                f"k = {k}: an optimal prefix code has expected length {opt:.6f}, yours has {L:.6f}. merge the TWO "
                "LEAST probable nodes each time (a min-heap).")
            H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
            assert H - 1e-12 <= L < H + 1, "Shannon: H <= L < H + 1"


def test_stage4_edge_cases(nb):
    assert _huffman(nb, [1.0]) == ["0"], "a single symbol still needs one bit: ['0']"
    assert sorted(len(c) for c in _huffman(nb, [0.9, 0.1])) == [1, 1], "two symbols: one bit each"
    assert sorted(len(c) for c in _huffman(nb, np.ones(8))) == [3] * 8, "8 equal weights: 3 bits each"
    code = _huffman(nb, [5, 5, 5, 5, 1, 1])
    assert _prefix_free(code) and abs(nb.expected_length([5, 5, 5, 5, 1, 1], code) - _optimal_length([5, 5, 5, 5, 1, 1])) < 1e-12, (
        "counts (weights) should work as well as probabilities, ties included")


# ---------------------------------------------------------------- stage 5: encoding and decoding


def test_stage5_hand_example(nb):
    code = ["0", "10", "11"]
    assert nb.encode([0, 2, 1, 0], code) == "011100", (
        f"with code (0, 10, 11), symbols (0, 2, 1, 0) encode to '0' + '11' + '10' + '0' = '011100', got "
        f"{nb.encode([0, 2, 1, 0], code)!r}")
    assert _decode(nb, "011100", code) == [0, 2, 1, 0], f"decoding '011100' should give [0, 2, 1, 0], got {_decode(nb, '011100', code)}"
    assert _decode(nb, "", code) == [], "decoding nothing gives nothing"
    try:
        _decode(nb, "01", code)
    except ValueError:
        return
    raise AssertionError("'01' decodes to 0 and then a dangling '1', which isn't a codeword: raise ValueError")


def test_stage5_roundtrip(nb):
    rng = np.random.default_rng(5)
    for code in (["1", "01", "001", "000"], ["00", "01", "10", "110", "111"]):
        msg = rng.integers(0, len(code), 3000)
        bits = nb.encode(msg, code)
        assert len(bits) == sum(len(code[s]) for s in msg), "the encoding is the concatenation of the codewords"
        back = _decode(nb, bits, code)
        assert list(back) == [int(s) for s in msg], (
            f"decode(encode(message)) must give the message back (code {code}). read bits until they form a codeword, "
            "emit it, and start over.")
