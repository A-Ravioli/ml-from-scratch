"""Checks for exercise.py (variational circuits).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Circuits are checked
against unitaries built gate by gate and against closed forms for one qubit (where <Z> = cos(a + x)
and <X> = sin(a + x) cos b); shot estimates with unbiasedness and variance tests over many seeds; the
parameter-shift rule by the exact points it evaluates and against analytic and finite-difference
gradients; training by Adam's exactly predictable first step and by learning a realizable task over
several seeds; gradient sampling by reproducing its random draws.
"""
import numpy as np
from scipy.linalg import expm

from mlfs.numgrad import numerical_gradient
from mlfs.testing import bounded

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def _ry(t):
    return expm(-0.5j * t * PAULI["Y"])


def _rz(t):
    return expm(-0.5j * t * PAULI["Z"])


def _embed(U, targets, n):
    """big-endian embedding of a 1- or 2-qubit gate on adjacent/any targets via kron + permutation."""
    k = len(targets)
    rest = [q for q in range(n) if q not in targets]
    full = np.kron(U, np.eye(2 ** (n - k)))  # acts on qubits (targets..., rest...) in that order
    perm = list(targets) + rest  # position i of the kron order holds qubit perm[i]
    T = full.reshape((2,) * (2 * n))
    inv = np.argsort(perm)
    T = T.transpose(list(inv) + [n + i for i in inv])
    return T.reshape(2 ** n, 2 ** n)


def _unitary_of(ops, n):
    U = np.eye(2 ** n, dtype=complex)
    for M, t in ops:
        U = _embed(np.asarray(M), list(t), n) @ U
    return U


def _expected_ansatz(theta):
    L, n, _ = theta.shape
    ops = []
    for l in range(L):
        for q in range(n):
            ops += [(_ry(theta[l, q, 0]), (q,)), (_rz(theta[l, q, 1]), (q,))]
        for q in range(n - 1):
            ops.append((CNOT, (q, q + 1)))
    return ops


def _enc(x):
    return [(_ry(v), (q,)) for q, v in enumerate(x)]


def _random_state(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def _pauli_matrix(p):
    M = np.array([[1]], dtype=complex)
    for c in p:
        M = np.kron(M, PAULI[c])
    return M


# --- stage 1 -------------------------------------------------------------------------------------

def test_stage1_ansatz_gate_sequence(nb):
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, size=(2, 3, 2))
    ops = nb.ansatz_ops(theta)
    want = _expected_ansatz(theta)
    assert len(ops) == len(want), (
        f"a 2-layer, 3-qubit ansatz has 2 x (3 RY + 3 RZ + 2 CNOT) = 16 gates, got {len(ops)}")
    for i, ((M, t), (W, tw)) in enumerate(zip(ops, want)):
        assert tuple(t) == tw and np.allclose(M, W, atol=1e-12), (
            f"gate {i} should act on {tw} with matrix\n{np.round(W, 3)}\ngot targets {tuple(t)} and\n"
            f"{np.round(np.asarray(M), 3)}. per layer: for each qubit q in order, RY(theta[l, q, 0]) then "
            "RZ(theta[l, q, 1]); then CNOT(0, 1), CNOT(1, 2), ...")


def test_stage1_ansatz_edge_cases(nb):
    ops = nb.ansatz_ops(np.zeros((1, 1, 2)))
    assert len(ops) == 2, f"one qubit, one layer: RY and RZ only (no CNOT), got {len(ops)} gates"
    try:
        nb.ansatz_ops(np.zeros((2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("ansatz_ops should raise ValueError when theta is not (L, n, 2)")


def test_stage1_encoding_and_model(nb):
    x = np.array([0.3, -1.2])
    enc = nb.encoding_ops(x)
    assert len(enc) == 2 and all(tuple(t) == (q,) for q, (_, t) in enumerate(enc)), (
        "encoding_ops(x) should be one RY per feature, RY(x[q]) on qubit q")
    assert np.allclose(_unitary_of(enc, 2), _unitary_of(_enc(x), 2)), "encoding_ops: RY(x[q]) on qubit q"
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, size=(2, 2, 2))
    got = _unitary_of(nb.model_ops(theta, x), 2)
    want = _unitary_of(_enc(x) + _expected_ansatz(theta), 2)
    assert np.allclose(got, want), "model_ops(theta, x) should be the encoding followed by the whole ansatz"
    got = _unitary_of(nb.model_ops(theta, x, reupload=True), 2)
    want = _unitary_of(_enc(x) + _expected_ansatz(theta[:1]) + _enc(x) + _expected_ansatz(theta[1:]), 2)
    assert np.allclose(got, want), (
        "with reupload=True the encoding comes before EVERY layer: enc, layer 0, enc, layer 1, ...")
    theta3 = rng.uniform(0, 2 * np.pi, size=(3, 2, 2))
    got = _unitary_of(nb.model_ops(theta3, x, reupload=True), 2)
    want = _unitary_of(sum((_enc(x) + _expected_ansatz(theta3[l:l + 1]) for l in range(3)), []), 2)
    assert np.allclose(got, want), "with 3 layers and reupload=True: enc, layer 0, enc, layer 1, enc, layer 2"
    try:
        nb.model_ops(theta, [0.1, 0.2, 0.3])
    except ValueError:
        pass
    else:
        raise AssertionError("model_ops should raise ValueError when len(x) != theta.shape[1]")


# --- stage 2 -------------------------------------------------------------------------------------

def test_stage2_one_qubit_closed_form(nb):
    for a, b, x in [(0.4, 1.1, 0.2), (2.5, -0.7, 1.3)]:
        theta = np.array([[[a, b]]])
        for obs, want in [("Z", np.cos(a + x)), ("X", np.sin(a + x) * np.cos(b)), ("Y", np.sin(a + x) * np.sin(b))]:
            got = nb.expectation(theta, [x], obs)
            assert isinstance(got, float), f"expectation should return a Python float, got {type(got).__name__}"
            assert np.isclose(got, want, atol=1e-10), (
                f"one qubit, RY(x={x}) then RY(a={a}) then RZ(b={b}): <{obs}> should be {want:.6f}, got {got:.6f}")
        assert np.isclose(nb.expectation(theta, [x]), np.cos(a + x)), "the default observable is Z on qubit 0"


def test_stage2_matches_unitary(nb):
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 2 * np.pi, size=(2, 3, 2))
    x = rng.uniform(0, np.pi, size=3)
    for reupload in (False, True):
        ops = _enc(x) + _expected_ansatz(theta) if not reupload else (
            _enc(x) + _expected_ansatz(theta[:1]) + _enc(x) + _expected_ansatz(theta[1:]))
        psi = _unitary_of(ops, 3)[:, 0]
        for obs in ("ZII", "IXY", "ZZZ"):
            want = np.real(psi.conj() @ _pauli_matrix(obs) @ psi)
            got = nb.expectation(theta, x, obs, reupload=reupload)
            assert np.isclose(got, want, atol=1e-10), (
                f"expectation(theta, x, {obs!r}, reupload={reupload}) should be {want:.6f}, got {got:.6f}")
        want = np.real(psi.conj() @ _pauli_matrix("ZII") @ psi)
        assert np.isclose(nb.expectation(theta, x, reupload=reupload), want, atol=1e-10), (
            "the default observable is Z on qubit 0, i.e. 'Z' + 'I' * (n - 1)")


def test_stage2_estimate_pauli_exact_cases(nb):
    s = 1 / np.sqrt(2)
    bell = np.array([s, 0, 0, s], dtype=complex)
    for p, want in [("ZZ", 1), ("XX", 1), ("YY", -1), ("II", 1)]:
        got = nb.estimate_pauli(bell, p, 200, rng=0)
        assert np.isclose(got, want), (
            f"every shot of {p} on a Bell pair gives {want:+d}, so the estimate is exactly {want}; got {got}. "
            "measure X by applying H first, Y by applying S^dagger then H, and multiply the +-1 outcomes of "
            "the qubits in the string.")
    assert nb.estimate_pauli(bell, "ZI", 300, rng=5) == nb.estimate_pauli(bell, "ZI", 300, rng=5), (
        "the same seed should give the same estimate")
    # the draw order: ONE sample_counts call on the state rotated into the Z basis
    psi = _random_state(2, np.random.default_rng(20))
    Sdg = np.diag([1, -1j])
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    rotated = np.kron(Hm, Hm @ Sdg) @ psi  # 'XY': H on qubit 0, S^dagger then H on qubit 1
    counts = nb.qsim.sample_counts(rotated, 64, np.random.default_rng(21))
    want = sum(c * (-1) ** (int(b[0]) + int(b[1])) for b, c in counts.items()) / 64
    got = nb.estimate_pauli(psi, "XY", 64, rng=21)
    assert np.isclose(got, want), (
        f"estimate_pauli should rotate into the Z basis and draw all shots with ONE qsim.sample_counts(state, shots, "
        f"rng) call; with seed 21 that gives {want}, got {got}")
    try:
        nb.estimate_pauli(bell, "ZZZ", 10, rng=0)
    except ValueError:
        pass
    else:
        raise AssertionError("estimate_pauli should raise ValueError for a Pauli string of the wrong length")


def test_stage2_estimate_pauli_unbiased_with_right_variance(nb):
    rng = np.random.default_rng(3)
    psi = _random_state(2, rng)
    shots, reps = 100, 300
    for p in ("XY", "YI", "ZX"):
        exact = np.real(psi.conj() @ _pauli_matrix(p) @ psi)
        est = np.array([nb.estimate_pauli(psi, p, shots, rng=1000 + r) for r in range(reps)])
        sd = np.sqrt((1 - exact ** 2) / shots)
        assert abs(est.mean() - exact) < 5 * sd / np.sqrt(reps), (
            f"estimates of <{p}> average {est.mean():.4f} over {reps} runs but the exact value is {exact:.4f}: "
            "the estimator is biased. check the basis change for X and Y.")
        ratio = est.var(ddof=1) / sd ** 2
        assert 0.7 < ratio < 1.4, (
            f"the variance of a {shots}-shot estimate of <{p}> should be (1 - <P>^2)/shots = {sd ** 2:.2e}; got "
            f"{est.var(ddof=1):.2e}. each shot is an independent +-1 outcome.")


# --- stage 3 -------------------------------------------------------------------------------------

def test_stage3_evaluates_at_shifted_points(nb):
    calls = []

    def f(t):
        calls.append(np.array(t, dtype=float).copy())
        return float(np.sum(np.sin(t)))

    theta = np.array([[0.2, -1.0, 0.7]])
    before = theta.copy()
    g = bounded(nb.parameter_shift, f, theta, what="parameter_shift")
    assert np.array_equal(theta, before), "parameter_shift must not modify theta"
    assert np.shape(g) == theta.shape, f"the gradient should have theta's shape {theta.shape}, got {np.shape(g)}"
    assert len(calls) == 2 * theta.size, (
        f"the rule needs exactly 2 evaluations per parameter ({2 * theta.size} here), got {len(calls)}")
    offsets = sorted(tuple(np.round((c - theta).ravel(), 10)) for c in calls)
    want = sorted(tuple(np.round(s * np.pi / 2 * np.eye(3)[i], 10)) for i in range(3) for s in (1, -1))
    assert offsets == want, (
        "parameter_shift should evaluate f at theta + pi/2 e_i and theta - pi/2 e_i, one parameter at a time. "
        f"got offsets {offsets}")
    assert np.allclose(g, np.cos(theta), atol=1e-12), (
        f"for f = sum(sin(theta)) the gradient is cos(theta) = {np.round(np.cos(theta), 6)}, got {np.round(g, 6)}. "
        "remember the factor 1/2: (f(t + pi/2) - f(t - pi/2)) / 2.")


def test_stage3_exact_on_circuits(nb):
    a, b, x = 0.9, 0.4, 0.3
    g = nb.parameter_shift(lambda t: nb.expectation(t, [x], "X"), np.array([[[a, b]]]))
    want = np.array([[[np.cos(a + x) * np.cos(b), -np.sin(a + x) * np.sin(b)]]])
    assert np.allclose(g, want, atol=1e-10), (
        f"d<X>/d(a, b) = (cos(a+x) cos b, -sin(a+x) sin b) = {want.ravel()}, got {np.ravel(g)}. the shift rule is "
        "exact (not an approximation), so these must agree to 1e-10.")
    rng = np.random.default_rng(4)
    theta = rng.uniform(0, 2 * np.pi, size=(2, 2, 2))
    xs = rng.uniform(0, np.pi, size=2)
    f = lambda t: nb.expectation(t, xs, "ZZ", reupload=True)  # noqa: E731
    g = nb.parameter_shift(f, theta)
    fd = numerical_gradient(f, theta, h=1e-5)
    assert np.allclose(g, fd, atol=1e-7), (
        f"parameter-shift and finite-difference gradients disagree by {np.max(np.abs(g - fd)):.1e} on a 2-qubit circuit")


# --- stage 4 -------------------------------------------------------------------------------------

def test_stage4_classify(nb):
    theta = np.zeros((1, 1, 2))
    got = nb.classify(theta, np.array([[0.0], [np.pi], [2.0], [1.0]]))
    assert np.array_equal(np.asarray(got), [1, -1, -1, 1]), (
        f"with theta = 0, f(x) = cos(x): labels sign(cos x) for x = 0, pi, 2, 1 are [1, -1, -1, 1]; got {got}")
    rng = np.random.default_rng(9)
    theta = rng.uniform(0, 2 * np.pi, size=(2, 2, 2))
    X = rng.uniform(0, np.pi, size=(12, 2))
    want = np.where([nb.expectation(theta, x, reupload=True) >= 0 for x in X], 1, -1)
    got = nb.classify(theta, X, reupload=True)
    assert np.array_equal(np.asarray(got), want), (
        f"classify(theta, X, reupload=True) should be the sign of expectation(theta, x, reupload=True); got {got}, "
        f"expected {want}. pass reupload through.")


def _adam_by_hand(nb, X, y, *, layers, steps, lr, seed, reupload):
    """Adam (0.9, 0.999, 1e-8, bias-corrected) on the MSE, with finite-difference gradients; (thetas, losses)."""

    def loss(t):
        f = np.array([nb.expectation(t, x, reupload=reupload) for x in X])
        return float(np.mean((f - y) ** 2))

    theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=(layers, X.shape[1], 2))
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    thetas, losses = [theta], []
    for t in range(1, steps + 1):
        losses.append(loss(theta))
        g = numerical_gradient(loss, theta, h=1e-5)
        g[np.abs(g) < 1e-7] = 0.0  # gradients that are exactly 0 (an RZ commuting with everything after it)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        theta = theta - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        thetas.append(theta)
    losses.append(loss(theta))
    return thetas, losses


def test_stage4_first_adam_step_is_exact(nb):
    rng = np.random.default_rng(5)
    X = rng.uniform(0, np.pi, size=(6, 2))
    y = np.array([1, -1, 1, 1, -1, -1])
    for reupload in (False, True):
        theta1, losses = nb.train_classifier(X, y, layers=2, steps=1, lr=0.05, reupload=reupload, rng=7)
        thetas, want_losses = _adam_by_hand(nb, X, y, layers=2, steps=1, lr=0.05, seed=7, reupload=reupload)
        assert np.shape(theta1) == (2, 2, 2), (
            f"theta should have shape (layers, n_features, 2) = (2, 2, 2), got {np.shape(theta1)}")
        assert len(losses) == 2, f"steps=1 should give 2 losses (before and after the step), got {len(losses)}"
        assert np.isclose(losses[0], want_losses[0]), (
            f"losses[0] should be the MSE at the initial theta = rng.uniform(0, 2 pi, size=(layers, n, 2)) drawn "
            f"first from as_rng(rng) (reupload={reupload}): {want_losses[0]:.6f}, got {losses[0]:.6f}")
        assert np.allclose(theta1, thetas[1], atol=1e-6), (
            f"with reupload={reupload}, Adam's first bias-corrected step moves every parameter by lr * sign(gradient) "
            f"(0 where the gradient is exactly 0). got a step of {np.round(np.ravel(theta1 - thetas[0]), 4)}, expected "
            f"{np.round(np.ravel(thetas[1] - thetas[0]), 4)}. check the gradient of the MSE (2 (f - y) / N times "
            "df/dtheta), that reupload is passed to every expectation, and the bias correction.")
        assert np.isclose(losses[1], want_losses[1]), "losses[-1] should be the loss at the returned theta"


def test_stage4_three_adam_steps_are_exact(nb):
    rng = np.random.default_rng(8)
    X = rng.uniform(0, np.pi, size=(5, 2))
    y = np.array([1, -1, -1, 1, 1])
    theta3, losses = nb.train_classifier(X, y, layers=2, steps=3, lr=0.2, rng=3)
    thetas, want_losses = _adam_by_hand(nb, X, y, layers=2, steps=3, lr=0.2, seed=3, reupload=False)
    assert len(losses) == 4, f"steps=3 should give 4 losses, got {len(losses)}"
    assert np.allclose(losses, want_losses, atol=1e-8), (
        f"losses after 0..3 Adam steps should be {np.round(want_losses, 6)}, got {np.round(losses, 6)}")
    assert np.allclose(theta3, thetas[3], atol=1e-6), (
        f"after 3 Adam steps theta is off by up to {np.max(np.abs(theta3 - thetas[3])):.1e}. step 1 can be right "
        "while later steps are wrong: carry m and v over between steps (beta1 = 0.9, beta2 = 0.999), bias-correct "
        "with the current step number t (1 - beta**t), and use lr on every step.")


def test_stage4_learns_a_realizable_task(nb):
    rng = np.random.default_rng(6)
    teacher = rng.uniform(0, 2 * np.pi, size=(1, 2, 2))
    X = rng.uniform(0, np.pi, size=(16, 2))
    f = np.array([nb.expectation(teacher, x) for x in X])
    keep = np.abs(f) > 0.15  # a margin, so the task is learnable to 100%
    X, y = X[keep], np.where(f[keep] > 0, 1, -1)
    accs, drops = [], []
    for seed in range(3):
        theta, losses = bounded(nb.train_classifier, X, y, layers=1, steps=30, lr=0.2, rng=seed, budget=20_000_000,
                                what="train_classifier")
        assert len(losses) == 31, f"steps=30 should give 31 losses, got {len(losses)}"
        accs.append(np.mean(nb.classify(theta, X) == y))
        drops.append(losses[-1] / losses[0])
    assert np.median(accs) >= 0.85, (
        f"labels made by a circuit of the same shape should be learned: training accuracy over 3 seeds was "
        f"{np.round(accs, 2)} (median should be >= 0.85). loss ratios final/initial: {np.round(drops, 2)}. is the "
        "gradient's sign right (you minimize the loss)?")
    try:
        nb.train_classifier(X, np.zeros(len(X)), steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("train_classifier should raise ValueError when labels aren't -1/+1")


# --- stage 5 -------------------------------------------------------------------------------------

def test_stage5_reproduces_draws(nb):
    g = nb.gradient_samples(1, 1, 50, observable="Z", rng=11)
    assert np.shape(g) == (50,), f"gradient_samples should return shape (n_samples,) = (50,), got {np.shape(g)}"
    gen = np.random.default_rng(11)
    a = np.array([gen.uniform(0, 2 * np.pi, size=(1, 1, 2))[0, 0, 0] for _ in range(50)])
    assert np.allclose(g, -np.sin(a), atol=1e-10), (
        "for one qubit, <Z> = cos(theta[0, 0, 0]), so each sample is -sin(theta[0, 0, 0]) with theta drawn as "
        "rng.uniform(0, 2 pi, size=(layers, n, 2)), one draw per sample, in order. your first samples: "
        f"{np.round(g[:3], 4)}, expected {np.round(-np.sin(a[:3]), 4)}")


def test_stage5_one_qubit_variance(nb):
    g = nb.gradient_samples(1, 1, 2000, observable="Z", rng=12)
    assert abs(g.mean()) < 0.07 and 0.45 < g.var() < 0.55, (
        f"for one qubit the gradient is -sin(a) with a uniform: mean 0, variance 1/2. got mean {g.mean():.3f}, "
        f"variance {g.var():.3f}")


def test_stage5_global_observable_has_smaller_gradients(nb):
    n = 4
    local = nb.gradient_samples(n, 2, 300, observable="ZZII", rng=13)
    glob = nb.gradient_samples(n, 2, 300, observable="ZZZZ", rng=14)
    assert abs(local.mean()) < 4 * local.std() / np.sqrt(300), (
        f"gradients over random parameters average to 0 (symmetry); got mean {local.mean():.3f}")
    assert local.var() > 2.5 * glob.var(), (
        f"on 4 qubits with 2 layers, the gradient variance for the local observable ZZII should be several times "
        f"that for the global ZZZZ: got {local.var():.3e} vs {glob.var():.3e}. check that the observable is "
        "passed through to the expectation.")
