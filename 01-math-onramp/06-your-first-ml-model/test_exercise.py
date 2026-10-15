"""Checks for exercise.py (your first ML model).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The gradient
checks use this file's own finite differences, and the exact fit is compared with np.polyfit.
"""
import numpy as np
import pytest

X3, Y3 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 5.0])  # the lesson's worked example


def _data(seed=0, n=40, w=1.7, b=-0.6):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 3.0, n)
    return x, w * x + b + 0.3 * rng.normal(size=n)


def _loss(x, y, w, b):
    return np.mean((w * x + b - y) ** 2)


# ---------------------------------------------------------------- stage 1: model and loss


def test_stage1_predict(nb):
    got = np.asarray(nb.predict([0.0, 1.0, 2.0], 2.0, 1.0))
    assert np.allclose(got, [1.0, 3.0, 5.0]), f"predict([0, 1, 2], w=2, b=1) = 2x + 1 = [1, 3, 5], got {got}"
    got = np.asarray(nb.predict([1, 2, 3], 0.5, 0.0))
    assert np.allclose(got, [0.5, 1.0, 1.5]), f"predict should work on integer lists too, expected [0.5, 1, 1.5], got {got}"


def test_stage1_mse_worked_example(nb):
    got = nb.mse_loss(X3, Y3, 1.0, 1.0)
    assert isinstance(got, float), f"mse_loss should return a float, got {type(got).__name__}"
    assert np.isclose(got, 2 / 3), (
        f"predictions [2, 3, 4] against targets [2, 4, 5] miss by [0, -1, -1], so the MSE is (0 + 1 + 1) / 3 = 0.6667; "
        f"got {got:.4f}. "
        + ("2.0 means you summed instead of averaging." if np.isclose(got, 2.0) else
           "0.8165 means you took a square root (that's the RMSE)." if np.isclose(got, np.sqrt(2 / 3)) else ""))


def test_stage1_mse_matches_definition(nb):
    x, y = _data(1)
    got = nb.mse_loss(x, y, 0.4, 2.0)
    assert np.isclose(got, _loss(x, y, 0.4, 2.0)), (
        f"mse_loss should be mean((w x + b - y)^2) = {_loss(x, y, 0.4, 2.0):.6f}, got {got:.6f}. "
        + ("you averaged |residuals|: square them." if np.isclose(got, np.mean(np.abs(0.4 * x + 2.0 - y))) else ""))
    assert nb.mse_loss(x, 3 * x - 1, 3.0, -1.0) == 0.0, "a line through every point has loss exactly 0"


# ---------------------------------------------------------------- stage 2: the gradient


def test_stage2_worked_example(nb):
    dw, db = nb.mse_gradient(X3, Y3, 1.0, 1.0)
    assert np.isclose(dw, -10 / 3) and np.isclose(db, -4 / 3), (
        f"with residuals r = [0, -1, -1]: dL/dw = (2/3)(0*1 - 1*2 - 1*3) = -3.333 and dL/db = (2/3)(0 - 1 - 1) = -1.333; "
        f"got ({dw:.4f}, {db:.4f}). "
        + ("the signs are flipped: use r = prediction - y." if np.isclose(dw, 10 / 3) else
           "half the right size: the derivative of r^2 is 2r." if np.isclose(dw, -5 / 3) else
           "3x too big: average over the n points, don't just sum." if np.isclose(dw, -10) else ""))


def test_stage2_matches_finite_differences(nb):
    x, y = _data(2)
    for w, b in ((0.3, -1.1), (2.5, 0.7), (-1.4, 3.2)):
        dw, db = nb.mse_gradient(x, y, w, b)
        h = 1e-6
        fdw = (_loss(x, y, w + h, b) - _loss(x, y, w - h, b)) / (2 * h)
        fdb = (_loss(x, y, w, b + h) - _loss(x, y, w, b - h)) / (2 * h)
        assert np.isclose(dw, fdw, rtol=1e-6, atol=1e-8) and np.isclose(db, fdb, rtol=1e-6, atol=1e-8), (
            f"at (w, b) = ({w}, {b}) your gradient is ({dw:.6f}, {db:.6f}) but finite differences give "
            f"({fdw:.6f}, {fdb:.6f}). dL/dw multiplies each residual by its x_i; dL/db doesn't.")


def test_stage2_zero_at_the_best_line(nb):
    x, y = _data(3)
    w, b = np.polyfit(x, y, 1)
    dw, db = nb.mse_gradient(x, y, w, b)
    assert abs(dw) < 1e-10 and abs(db) < 1e-10, (
        f"at the best-fitting line the loss is at its minimum, so both partial derivatives are 0; got ({dw:.2e}, {db:.2e})")


# ---------------------------------------------------------------- stage 3: gradient descent


def test_stage3_first_step_by_hand(nb):
    w, b, losses = nb.gradient_descent(X3, Y3, lr=0.01, n_steps=1)
    # at (0, 0): r = -y = [-2, -4, -5], dL/dw = (2/3)(-25) = -50/3, dL/db = (2/3)(-11) = -22/3
    assert np.isclose(w, 0.5 / 3) and np.isclose(b, 0.22 / 3), (
        f"one step with lr = 0.01 from (0, 0): w = 0 - 0.01 * (-50/3) = 0.1667, b = 0 - 0.01 * (-22/3) = 0.0733; "
        f"got ({w:.4f}, {b:.4f}). "
        + ("b is off: compute BOTH partial derivatives at the old (w, b) before changing either."
           if np.isclose(w, 0.5 / 3) else
           "your step is 10x too big: use the lr argument, not a fixed learning rate." if np.isclose(w, 5 / 3) else
           "move AGAINST the gradient: w - lr * dL/dw."))


def test_stage3_losses_array(nb):
    w, b, losses = nb.gradient_descent(X3, Y3, lr=0.05, n_steps=10, w0=1.0, b0=1.0)
    losses = np.asarray(losses)
    assert losses.shape == (11,), f"n_steps = 10 should give 11 losses (before the first step, then after each), got {losses.shape}"
    assert np.isclose(losses[0], 2 / 3), f"losses[0] is the loss at the start (w0, b0) = (1, 1), i.e. 0.6667; got {losses[0]:.4f}"
    assert np.isclose(losses[-1], _loss(X3, Y3, w, b)), "losses[-1] should be the loss at the (w, b) you return"
    assert np.all(np.diff(losses) < 0), f"with a small learning rate the loss should go down at every step: {np.round(losses, 4)}"


def test_stage3_converges_to_the_best_line(nb):
    x, y = _data(4)
    w, b, losses = nb.gradient_descent(x, y, lr=0.05, n_steps=2000)
    w_star, b_star = np.polyfit(x, y, 1)
    assert abs(w - w_star) < 1e-6 and abs(b - b_star) < 1e-6, (
        f"after 2000 steps with lr = 0.05, gradient descent should reach the best line ({w_star:.5f}, {b_star:.5f}); "
        f"got ({w:.5f}, {b:.5f})")


def test_stage3_too_large_a_step_diverges(nb):
    x, y = _data(5)
    _, _, losses = nb.gradient_descent(x, y, lr=1.0, n_steps=20)
    losses = np.asarray(losses)
    assert losses[-1] > 100 * losses[0], (
        f"with lr = 1.0 on this data each step overshoots further, so the loss should explode; it went from "
        f"{losses[0]:.3g} to {losses[-1]:.3g}. are you really using the lr argument?")


# ---------------------------------------------------------------- stage 4: the closed form


def test_stage4_worked_example(nb):
    w, b = nb.closed_form_fit(X3, Y3)
    assert np.isclose(w, 1.5) and np.isclose(b, 2 / 3), (
        f"for (1, 2), (2, 4), (3, 5): x_mean = 2, y_mean = 11/3, w = 3/2 = 1.5 and b = 11/3 - 1.5 * 2 = 0.6667; "
        f"got ({w:.4f}, {b:.4f})")


def test_stage4_matches_polyfit_and_gd(nb):
    x, y = _data(6, n=80)
    w, b = nb.closed_form_fit(x, y)
    w_ref, b_ref = np.polyfit(x, y, 1)
    assert np.isclose(w, w_ref) and np.isclose(b, b_ref), (
        f"the exact least-squares line is ({w_ref:.6f}, {b_ref:.6f}), got ({w:.6f}, {b:.6f}). "
        "w = sum(dx * dy) / sum(dx * dx) with dx, dy measured from the means.")
    w_gd, b_gd, _ = nb.gradient_descent(x, y, lr=0.05, n_steps=3000)
    assert abs(w_gd - w) < 1e-6 and abs(b_gd - b) < 1e-6, "your gradient descent and your closed form should agree"


def test_stage4_rejects_constant_x(nb):
    with pytest.raises(ValueError):
        nb.closed_form_fit([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------- stage 5: standardizing


def test_stage5_standardize(nb):
    z, m, s = nb.standardize([1.0, 2.0, 3.0])
    assert np.isclose(m, 2.0) and np.isclose(s, np.sqrt(2 / 3)), (
        f"mean 2 and std sqrt(2/3) = 0.8165 (np.std's default divides by n), got mean {m}, std {s}")
    assert np.allclose(z, [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]), f"z = (x - 2) / 0.8165 = [-1.2247, 0, 1.2247], got {z}"
    x = np.random.default_rng(7).uniform(50, 250, 100)
    z, m, s = nb.standardize(x)
    assert abs(np.mean(z)) < 1e-12 and np.isclose(np.std(z), 1.0), "standardized values should have mean 0 and std 1"
    with pytest.raises(ValueError):
        nb.standardize([4.0, 4.0, 4.0])


def test_stage5_unstandardize_by_hand(nb):
    w, b = nb.unstandardize_params(2.0, 5.0, 10.0, 4.0)
    assert np.isclose(w, 0.5) and np.isclose(b, 0.0), (
        f"y = 2 z + 5 with z = (x - 10) / 4 is y = 0.5 x - 5 + 5 = 0.5 x + 0: expected (0.5, 0.0), got ({w}, {b})")


def test_stage5_round_trip_matches_closed_form(nb):
    rng = np.random.default_rng(8)
    x = rng.uniform(50, 250, 60)              # house sizes in square meters: big numbers
    y = 0.02 * x + 0.5 + 0.3 * rng.normal(size=60)
    z, m, s = nb.standardize(x)
    w_z, b_z, _ = nb.gradient_descent(z, y, lr=0.1, n_steps=300)
    w, b = nb.unstandardize_params(w_z, b_z, m, s)
    w_ref, b_ref = np.polyfit(x, y, 1)
    assert np.isclose(w, w_ref, rtol=1e-6) and np.isclose(b, b_ref, rtol=1e-6, atol=1e-8), (
        f"gradient descent on standardized x, converted back, should match the exact line on the raw x: "
        f"({w_ref:.6f}, {b_ref:.6f}); got ({w:.6f}, {b:.6f})")
