import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.utils import sigmoid  # yours, from ch 00 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # generalized linear models · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    classes and functions to write, and checks them live as you edit. `GLM` joins
    `mlfs.linear`. it uses your `Regressor` and `check_X_y` (ch 07) and `sigmoid` (ch 00);
    if those aren't finished, launch with `MLFS_REFERENCE=base,utils marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `Gaussian`, `Bernoulli`, `Poisson`: three exponential families with their canonical links |
    | 2 | `glm_loss`, `glm_gradient`, `irls_step`: one objective and one update for every family |
    | 3 | `GLM`: the estimator, fitted by IRLS with step halving → `mlfs.linear` |
    | 4 | `d2_score`, `pearson_dispersion`: how good the fit is, and whether the variance is right |

    as in topics 04–05, `A` is a design matrix with a first column of ones when there's an
    intercept, `theta` = [intercept, weights], and `penalty` holds each entry's ridge strength.

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Gaussian:
    """normal responses with the identity link: mu = eta, V(mu) = 1."""

    name = "gaussian"

    def inverse_link(self, eta):
        """stage 1: the mean mu as a function of the linear predictor eta."""
        return np.asarray(eta, dtype=np.float64)

    def link(self, mu):
        """stage 1: eta as a function of mu (the inverse of inverse_link)."""
        return np.asarray(mu, dtype=np.float64)

    def variance(self, mu):
        """stage 1: the variance function V(mu), same shape as mu."""
        return np.ones_like(np.asarray(mu, dtype=np.float64))

    def deviance(self, y, mu):
        """stage 1: the unit deviance d(y, mu) per sample: (y - mu)^2."""
        return (np.asarray(y, dtype=np.float64) - mu) ** 2


@app.class_definition
class Bernoulli:
    """0/1 responses with the logit link: mu = sigmoid(eta), V(mu) = mu (1 - mu)."""

    name = "bernoulli"

    def inverse_link(self, eta):
        """stage 1: mu = sigmoid(eta) (stable: use mlfs.utils.sigmoid)."""
        return sigmoid(np.asarray(eta, dtype=np.float64))

    def link(self, mu):
        """stage 1: the logit log(mu / (1 - mu))."""
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(mu) - np.log1p(-mu)

    def variance(self, mu):
        """stage 1: mu (1 - mu)."""
        mu = np.asarray(mu, dtype=np.float64)
        return mu * (1.0 - mu)

    def deviance(self, y, mu):
        """stage 1: 2 [y log(y / mu) + (1 - y) log((1 - y) / (1 - mu))], with 0 log 0 = 0."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            b = np.where(y < 1, (1 - y) * np.log(np.where(y < 1, 1 - y, 1.0) / (1 - mu)), 0.0)
        return 2.0 * (a + b)


@app.class_definition
class Poisson:
    """counts with the log link: mu = exp(eta), V(mu) = mu."""

    name = "poisson"

    def inverse_link(self, eta):
        """stage 1: mu = exp(eta)."""
        return np.exp(np.asarray(eta, dtype=np.float64))

    def link(self, mu):
        """stage 1: log(mu)."""
        return np.log(np.asarray(mu, dtype=np.float64))

    def variance(self, mu):
        """stage 1: mu."""
        return np.asarray(mu, dtype=np.float64).copy()

    def deviance(self, y, mu):
        """stage 1: 2 [y log(y / mu) - (y - mu)], with 0 log 0 = 0 (so y = 0 gives 2 mu)."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ylog = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        return 2.0 * (ylog - (y - mu))


@app.function
def get_family(name):
    """given: 'gaussian' | 'bernoulli' | 'poisson' -> a family object; anything else -> ValueError."""
    families = {"gaussian": Gaussian, "bernoulli": Bernoulli, "poisson": Poisson}
    if name not in families:
        raise ValueError(f"family must be one of {sorted(families)}, got {name!r}")
    return families[name]()


@app.cell(hide_code=True)
def _():
    checks.run(1, Gaussian=Gaussian, Bernoulli=Bernoulli, Poisson=Poisson)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def glm_loss(theta, A, y, family, penalty):
    """stage 2: mean half unit deviance of y against mu = inverse_link(A @ theta), plus
    0.5 * sum(penalty * theta^2). a float."""
    mu = family.inverse_link(A @ theta)
    return float(0.5 * np.mean(family.deviance(y, mu)) + 0.5 * np.sum(penalty * theta**2))


@app.function
def glm_gradient(theta, A, y, family, penalty):
    """stage 2: its gradient A^T (mu - y) / n + penalty * theta (canonical links only)."""
    mu = family.inverse_link(A @ theta)
    return A.T @ (mu - y) / len(y) + penalty * theta


@app.function
def irls_step(theta, A, y, family, penalty):
    """stage 2: one IRLS update: the weighted ridge solution
    (A^T W A + n diag(penalty)) theta_new = A^T W z, with weights W = V(mu) and working
    response z = eta + (y - mu) / V(mu). compute A^T W z as A^T (W eta + (y - mu)) so that a
    tiny V(mu) never divides anything. returns theta_new (p,)."""
    eta = A @ theta
    mu = family.inverse_link(eta)
    w = family.variance(mu)
    lhs = (A.T * w) @ A + len(y) * np.diag(penalty)
    rhs = A.T @ (w * eta + (y - mu))
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


@app.cell(hide_code=True)
def _():
    checks.run(2, glm_loss=glm_loss, glm_gradient=glm_gradient, irls_step=irls_step, Gaussian=Gaussian,
               Bernoulli=Bernoulli, Poisson=Poisson)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class GLM(Regressor):
    """a generalized linear model with a canonical link: E[y | x] = inverse_link(x @ coef_ + intercept_)."""

    def __init__(self, family="poisson", *, alpha=0.0, max_iters=100, tol=1e-08, fit_intercept=True):
        self.family = family
        self.alpha = alpha
        self.max_iters = max_iters
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """stage 3: minimize glm_loss by IRLS steps with step halving; returns self.

        family_ = get_family(family) (ValueError if unknown); alpha < 0 raises ValueError; y
        must be valid for the family (bernoulli: 0/1; poisson: >= 0), else ValueError.
        start from theta = 0 except the intercept, link(mean(y)) (clipped to keep it finite);
        each step goes towards irls_step's theta, halving until the loss is finite and at most
        f + 1e-12 (1 + |f|) (no increase, up to rounding; at most 30 halvings, then stop). stop when ||glm_gradient|| <= tol or after max_iters
        steps. sets coef_ (d,), intercept_ (float), n_iter_, converged_.
        """
        self.family_ = get_family(self.family)
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        if self.family_.name == "bernoulli" and not np.all((y == 0) | (y == 1)):
            raise ValueError("the bernoulli family needs y in {0, 1}")
        if self.family_.name == "poisson" and np.any(y < 0):
            raise ValueError("the poisson family needs y >= 0")
        A = np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X
        pen = np.full(A.shape[1], float(self.alpha))
        theta = np.zeros(A.shape[1])
        if self.fit_intercept:
            pen[0] = 0.0
            m = float(np.mean(y))
            if self.family_.name == "bernoulli":
                m = min(max(m, 1e-3), 1 - 1e-3)
            elif self.family_.name == "poisson":
                m = max(m, 1e-3)
            theta[0] = float(self.family_.link(m))
        f = glm_loss(theta, A, y, self.family_, pen)
        g = glm_gradient(theta, A, y, self.family_, pen)
        steps = 0
        while np.linalg.norm(g) > self.tol and steps < self.max_iters:
            target = irls_step(theta, A, y, self.family_, pen)
            t = 1.0
            for _ in range(31):
                candidate = theta + t * (target - theta)
                with np.errstate(over="ignore"):
                    f_new = glm_loss(candidate, A, y, self.family_, pen)
                if np.isfinite(f_new) and f_new <= f + 1e-12 * (1.0 + abs(f)):
                    break
                t /= 2
            else:
                break
            theta, f = candidate, f_new
            g = glm_gradient(theta, A, y, self.family_, pen)
            steps += 1
        self.n_iter_, self.converged_ = steps, bool(np.linalg.norm(g) <= self.tol)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(theta[0]), np.asarray(theta[1:], dtype=np.float64)
        else:
            self.intercept_, self.coef_ = 0.0, np.asarray(theta, dtype=np.float64)
        return self

    def predict(self, X):
        """stage 3: the fitted mean mu = inverse_link(X @ coef_ + intercept_), shape (n,)."""
        X, _ = check_X_y(X)
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"fitted with {len(self.coef_)} features, got {X.shape[1]}")
        return self.family_.inverse_link(X @ self.coef_ + self.intercept_)


@app.cell(hide_code=True)
def _():
    checks.run(3, GLM=GLM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def d2_score(y, mu, family):
    """stage 4: the fraction of deviance explained, 1 - sum d(y, mu) / sum d(y, mean(y)).
    family is a family object. a float (1 is perfect, 0 is no better than the mean)."""
    y = np.asarray(y, dtype=np.float64)
    null = np.sum(family.deviance(y, np.full_like(y, y.mean())))
    return float(1.0 - np.sum(family.deviance(y, mu)) / null)


@app.function
def pearson_dispersion(y, mu, family, n_params):
    """stage 4: the Pearson estimate of the dispersion, sum (y - mu)^2 / V(mu) / (n - n_params).
    about 1 if the family's variance is right; much bigger means overdispersion."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if n_params >= len(y):
        raise ValueError(f"need more rows ({len(y)}) than parameters ({n_params})")
    return float(np.sum((y - mu) ** 2 / family.variance(mu)) / (len(y) - n_params))


@app.cell(hide_code=True)
def _():
    checks.run(4, d2_score=d2_score, pearson_dispersion=pearson_dispersion, Poisson=Poisson, Gaussian=Gaussian,
               Bernoulli=Bernoulli, GLM=GLM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
