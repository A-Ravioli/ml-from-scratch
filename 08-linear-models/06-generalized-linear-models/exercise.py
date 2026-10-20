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
    # generalized linear models · build

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
        raise NotImplementedError("stage1: implement Gaussian.inverse_link")

    def link(self, mu):
        """stage 1: eta as a function of mu (the inverse of inverse_link)."""
        raise NotImplementedError("stage1: implement Gaussian.link")

    def variance(self, mu):
        """stage 1: the variance function V(mu), same shape as mu."""
        raise NotImplementedError("stage1: implement Gaussian.variance")

    def deviance(self, y, mu):
        """stage 1: the unit deviance d(y, mu) per sample: (y - mu)^2."""
        raise NotImplementedError("stage1: implement Gaussian.deviance")


@app.class_definition
class Bernoulli:
    """0/1 responses with the logit link: mu = sigmoid(eta), V(mu) = mu (1 - mu)."""

    name = "bernoulli"

    def inverse_link(self, eta):
        """stage 1: mu = sigmoid(eta) (stable: use mlfs.utils.sigmoid)."""
        raise NotImplementedError("stage1: implement Bernoulli.inverse_link")

    def link(self, mu):
        """stage 1: the logit log(mu / (1 - mu))."""
        raise NotImplementedError("stage1: implement Bernoulli.link")

    def variance(self, mu):
        """stage 1: mu (1 - mu)."""
        raise NotImplementedError("stage1: implement Bernoulli.variance")

    def deviance(self, y, mu):
        """stage 1: 2 [y log(y / mu) + (1 - y) log((1 - y) / (1 - mu))], with 0 log 0 = 0."""
        raise NotImplementedError("stage1: implement Bernoulli.deviance")


@app.class_definition
class Poisson:
    """counts with the log link: mu = exp(eta), V(mu) = mu."""

    name = "poisson"

    def inverse_link(self, eta):
        """stage 1: mu = exp(eta)."""
        raise NotImplementedError("stage1: implement Poisson.inverse_link")

    def link(self, mu):
        """stage 1: log(mu)."""
        raise NotImplementedError("stage1: implement Poisson.link")

    def variance(self, mu):
        """stage 1: mu."""
        raise NotImplementedError("stage1: implement Poisson.variance")

    def deviance(self, y, mu):
        """stage 1: 2 [y log(y / mu) - (y - mu)], with 0 log 0 = 0 (so y = 0 gives 2 mu)."""
        raise NotImplementedError("stage1: implement Poisson.deviance")


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
    raise NotImplementedError("stage2: implement glm_loss")


@app.function
def glm_gradient(theta, A, y, family, penalty):
    """stage 2: its gradient A^T (mu - y) / n + penalty * theta (canonical links only)."""
    raise NotImplementedError("stage2: implement glm_gradient")


@app.function
def irls_step(theta, A, y, family, penalty):
    """stage 2: one IRLS update: the weighted ridge solution
    (A^T W A + n diag(penalty)) theta_new = A^T W z, with weights W = V(mu) and working
    response z = eta + (y - mu) / V(mu). compute A^T W z as A^T (W eta + (y - mu)) so that a
    tiny V(mu) never divides anything. returns theta_new (p,)."""
    raise NotImplementedError("stage2: implement irls_step")


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
        raise NotImplementedError("stage3: implement GLM.fit")

    def predict(self, X):
        """stage 3: the fitted mean mu = inverse_link(X @ coef_ + intercept_), shape (n,)."""
        raise NotImplementedError("stage3: implement GLM.predict")


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
    raise NotImplementedError("stage4: implement d2_score")


@app.function
def pearson_dispersion(y, mu, family, n_params):
    """stage 4: the Pearson estimate of the dispersion, sum (y - mu)^2 / V(mu) / (n - n_params).
    about 1 if the family's variance is right; much bigger means overdispersion."""
    raise NotImplementedError("stage4: implement pearson_dispersion")


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
