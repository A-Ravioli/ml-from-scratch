import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.base import Classifier, check_X_y  # yours, from topic 01
    from mlfs.model_selection import StratifiedKFold, cross_val_score  # yours, from topic 02
    from mlfs.utils import pairwise_distances  # yours, from ch 00

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.class_definition
class KNNClassifier(Classifier):
    """majority vote of the k nearest training rows (Euclidean distance). a lab helper:
    chapter 09 builds k-NN properly."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_, y = check_X_y(X, y)
        self.classes_, self.y_idx_ = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, X):
        X, _ = check_X_y(X)
        nearest = np.argsort(pairwise_distances(X, self.X_), axis=1)[:, : self.k]
        votes = np.zeros((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            votes[:, c] = np.mean(self.y_idx_[nearest] == c, axis=1)
        return votes


@app.function
def loan_data(n, rng, income_unit=1.0, outlier_frac=0.0):
    """columns: age (years), income (in units of `income_unit` dollars), years at the job.
    the label depends on age and job tenure only; income is irrelevant. a fraction
    `outlier_frac` of rows get a data-entry error: 500 years at the job."""
    age = rng.normal(45, 12, n)
    income = rng.normal(60_000, 20_000, n) / income_unit
    job = np.clip(rng.normal(8, 5, n), 0, None)
    y = ((age - 45) / 12 + (job - 8) / 5 + 0.6 * rng.normal(size=n) > 0).astype(int)
    bad = rng.random(n) < outlier_frac
    job = np.where(bad, 500.0, job)
    return np.column_stack([age, income, job]), y


@app.function
def cv_accuracy(model, X, y, seed=0):
    """mean 5-fold stratified CV accuracy, with YOUR splitters and cross_val_score."""
    return float(cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, rng=seed)).mean())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # features & preprocessing · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the model throughout is $k$-nearest neighbours
    ($k = 5$), a small helper defined in this notebook, which classifies a point by the
    labels of the training points closest to it in Euclidean distance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: kNN with and without scaling

    600 loan applicants with three features: age (years), income, and years at their job.
    whether they repay depends on age and job tenure; income is irrelevant. three pipelines
    built with your `make_pipeline`, each scored by 5-fold CV with your `cross_val_score`:
    kNN on the raw features, after your `StandardScaler`, and after your `MinMaxScaler`.
    the dropdown changes only the *unit* income is recorded in.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with income in dollars, what CV accuracy will kNN get on the raw features? with "
        "standardized features? does the unit of income matter after scaling?"
    )
    p1
    return (p1,)


@app.cell
def _():
    unit = mo.ui.dropdown({"dollars": 1.0, "thousands of dollars": 1_000.0, "millions of dollars": 1e6},
                          value="dollars", label="income recorded in")
    unit
    return (unit,)


@app.cell
def _(p1, unit):
    ui.gate(p1)
    _X, _y = loan_data(600, np.random.default_rng(0), income_unit=unit.value)
    _models = {
        "raw features": KNNClassifier(5),
        "StandardScaler → kNN": ex.make_pipeline(ex.StandardScaler(), KNNClassifier(5)),
        "MinMaxScaler → kNN": ex.make_pipeline(ex.MinMaxScaler(), KNNClassifier(5)),
    }
    _rows = [f"| {k} | {cv_accuracy(m, _X, _y):.3f} |" for k, m in _models.items()]
    _std = _X.std(axis=0)
    mo.md(f"feature standard deviations (age, income, job): {np.round(_std, 2).tolist()}\n\n"
          "| model | 5-fold CV accuracy |\n|---|---|\n" + "\n".join(_rows)
          + f"\n\nmajority-class baseline: {max(_y.mean(), 1 - _y.mean()):.3f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which feature decides who the nearest neighbours are on the raw data, and
    why? why does changing income from dollars to millions change the raw model so much, and
    the scaled models not at all? what does that say about distance-based models (and about
    gradient descent's conditioning, chapter 06) versus, say, a decision tree?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: one typo in a hundred

    the same data (income in dollars), but in a fraction of rows "years at the job" was typed
    as 500. the label is unaffected. we compare the two scalers as the fraction of corrupted
    rows grows.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 1% of rows corrupted, which scaler suffers more, StandardScaler or MinMaxScaler? "
        "by how much does accuracy drop?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _fracs = [0.0, 0.002, 0.005, 0.01, 0.02]
    _res = {"StandardScaler": [], "MinMaxScaler": []}
    for _f in _fracs:
        _X, _y = loan_data(600, np.random.default_rng(1), outlier_frac=_f)
        _res["StandardScaler"].append(cv_accuracy(ex.make_pipeline(ex.StandardScaler(), KNNClassifier(5)), _X, _y))
        _res["MinMaxScaler"].append(cv_accuracy(ex.make_pipeline(ex.MinMaxScaler(), KNNClassifier(5)), _X, _y))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _k, _v in _res.items():
        _ax.plot(np.array(_fracs) * 100, _v, "o-", label=_k)
    _ax.set(xlabel="% of rows with a 500-year job tenure", ylabel="5-fold CV accuracy", ylim=(0.5, 1.0))
    _ax.legend()
    _fig.tight_layout()
    _X, _ = loan_data(600, np.random.default_rng(1), outlier_frac=0.01)
    _job_scaled = ex.MinMaxScaler().fit(_X).transform(_X)[:, 2]
    mo.vstack([_fig, mo.md(
        f"with 1% corrupted, MinMaxScaler squeezes the 99% of normal job tenures into "
        f"[{np.min(_job_scaled[_X[:, 2] < 100]):.3f}, {np.max(_job_scaled[_X[:, 2] < 100]):.3f}] "
        f"while age spans [0, 1].")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does a single extreme value change `data_max_`, and so the scale of
    every other row? why is the standard deviation also affected, but less? both scalers are
    fitted on the training folds only, so this isn't leakage. what would you do about it in a
    real project (before reaching for a fancier scaler)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how to encode a category

    300 customers, each with a city code (6 cities, numbered 0–5) and four numeric features.
    whether a customer churns depends mostly on the city (cities 0, 2 and 4 churn more), plus
    a weak signal in the first numeric feature; the other three are noise. two encodings,
    each followed by your `StandardScaler` and kNN: the city as one integer column, and the
    city one-hot encoded with your `OneHotEncoder`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("which encoding gives kNN the better CV accuracy, and why?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _n = 300
    _city = _rng.integers(0, 6, _n)
    _num = _rng.normal(size=(_n, 4))
    _y = ((_city % 2 == 0) * 1.5 + 0.5 * _num[:, 0] + 0.7 * _rng.normal(size=_n) > 0.75).astype(int)
    _X_int = np.column_stack([_city, _num])
    _X_hot = np.column_stack([ex.OneHotEncoder().fit_transform(_city.reshape(-1, 1)), _num])
    _acc_int = cv_accuracy(ex.make_pipeline(ex.StandardScaler(), KNNClassifier(5)), _X_int, _y)
    _acc_hot = cv_accuracy(ex.make_pipeline(ex.StandardScaler(), KNNClassifier(5)), _X_hot, _y)
    _rates = [float(np.mean(_y[_city == c])) for c in range(6)]
    mo.md("churn rate by city: " + ", ".join(f"{c}: {r:.2f}" for c, r in enumerate(_rates)) +
          f"\n\n| encoding | columns | 5-fold CV accuracy |\n|---|---|---|\n"
          f"| integer code | {_X_int.shape[1]} | {_acc_int:.3f} |\n"
          f"| one-hot | {_X_hot.shape[1]} | {_acc_hot:.3f} |")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** under the integer code, which cities are "close" to city 2, and does that
    closeness mean anything? what are the distances between any two cities after one-hot
    encoding? the one-hot encoder here was fitted on all rows before cross-validation: is
    that a leak? (what does it learn, and could it matter?)
    """)
    return


if __name__ == "__main__":
    app.run()
