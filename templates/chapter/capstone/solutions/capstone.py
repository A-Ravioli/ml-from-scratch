import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # __CHAPTER_NUM__ · capstone (reference solution)

    <!-- TEMPLATE: replace the nearest-centroid example with this chapter's capstone. -->

    read [`brief.md`](brief.md) first. implement the required functions below, then write up
    your results at the end of this notebook.
    """)
    return


@app.function
def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Fit on (X_train, y_train) and return integer class predictions for X_test."""
    classes = np.unique(y_train)
    centroids = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])
    dists = ((X_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return classes[dists.argmin(axis=1)]


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", fit_predict=fit_predict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    results (with uncertainty), baselines, ablations, and what you learned.
    """)
    return


if __name__ == "__main__":
    app.run()
