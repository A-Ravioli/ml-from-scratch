import numpy as np

from exercise import BayesianLinearRegression, GaussianProcess, NaiveBayesClassifier, bayesian_model_selection


def test_naive_bayes_and_bayesian_linear_regression():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    nb = NaiveBayesClassifier()
    nb.fit(X, y)
    assert nb.predict(X).shape == y.shape

    X_reg = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    y_reg = 2.0 * X_reg.ravel() + 1.0
    model = BayesianLinearRegression(alpha=1.0, beta=10.0)
    model.fit(X_reg, y_reg)
    mean = model.predict(X_reg)
    assert mean.shape == y_reg.shape


def test_gaussian_process_and_model_selection():
    X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = np.sin(X).ravel()
    gp = GaussianProcess(kernel="rbf", noise_level=1e-4)
    gp.fit(X, y)
    mean = gp.predict(X)
    assert mean.shape == y.shape

    best_idx, scores = bayesian_model_selection(X, y, [BayesianLinearRegression(), GaussianProcess()])
    assert len(scores) == 2
    assert int(best_idx) in {0, 1}
