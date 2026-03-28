import numpy as np

from exercise import CrossValidation, LinearRegression, PolynomialFeatures, RidgeRegression


def test_linear_regression_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5
    model = LinearRegression(solver="normal_equation")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.mean((preds - y) ** 2) < 1e-8


def test_ridge_regression_and_polynomial_features():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    ridge = RidgeRegression(lambda_reg=0.1)
    ridge.fit(X, y)
    assert ridge.predict(X).shape == y.shape

    poly = PolynomialFeatures(degree=2, include_bias=True)
    transformed = poly.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] >= 2


def test_cross_validation_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] - X[:, 1]
    cv = CrossValidation(n_folds=3, scoring="mse", random_state=0)
    scores = cv.cross_validate(LinearRegression(), X, y)
    assert len(scores) == 3
