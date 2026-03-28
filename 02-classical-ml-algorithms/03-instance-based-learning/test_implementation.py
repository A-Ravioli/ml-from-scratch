import numpy as np

from exercise import KNearestNeighbors, LocallyWeightedRegression, NearestCentroid, analyze_curse_of_dimensionality


def test_knn_and_nearest_centroid():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    knn = KNearestNeighbors(k=1)
    knn.fit(X, y)
    assert np.array_equal(knn.predict(X), y)

    centroid = NearestCentroid()
    centroid.fit(X, y)
    assert centroid.predict(X).shape == y.shape


def test_locally_weighted_regression_and_analysis():
    X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    y = 2.0 * X.ravel()
    model = LocallyWeightedRegression(bandwidth=0.2)
    model.fit(X, y)
    pred = model.predict(np.array([[0.5]]))
    assert pred.shape == (1,)

    analysis = analyze_curse_of_dimensionality(n_samples=20, max_dim=3)
    assert len(analysis) == 2
