import numpy as np

from exercise import (
    DecisionStumps,
    PolynomialClassifiers,
    UnionOfIntervals,
    sauer_shelah_bound,
)


def test_polynomial_features_and_theoretical_vc_dimension():
    model = PolynomialClassifiers(dimension=2, degree=2)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    features = model._polynomial_features(X)

    assert features.shape == (2, 6)
    assert np.allclose(features[0], np.array([1.0, 1.0, 2.0, 1.0, 2.0, 4.0]))
    assert model.compute_vc_dimension_theoretical() == 6


def test_union_of_intervals_and_decision_stump_predictions():
    union = UnionOfIntervals(k=2)
    X = np.array([-1.0, 0.25, 1.5, 3.0])
    params = np.array([-0.5, 0.5, 2.5, 3.5])

    union_predictions = union.predict(params, X)
    assert np.array_equal(union_predictions, np.array([0, 1, 0, 1]))

    stump = DecisionStumps(dimension=3)
    X_stump = np.array([[0.0, -1.0, 2.0], [0.0, 1.0, -2.0]])
    stump_predictions = stump.predict(np.array([1, 0.0, 1]), X_stump)

    assert np.array_equal(stump_predictions, np.array([-1.0, 1.0]))
    assert stump.compute_vc_dimension_theoretical() == 3


def test_sauer_shelah_bound_matches_closed_form_cases():
    assert sauer_shelah_bound(4, 2) == 11
    assert sauer_shelah_bound(5, 0) == 1
    assert sauer_shelah_bound(3, 5) == 8
