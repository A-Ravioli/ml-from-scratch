import numpy as np
from sklearn.tree import DecisionTreeClassifier

from exercise import BaggingEnsemble, RandomForestAdvanced, VotingEnsemble, calculate_ensemble_diversity


def test_bagging_and_random_forest_shapes():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    bagging = BaggingEnsemble(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3, random_state=0)
    bagging.fit(X, y)
    assert bagging.predict(X).shape == y.shape

    forest = RandomForestAdvanced(n_estimators=3, max_depth=2, random_state=0)
    forest.fit(X, y)
    assert forest.predict(X).shape == y.shape


def test_voting_and_diversity():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    voting = VotingEnsemble(
        [
            ("a", DecisionTreeClassifier(max_depth=1, random_state=0)),
            ("b", DecisionTreeClassifier(max_depth=2, random_state=1)),
        ]
    )
    voting.fit(X, y)
    assert voting.predict(X).shape == y.shape

    metrics = calculate_ensemble_diversity(np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]]))
    assert "q_statistic" in metrics
