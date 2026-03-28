import numpy as np
from sklearn.linear_model import LogisticRegression

from exercise import LabelSpreading, SelfTraining, evaluate_semi_supervised, generate_semi_supervised_data


def test_generate_data_shapes():
    data = generate_semi_supervised_data(n_labeled=10, n_unlabeled=20, n_test=10, n_features=6, random_state=0)
    X_labeled, y_labeled, X_unlabeled, X_test, y_test = data[:5]
    assert X_labeled.shape[0] == 10
    assert X_unlabeled.shape[0] == 20
    assert X_test.shape[0] == 10
    assert y_labeled.shape[0] == 10
    assert y_test.shape[0] == 10


def test_self_training_and_label_spreading():
    X_labeled, y_labeled, X_unlabeled, X_test, y_test = generate_semi_supervised_data(
        n_labeled=10, n_unlabeled=20, n_test=10, n_features=6, random_state=1
    )[:5]

    model = SelfTraining(LogisticRegression(max_iter=200), threshold=0.6, max_iterations=2)
    model.fit(X_labeled, y_labeled, X_unlabeled)
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape

    graph_model = LabelSpreading(gamma=0.5, alpha=0.8, max_iter=30)
    X_all = np.vstack([X_labeled, X_unlabeled])
    y_all = np.concatenate([y_labeled, -np.ones(len(X_unlabeled), dtype=int)])
    graph_model.fit(X_all, y_all)
    assert graph_model.predict(X_test).shape == y_test.shape


def test_evaluation_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    accuracy, precision, recall, f1 = evaluate_semi_supervised(None, None, None, y_true=y_true, y_pred=y_pred)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
