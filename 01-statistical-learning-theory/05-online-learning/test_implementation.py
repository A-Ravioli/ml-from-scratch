import numpy as np

from exercise import (
    FollowRegularizedLeader,
    MultiplicativeWeights,
    OnlineGradientDescent,
    OnlineToBatchConverter,
    UCBBandit,
    adaptive_learning_rates,
    entropy_regularizer,
    entropy_regularizer_grad,
    l2_regularizer,
    l2_regularizer_grad,
    online_svm,
    project_box,
    project_l2_ball,
    project_simplex,
)


def test_projection_and_regularizer_helpers():
    simplex_projection = project_simplex(np.array([0.2, -0.5, 1.7]))
    assert np.all(simplex_projection >= 0.0)
    assert np.isclose(simplex_projection.sum(), 1.0)

    l2_projection = project_l2_ball(np.array([3.0, 4.0]), radius=2.0)
    assert np.linalg.norm(l2_projection) <= 2.0 + 1e-8

    assert np.allclose(
        project_box(np.array([-2.0, 0.0, 3.0]), -1.0, 1.0),
        np.array([-1.0, 0.0, 1.0]),
    )

    x = np.array([0.5, -0.5])
    assert np.isclose(l2_regularizer(x, eta=2.0), 0.125)
    assert np.allclose(l2_regularizer_grad(x, eta=2.0), np.array([0.25, -0.25]))

    p = np.array([0.25, 0.75])
    assert np.isfinite(entropy_regularizer(p, eta=1.0))
    assert np.all(np.isfinite(entropy_regularizer_grad(p, eta=1.0)))


def test_online_gradient_descent_and_ftrl_updates():
    target = np.array([1.0, -1.0])
    ogd = OnlineGradientDescent(dimension=2, constraint_set_radius=1.0, learning_rate=0.5)

    action = ogd.predict()
    updated = ogd.update(
        action,
        loss_function=lambda x: 0.5 * np.sum((x - target) ** 2),
        gradient=action - target,
    )

    assert updated.shape == (2,)
    assert np.linalg.norm(updated) <= 1.0 + 1e-8
    assert not np.allclose(updated, action)

    loss_sequence = [
        (
            lambda x, g=np.array([1.0, 0.0]): float(np.dot(g, x)),
            lambda x, g=np.array([1.0, 0.0]): g,
        ),
        (
            lambda x, g=np.array([0.0, 1.0]): float(np.dot(g, x)),
            lambda x, g=np.array([0.0, 1.0]): g,
        ),
    ]
    assert np.isfinite(ogd.run_online_learning(loss_sequence, T=2))

    ftrl = FollowRegularizedLeader(dimension=2, regularizer='l2', regularization_strength=1.0)
    ftrl_start = ftrl.predict()
    ftrl_next = ftrl.update(
        ftrl_start,
        loss_function=lambda x: float(np.dot(np.array([1.0, 0.0]), x)),
        gradient=np.array([1.0, 0.0]),
    )

    assert ftrl_next.shape == (2,)
    assert ftrl_next[0] <= 0.0


def test_multiplicative_weights_bandits_and_online_to_batch():
    weights = MultiplicativeWeights(n_experts=3, learning_rate=0.3)
    regret = weights.run_expert_learning(
        [
            np.array([0.0, 1.0, 1.0]),
            np.array([0.1, 1.0, 0.9]),
            np.array([0.0, 0.8, 0.9]),
        ],
        T=3,
    )

    probabilities = weights.get_expert_weights()
    assert np.isfinite(regret)
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1]

    ucb = UCBBandit(n_arms=2)
    environment = [lambda arm: 1.0 if arm == 0 else 0.0 for _ in range(8)]
    ucb.run_bandit_learning(environment, T=8)
    counts = ucb.get_arm_counts()
    assert counts[0] >= counts[1]

    converter = OnlineToBatchConverter(
        OnlineGradientDescent(dimension=2, constraint_set_radius=1.0, learning_rate=0.1)
    )
    conversion = converter.convert_online_to_batch(
        target_function=lambda x: float(np.sum(x)),
        data_distribution=lambda n: np.full((n, 2), 0.5),
        T=4,
        n_samples=4,
    )
    assert "batch_excess_risk" in conversion
    assert conversion["averaged_solution"].shape == (2,)

    rates = adaptive_learning_rates(np.array([1.0, 2.0, 2.0]), base_rate=1.0)
    assert len(rates) == 3
    assert rates[-1] <= rates[0]

    classifier = online_svm(dimension=2, learning_rate=0.1)
    mistakes = classifier.run_online_classification(
        [
            (np.array([1.0, 1.0]), 1),
            (np.array([2.0, 2.0]), 1),
            (np.array([-1.0, -1.0]), -1),
            (np.array([-2.0, -2.0]), -1),
        ],
        T=4,
    )
    assert mistakes >= 0
