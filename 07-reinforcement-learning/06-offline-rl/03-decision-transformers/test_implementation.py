"""
Decision Transformers - Tests
"""

from __future__ import annotations

import numpy as np

from exercise import TopicConfig, TopicModel, build_feature_map, set_seed, topic_similarity


def test_build_feature_map_shape_and_bias():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    features = build_feature_map(x, scale=0.5)
    assert features.shape == (2, 5)
    np.testing.assert_allclose(features[:, -1], np.ones(2))


def test_topic_similarity_is_normalized():
    x = np.array([1.0, 0.0, 1.0])
    y = np.array([1.0, 1.0, 0.0])
    score = topic_similarity(x, y)
    assert -1.0 <= score <= 1.0


def test_topic_model_fit_transform_and_score():
    set_seed(0)
    x = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    model = TopicModel(TopicConfig(input_dim=3, scale=0.25))
    model.fit(x)
    projections = model.transform(x)
    assert projections.shape == (4,)
    assert np.isfinite(projections).all()
    assert model.score(x) >= 0.0
