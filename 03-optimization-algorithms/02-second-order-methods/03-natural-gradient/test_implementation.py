"""
Deterministic tests for natural gradient implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import GaussianModel, CategoricalModel, NaturalGradientOptimizer


class TestGaussianNaturalGradient:
    def test_natural_gradient_improves_log_likelihood(self):
        rng = np.random.default_rng(0)
        dim = 2
        model = GaussianModel(dim=dim)

        mu_true = np.array([0.5, -0.25])
        L_true = np.array([[1.2, 0.0], [0.3, 0.8]])
        params_true = model._pack_params(mu_true, L_true)
        data = model.sample(params_true, n_samples=80)

        mu0 = np.zeros(dim)
        L0 = np.eye(dim)
        params0 = model._pack_params(mu0, L0)

        opt = NaturalGradientOptimizer(learning_rate=0.3, regularization=1e-4, fisher_estimation="empirical", max_iterations=40)
        ll0 = model.log_likelihood(params0, data)
        params_f, _ = opt.optimize(model, data, params0)
        llf = model.log_likelihood(params_f, data)
        assert llf > ll0


class TestCategoricalNaturalGradient:
    def test_categorical_improves_log_likelihood(self):
        rng = np.random.default_rng(0)
        model = CategoricalModel(n_categories=3)

        probs = np.array([0.2, 0.5, 0.3])
        logits = np.log(probs + 1e-12)
        params_true = logits[:2] - logits[2]  # last logit fixed at 0
        data = model.sample(params_true, n_samples=200)

        params0 = np.array([0.0, 0.0])
        opt = NaturalGradientOptimizer(learning_rate=0.5, regularization=1e-4, fisher_estimation="exact", max_iterations=60)

        ll0 = model.log_likelihood(params0, data)
        params_f, _ = opt.optimize(model, data, params0)
        llf = model.log_likelihood(params_f, data)
        assert llf > ll0

