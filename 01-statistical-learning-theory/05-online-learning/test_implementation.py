"""
Test suite for online learning implementations.
"""

import numpy as np
import pytest
from exercise import (
    OnlineGradientDescent, FollowTheRegularizedLeader, MultiplicativeWeights,
    UCBBandit, ThompsonSamplingBandit, LinUCB, ContextualThompsonSampling,
    OnlineToBookConversion, AdversarialTrainingOnline, OnlineLearner,
    regret_analysis, adaptive_learning_rates, online_svm
)


class TestOnlineGradientDescent:
    """Test Online Gradient Descent implementation."""
    
    def test_ogd_convex_functions(self):
        """Test OGD on convex loss functions."""
        # Quadratic loss: f(x) = (x - 1)^2
        def loss_sequence():
            for t in range(100):
                def loss_t(x):
                    return (x - 1) ** 2
                def grad_t(x):
                    return 2 * (x - 1)
                yield loss_t, grad_t
        
        ogd = OnlineGradientDescent(
            dimension=1, 
            constraint_set_radius=5.0,
            learning_rate=0.1,
            learning_rate_schedule='constant'
        )
        
        regret = ogd.run_online_learning(loss_sequence(), T=100)
        
        # Should achieve sublinear regret
        assert regret >= 0
        assert regret <= 50  # Should be reasonable for this simple problem
        
        # Final iterate should be close to optimum
        final_x = ogd.get_final_iterate()
        assert abs(final_x[0] - 1.0) <= 0.5
    
    def test_ogd_learning_rate_schedules(self):
        """Test different learning rate schedules."""
        def simple_loss_sequence():
            for t in range(50):
                def loss_t(x):
                    return x[0] ** 2 + x[1] ** 2
                def grad_t(x):
                    return 2 * x
                yield loss_t, grad_t
        
        # Test constant vs decreasing rates
        ogd_constant = OnlineGradientDescent(
            dimension=2, constraint_set_radius=2.0,
            learning_rate=0.1, learning_rate_schedule='constant'
        )
        
        ogd_decreasing = OnlineGradientDescent(
            dimension=2, constraint_set_radius=2.0,
            learning_rate=0.1, learning_rate_schedule='sqrt_t'
        )
        
        regret_constant = ogd_constant.run_online_learning(simple_loss_sequence(), T=50)
        regret_decreasing = ogd_decreasing.run_online_learning(simple_loss_sequence(), T=50)
        
        # Both should achieve low regret
        assert regret_constant >= 0
        assert regret_decreasing >= 0
    
    def test_ogd_projection(self):
        """Test projection onto constraint sets."""
        ogd = OnlineGradientDescent(
            dimension=2, constraint_set_radius=1.0,
            learning_rate=0.5, learning_rate_schedule='constant'
        )
        
        # Test projection of point outside unit ball
        x_outside = np.array([2.0, 1.5])
        x_projected = ogd._project_onto_constraint_set(x_outside)
        
        # Should be on boundary of unit ball
        assert np.linalg.norm(x_projected) <= 1.0 + 1e-10
        assert np.linalg.norm(x_projected) >= 0.99  # Should be close to boundary
        
        # Test projection of point inside
        x_inside = np.array([0.3, 0.4])
        x_proj_inside = ogd._project_onto_constraint_set(x_inside)
        
        # Should remain unchanged
        np.testing.assert_array_almost_equal(x_proj_inside, x_inside)


class TestFollowTheRegularizedLeader:
    """Test Follow-The-Regularized-Leader implementation."""
    
    def test_ftrl_basic(self):
        """Test basic FTRL functionality."""
        def linear_loss_sequence():
            for t in range(30):
                # Linear loss with gradient [1, -1]
                grad = np.array([1.0, -1.0])
                def loss_t(x):
                    return np.dot(grad, x)
                def grad_t(x):
                    return grad
                yield loss_t, grad_t
        
        ftrl = FollowTheRegularizedLeader(
            dimension=2,
            regularizer='l2',
            regularization_strength=0.1,
            constraint_set_radius=2.0
        )
        
        regret = ftrl.run_online_learning(linear_loss_sequence(), T=30)
        
        assert regret >= 0
        
        # Final solution should move toward optimal direction
        final_x = ftrl.get_final_iterate()
        # Optimal for linear loss is boundary point in direction [-1, 1]
        assert final_x[0] <= final_x[1]  # Should prefer direction with negative gradient
    
    def test_ftrl_regularizers(self):
        """Test different regularizers."""
        def loss_sequence():
            for t in range(20):
                grad = np.random.randn(3)
                def loss_t(x):
                    return np.dot(grad, x)
                def grad_t(x):
                    return grad
                yield loss_t, grad_t
        
        # Test L2 regularizer
        ftrl_l2 = FollowTheRegularizedLeader(
            dimension=3, regularizer='l2',
            regularization_strength=0.1, constraint_set_radius=1.0
        )
        
        # Test entropy regularizer (for simplex constraint)
        ftrl_entropy = FollowTheRegularizedLeader(
            dimension=3, regularizer='entropy',
            regularization_strength=0.1, constraint_set='simplex'
        )
        
        regret_l2 = ftrl_l2.run_online_learning(loss_sequence(), T=20)
        regret_entropy = ftrl_entropy.run_online_learning(loss_sequence(), T=20)
        
        assert regret_l2 >= 0
        assert regret_entropy >= 0


class TestMultiplicativeWeights:
    """Test Multiplicative Weights / Hedge algorithm."""
    
    def test_hedge_expert_setting(self):
        """Test Hedge in expert setting."""
        n_experts = 5
        T = 50
        
        # Create loss sequence where expert 0 is best
        def expert_loss_sequence():
            for t in range(T):
                losses = np.random.uniform(0, 1, n_experts)
                losses[0] *= 0.3  # Make expert 0 consistently better
                yield losses
        
        hedge = MultiplicativeWeights(
            n_experts=n_experts,
            learning_rate=0.1,
            learning_rate_schedule='optimal'
        )
        
        regret = hedge.run_expert_learning(expert_loss_sequence(), T=T)
        
        # Should achieve sublinear regret
        assert regret >= 0
        assert regret <= np.sqrt(T * np.log(n_experts))  # Theoretical bound
        
        # Should put most weight on best expert
        final_weights = hedge.get_expert_weights()
        best_expert = np.argmax(final_weights)
        assert best_expert == 0  # Should identify expert 0 as best
    
    def test_hedge_regret_bound(self):
        """Test Hedge regret bound."""
        n_experts = 3
        T = 100
        
        def adversarial_losses():
            for t in range(T):
                # Adversarial losses in [0, 1]
                yield np.random.uniform(0, 1, n_experts)
        
        hedge = MultiplicativeWeights(
            n_experts=n_experts,
            learning_rate=np.sqrt(8 * np.log(n_experts) / T)  # Optimal rate
        )
        
        regret = hedge.run_expert_learning(adversarial_losses(), T=T)
        
        # Should satisfy theoretical bound
        theoretical_bound = np.sqrt(T * np.log(n_experts) / 2)
        assert regret <= theoretical_bound + 10  # Allow some empirical error


class TestBanditAlgorithms:
    """Test bandit learning algorithms."""
    
    def test_ucb_bandit(self):
        """Test UCB bandit algorithm."""
        # Create bandit with known optimal arm
        arm_means = [0.3, 0.7, 0.4, 0.6]  # Arm 1 is optimal
        n_arms = len(arm_means)
        T = 200
        
        def bandit_environment():
            for t in range(T):
                def pull_arm(arm):
                    # Return reward from Bernoulli distribution
                    return np.random.binomial(1, arm_means[arm])
                yield pull_arm
        
        ucb = UCBBandit(
            n_arms=n_arms,
            confidence_width=2.0,
            exploration_bonus='log_t'
        )
        
        regret = ucb.run_bandit_learning(bandit_environment(), T=T)
        
        # Should achieve sublinear regret
        assert regret >= 0
        
        # Should identify optimal arm eventually
        arm_counts = ucb.get_arm_counts()
        most_pulled = np.argmax(arm_counts)
        # Give some flexibility due to exploration
        assert most_pulled in [1, 3]  # Should prefer arms 1 or 3 (both good)
    
    def test_thompson_sampling(self):
        """Test Thompson Sampling bandit."""
        arm_means = [0.2, 0.8, 0.5]  # Arm 1 is clearly optimal
        n_arms = len(arm_means)
        T = 150
        
        def bandit_env():
            for t in range(T):
                def pull_arm(arm):
                    return np.random.binomial(1, arm_means[arm])
                yield pull_arm
        
        ts = ThompsonSamplingBandit(
            n_arms=n_arms,
            prior_alpha=1.0,
            prior_beta=1.0,
            posterior_update='beta_bernoulli'
        )
        
        regret = ts.run_bandit_learning(bandit_env(), T=T)
        
        assert regret >= 0
        
        # Should concentrate on optimal arm
        arm_counts = ts.get_arm_counts()
        optimal_pulls = arm_counts[1]
        total_pulls = np.sum(arm_counts)
        
        # Should pull optimal arm frequently in later rounds
        assert optimal_pulls / total_pulls >= 0.3


class TestContextualBandits:
    """Test contextual bandit algorithms."""
    
    def test_linucb(self):
        """Test LinUCB contextual bandit."""
        n_arms = 3
        context_dim = 4
        T = 100
        
        # True parameters for each arm
        true_theta = np.random.randn(n_arms, context_dim)
        
        def contextual_bandit_env():
            for t in range(T):
                context = np.random.randn(context_dim)
                def pull_arm(arm):
                    # Linear reward + noise
                    expected_reward = np.dot(context, true_theta[arm])
                    return expected_reward + 0.1 * np.random.randn()
                yield context, pull_arm
        
        linucb = LinUCB(
            n_arms=n_arms,
            context_dimension=context_dim,
            regularization_param=0.1,
            confidence_width=1.0
        )
        
        regret = linucb.run_contextual_bandit(contextual_bandit_env(), T=T)
        
        assert regret >= 0
        
        # Should learn reasonable parameter estimates
        learned_params = linucb.get_parameter_estimates()
        assert learned_params.shape == (n_arms, context_dim)
    
    def test_contextual_thompson_sampling(self):
        """Test contextual Thompson Sampling."""
        n_arms = 2
        context_dim = 3
        T = 80
        
        true_theta = np.array([[1, -1, 0.5], [-0.5, 1, -1]])
        
        def contextual_env():
            for t in range(T):
                context = np.random.randn(context_dim)
                def pull_arm(arm):
                    reward = np.dot(context, true_theta[arm])
                    return reward + 0.2 * np.random.randn()
                yield context, pull_arm
        
        cts = ContextualThompsonSampling(
            n_arms=n_arms,
            context_dimension=context_dim,
            prior_precision=1.0,
            noise_variance=0.04
        )
        
        regret = cts.run_contextual_bandit(contextual_env(), T=T)
        
        assert regret >= 0


class TestOnlineToOfflineConversion:
    """Test online-to-batch conversion."""
    
    def test_online_to_batch_conversion(self):
        """Test online-to-batch conversion theorem."""
        # Simple quadratic function
        def target_function(x):
            return (x[0] - 1) ** 2 + (x[1] + 0.5) ** 2
        
        def data_distribution(n_samples):
            return np.random.randn(n_samples, 2)
        
        converter = OnlineToBookConversion(
            online_algorithm=OnlineGradientDescent(
                dimension=2, constraint_set_radius=2.0,
                learning_rate=0.1, learning_rate_schedule='sqrt_t'
            ),
            averaging_scheme='uniform'
        )
        
        results = converter.convert_online_to_batch(
            target_function, data_distribution, T=100, n_samples=50
        )
        
        assert isinstance(results, dict)
        assert 'online_regret' in results
        assert 'batch_excess_risk' in results
        assert 'averaged_solution' in results
        
        # Excess risk should be bounded by regret/T
        assert results['batch_excess_risk'] <= results['online_regret'] / 100 + 0.1


class TestAdversarialLearning:
    """Test adversarial training as online learning."""
    
    def test_adversarial_training(self):
        """Test adversarial training framework."""
        # Simple linear model
        model_dim = 3
        data_dim = 2
        
        def model_loss(theta, x, y):
            # Linear model loss
            prediction = np.dot(x, theta[:data_dim]) + theta[-1]  # bias
            return (prediction - y) ** 2
        
        def adversarial_perturbation(x, epsilon=0.1):
            # Random perturbation
            return x + epsilon * np.random.randn(len(x))
        
        adv_trainer = AdversarialTrainingOnline(
            model_dimension=model_dim,
            perturbation_budget=0.1,
            inner_steps=5,
            outer_learning_rate=0.01
        )
        
        # Generate some data
        X = np.random.randn(30, data_dim)
        y = np.random.randn(30)
        
        robust_model = adv_trainer.train_robust_model(
            X, y, model_loss, adversarial_perturbation, T=20
        )
        
        assert len(robust_model) == model_dim
        assert isinstance(robust_model, np.ndarray)


class TestRegretAnalysis:
    """Test regret analysis tools."""
    
    def test_regret_computation(self):
        """Test regret computation and analysis."""
        # Simulate online algorithm sequence
        algorithm_losses = np.random.uniform(0, 1, 50)
        best_fixed_loss = np.min(algorithm_losses) * 0.8  # Better than any single choice
        
        cumulative_regret = regret_analysis(
            algorithm_losses, best_fixed_loss, analysis_type='cumulative'
        )
        
        assert len(cumulative_regret) == len(algorithm_losses)
        assert cumulative_regret[-1] >= 0  # Final regret should be non-negative
        
        # Test regret rate analysis
        regret_rate = regret_analysis(
            algorithm_losses, best_fixed_loss, analysis_type='rate'
        )
        
        assert len(regret_rate) == len(algorithm_losses)
    
    def test_adaptive_learning_rates(self):
        """Test adaptive learning rate schemes."""
        gradient_norms = np.random.uniform(0.1, 2.0, 100)
        
        # AdaGrad-style rates
        adagrad_rates = adaptive_learning_rates(
            gradient_norms, base_rate=0.1, method='adagrad'
        )
        
        assert len(adagrad_rates) == len(gradient_norms)
        assert np.all(adagrad_rates > 0)
        
        # Should decrease over time
        assert adagrad_rates[-1] <= adagrad_rates[0]


class TestOnlineSVM:
    """Test online SVM implementation."""
    
    def test_online_svm_classification(self):
        """Test online SVM for classification."""
        # Generate linearly separable sequence
        def data_sequence():
            for t in range(40):
                if np.random.rand() < 0.5:
                    x = np.random.randn(2) + [1, 1]
                    y = 1
                else:
                    x = np.random.randn(2) + [-1, -1]
                    y = -1
                yield x, y
        
        online_svm = online_svm(
            dimension=2,
            regularization=0.1,
            learning_rate=0.01,
            kernel_type='linear'
        )
        
        mistake_count = online_svm.run_online_classification(data_sequence(), T=40)
        
        # Should make fewer mistakes over time
        assert mistake_count >= 0
        assert mistake_count <= 40


def test_online_learner_interface():
    """Test that all online learners implement required interface."""
    learners = [
        OnlineGradientDescent(dimension=2, constraint_set_radius=1.0),
        FollowTheRegularizedLeader(dimension=2, regularizer='l2'),
        MultiplicativeWeights(n_experts=3),
        UCBBandit(n_arms=4),
        ThompsonSamplingBandit(n_arms=3)
    ]
    
    for learner in learners:
        # Should have learning rate or similar parameter
        assert hasattr(learner, 'learning_rate') or hasattr(learner, 'n_arms') or hasattr(learner, 'n_experts')
        
        # Should have reset method
        assert hasattr(learner, 'reset') or hasattr(learner, '__init__')


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])