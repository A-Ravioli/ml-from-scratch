"""
Online Learning and Regret Minimization

Implementation of fundamental online learning algorithms including:
- Online Gradient Descent (OGD)
- Follow the Regularized Leader (FTRL)
- Multiplicative Weights / Hedge
- Multi-Armed Bandits with UCB
- Online-to-batch conversion

Key theoretical concepts:
- Regret bounds and no-regret learning
- Adversarial vs stochastic settings
- Projection algorithms
- Concentration inequalities in online learning

Author: ML-from-Scratch Course
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Union
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings


class OnlineLearner(ABC):
    """
    Abstract base class for online learning algorithms.
    
    Online learning protocol:
    1. Learner chooses action xₜ ∈ X
    2. Environment reveals loss function ℓₜ: X → ℝ
    3. Learner suffers loss ℓₜ(xₜ)
    4. Learner updates internal state
    """
    
    def __init__(self, name: str):
        self.name = name
        self.actions = []  # History of chosen actions
        self.losses = []   # History of suffered losses
        self.regrets = []  # History of regret values
    
    @abstractmethod
    def predict(self, t: int) -> np.ndarray:
        """
        Choose action at time t.
        
        Args:
            t: Current time step
            
        Returns:
            Action xₜ
        """
        pass
    
    @abstractmethod
    def update(self, action: np.ndarray, loss_function: Callable[[np.ndarray], float], 
               gradient: Optional[np.ndarray] = None):
        """
        Update learner's state after observing loss.
        
        Args:
            action: Action that was taken
            loss_function: The loss function ℓₜ
            gradient: Gradient ∇ℓₜ(xₜ) if available
        """
        pass
    
    def compute_regret(self, comparator: np.ndarray, loss_functions: List[Callable]) -> float:
        """
        TODO: Compute regret against fixed comparator.
        
        Regret: R_T(u) = Σₜ ℓₜ(xₜ) - Σₜ ℓₜ(u)
        
        Args:
            comparator: Fixed action u ∈ X
            loss_functions: Sequence of loss functions ℓ₁, ..., ℓₜ
            
        Returns:
            Regret value
        """
        pass
    
    def compute_worst_case_regret(self, loss_functions: List[Callable], 
                                 feasible_set: List[np.ndarray]) -> float:
        """
        TODO: Compute worst-case regret.
        
        R_T = max_u∈X R_T(u) = Σₜ ℓₜ(xₜ) - min_u∈X Σₜ ℓₜ(u)
        
        Args:
            loss_functions: Sequence of loss functions
            feasible_set: Set of feasible actions X
            
        Returns:
            Worst-case regret
        """
        pass
    
    def reset(self):
        """Reset learner state."""
        self.actions = []
        self.losses = []
        self.regrets = []


class OnlineGradientDescent(OnlineLearner):
    """
    Online Gradient Descent algorithm.
    
    Update rule: xₜ₊₁ = Π_X(xₜ - ηₜ∇ℓₜ(xₜ))
    
    Regret bound: R_T ≤ DG√T with η = D/(G√T)
    where D = diam(X), G = max ||∇ℓₜ(x)||
    """
    
    def __init__(self, feasible_set_constraint: Callable[[np.ndarray], np.ndarray],
                 learning_rate: Union[float, Callable[[int], float]] = 0.1,
                 initial_point: Optional[np.ndarray] = None):
        """
        Initialize Online Gradient Descent.
        
        Args:
            feasible_set_constraint: Projection function Π_X(x)
            learning_rate: Step size ηₜ (constant or function of t)
            initial_point: Starting point x₁
        """
        super().__init__("Online Gradient Descent")
        self.project = feasible_set_constraint
        self.learning_rate = learning_rate
        self.current_point = initial_point
        self.t = 0
    
    def predict(self, t: int) -> np.ndarray:
        """
        TODO: Return current action.
        
        Args:
            t: Time step
            
        Returns:
            Current point xₜ
        """
        pass
    
    def update(self, action: np.ndarray, loss_function: Callable[[np.ndarray], float], 
               gradient: Optional[np.ndarray] = None):
        """
        TODO: Perform OGD update.
        
        Update: xₜ₊₁ = Π_X(xₜ - ηₜ∇ℓₜ(xₜ))
        
        Args:
            action: Action xₜ that was taken
            loss_function: Loss function ℓₜ
            gradient: Gradient ∇ℓₜ(xₜ), computed if not provided
        """
        pass
    
    def _compute_gradient(self, loss_function: Callable[[np.ndarray], float], 
                         point: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """
        TODO: Compute numerical gradient if analytical not provided.
        
        Args:
            loss_function: Function to differentiate
            point: Point at which to compute gradient
            h: Step size for finite differences
            
        Returns:
            Approximate gradient
        """
        pass
    
    def get_optimal_learning_rate(self, diameter: float, gradient_bound: float, 
                                 horizon: int) -> float:
        """
        TODO: Compute optimal learning rate for regret bound.
        
        Optimal η = D/(G√T) for regret bound DG√T.
        
        Args:
            diameter: Diameter of feasible set D
            gradient_bound: Bound on gradient norms G
            horizon: Time horizon T
            
        Returns:
            Optimal learning rate
        """
        pass


class FollowRegularizedLeader(OnlineLearner):
    """
    Follow the Regularized Leader (FTRL) algorithm.
    
    Update: xₜ₊₁ = argmin_{x∈X} [Σₛ₌₁ᵗ ℓₛ(x) + R(x)]
    
    Common regularizers:
    - L2: R(x) = (1/2η)||x||²
    - Entropy: R(x) = η Σᵢ xᵢ log xᵢ
    """
    
    def __init__(self, regularizer: Callable[[np.ndarray], float],
                 regularizer_grad: Callable[[np.ndarray], np.ndarray],
                 feasible_set_constraint: Callable[[np.ndarray], np.ndarray],
                 solver: str = 'gradient_descent'):
        """
        Initialize FTRL.
        
        Args:
            regularizer: Regularization function R(x)
            regularizer_grad: Gradient of regularizer ∇R(x)
            feasible_set_constraint: Projection onto feasible set
            solver: Optimization method for FTRL subproblem
        """
        super().__init__("Follow the Regularized Leader")
        self.regularizer = regularizer
        self.regularizer_grad = regularizer_grad
        self.project = feasible_set_constraint
        self.solver = solver
        
        # Accumulated gradients
        self.cumulative_gradient = None
        self.current_point = None
        self.t = 0
    
    def predict(self, t: int) -> np.ndarray:
        """
        TODO: Solve FTRL optimization problem.
        
        xₜ₊₁ = argmin_{x∈X} [⟨Σₛ₌₁ᵗ⁻¹ ∇ℓₛ(xₛ), x⟩ + R(x)]
        
        Args:
            t: Time step
            
        Returns:
            Optimal action
        """
        pass
    
    def update(self, action: np.ndarray, loss_function: Callable[[np.ndarray], float], 
               gradient: Optional[np.ndarray] = None):
        """
        TODO: Update cumulative gradient.
        
        Args:
            action: Action taken
            loss_function: Loss function
            gradient: Loss gradient at action
        """
        pass
    
    def _solve_ftrl_subproblem(self, cumulative_grad: np.ndarray, 
                              initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        TODO: Solve FTRL optimization subproblem.
        
        minimize ⟨g, x⟩ + R(x) subject to x ∈ X
        
        Args:
            cumulative_grad: Accumulated gradient g
            initial_guess: Starting point for optimization
            
        Returns:
            Optimal solution
        """
        pass


class MultiplicativeWeights(OnlineLearner):
    """
    Multiplicative Weights / Hedge algorithm.
    
    For experts setting with linear losses ℓₜ(p) = pᵀlₜ.
    
    Update: wₜ₊₁,ᵢ = wₜᵢ exp(-ηlₜᵢ)
    Action: pₜᵢ = wₜᵢ / Σⱼ wₜⱼ
    
    Regret bound: R_T ≤ √(T log n / 2) with η = √(8 log n / T)
    """
    
    def __init__(self, n_experts: int, learning_rate: Union[float, Callable[[int], float]]):
        """
        Initialize Multiplicative Weights.
        
        Args:
            n_experts: Number of experts/actions
            learning_rate: Step size η
        """
        super().__init__("Multiplicative Weights")
        self.n_experts = n_experts
        self.learning_rate = learning_rate
        
        # Initialize uniform weights
        self.weights = np.ones(n_experts)
        self.t = 0
    
    def predict(self, t: int) -> np.ndarray:
        """
        TODO: Compute probability distribution over experts.
        
        pₜᵢ = wₜᵢ / Σⱼ wₜⱼ
        
        Args:
            t: Time step
            
        Returns:
            Probability distribution over experts
        """
        pass
    
    def update(self, action: np.ndarray, loss_function: Callable[[np.ndarray], float], 
               gradient: Optional[np.ndarray] = None):
        """
        TODO: Update expert weights.
        
        For linear loss ℓₜ(p) = pᵀlₜ, we need loss vector lₜ.
        Update: wₜ₊₁,ᵢ = wₜᵢ exp(-η lₜᵢ)
        
        Args:
            action: Probability distribution used
            loss_function: Should be linear: ℓₜ(p) = pᵀlₜ
            gradient: Loss vector lₜ (negative of reward vector)
        """
        pass
    
    def get_optimal_learning_rate(self, horizon: int) -> float:
        """
        TODO: Compute optimal learning rate.
        
        η* = √(8 log n / T) for regret bound √(T log n / 2)
        
        Args:
            horizon: Time horizon T
            
        Returns:
            Optimal learning rate
        """
        pass


class UCBBandit:
    """
    Upper Confidence Bound algorithm for multi-armed bandits.
    
    UCB action selection: aₜ = argmax_i [μ̂ᵢ + √(2 log t / nᵢ)]
    
    Regret bound: O(√(K log T)) where K is number of arms.
    """
    
    def __init__(self, n_arms: int, confidence_parameter: float = 2.0):
        """
        Initialize UCB bandit.
        
        Args:
            n_arms: Number of bandit arms
            confidence_parameter: Confidence width parameter
        """
        self.n_arms = n_arms
        self.confidence_param = confidence_parameter
        
        # Statistics for each arm
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_rounds = 0
    
    def select_arm(self, t: int) -> int:
        """
        TODO: Select arm using UCB criterion.
        
        UCB(i) = μ̂ᵢ + √(c log t / nᵢ)
        where μ̂ᵢ is empirical mean, nᵢ is arm count.
        
        Args:
            t: Current round
            
        Returns:
            Selected arm index
        """
        pass
    
    def update(self, arm: int, reward: float):
        """
        TODO: Update arm statistics.
        
        Args:
            arm: Arm that was pulled
            reward: Observed reward
        """
        pass
    
    def get_regret(self, true_means: np.ndarray) -> float:
        """
        TODO: Compute cumulative regret.
        
        Regret = T·max_i μᵢ - Σₜ μₐₜ
        
        Args:
            true_means: True reward means for each arm
            
        Returns:
            Cumulative regret
        """
        pass
    
    def reset(self):
        """Reset bandit state."""
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)
        self.total_rounds = 0


class OnlineToBatchConverter:
    """
    Convert online learning algorithms to batch learning.
    
    If online algorithm achieves regret R_T, then:
    E[f(x̄) - min_u f(u)] ≤ R_T/T
    where x̄ = (1/T)Σₜ xₜ is average action.
    """
    
    def __init__(self, online_learner: OnlineLearner):
        """
        Initialize converter.
        
        Args:
            online_learner: Online learning algorithm
        """
        self.learner = online_learner
    
    def batch_learn(self, data: List[Tuple[np.ndarray, float]], 
                   loss_function: Callable[[np.ndarray, Tuple], float]) -> np.ndarray:
        """
        TODO: Apply online learner to batch data.
        
        Simulate online learning on batch data, return average action.
        
        Args:
            data: List of (feature, label) pairs
            loss_function: Loss function ℓ(x, (feature, label))
            
        Returns:
            Average action x̄ = (1/T)Σₜ xₜ
        """
        pass
    
    def generalization_bound(self, regret: float, n_samples: int) -> float:
        """
        TODO: Compute generalization bound.
        
        From online regret to expected risk:
        E[f(x̄) - min_u f(u)] ≤ R_T/T
        
        Args:
            regret: Online regret R_T
            n_samples: Number of samples T
            
        Returns:
            Generalization bound
        """
        pass


# ============================================================================
# PROJECTION FUNCTIONS
# ============================================================================

def project_simplex(x: np.ndarray) -> np.ndarray:
    """
    TODO: Project onto probability simplex.
    
    Simplex: {x : Σᵢ xᵢ = 1, xᵢ ≥ 0}
    
    Efficient algorithm using sorting.
    
    Args:
        x: Point to project
        
    Returns:
        Projected point
    """
    pass


def project_l2_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    TODO: Project onto L2 ball.
    
    Ball: {x : ||x||₂ ≤ radius}
    
    Args:
        x: Point to project
        radius: Ball radius
        
    Returns:
        Projected point
    """
    pass


def project_box(x: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """
    TODO: Project onto box constraints.
    
    Box: {x : lower ≤ xᵢ ≤ upper for all i}
    
    Args:
        x: Point to project
        lower: Lower bound
        upper: Upper bound
        
    Returns:
        Projected point
    """
    pass


# ============================================================================
# REGULARIZERS
# ============================================================================

def l2_regularizer(x: np.ndarray, eta: float = 1.0) -> float:
    """
    TODO: L2 regularizer.
    
    R(x) = (1/2η)||x||²
    
    Args:
        x: Input point
        eta: Regularization parameter
        
    Returns:
        Regularizer value
    """
    pass


def l2_regularizer_grad(x: np.ndarray, eta: float = 1.0) -> np.ndarray:
    """
    TODO: Gradient of L2 regularizer.
    
    ∇R(x) = (1/η)x
    
    Args:
        x: Input point
        eta: Regularization parameter
        
    Returns:
        Gradient
    """
    pass


def entropy_regularizer(x: np.ndarray, eta: float = 1.0) -> float:
    """
    TODO: Entropy regularizer for simplex.
    
    R(x) = η Σᵢ xᵢ log xᵢ
    
    Args:
        x: Probability vector
        eta: Regularization parameter
        
    Returns:
        Entropy value
    """
    pass


def entropy_regularizer_grad(x: np.ndarray, eta: float = 1.0) -> np.ndarray:
    """
    TODO: Gradient of entropy regularizer.
    
    ∇R(x) = η(log x + 1)
    
    Args:
        x: Probability vector
        eta: Regularization parameter
        
    Returns:
        Gradient
    """
    pass


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_online_gradient_descent():
    """
    Exercise 1: Implement and analyze Online Gradient Descent.
    
    Tasks:
    1. Implement OGD with different projection operators
    2. Test on online convex optimization problems
    3. Verify regret bounds empirically
    4. Compare different learning rate schedules
    """
    print("Exercise 1: Online Gradient Descent")
    print("=" * 50)
    
    # Define online learning problem
    np.random.seed(42)
    
    # Quadratic losses with random linear terms
    def generate_loss_function(t):
        """Generate loss ℓₜ(x) = (1/2)(x - aₜ)ᵀQ(x - aₜ) + bₜᵀx"""
        d = 2
        Q = np.eye(d)  # Identity for simplicity
        a_t = np.random.randn(d)
        b_t = 0.1 * np.random.randn(d)
        
        def loss(x):
            return 0.5 * np.dot(x - a_t, Q @ (x - a_t)) + np.dot(b_t, x)
        
        def grad(x):
            return Q @ (x - a_t) + b_t
        
        return loss, grad
    
    # Test on L2 ball constraint
    radius = 1.0
    projection = lambda x: project_l2_ball(x, radius)
    
    print(f"Problem: Online convex optimization on L2 ball (radius={radius})")
    
    # TODO: Test different learning rates
    T = 1000
    learning_rates = [0.01, 0.1, lambda t: 1.0/np.sqrt(t+1)]
    
    for lr in learning_rates:
        print(f"\nLearning rate: {lr if isinstance(lr, (int, float)) else 'O(1/√t)'}")
        
        # TODO: Initialize OGD
        initial_point = np.zeros(2)
        ogd = OnlineGradientDescent(projection, lr, initial_point)
        
        total_loss = 0
        loss_functions = []
        
        for t in range(1, T+1):
            # TODO: Generate loss function
            loss_fn, grad_fn = generate_loss_function(t)
            loss_functions.append(loss_fn)
            
            # TODO: Get action and update
            # action = ogd.predict(t)
            # loss_val = loss_fn(action)
            # total_loss += loss_val
            # gradient = grad_fn(action)
            # ogd.update(action, loss_fn, gradient)
        
        print(f"  Total loss: TODO")
        
        # TODO: Compute regret against best fixed point
        # Find best fixed point in hindsight
        # best_point = minimize best retrospective action
        # regret = ogd.compute_regret(best_point, loss_functions)
        print(f"  Regret: TODO")
        print(f"  Regret/T: TODO")
    
    print("\nTODO: Implement OGD and regret computation")


def exercise_2_ftrl_implementation():
    """
    Exercise 2: Follow the Regularized Leader.
    
    Tasks:
    1. Implement FTRL with L2 and entropy regularizers
    2. Compare with OGD on same problems
    3. Test on expert advice setting
    4. Analyze computational complexity
    """
    print("\nExercise 2: Follow the Regularized Leader")
    print("=" * 50)
    
    # Expert advice problem
    n_experts = 5
    T = 500
    
    print(f"Expert advice with {n_experts} experts, {T} rounds")
    
    # Generate expert advice sequence
    np.random.seed(42)
    expert_losses = np.random.uniform(0, 1, (T, n_experts))
    
    # TODO: FTRL with entropy regularization (for simplex)
    eta = 0.1
    regularizer = lambda x: entropy_regularizer(x, eta)
    reg_grad = lambda x: entropy_regularizer_grad(x, eta)
    projection = project_simplex
    
    ftrl = FollowRegularizedLeader(regularizer, reg_grad, projection)
    
    # TODO: Run FTRL algorithm
    total_loss = 0
    regrets = []
    
    for t in range(T):
        # TODO: Get probability distribution over experts
        # prob_dist = ftrl.predict(t+1)
        
        # TODO: Compute expected loss
        # expected_loss = np.dot(prob_dist, expert_losses[t])
        # total_loss += expected_loss
        
        # TODO: Update with linear loss
        # def linear_loss(p):
        #     return np.dot(p, expert_losses[t])
        # ftrl.update(prob_dist, linear_loss, expert_losses[t])
        
        # TODO: Compute regret against best expert so far
        # best_expert_loss = np.min(np.cumsum(expert_losses[:t+1], axis=0)[-1])
        # current_regret = total_loss - best_expert_loss
        # regrets.append(current_regret)
    
    print("FTRL Results:")
    print(f"  Total loss: TODO")
    print(f"  Final regret: TODO")
    print(f"  Regret/T: TODO")
    
    # TODO: Compare with Multiplicative Weights
    print("\nComparison with Multiplicative Weights:")
    mw = MultiplicativeWeights(n_experts, eta)
    
    # TODO: Run MW algorithm on same sequence
    print("  MW total loss: TODO")
    print("  MW regret: TODO")
    
    print("\nTODO: Implement FTRL and MW algorithms")


def exercise_3_multiplicative_weights():
    """
    Exercise 3: Multiplicative Weights algorithm.
    
    Tasks:
    1. Implement MW/Hedge algorithm
    2. Verify regret bound √(T log n)
    3. Test optimal learning rate selection
    4. Apply to game-theoretic scenarios
    """
    print("\nExercise 3: Multiplicative Weights")
    print("=" * 50)
    
    # Prediction with expert advice
    n_experts = 10
    T = 1000
    
    print(f"Prediction setting: {n_experts} experts, {T} rounds")
    
    # Generate expert predictions and true outcomes
    np.random.seed(42)
    
    # Each expert has some accuracy
    expert_accuracies = np.random.uniform(0.5, 0.9, n_experts)
    true_outcomes = np.random.choice([0, 1], T)
    
    # Generate expert predictions based on accuracies
    expert_predictions = np.zeros((T, n_experts))
    for t in range(T):
        for i in range(n_experts):
            if np.random.random() < expert_accuracies[i]:
                expert_predictions[t, i] = true_outcomes[t]
            else:
                expert_predictions[t, i] = 1 - true_outcomes[t]
    
    # Convert to losses (0 = correct, 1 = incorrect)
    expert_losses = np.abs(expert_predictions - true_outcomes.reshape(-1, 1))
    
    # TODO: Test different learning rates
    learning_rates = [0.01, 0.1, 0.5]
    
    for eta in learning_rates:
        print(f"\nLearning rate η = {eta}")
        
        mw = MultiplicativeWeights(n_experts, eta)
        
        total_loss = 0
        mistakes = 0
        
        for t in range(T):
            # TODO: Get probability distribution
            # prob_dist = mw.predict(t+1)
            
            # TODO: Make prediction (majority vote weighted by probabilities)
            # prediction = 1 if np.dot(prob_dist, expert_predictions[t]) > 0.5 else 0
            # mistake = (prediction != true_outcomes[t])
            # mistakes += mistake
            
            # TODO: Compute loss and update
            # loss_val = np.dot(prob_dist, expert_losses[t])
            # total_loss += loss_val
            
            # def linear_loss(p):
            #     return np.dot(p, expert_losses[t])
            # mw.update(prob_dist, linear_loss, expert_losses[t])
        
        print(f"  Total loss: TODO")
        print(f"  Mistakes: TODO / {T}")
        print(f"  Error rate: TODO")
        
        # TODO: Compare with best expert
        best_expert_losses = np.sum(expert_losses, axis=0)
        best_expert_loss = np.min(best_expert_losses)
        regret = total_loss - best_expert_loss
        
        print(f"  Best expert loss: {best_expert_loss:.3f}")
        print(f"  Regret: TODO")
    
    # TODO: Optimal learning rate
    optimal_eta = np.sqrt(8 * np.log(n_experts) / T)
    theoretical_regret = np.sqrt(T * np.log(n_experts) / 2)
    
    print(f"\nTheoretical analysis:")
    print(f"  Optimal η = √(8 log n / T) = {optimal_eta:.4f}")
    print(f"  Theoretical regret bound: √(T log n / 2) = {theoretical_regret:.3f}")
    
    print("\nTODO: Implement MW algorithm and analysis")


def exercise_4_bandit_algorithms():
    """
    Exercise 4: Multi-armed bandits with UCB.
    
    Tasks:
    1. Implement UCB algorithm
    2. Test on different bandit instances
    3. Compare with epsilon-greedy and random
    4. Verify regret bounds
    """
    print("\nExercise 4: Multi-Armed Bandits")
    print("=" * 50)
    
    # Define bandit instance
    n_arms = 5
    true_means = np.array([0.1, 0.3, 0.5, 0.6, 0.2])  # Arm 3 is best
    T = 2000
    
    print(f"Bandit instance: {n_arms} arms, {T} rounds")
    print(f"True means: {true_means}")
    print(f"Optimal arm: {np.argmax(true_means)} (mean = {np.max(true_means):.1f})")
    
    # TODO: Test different algorithms
    algorithms = {
        'UCB': UCBBandit(n_arms, confidence_parameter=2.0),
        # Add epsilon-greedy for comparison
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\nRunning {name}:")
        
        algorithm.reset()
        total_reward = 0
        arm_pulls = np.zeros(n_arms)
        
        for t in range(1, T+1):
            # TODO: Select arm
            if name == 'UCB':
                # arm = algorithm.select_arm(t)
                pass
            
            # TODO: Generate reward
            # reward = np.random.normal(true_means[arm], 0.1)  # Gaussian rewards
            # total_reward += reward
            # arm_pulls[arm] += 1
            
            # TODO: Update algorithm
            # algorithm.update(arm, reward)
            
            # Print progress
            if t % 500 == 0:
                print(f"  Round {t}: Total reward = TODO, Regret = TODO")
        
        # TODO: Compute final statistics
        # regret = algorithm.get_regret(true_means)
        results[name] = {
            'total_reward': total_reward,
            'regret': 'TODO',  # regret
            'arm_pulls': arm_pulls
        }
        
        print(f"  Final total reward: {total_reward:.2f}")
        print(f"  Final regret: TODO")
        print(f"  Arm pulls: {arm_pulls}")
    
    # TODO: Theoretical comparison
    optimal_regret = np.sqrt(2 * n_arms * np.log(T))
    print(f"\nTheoretical UCB regret bound: O(√(K log T)) ≈ {optimal_regret:.2f}")
    
    print("\nTODO: Implement bandit algorithms and analysis")


def exercise_5_online_to_batch():
    """
    Exercise 5: Online-to-batch conversion.
    
    Tasks:
    1. Implement online-to-batch converter
    2. Apply to classification and regression
    3. Compare with batch algorithms
    4. Verify generalization bounds
    """
    print("\nExercise 5: Online-to-Batch Conversion")
    print("=" * 50)
    
    # Generate synthetic dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                              n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Convert labels to {-1, +1}
    y = 2 * y - 1
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split into train/test
    split_idx = 800
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Define loss function for binary classification
    def logistic_loss(w, data_point):
        x, label = data_point
        margin = label * np.dot(w, x)
        return np.log(1 + np.exp(-margin))
    
    def logistic_loss_grad(w, data_point):
        x, label = data_point
        margin = label * np.dot(w, x)
        prob = 1 / (1 + np.exp(margin))
        return -prob * label * x
    
    # TODO: Apply online learning algorithms
    projection = lambda w: project_l2_ball(w, radius=5.0)
    
    algorithms = {
        'OGD': OnlineGradientDescent(projection, learning_rate=0.1, 
                                   initial_point=np.zeros(X.shape[1])),
        # Add FTRL if desired
    }
    
    for name, online_alg in algorithms.items():
        print(f"\n{name} Online-to-Batch:")
        
        # TODO: Run online learning
        converter = OnlineToBatchConverter(online_alg)
        
        # Prepare data for online learning
        train_data = list(zip(X_train, y_train))
        
        # TODO: Apply batch learning
        # avg_weights = converter.batch_learn(train_data, logistic_loss)
        
        # TODO: Evaluate on test set
        # def make_predictions(weights, X_test):
        #     scores = X_test @ weights
        #     return np.sign(scores)
        
        # predictions = make_predictions(avg_weights, X_test)
        # accuracy = np.mean(predictions == y_test)
        
        print(f"  Test accuracy: TODO")
        
        # TODO: Compute generalization bound
        # online_regret = # compute from online learning
        # gen_bound = converter.generalization_bound(online_regret, len(train_data))
        print(f"  Generalization bound: TODO")
    
    # TODO: Compare with batch methods
    from sklearn.linear_model import LogisticRegression
    
    batch_lr = LogisticRegression()
    batch_lr.fit(X_train, y_train)
    batch_accuracy = batch_lr.score(X_test, y_test)
    
    print(f"\nBatch Logistic Regression accuracy: {batch_accuracy:.3f}")
    
    print("\nTODO: Implement online-to-batch conversion and comparison")


def exercise_6_adversarial_examples():
    """
    Exercise 6: Adversarial robustness in online learning.
    
    Tasks:
    1. Generate adversarial loss sequences
    2. Test robustness of different algorithms
    3. Implement adaptive adversaries
    4. Compare worst-case vs average-case performance
    """
    print("\nExercise 6: Adversarial Robustness")
    print("=" * 50)
    
    # TODO: Design adversarial loss sequences
    d = 2  # Dimension
    T = 500
    
    print(f"Adversarial online learning in {d}D, {T} rounds")
    
    # Different adversarial strategies
    def constant_adversary(t, past_actions):
        """Always returns the same loss function."""
        def loss(x):
            return np.dot(x, [1, -1])  # Prefer x₁ small, x₂ large
        def grad(x):
            return np.array([1, -1])
        return loss, grad
    
    def adaptive_adversary(t, past_actions):
        """Adapts based on learner's past actions."""
        if len(past_actions) == 0:
            direction = np.array([1, 0])
        else:
            # Choose direction opposite to recent average
            recent_avg = np.mean(past_actions[-10:], axis=0)
            direction = -recent_avg / (np.linalg.norm(recent_avg) + 1e-8)
        
        def loss(x):
            return np.dot(x, direction)
        def grad(x):
            return direction
        return loss, grad
    
    def random_adversary(t, past_actions):
        """Random loss functions."""
        direction = np.random.randn(d)
        direction /= np.linalg.norm(direction)
        
        def loss(x):
            return np.dot(x, direction)
        def grad(x):
            return direction
        return loss, grad
    
    adversaries = {
        'Constant': constant_adversary,
        'Adaptive': adaptive_adversary,
        'Random': random_adversary
    }
    
    # TODO: Test algorithms against different adversaries
    projection = lambda x: project_l2_ball(x, radius=1.0)
    
    algorithms = {
        'OGD-0.1': OnlineGradientDescent(projection, 0.1, np.zeros(d)),
        'OGD-0.01': OnlineGradientDescent(projection, 0.01, np.zeros(d)),
        'OGD-Adaptive': OnlineGradientDescent(projection, lambda t: 1.0/np.sqrt(t), np.zeros(d))
    }
    
    for adv_name, adversary in adversaries.items():
        print(f"\nAdversary: {adv_name}")
        
        for alg_name, algorithm in algorithms.items():
            algorithm.reset()
            
            total_loss = 0
            actions = []
            
            for t in range(1, T+1):
                # TODO: Get action
                # action = algorithm.predict(t)
                # actions.append(action.copy())
                
                # TODO: Adversary chooses loss
                # loss_fn, grad_fn = adversary(t, actions)
                # loss_val = loss_fn(action)
                # total_loss += loss_val
                
                # TODO: Update algorithm
                # gradient = grad_fn(action)
                # algorithm.update(action, loss_fn, gradient)
            
            print(f"  {alg_name}: Total loss = TODO")
    
    print("\nTODO: Implement adversarial testing framework")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Online Learning and Regret Minimization - Comprehensive Exercises")
    print("=" * 70)
    
    # Run all exercises
    exercise_1_online_gradient_descent()
    exercise_2_ftrl_implementation()
    exercise_3_multiplicative_weights()
    exercise_4_bandit_algorithms()
    exercise_5_online_to_batch()
    exercise_6_adversarial_examples()
    
    print("\n" + "=" * 70)
    print("COMPLETION CHECKLIST:")
    print("1. ✓ Implement Online Gradient Descent with projections")
    print("2. ✓ Implement Follow the Regularized Leader")
    print("3. ✓ Implement Multiplicative Weights / Hedge")
    print("4. ✓ Implement UCB for multi-armed bandits")
    print("5. ✓ Implement online-to-batch conversion")
    print("6. ✓ Test adversarial robustness")
    print("7. ✓ Verify regret bounds empirically")
    print("8. ✓ Add comprehensive visualizations")
    
    print("\nKey theoretical insights to verify:")
    print("- OGD regret bound: O(√T) with optimal learning rate")
    print("- MW regret bound: O(√(T log n)) for expert advice")
    print("- UCB regret bound: O(√(K log T)) for bandits")
    print("- Online-to-batch conversion: regret/T → generalization")
    print("- Projection algorithms preserve regret bounds")
    print("- Adaptive vs oblivious adversaries") 