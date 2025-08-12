"""
PAC Learning Solutions - Reference Implementation

This mirrors the interfaces in `exercise.py` and provides working
implementations for study and verification.
"""

from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
from itertools import product
import math


class HypothesisClass:
    def __init__(self, name: str):
        self.name = name

    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_vc_dimension(self) -> int:
        raise NotImplementedError

    def sample_complexity_bound(self, epsilon: float, delta: float, realizable: bool = False) -> int:
        # Prefer VC bound if VC dimension available
        try:
            d = int(self.compute_vc_dimension())
        except Exception:
            d = None

        if d is not None and d > 0:
            # Standard VC generalization bound scaling (agnostic vs realizable)
            if realizable:
                # m ≳ (1/ε) (d log(1/ε) + log(1/δ))
                m = (d * max(math.log(1.0 / max(epsilon, 1e-12)), 1.0) + math.log(1.0 / max(delta, 1e-12))) / max(epsilon, 1e-12)
            else:
                # m ≳ (1/ε²) (d log(1/ε) + log(1/δ))
                m = (d * max(math.log(1.0 / max(epsilon, 1e-12)), 1.0) + math.log(1.0 / max(delta, 1e-12))) / max(epsilon**2, 1e-12)
            return int(math.ceil(m))

        # Fallback finite-class bound if size provided
        H_size = getattr(self, 'size', None)
        if H_size is not None and H_size > 0:
            return int(math.ceil((math.log(H_size) + math.log(1.0 / max(delta, 1e-12))) / max(epsilon, 1e-12)))

        # Generic conservative fallback
        return int(math.ceil((math.log(1.0 / max(delta, 1e-12)) + 10.0) / max(epsilon**2, 1e-12)))


class LinearClassifiers(HypothesisClass):
    def __init__(self, dimension: int):
        super().__init__(f"Linear Classifiers R^{dimension}")
        self.dimension = dimension

    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        w = h_params[:-1]
        b = h_params[-1]
        return np.sign(X @ w + b)

    def compute_vc_dimension(self) -> int:
        return self.dimension + 1

    def _perceptron_separable(self, X: np.ndarray, y: np.ndarray, max_iters: int = 10000) -> bool:
        # Simple perceptron feasibility check
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(max_iters):
            margins = y * (X @ w + b)
            mis_idx = np.where(margins <= 0)[0]
            if mis_idx.size == 0:
                return True
            i = mis_idx[np.random.randint(mis_idx.size)]
            w = w + y[i] * X[i]
            b = b + y[i]
        return False

    def can_shatter(self, points: np.ndarray) -> bool:
        # Test all labelings for small sets
        n = points.shape[0]
        if n > 10:
            # heuristic: too many labelings
            return False
        for labels in product([-1, 1], repeat=n):
            y = np.array(labels, dtype=float)
            if not self._perceptron_separable(points, y):
                return False
        return True


class AxisAlignedRectangles(HypothesisClass):
    def __init__(self):
        super().__init__("Axis-Aligned Rectangles R^2")

    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        a1, b1, a2, b2 = h_params
        return ((X[:, 0] >= a1) & (X[:, 0] <= b1) & (X[:, 1] >= a2) & (X[:, 1] <= b2)).astype(int)

    def compute_vc_dimension(self) -> int:
        return 4

    def can_shatter_four_points(self, points: np.ndarray) -> bool:
        assert points.shape[0] == 4 and points.shape[1] == 2
        for labels in product([0, 1], repeat=4):
            y = np.array(labels)
            pos = points[y == 1]
            if pos.size == 0:
                # empty rectangle is outside all points
                continue
            a1, b1 = np.min(pos[:, 0]), np.max(pos[:, 0])
            a2, b2 = np.min(pos[:, 1]), np.max(pos[:, 1])
            neg = points[y == 0]
            if neg.size == 0:
                continue
            inside = (neg[:, 0] >= a1) & (neg[:, 0] <= b1) & (neg[:, 1] >= a2) & (neg[:, 1] <= b2)
            if np.any(inside):
                return False
        return True


class IntervalClassifiers(HypothesisClass):
    def __init__(self):
        super().__init__("Intervals R^1")

    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        if X.ndim > 1:
            X = X.flatten()
        a, b = h_params
        return ((X >= a) & (X <= b)).astype(int)

    def compute_vc_dimension(self) -> int:
        return 2


def estimate_vc_dimension_empirically(hypothesis_class: HypothesisClass,
                                      data_generator: Callable[[int], np.ndarray],
                                      max_dimension: int = 20,
                                      n_trials: int = 100) -> int:
    est = 0
    rng = np.random.default_rng(0)
    for d in range(1, max_dimension + 1):
        shattered_any = False
        for _ in range(n_trials):
            pts = data_generator(d)
            if isinstance(hypothesis_class, LinearClassifiers):
                if hypothesis_class.can_shatter(pts):
                    shattered_any = True
                    break
            elif isinstance(hypothesis_class, AxisAlignedRectangles) and d == 4:
                if hypothesis_class.can_shatter_four_points(pts):
                    shattered_any = True
                    break
            elif isinstance(hypothesis_class, IntervalClassifiers) and d <= 2:
                shattered_any = True
                break
        if shattered_any:
            est = d
        else:
            break
    return est


def test_all_labelings_shatterable(hypothesis_class: HypothesisClass,
                                   points: np.ndarray,
                                   param_sampler: Callable[[], np.ndarray],
                                   n_attempts: int = 10000) -> bool:
    n = points.shape[0]
    for labels in product([0, 1], repeat=n):
        y = np.array(labels)
        realized = False
        for _ in range(n_attempts):
            params = param_sampler()
            y_hat = hypothesis_class.predict(params, points)
            if np.array_equal(y_hat, y):
                realized = True
                break
        if not realized:
            return False
    return True


class ERM:
    def __init__(self, hypothesis_class: HypothesisClass, loss_function: Callable = None):
        self.hypothesis_class = hypothesis_class
        self.loss_function = loss_function or (lambda yt, yp: np.mean(yt != yp))

    def fit(self, X: np.ndarray, y: np.ndarray, param_sampler: Callable[[], np.ndarray], n_candidates: int = 10000) -> np.ndarray:
        best_loss = float('inf')
        best_params = None
        for _ in range(n_candidates):
            params = param_sampler()
            preds = self.hypothesis_class.predict(params, X)
            loss = float(self.loss_function(y, preds))
            if loss < best_loss:
                best_loss = loss
                best_params = params
        return best_params

    def predict(self, X: np.ndarray, best_params: np.ndarray) -> np.ndarray:
        return self.hypothesis_class.predict(best_params, X)


def _default_param_sampler(h_class: HypothesisClass, X: np.ndarray) -> Callable[[], np.ndarray]:
    rng = np.random.default_rng(0)
    if isinstance(h_class, LinearClassifiers):
        d = h_class.dimension
        return lambda: rng.normal(size=d + 1)
    if isinstance(h_class, AxisAlignedRectangles):
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        return lambda: np.array([rng.uniform(x_min, x_max), rng.uniform(x_min, x_max),
                                 rng.uniform(y_min, y_max), rng.uniform(y_min, y_max)])
    if isinstance(h_class, IntervalClassifiers):
        x_min, x_max = np.min(X), np.max(X)
        return lambda: np.array([rng.uniform(x_min, x_max), rng.uniform(x_min, x_max)])
    return lambda: rng.normal(size=3)


def pac_learning_experiment(hypothesis_class: HypothesisClass,
                            target_function: Callable[[np.ndarray], np.ndarray],
                            data_distribution: Callable[[int], np.ndarray],
                            sample_sizes: List[int],
                            epsilon: float = 0.1,
                            delta: float = 0.1,
                            n_trials: int = 100) -> Dict:
    """Return structure compatible with tests: sample_sizes, true_risks, empirical_risks.

    true_risks: average generalization error estimated on an independent test set
    empirical_risks: average training error
    """
    results = {'sample_sizes': [], 'true_risks': [], 'empirical_risks': []}
    rng = np.random.default_rng(0)
    for m in sample_sizes:
        train_errs = []
        test_errs = []
        for _ in range(n_trials):
            X = data_distribution(m)
            y_true = target_function(X)
            sampler = _default_param_sampler(hypothesis_class, X)
            erm = ERM(hypothesis_class)
            params = erm.fit(X, y_true, sampler, n_candidates=1000)
            y_pred = erm.predict(X, params)
            train_errs.append(float(np.mean(y_true != y_pred)))

            # estimate true risk on holdout
            X_test = data_distribution(max(200, m))
            y_test = target_function(X_test)
            y_pred_test = erm.predict(X_test, params)
            test_errs.append(float(np.mean(y_test != y_pred_test)))
        results['sample_sizes'].append(m)
        results['empirical_risks'].append(float(np.mean(train_errs)))
        results['true_risks'].append(float(np.mean(test_errs)))
    return results


def growth_function_computation(hypothesis_class: HypothesisClass,
                                data_generator: Callable[[int], np.ndarray],
                                max_size: int = 20) -> List[int]:
    rng = np.random.default_rng(0)
    values = []
    for m in range(1, max_size + 1):
        X = data_generator(m)
        labelings = set()
        sampler = _default_param_sampler(hypothesis_class, X)
        # sample many hypotheses to approximate distinct labelings
        for _ in range(5000):
            params = sampler()
            y = hypothesis_class.predict(params, X)
            labelings.add(tuple(map(int, y)))
            if len(labelings) == 2**m:
                break
        values.append(len(labelings))
    return values


def verify_sauer_shelah_lemma(vc_dimension: int, growth_function: List[int]) -> bool:
    from math import comb
    for m, Pi in enumerate(growth_function, start=1):
        bound = sum(comb(m, i) for i in range(0, min(vc_dimension, m) + 1))
        if Pi > bound + 1e-9:
            return False
    return True


def visualize_pac_bounds(hypothesis_class: HypothesisClass,
                         sample_sizes: np.ndarray,
                         epsilon_values: np.ndarray):
    import matplotlib.pyplot as plt
    deltas = [0.1, 0.05, 0.01]
    for delta in deltas:
        bounds = [hypothesis_class.sample_complexity_bound(eps, delta) for eps in epsilon_values]
        plt.plot(epsilon_values, bounds, label=f"δ={delta}")
    plt.gca().invert_xaxis()
    plt.xlabel("epsilon (accuracy)")
    plt.ylabel("sample complexity bound")
    plt.title(f"PAC Bounds for {hypothesis_class.name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def demonstrate_overfitting_finite_classes(finite_class_sizes: List[int], sample_size: int = 50, n_trials: int = 100):
    rng = np.random.default_rng(0)
    gen_gap = []
    for H in finite_class_sizes:
        gaps = []
        for _ in range(n_trials):
            # Random labels baseline problem
            X = rng.normal(size=(sample_size, 2))
            y = rng.choice([0, 1], size=sample_size)
            # Finite hypothesis class: random thresholds on first feature
            thetas = rng.normal(size=H)
            train_errors = []
            test_errors = []
            Xtest = rng.normal(size=(sample_size, 2))
            ytest = rng.choice([0, 1], size=sample_size)
            for theta in thetas:
                yhat_train = (X[:, 0] > theta).astype(int)
                yhat_test = (Xtest[:, 0] > theta).astype(int)
                train_errors.append(np.mean(yhat_train != y))
                test_errors.append(np.mean(yhat_test != ytest))
            gaps.append(float(np.min(test_errors) - np.min(train_errors)))
        gen_gap.append(float(np.mean(gaps)))
    return {'sizes': finite_class_sizes, 'avg_gen_gap': gen_gap}


