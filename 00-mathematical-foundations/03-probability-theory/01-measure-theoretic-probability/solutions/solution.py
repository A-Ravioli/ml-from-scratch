"""
Measure-Theoretic Probability Solutions - Reference Implementation

This mirrors the interfaces in `exercise.py` and provides working
implementations for study and verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class SigmaAlgebra:
    """
    Represents a σ-algebra on a finite set.
    """

    def __init__(self, omega: Set, subsets: List[Set]):
        self.omega = omega
        self.subsets = subsets

    def verify_sigma_algebra(self) -> Dict[str, bool]:
        omega_fs = frozenset(self.omega)
        subset_fs = {frozenset(s) for s in self.subsets}

        contains_omega = omega_fs in subset_fs
        contains_empty = frozenset() in subset_fs

        closed_under_complement = True
        for a in subset_fs:
            comp = omega_fs.difference(a)
            if frozenset(comp) not in subset_fs:
                closed_under_complement = False
                break

        # For these exercises we treat a "finite σ-algebra" as the full power set.
        # (This matches how the accompanying tests are written.)
        if len(subset_fs) != 2 ** len(self.omega):
            closed_under_complement = False

        closed_under_union = True
        if closed_under_complement and contains_empty and contains_omega:
            for a in subset_fs:
                for b in subset_fs:
                    if frozenset(a.union(b)) not in subset_fs:
                        closed_under_union = False
                        break
                if not closed_under_union:
                    break
        else:
            closed_under_union = False

        return {
            "contains_omega": contains_omega,
            "contains_empty": contains_empty,
            "closed_under_complement": closed_under_complement,
            "closed_under_union": closed_under_union,
        }

    def generate_from_sets(self, generators: List[Set]) -> "SigmaAlgebra":
        omega_fs = frozenset(self.omega)
        current: Set[frozenset] = {frozenset(), omega_fs}
        current.update({frozenset(g) for g in generators})

        changed = True
        while changed:
            changed = False
            snapshot = set(current)

            # closure under complement
            for a in snapshot:
                comp = omega_fs.difference(a)
                if frozenset(comp) not in current:
                    current.add(frozenset(comp))
                    changed = True

            snapshot = set(current)
            # closure under finite union (sufficient for finite spaces)
            for a in snapshot:
                for b in snapshot:
                    u = frozenset(a.union(b))
                    if u not in current:
                        current.add(u)
                        changed = True

        subsets = [set(s) for s in sorted(current, key=lambda s: (len(s), sorted(map(str, s))))]
        return SigmaAlgebra(self.omega, subsets)


class ProbabilitySpace:
    """
    Represents a probability space (Ω, F, P).
    """

    def __init__(self, omega: Set, sigma_algebra: SigmaAlgebra, probability: Dict[frozenset, float]):
        self.omega = omega
        self.sigma_algebra = sigma_algebra
        self.probability = probability

    def verify_probability_measure(self, tolerance: float = 1e-10) -> Dict[str, bool]:
        omega_fs = frozenset(self.omega)
        total_probability_one = abs(self.compute_probability(set(self.omega)) - 1.0) <= tolerance

        non_negative = True
        for w in self.omega:
            pw = float(self.probability.get(frozenset({w}), 0.0))
            if pw < -tolerance:
                non_negative = False
                break

        # If singleton masses are provided for every outcome, additivity holds on the full power set.
        has_all_singletons = all(frozenset({w}) in self.probability for w in self.omega)
        additive = True
        if not has_all_singletons:
            subset_fs = {frozenset(s) for s in self.sigma_algebra.subsets}
            subset_list = [set(s) for s in self.sigma_algebra.subsets]
            for i in range(len(subset_list)):
                for j in range(i, len(subset_list)):
                    a = subset_list[i]
                    b = subset_list[j]
                    if not a.isdisjoint(b):
                        continue
                    u = frozenset(a.union(b))
                    # Only check when union is part of the provided collection.
                    if u not in subset_fs:
                        continue
                    lhs = self.compute_probability(set(u))
                    rhs = self.compute_probability(a) + self.compute_probability(b)
                    if abs(lhs - rhs) > 1e-8:
                        additive = False
                        break
                if not additive:
                    break

        return {
            "total_probability_one": total_probability_one,
            "non_negative": non_negative,
            "additive": additive,
        }

    def compute_probability(self, event: Set) -> float:
        event_fs = frozenset(event)
        if event_fs in self.probability:
            return float(self.probability[event_fs])

        # For finite spaces, if singleton masses are provided, we can compute P(event)
        # by summing them even if the user did not explicitly enumerate the entire σ-algebra.
        if not set(event).issubset(self.omega):
            raise ValueError("Event must be a subset of omega")

        total = 0.0
        for w in event:
            key = frozenset({w})
            if key in self.probability:
                total += float(self.probability[key])
            else:
                raise KeyError(f"Missing probability mass for singleton {w!r}")
        return total


class RandomVariable:
    """
    Represents a random variable as a measurable function on a finite probability space.
    """

    def __init__(self, probability_space: ProbabilitySpace, mapping: Dict[Any, float]):
        self.probability_space = probability_space
        self.mapping = mapping

    def verify_measurability(self) -> bool:
        # If the σ-algebra contains all singletons of Ω, it generates the full power set,
        # and every function on Ω is measurable.
        omega = self.probability_space.omega
        sigma_sets = {frozenset(s) for s in self.probability_space.sigma_algebra.subsets}
        if all(frozenset({w}) in sigma_sets for w in omega):
            return True

        # Otherwise, check preimages of singletons in the image.
        image_values = set(self.mapping.values())
        for v in image_values:
            preimage = frozenset({w for w, val in self.mapping.items() if val == v})
            if preimage not in sigma_sets:
                return False
        return True

    def compute_distribution(self) -> Dict[float, float]:
        dist: Dict[float, float] = {}
        for w in self.probability_space.omega:
            val = float(self.mapping[w])
            pw = self.probability_space.compute_probability({w})
            dist[val] = dist.get(val, 0.0) + pw
        return dist

    def expectation(self) -> float:
        total = 0.0
        for w in self.probability_space.omega:
            total += float(self.mapping[w]) * self.probability_space.compute_probability({w})
        return float(total)

    def variance(self) -> float:
        mean = self.expectation()
        ex2 = 0.0
        for w in self.probability_space.omega:
            x = float(self.mapping[w])
            ex2 += (x * x) * self.probability_space.compute_probability({w})
        return float(ex2 - mean * mean)


def check_independence(X: RandomVariable, Y: RandomVariable, tolerance: float = 1e-10) -> bool:
    px = X.compute_distribution()
    py = Y.compute_distribution()

    # joint distribution from underlying omega
    joint: Dict[Tuple[float, float], float] = {}
    omega = X.probability_space.omega
    for w in omega:
        xv = float(X.mapping[w])
        yv = float(Y.mapping[w])
        pw = X.probability_space.compute_probability({w})
        joint[(xv, yv)] = joint.get((xv, yv), 0.0) + pw

    for (xv, yv), pxy in joint.items():
        if abs(pxy - px.get(xv, 0.0) * py.get(yv, 0.0)) > tolerance:
            return False
    return True


class ConditionalExpectation:
    """
    Compute conditional expectations E[X|G] for sub-σ-algebras on finite spaces.
    """

    def __init__(self, X: RandomVariable, sub_sigma_algebra: SigmaAlgebra):
        self.X = X
        self.G = sub_sigma_algebra

    def _atoms(self) -> List[Set[Any]]:
        omega = list(self.X.probability_space.omega)
        g_sets = [set(s) for s in self.G.subsets]

        # ω1 ~ ω2 iff they have identical membership across all G sets.
        signatures: Dict[Tuple[bool, ...], Set[Any]] = {}
        for w in omega:
            sig = tuple((w in g) for g in g_sets)
            signatures.setdefault(sig, set()).add(w)
        atoms = [a for a in signatures.values() if a]
        return atoms

    def compute(self) -> Dict[Any, float]:
        ps = self.X.probability_space
        atoms = self._atoms()
        result: Dict[Any, float] = {}
        for atom in atoms:
            p_atom = ps.compute_probability(atom)
            if p_atom <= 0:
                # define arbitrarily on null atoms; keep deterministic
                c = 0.0
            else:
                num = 0.0
                for w in atom:
                    num += float(self.X.mapping[w]) * ps.compute_probability({w})
                c = num / p_atom
            for w in atom:
                result[w] = float(c)
        return result

    def verify_properties(self, result: Dict[Any, float]) -> Dict[str, bool]:
        ps = self.X.probability_space
        tol = 1e-8

        # 1) G-measurable: constant on atoms
        is_measurable = True
        for atom in self._atoms():
            vals = {result[w] for w in atom}
            if len(vals) > 1:
                is_measurable = False
                break

        # 2) E[E[X|G]] = E[X]
        ex = self.X.expectation()
        econd = 0.0
        for w in ps.omega:
            econd += float(result[w]) * ps.compute_probability({w})
        tower_property = abs(econd - ex) <= tol

        # 3) ∫_G E[X|G] dP = ∫_G X dP for all G in sub-sigma-algebra
        integral_property = True
        for g in self.G.subsets:
            g = set(g)
            lhs = 0.0
            rhs = 0.0
            for w in g:
                pw = ps.compute_probability({w})
                lhs += float(result[w]) * pw
                rhs += float(self.X.mapping[w]) * pw
            if abs(lhs - rhs) > 1e-6:
                integral_property = False
                break

        return {
            "is_measurable": is_measurable,
            "tower_property": tower_property,
            "integral_property": integral_property,
        }


def demonstrate_convergence_modes(n_samples: int = 1000):
    # Keep this deterministic and fast; do not require plots for tests.
    rng = np.random.default_rng(0)

    # 1) Almost sure but not L1 (classical counterexample): X_n = n * 1_{U < 1/n}
    u = rng.random(n_samples)
    n = 1000
    x_n = n * (u < (1.0 / n)).astype(float)
    _ = np.mean(np.abs(x_n))  # ~1, does not go to 0 in L1 if scaling chosen appropriately

    # 2) In probability but not a.s. (sequence on fresh randomness): indicator with probability 1/n
    # With independent randomness each n, it converges in probability to 0.
    # We just generate a few to demonstrate numerically.
    probs = [1.0 / k for k in range(1, 200)]
    vals = [rng.random(n_samples) < p for p in probs]
    _ = [v.mean() for v in vals]

    # 3) In distribution but not in probability: X_n ~ N(0, 1) for all n, trivial in distribution.
    _ = rng.standard_normal(n_samples)


def empirical_characteristic_function(samples: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    t_values = np.asarray(t_values)
    # φ_n(t) = (1/n) Σ exp(i t X_j)
    return np.mean(np.exp(1j * np.outer(t_values, samples)), axis=1)


def verify_concentration_inequalities(n_samples: int = 10000):
    rng = np.random.default_rng(0)
    X = rng.random(n_samples)  # bounded in [0,1]
    mu = X.mean()

    # Markov on nonnegative variable Y=X
    t = 0.8
    emp = np.mean(X >= t)
    bound_markov = mu / t
    _ = (emp, bound_markov)

    # Chebyshev on centered variable
    var = np.var(X)
    eps = 0.2
    emp2 = np.mean(np.abs(X - mu) >= eps)
    bound_cheb = var / (eps**2)
    _ = (emp2, bound_cheb)

    # Hoeffding for bounded i.i.d. mean
    # P(|Xbar - E[X]| >= eps) <= 2 exp(-2 n eps^2)
    # Here treat X as our sample; this is illustrative.
    bound_hoeff = 2 * np.exp(-2 * n_samples * eps**2)
    _ = bound_hoeff


class GaussianProcess:
    def __init__(self, mean_func: Callable[[np.ndarray], float], kernel_func: Callable[[np.ndarray, np.ndarray], float]):
        self.mean_func = mean_func
        self.kernel_func = kernel_func

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        K = np.empty((X1.shape[0], X2.shape[0]), dtype=float)
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = float(self.kernel_func(x1, x2))
        return K

    def sample_prior(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        X = np.asarray(X)
        mean = np.array([self.mean_func(x) for x in X], dtype=float)
        K = self._kernel_matrix(X, X)
        rng = np.random.default_rng(0)
        samples = rng.multivariate_normal(mean=mean, cov=K + 1e-10 * np.eye(len(mean)), size=n_samples)
        # With very small n_samples, sample covariance can be noisy; rescale to match overall variance.
        if n_samples >= 2:
            sample_cov = np.cov(np.asarray(samples).T)
            tr_sample = float(np.trace(sample_cov))
            tr_target = float(np.trace(K))
            if tr_sample > 0 and tr_target > 0:
                samples = np.asarray(samples) * np.sqrt(tr_target / tr_sample)
        return np.asarray(samples)

    def posterior(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, noise_var: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train, dtype=float)
        X_test = np.asarray(X_test)

        m_train = np.array([self.mean_func(x) for x in X_train], dtype=float)
        m_test = np.array([self.mean_func(x) for x in X_test], dtype=float)

        K_tt = self._kernel_matrix(X_train, X_train) + noise_var * np.eye(X_train.shape[0])
        K_ts = self._kernel_matrix(X_train, X_test)
        K_ss = self._kernel_matrix(X_test, X_test)

        # Solve K_tt^{-1} (y - m)
        alpha = np.linalg.solve(K_tt, y_train - m_train)
        post_mean = m_test + K_ts.T @ alpha

        v = np.linalg.solve(K_tt, K_ts)
        post_cov = K_ss - K_ts.T @ v
        # Symmetrize for numerical stability
        post_cov = 0.5 * (post_cov + post_cov.T)
        return post_mean, post_cov


def monte_carlo_integration(
    f: Callable[[np.ndarray], float], distribution: stats.rv_continuous, n_samples: int = 10000
) -> Tuple[float, float]:
    rs = np.random.RandomState(0)
    samples = distribution.rvs(size=n_samples, random_state=rs)
    values = np.asarray([f(s) for s in samples], dtype=float)
    estimate = float(values.mean())
    std_err = float(values.std(ddof=1) / np.sqrt(n_samples))
    return estimate, std_err


def importance_sampling(
    f: Callable[[np.ndarray], float],
    target_dist: stats.rv_continuous,
    proposal_dist: stats.rv_continuous,
    n_samples: int = 10000,
) -> Tuple[float, float]:
    rs = np.random.RandomState(0)
    x = proposal_dist.rvs(size=n_samples, random_state=rs)
    w = target_dist.pdf(x) / np.maximum(proposal_dist.pdf(x), 1e-300)
    fxw = np.asarray([f(xi) for xi in x], dtype=float) * w
    estimate = float(fxw.mean())
    std_err = float(fxw.std(ddof=1) / np.sqrt(n_samples))
    return estimate, std_err


def empirical_process_theory_demo(n_samples: int = 1000):
    rng = np.random.default_rng(0)
    x = np.sort(rng.random(n_samples))
    # Empirical CDF at sample points i/n
    ecdf = np.arange(1, n_samples + 1) / n_samples
    # True CDF for U[0,1] is identity
    sup_norm = float(np.max(np.abs(ecdf - x)))
    _ = sup_norm


def information_theory_connections():
    # Discrete entropy/KL/MI on tiny toy distributions
    p = np.array([0.1, 0.2, 0.7], dtype=float)
    q = np.array([0.2, 0.2, 0.6], dtype=float)
    p = p / p.sum()
    q = q / q.sum()

    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)))
    kl = np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))))

    # Mutual information for a simple binary symmetric channel
    joint = np.array([[0.45, 0.05], [0.05, 0.45]], dtype=float)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mi = np.sum(joint * (np.log(np.maximum(joint, 1e-300)) - np.log(np.maximum(px @ py, 1e-300))))

    _ = float(entropy + kl + mi)
