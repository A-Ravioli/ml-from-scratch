#!/usr/bin/env python3
"""
Fill empty topic directories and lesson-only topics with consistent curriculum files.

This script is intentionally conservative:
- It creates missing chapter intros for late chapters.
- It fully scaffolds topics that currently have no required files.
- It fills lesson-only topics with exercise/test/solution files.
- It does not overwrite existing exercises, tests, or solutions.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("lesson.md", "exercise.py", "test_implementation.py", "solutions/solution.py")

ACRONYMS = {
    "a2c": "A2C",
    "a3c": "A3C",
    "ai": "AI",
    "bert": "BERT",
    "clip": "CLIP",
    "cnn": "CNN",
    "cnns": "CNNs",
    "cql": "CQL",
    "ddpm": "DDPM",
    "dqn": "DQN",
    "ebm": "EBM",
    "gat": "GAT",
    "gcn": "GCN",
    "gnn": "GNN",
    "gnns": "GNNs",
    "gpt": "GPT",
    "gru": "GRU",
    "kl": "KL",
    "lqr": "LQR",
    "lstm": "LSTM",
    "maml": "MAML",
    "mdp": "MDP",
    "ml": "ML",
    "mnist": "MNIST",
    "muzero": "MuZero",
    "ntk": "NTK",
    "ode": "ODE",
    "odes": "ODEs",
    "pac": "PAC",
    "pde": "PDE",
    "ppo": "PPO",
    "q": "Q",
    "qlearning": "Q-learning",
    "rademacher": "Rademacher",
    "relu": "ReLU",
    "resnet": "ResNet",
    "rhlf": "RHLF",
    "rl": "RL",
    "rlhf": "RLHF",
    "rnn": "RNN",
    "rnns": "RNNs",
    "sam": "SAM",
    "sde": "SDE",
    "sdes": "SDEs",
    "svd": "SVD",
    "t5": "T5",
    "td": "TD",
    "trpo": "TRPO",
    "vae": "VAE",
    "vaes": "VAEs",
    "vc": "VC",
    "vq": "VQ",
}

CHAPTER_INTROS = {
    "08-advanced-optimization": {
        "title": "Advanced Optimization",
        "overview": (
            "This chapter studies modern optimization phenomena that sit beyond the classical "
            "first-order and second-order toolbox. The emphasis is on geometry, robustness, "
            "distributed settings, and the algorithmic ideas currently shaping large-scale ML."
        ),
        "focus": (
            "Connect optimization theory to empirical behavior, implement compact research-flavored "
            "algorithms, and reason about tradeoffs between stability, communication, and compute."
        ),
    },
    "09-theoretical-deep-learning": {
        "title": "Theoretical Deep Learning",
        "overview": (
            "This chapter connects deep-learning practice to the main theoretical lenses used to "
            "analyze approximation, optimization, generalization, and representation dynamics."
        ),
        "focus": (
            "Treat each topic as a bridge between proof-oriented theory and computational intuition, "
            "with small numerical experiments that make abstract claims concrete."
        ),
    },
    "10-specialized-topics": {
        "title": "Specialized Topics",
        "overview": (
            "This chapter branches into specialized research directions that extend core ML ideas "
            "into new scientific, physical, and biological regimes."
        ),
        "focus": (
            "Preserve the same first-principles style as earlier chapters while making explicit which "
            "assumptions, symmetries, or hardware constraints make each area distinctive."
        ),
    },
    "11-advanced-topics": {
        "title": "Advanced Topics",
        "overview": (
            "This chapter explores advanced topics that often appear at the boundary between mature "
            "research communities and production ML systems."
        ),
        "focus": (
            "Prioritize transferable abstractions, compact from-scratch implementations, and careful "
            "comparisons between neighboring methods instead of encyclopedic coverage."
        ),
    },
    "12-frontier-research": {
        "title": "Frontier Research",
        "overview": (
            "This chapter surveys frontier ML research areas where the conceptual foundations are "
            "still moving and the implementation questions are tightly coupled to open problems."
        ),
        "focus": (
            "Frame each lesson as a research-oriented tutorial: define the core mechanism, explain "
            "why it matters, and identify the main limitations or unresolved questions."
        ),
    },
}


def slug_to_title(slug: str) -> str:
    slug = slug.strip().replace("_", "-")
    parts = [p for p in slug.split("-") if p]
    title_parts: list[str] = []
    for part in parts:
        lower = part.lower()
        if lower.isdigit():
            continue
        if lower in ACRONYMS:
            title_parts.append(ACRONYMS[lower])
            continue
        if lower == "of":
            title_parts.append("of")
            continue
        title_parts.append(lower.capitalize())
    return " ".join(title_parts) if title_parts else slug


def topic_title(topic_dir: Path) -> str:
    return slug_to_title(topic_dir.name)


def chapter_intro_template(chapter_name: str) -> str:
    info = CHAPTER_INTROS[chapter_name]
    return f"""# Chapter {chapter_name[:2]}: {info["title"]} - Learning Approach Guide

## Overview
{info["overview"]}

## Prerequisites
- Solid command of the earlier foundational chapters
- Comfort with reading mathematical derivations and implementing small numerical experiments
- Familiarity with the optimization, probability, and deep-learning tools introduced earlier in the curriculum

## Learning Philosophy
This chapter should be approached as a bridge between **theory**, **implementation**, and **research taste**.
1. Start from the governing assumptions and invariances.
2. Reduce each topic to a small computational core you can implement from scratch.
3. Use controlled experiments to understand where the method helps and where it breaks.
4. Keep track of unresolved questions rather than pretending the field is settled.

## How To Work Through The Chapter
1. Read each `lesson.md` end-to-end before coding.
2. Translate the main derivation into the small public API exposed by `exercise.py`.
3. Use the test suite as a correctness guard, not as a substitute for conceptual understanding.
4. Compare neighboring topics to understand what truly changes from one method family to the next.

## Chapter Outcomes
- Build intuition for the mathematical object each topic is really manipulating
- Implement compact, deterministic reference versions of current research ideas
- Develop the habit of comparing methods by assumptions, compute profile, and failure modes
- Leave each section with both a working implementation and a research question worth pursuing
"""


def lesson_template(title: str, topic_dir: Path) -> str:
    family = slug_to_title(topic_dir.parent.name)
    chapter = slug_to_title(topic_dir.parts[0])
    return f"""# {title}

## Prerequisites
- Earlier material from {chapter}
- Comfort with linear algebra, probability, and optimization at the level used in nearby topics
- Ability to translate equations into small numerical experiments

## Learning Objectives
- Understand the central mechanism behind {title}
- Derive a compact first-principles formulation instead of treating the method as a black box
- Implement a small deterministic version from scratch
- Connect the implementation to broader ML tradeoffs and research questions

## Mathematical Foundations

### 1. Problem Setup
{title} sits inside the broader family of {family}. The core goal is to define a transformation, update rule, or representation that captures the structure we care about while remaining computationally tractable.

At a high level we work with:
- Inputs `x` that carry the relevant structure
- Parameters `theta` that encode the method
- A score, value, loss, or transition rule that determines what "good" behavior looks like

### 2. Core Objective
The generic learning pattern is to combine:
- A signal term that rewards the desired behavior
- A regularizing or stabilizing term that prevents degenerate solutions
- A computational procedure that turns the objective into an implementable algorithm

The exact algebra differs from topic to topic, but the recurring questions are the same:
1. What information is preserved?
2. What inductive bias is added?
3. What numerical approximation makes the method practical?

### 3. Algorithmic Intuition
Think of {title} as an iterative transformation on a state or representation:
1. Start from a simple initial object
2. Apply a structured update that incorporates local evidence
3. Normalize, regularize, or project the result to keep it stable
4. Measure whether the updated representation improved the target criterion

This perspective is useful because it exposes the implementation as a small set of composable primitives instead of a monolithic block of code.

### 4. Practical Failure Modes
Important failure modes to look for:
- Numerical instability from poorly scaled updates
- Collapse to trivial solutions when the signal term dominates or vanishes
- Over-smoothing, under-fitting, or over-parameterization depending on the topic family
- Mismatch between the elegant theory and the finite-sample or finite-step implementation

## Implementation Details
The companion exercise focuses on a compact pedagogical version of {title}. The public API is intentionally small:
- one configuration object
- one core model class
- one or two helper functions that expose the mathematical primitive directly

This keeps the topic testable and makes it easier to compare with neighboring lessons.

## Suggested Experiments
1. Perturb the scale or regularization strength and observe the stability of the outputs.
2. Compare the method against a naive baseline on a tiny synthetic problem.
3. Measure how the representation or score changes over repeated updates.
4. Identify a regime where the method clearly fails and explain why.

## Research Connections
- Relate the toy implementation to the larger research literature in {family}.
- Track which simplifying assumptions were introduced for pedagogy.
- Note at least one open question where the clean mathematical story becomes difficult in large-scale practice.
"""


EXERCISE_TEMPLATE = '''"""
{title} - Exercises

Implement a compact, deterministic version of the core ideas from the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Set the NumPy seed used by the exercises."""
    np.random.seed(seed)


@dataclass
class TopicConfig:
    input_dim: int
    hidden_dim: int = 4
    scale: float = 0.5


def build_feature_map(x: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Build a simple nonlinear feature map.

    TODO: return a concatenation of:
    - the original features
    - scaled squared features
    - a bias column of ones
    """
    raise NotImplementedError


def topic_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a normalized similarity score between two vectors.

    TODO: implement cosine similarity with a small numerical stabilizer.
    """
    raise NotImplementedError


class TopicModel:
    """
    Small reference model used throughout the generated topics.
    """

    def __init__(self, config: TopicConfig):
        self.config = config
        self.weight_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TopicModel":
        """
        Fit a compact linear projector on the generated feature map.

        TODO:
        1. Build the feature map.
        2. Compute the mean feature vector.
        3. Normalize it to unit norm and store it in `weight_`.
        """
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Project data onto the learned direction.

        TODO: raise a ValueError if the model is not fitted.
        """
        raise NotImplementedError

    def score(self, x: np.ndarray) -> float:
        """
        Return the average absolute projected magnitude.
        """
        raise NotImplementedError
'''


SOLUTION_TEMPLATE = '''"""
{title} - Reference Solution
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


@dataclass
class TopicConfig:
    input_dim: int
    hidden_dim: int = 4
    scale: float = 0.5


def build_feature_map(x: np.ndarray, scale: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    squared = scale * np.square(x)
    bias = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([x, squared, bias], axis=1)


def topic_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
    return float(np.dot(x, y) / denom)


class TopicModel:
    def __init__(self, config: TopicConfig):
        self.config = config
        self.weight_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TopicModel":
        features = build_feature_map(x, scale=self.config.scale)
        weight = features.mean(axis=0)
        norm = np.linalg.norm(weight)
        self.weight_ = weight / norm if norm > 0 else weight
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.weight_ is None:
            raise ValueError("Model must be fitted before calling transform")
        features = build_feature_map(x, scale=self.config.scale)
        return features @ self.weight_

    def score(self, x: np.ndarray) -> float:
        projections = self.transform(x)
        return float(np.mean(np.abs(projections)))
'''


TEST_TEMPLATE = '''"""
{title} - Tests
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
'''


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def is_topic_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    parts = path.relative_to(ROOT).parts
    if not parts:
        return False
    if any(part.startswith(".") for part in parts):
        return False
    if "solutions" in parts:
        return False
    if not parts[0][:2].isdigit():
        return False
    subdirs = [d for d in path.iterdir() if d.is_dir() and d.name != "solutions" and not d.name.startswith(".")]
    if subdirs:
        return False
    return True


def required_state(topic_dir: Path) -> tuple[bool, bool]:
    present = [(topic_dir / rel).exists() for rel in REQUIRED]
    return all(not item for item in present), present[0] and not any(present[1:])


def main() -> int:
    generated = 0
    chapter_intros = 0

    for chapter_name in CHAPTER_INTROS:
        intro_path = ROOT / chapter_name / "CHAPTER_INTRO.md"
        if write_if_missing(intro_path, chapter_intro_template(chapter_name)):
            chapter_intros += 1

    for topic_dir in sorted(p for p in ROOT.rglob("*") if is_topic_dir(p)):
        empty_topic, lesson_only = required_state(topic_dir)
        if not empty_topic and not lesson_only:
            continue

        title = topic_title(topic_dir)
        lesson = lesson_template(title, topic_dir.relative_to(ROOT))
        exercise = EXERCISE_TEMPLATE.format(title=title)
        solution = SOLUTION_TEMPLATE.format(title=title)
        tests = TEST_TEMPLATE.format(title=title)

        if empty_topic:
            generated += write_if_missing(topic_dir / "lesson.md", lesson)
            generated += write_if_missing(topic_dir / "exercise.py", exercise)
            generated += write_if_missing(topic_dir / "test_implementation.py", tests)
            generated += write_if_missing(topic_dir / "solutions" / "solution.py", solution)
        elif lesson_only:
            generated += write_if_missing(topic_dir / "exercise.py", exercise)
            generated += write_if_missing(topic_dir / "test_implementation.py", tests)
            generated += write_if_missing(topic_dir / "solutions" / "solution.py", solution)

    print(f"Created chapter intros: {chapter_intros}")
    print(f"Created files: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
