"""
Reference solution for Module 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    attention_scores,
    compose_rotations,
    gram_schmidt,
    plot_projection_2d,
    preserves_norm,
    project_onto_subspace,
    project_onto_vector,
    projection_matrix,
    qr_condition_numbers,
    qr_via_gram_schmidt,
    rotation_matrix,
)

__all__ = [
    "attention_scores",
    "compose_rotations",
    "gram_schmidt",
    "plot_projection_2d",
    "preserves_norm",
    "project_onto_subspace",
    "project_onto_vector",
    "projection_matrix",
    "qr_condition_numbers",
    "qr_via_gram_schmidt",
    "rotation_matrix",
]
