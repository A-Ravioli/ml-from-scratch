import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # causal discovery · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. variables are the columns
    `0..p-1` of a data matrix. a graph is a `(p, p)` 0/1 integer matrix `G`: `G[i, j] = 1`
    and `G[j, i] = 0` means $i \to j$; `G[i, j] = G[j, i] = 1` means an undirected edge
    $i - j$. a DAG has no undirected edges; a CPDAG may have both kinds.

    | stage | you build |
    |---|---|
    | 1 | `partial_correlation`, `fisher_z_test`: a conditional independence test |
    | 2 | `pc_skeleton`: the PC algorithm's adjacency search, with separating sets |
    | 3 | `apply_meek_rules`, `orient`, `pc`: Meek's rules, v-structures, then the whole algorithm |
    | 4 | `dag_to_cpdag`, `structural_hamming_distance`: the target PC aims for, and how far off you are |
    | 5 | `possible_parent_sets`, `ida_effects`: estimating an effect when the graph is only partly oriented |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def partial_correlation(data, i, j, cond=()):
    """stage 1: correlation of columns i and j of data (n, p) after linearly regressing out the columns in cond."""
    raise NotImplementedError("stage1: implement partial_correlation")


@app.function
def fisher_z_test(data, i, j, cond=()):
    """stage 1: two-sided p-value of H0: partial correlation of i, j given cond = 0, by Fisher's z transform."""
    raise NotImplementedError("stage1: implement fisher_z_test")


@app.cell(hide_code=True)
def _():
    checks.run(1, partial_correlation=partial_correlation, fisher_z_test=fisher_z_test)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def pc_skeleton(n_vars, ci_test, *, alpha=0.05, max_cond=None):
    """stage 2: stable-PC adjacency search; returns (adj bool (p, p) symmetric, sepsets {(i, j) i < j: tuple})."""
    raise NotImplementedError("stage2: implement pc_skeleton")


@app.cell(hide_code=True)
def _():
    checks.run(2, pc_skeleton=pc_skeleton)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def apply_meek_rules(G):
    """stage 3: a copy of the graph matrix G with Meek's rules R1-R3 applied until nothing changes."""
    raise NotImplementedError("stage3: implement apply_meek_rules")


@app.function
def orient(adj, sepsets):
    """stage 3: CPDAG from a skeleton and separating sets: v-structures, then Meek's rules R1-R3."""
    raise NotImplementedError("stage3: implement orient")


@app.function
def pc(data, *, alpha=0.05, max_cond=None):
    """stage 3: the PC algorithm on data (n, p) with Fisher-z tests; returns the estimated CPDAG (p, p)."""
    raise NotImplementedError("stage3: implement pc")


@app.cell(hide_code=True)
def _():
    checks.run(3, apply_meek_rules=apply_meek_rules, orient=orient, pc=pc, pc_skeleton=pc_skeleton)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def dag_to_cpdag(dag):
    """stage 4: the CPDAG of a DAG's Markov equivalence class: keep its v-structures, undirect the rest, apply Meek."""
    raise NotImplementedError("stage4: implement dag_to_cpdag")


@app.function
def structural_hamming_distance(G1, G2):
    """stage 4: number of node pairs whose edge (none, i->j, j->i, i-j) differs between graph matrices G1 and G2."""
    raise NotImplementedError("stage4: implement structural_hamming_distance")


@app.cell(hide_code=True)
def _():
    checks.run(4, dag_to_cpdag=dag_to_cpdag, structural_hamming_distance=structural_hamming_distance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def possible_parent_sets(cpdag, x):
    """stage 5: every parent set of x consistent with the CPDAG (adds no new v-structure at x), as sorted tuples."""
    raise NotImplementedError("stage5: implement possible_parent_sets")


@app.function
def ida_effects(data, x, y, cpdag):
    """stage 5: one OLS estimate of the effect of x on y per possible parent set (coefficient of x in y ~ x + parents)."""
    raise NotImplementedError("stage5: implement ida_effects")


@app.cell(hide_code=True)
def _():
    checks.run(5, possible_parent_sets=possible_parent_sets, ida_effects=ida_effects)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`), which ends with the
    block's project: discover a graph on semi-synthetic data and estimate an effect from it.
    """)
    return


if __name__ == "__main__":
    app.run()
