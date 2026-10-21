import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # decoding & inference · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `apply_repetition_penalty`, `min_p_filter`: two more ways to shape the next-token distribution |
    | 2 | `speculative_accept`: the rejection step that makes a draft model's guesses exact |
    | 3 | `speculative_decode`: draft k tokens, verify them in one target pass → `mlfs.llm` |
    | 4 | `constrained_decode`: sampling only from tokens a grammar allows |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def apply_repetition_penalty(logits, prev_tokens, penalty):
    """stage 1: logits (B, V); for tokens in prev_tokens (B, L) divide positive logits by penalty, multiply negative ones."""
    raise NotImplementedError("stage1: implement apply_repetition_penalty")


@app.function
def min_p_filter(logits, min_p):
    """stage 1: set logits to -inf where softmax prob < min_p * (max prob of that row); logits (B, V)."""
    raise NotImplementedError("stage1: implement min_p_filter")


@app.cell(hide_code=True)
def _():
    checks.run(1, apply_repetition_penalty=apply_repetition_penalty, min_p_filter=min_p_filter)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def speculative_accept(p, q, draft_tokens, *, generator=None):
    """stage 2: p (k+1, V) target probs, q (k, V) draft probs, draft_tokens (k,) -> (n_accepted, next_token int)."""
    raise NotImplementedError("stage2: implement speculative_accept")


@app.cell(hide_code=True)
def _():
    checks.run(2, speculative_accept=speculative_accept)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
@torch.no_grad()
def speculative_decode(target, draft, idx, max_new_tokens, *, k=4, temperature=1.0, generator=None):
    """stage 3: batch 1; draft proposes k tokens, target verifies in one forward; exact; (tokens (1, T + max_new_tokens), acceptance_rate)."""
    raise NotImplementedError("stage3: implement speculative_decode")


@app.cell(hide_code=True)
def _():
    checks.run(3, speculative_decode=speculative_decode, speculative_accept=speculative_accept)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
@torch.no_grad()
def constrained_decode(model, idx, max_new_tokens, allowed, *, temperature=1.0, generator=None):
    """stage 4: batch 1; at each step sample only from tokens where allowed(generated: list[int]) (bool (V,)) is True; stop early if none."""
    raise NotImplementedError("stage4: implement constrained_decode")


@app.cell(hide_code=True)
def _():
    checks.run(4, constrained_decode=constrained_decode)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import speculative_decode` generates from any pair of causal LMs with the
    same vocabulary, exactly as the big one would, only faster when the small one guesses well.
    """)
    return


if __name__ == "__main__":
    app.run()
