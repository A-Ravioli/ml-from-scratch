import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # mechanistic interpretability · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `run_with_cache`: read every activation of a forward pass with hooks → `mlfs.safety` |
    | 2 | `patch_activation`: run the model with one activation replaced → `mlfs.safety` |
    | 3 | `activation_patching`, `position_patching`: clean-into-corrupt patching, per layer and per position → `mlfs.safety` |
    | 4 | `attention_patterns`, `prev_token_score`, `induction_score`: read attention heads and score what they do |
    | 5 | `ablate_heads`: knock heads out to test what they cause |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def run_with_cache(model, idx, *, names=None):
    """stage 1: model(idx) with forward hooks on named submodules (None: all but the root); (output, {name: detached output})."""
    raise NotImplementedError("stage1: implement run_with_cache")


@app.cell(hide_code=True)
def _():
    checks.run(1, run_with_cache=run_with_cache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def patch_activation(model, idx, name, value, *, positions=None):
    """stage 2: model(idx) with module `name`'s output replaced by value (broadcastable), only at positions (dim 1) if given."""
    raise NotImplementedError("stage2: implement patch_activation")


@app.cell(hide_code=True)
def _():
    checks.run(2, patch_activation=patch_activation, run_with_cache=run_with_cache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def activation_patching(model, clean_idx, corrupt_idx, metric, names):
    """stage 3: {name: metric(corrupt run with name's clean output patched in)}; eval mode (restored), no grad."""
    raise NotImplementedError("stage3: implement activation_patching")


@app.function
def position_patching(model, clean_idx, corrupt_idx, metric, names):
    """stage 3: ndarray (len(names), T): metric after patching one name's clean output at one position t."""
    raise NotImplementedError("stage3: implement position_patching")


@app.cell(hide_code=True)
def _():
    checks.run(3, activation_patching=activation_patching, position_patching=position_patching,
               patch_activation=patch_activation, run_with_cache=run_with_cache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def attention_patterns(model, idx):
    """stage 4: [(B, H, T, T) attention weights of each block in model.blocks], via attn.save_weights; eval mode, flags restored."""
    raise NotImplementedError("stage4: implement attention_patterns")


@app.function
def prev_token_score(weights):
    """stage 4: (H,) mean attention from position t to t - 1, over t >= 1 and the batch; weights (B, H, T, T)."""
    raise NotImplementedError("stage4: implement prev_token_score")


@app.function
def induction_score(weights, period):
    """stage 4: (H,) mean attention from t to t - period + 1 over t >= period (sequences repeating every `period`)."""
    raise NotImplementedError("stage4: implement induction_score")


@app.cell(hide_code=True)
def _():
    checks.run(4, attention_patterns=attention_patterns, prev_token_score=prev_token_score,
               induction_score=induction_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def ablate_heads(model, idx, heads):
    """stage 5: model(idx) with each (layer, head) in heads zeroed at the input of blocks[layer].attn.out_proj (or o_proj)."""
    raise NotImplementedError("stage5: implement ablate_heads")


@app.cell(hide_code=True)
def _():
    checks.run(5, ablate_heads=ablate_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import run_with_cache, patch_activation, activation_patching` lets you
    read and edit any model's activations (TransformerLens-style hooks are unlocked). topic 03
    asks what the directions in those activations mean when there are more features than
    dimensions.
    """)
    return


if __name__ == "__main__":
    app.run()
