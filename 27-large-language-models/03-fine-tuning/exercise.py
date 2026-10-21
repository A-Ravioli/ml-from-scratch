import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from scipy.stats import norm
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # fine-tuning · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `LoRALinear`: a frozen linear layer plus a trainable low-rank update → `mlfs.llm` |
    | 2 | `apply_lora`: wrap a model's linear layers in place and freeze everything else → `mlfs.llm` |
    | 3 | `merge_lora`: fold the updates back into plain linear layers, for free inference → `mlfs.llm` |
    | 4 | `build_sft_example`, `collate_sft`, `sft_loss`: instruction data with the loss on the response only → `mlfs.llm` |
    | 5 | `nf4_levels`, `nf4_quantize`: QLoRA's 4-bit NormalFloat format for the frozen base |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class LoRALinear(nn.Module):
    """base(x) + (alpha / r) * B A dropout(x), with base frozen, A (r, in) Kaiming-uniform and B (out, r) zero."""

    def __init__(self, base, r=8, *, alpha=16.0, dropout=0.0):
        """stage 1: keep base (frozen), lora_A (r, in_features), lora_B (out_features, r) = 0, scaling = alpha / r, dropout."""
        raise NotImplementedError("stage1: implement LoRALinear.__init__")

    def forward(self, x):
        """stage 1: base(x) + scaling * dropout(x) @ A^T @ B^T."""
        raise NotImplementedError("stage1: implement LoRALinear.forward")

    @torch.no_grad()
    def merge(self):
        """stage 1: a new nn.Linear with weight W + scaling * B A (and base's bias), with base's requires_grad."""
        raise NotImplementedError("stage1: implement LoRALinear.merge")


@app.cell(hide_code=True)
def _():
    checks.run(1, LoRALinear=LoRALinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def apply_lora(model, *, r=8, alpha=16.0, dropout=0.0, targets=None):
    """stage 2: in place, wrap every nn.Linear whose dotted name contains a target (None: all but lm_head) in LoRALinear; freeze the rest; return model."""
    raise NotImplementedError("stage2: implement apply_lora")


@app.cell(hide_code=True)
def _():
    checks.run(2, apply_lora=apply_lora, LoRALinear=LoRALinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def merge_lora(model):
    """stage 3: replace every LoRALinear in model by its merge() in place; return model."""
    raise NotImplementedError("stage3: implement merge_lora")


@app.cell(hide_code=True)
def _():
    checks.run(3, merge_lora=merge_lora, apply_lora=apply_lora, LoRALinear=LoRALinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def build_sft_example(tokenizer, prompt, response, *, max_len=None):
    """stage 4: 'User: {prompt}\\nAssistant: {response}<|endoftext|>' -> (input_ids, targets), targets shifted with -100 on the prompt."""
    raise NotImplementedError("stage4: implement build_sft_example")


@app.function
def collate_sft(examples, *, pad_id=0):
    """stage 4: list of (input_ids, targets) -> (B, T) int64 tensors, right-padded with pad_id and -100."""
    raise NotImplementedError("stage4: implement collate_sft")


@app.function
def sft_loss(logits, targets):
    """stage 4: mean cross-entropy of logits (B, T, V) against targets (B, T), ignoring -100."""
    raise NotImplementedError("stage4: implement sft_loss")


@app.cell(hide_code=True)
def _():
    checks.run(4, build_sft_example=build_sft_example, collate_sft=collate_sft, sft_loss=sft_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def nf4_levels():
    """stage 5: the 16 NormalFloat-4 levels in [-1, 1], ascending: normal quantiles with an exact zero (Dettmers et al., 2023)."""
    raise NotImplementedError("stage5: implement nf4_levels")


@app.function
def nf4_quantize(w, *, block_size=64):
    """stage 5: round each block of block_size consecutive weights to absmax * nearest NF4 level; same shape, w's dtype."""
    raise NotImplementedError("stage5: implement nf4_quantize")


@app.cell(hide_code=True)
def _():
    checks.run(5, nf4_levels=nf4_levels, nf4_quantize=nf4_quantize)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import apply_lora, merge_lora, build_sft_example, sft_loss` turns any
    pretrained `Llama` into an instruction follower; topic 04 starts its preference
    training from exactly such a model.
    """)
    return


if __name__ == "__main__":
    app.run()
