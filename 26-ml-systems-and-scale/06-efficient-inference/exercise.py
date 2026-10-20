import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # efficient inference · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `quantize_int8`, `dequantize_int8`, `fake_quantize`: symmetric weight quantization, per tensor, channel or group → `mlfs.systems` |
    | 2 | `Int8Linear`, `quantize_linears`: a model whose weights are stored in int8 → `mlfs.systems` |
    | 3 | `magnitude_prune`: zeroing the smallest weights, per layer or globally |
    | 4 | `distillation_loss`: training a small model on a large one's soft predictions → `mlfs.systems` |
    | 5 | `decode_tokens_per_second`: why batching makes generation cheaper per token |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def quantize_int8(w, *, axis=0):
    """stage 1: symmetric per-channel int8 along `axis`: (q int8, scale float, broadcastable), w ~ q * scale."""
    raise NotImplementedError("stage1: implement quantize_int8")


@app.function
def dequantize_int8(q, scale):
    """stage 1: q.float() * scale."""
    raise NotImplementedError("stage1: implement dequantize_int8")


@app.function
def fake_quantize(w, bits, *, axis=0, group_size=None):
    """stage 1: round w to symmetric `bits`-bit levels and back; scales per tensor (axis None), per channel, or per group."""
    raise NotImplementedError("stage1: implement fake_quantize")


@app.cell(hide_code=True)
def _():
    checks.run(1, quantize_int8=quantize_int8, dequantize_int8=dequantize_int8, fake_quantize=fake_quantize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Int8Linear(nn.Module):
    """a linear layer whose weight is stored as int8 with per-output-channel float scales."""

    def __init__(self, weight_q, scale, bias=None):
        """stage 2: register weight_q (int8, (out, in)), scale ((out, 1)) and bias as buffers."""
        raise NotImplementedError("stage2: implement Int8Linear.__init__")

    @classmethod
    def from_linear(cls, linear):
        """stage 2: quantize an nn.Linear's weight per output channel (axis 0) and keep its bias."""
        raise NotImplementedError("stage2: implement Int8Linear.from_linear")

    def forward(self, x):
        """stage 2: F.linear(x, dequantized weight in x's dtype, bias)."""
        raise NotImplementedError("stage2: implement Int8Linear.forward")


@app.function
def quantize_linears(model, *, skip=()):
    """stage 2: a deep copy of model with every nn.Linear (except those whose qualified name is in skip) as Int8Linear."""
    raise NotImplementedError("stage2: implement quantize_linears")


@app.cell(hide_code=True)
def _():
    checks.run(2, Int8Linear=Int8Linear, quantize_linears=quantize_linears, quantize_int8=quantize_int8,
               dequantize_int8=dequantize_int8)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
@torch.no_grad()
def magnitude_prune(model, sparsity, *, scope="global"):
    """stage 3: zero the fraction `sparsity` of nn.Linear weights with the smallest |w| (scope 'global' or 'layer'); {name: mask}."""
    raise NotImplementedError("stage3: implement magnitude_prune")


@app.cell(hide_code=True)
def _():
    checks.run(3, magnitude_prune=magnitude_prune)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def distillation_loss(student_logits, teacher_logits, targets, *, temperature=2.0, alpha=0.5):
    """stage 4: alpha T^2 KL(softmax(t/T) || softmax(s/T)) (batch mean) + (1 - alpha) CE(s, targets); teacher is a constant."""
    raise NotImplementedError("stage4: implement distillation_loss")


@app.cell(hide_code=True)
def _():
    checks.run(4, distillation_loss=distillation_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def decode_tokens_per_second(n_params, batch_size, *, bytes_per_param=2, bandwidth, peak_flops):
    """stage 5: roofline for one decode step: batch / max(weight bytes / bandwidth, 2 N batch / peak)."""
    raise NotImplementedError("stage5: implement decode_tokens_per_second")


@app.cell(hide_code=True)
def _():
    checks.run(5, decode_tokens_per_second=decode_tokens_per_second)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.systems import quantize_int8, Int8Linear, distillation_loss` gives you the core
    of model compression; chapter 27 distils and quantizes its Llama.
    """)
    return


if __name__ == "__main__":
    app.run()
