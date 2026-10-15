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
    # efficient inference · build (reference solution)

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
    dims = [d for d in range(w.dim()) if d != axis % w.dim()]
    amax = w.detach().abs().amax(dim=dims, keepdim=True) if dims else w.detach().abs()
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).float()
    q = torch.round(w.detach().float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


@app.function
def dequantize_int8(q, scale):
    """stage 1: q.float() * scale."""
    return q.float() * scale


@app.function
def fake_quantize(w, bits, *, axis=0, group_size=None):
    """stage 1: round w to symmetric `bits`-bit levels and back; scales per tensor (axis None), per channel, or per group."""
    qmax = 2 ** (bits - 1) - 1
    x = w.detach().float()
    if group_size is not None:  # groups of consecutive elements along the last dimension
        shape = x.shape
        g = x.reshape(-1, group_size)
        amax = g.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
        return (torch.round(g / scale).clamp(-qmax, qmax) * scale).reshape(shape)
    if axis is None:
        amax = x.abs().amax()
    else:
        dims = [d for d in range(x.dim()) if d != axis % x.dim()]
        amax = x.abs().amax(dim=dims, keepdim=True) if dims else x.abs()
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


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
        super().__init__()
        self.register_buffer("weight_q", weight_q.to(torch.int8))
        self.register_buffer("scale", scale.float())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.out_features, self.in_features = weight_q.shape

    @classmethod
    def from_linear(cls, linear):
        """stage 2: quantize an nn.Linear's weight per output channel (axis 0) and keep its bias."""
        q, scale = quantize_int8(linear.weight, axis=0)
        return cls(q, scale, linear.bias)

    def forward(self, x):
        """stage 2: F.linear(x, dequantized weight in x's dtype, bias)."""
        w = dequantize_int8(self.weight_q, self.scale).to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


@app.function
def quantize_linears(model, *, skip=()):
    """stage 2: a deep copy of model with every nn.Linear (except those whose qualified name is in skip) as Int8Linear."""
    model = copy.deepcopy(model)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full not in skip:
                setattr(module, child_name, Int8Linear.from_linear(child))
    return model


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
    if scope not in ("global", "layer"):
        raise ValueError(f"scope must be 'global' or 'layer', got {scope!r}")
    layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    def threshold(values, frac):
        k = int(round(frac * values.numel()))
        if k == 0:
            return None
        return torch.kthvalue(values.flatten(), k).values

    masks = {}
    if scope == "global":
        t = threshold(torch.cat([m.weight.abs().flatten() for m in layers.values()]), sparsity)
    for name, m in layers.items():
        tl = t if scope == "global" else threshold(m.weight.abs(), sparsity)
        mask = torch.ones_like(m.weight, dtype=torch.bool) if tl is None else m.weight.abs() > tl
        m.weight.mul_(mask)
        masks[name] = mask
    return masks


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
    T = temperature
    t = teacher_logits.detach()
    kl = F.kl_div(F.log_softmax(student_logits / T, dim=-1), F.log_softmax(t / T, dim=-1), log_target=True,
                  reduction="batchmean")
    return alpha * T * T * kl + (1 - alpha) * F.cross_entropy(student_logits, targets)


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
    t_memory = n_params * bytes_per_param / bandwidth
    t_compute = 2 * n_params * batch_size / peak_flops
    return batch_size / max(t_memory, t_compute)


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
