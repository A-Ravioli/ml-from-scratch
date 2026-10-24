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
    # fine-tuning · build (reference solution)

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
        super().__init__()
        if r < 1:
            raise ValueError(f"rank r must be >= 1, got {r}")
        self.base, self.r, self.alpha = base, r, alpha
        self.scaling = alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # nn.Linear's default init
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """stage 1: base(x) + scaling * dropout(x) @ A^T @ B^T."""
        return self.base(x) + self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)

    @torch.no_grad()
    def merge(self):
        """stage 1: a new nn.Linear with weight W + scaling * B A (and base's bias), with base's requires_grad."""
        out_f, in_f = self.base.weight.shape
        merged = nn.Linear(in_f, out_f, bias=self.base.bias is not None, device=self.base.weight.device,
                           dtype=self.base.weight.dtype)
        merged.weight.copy_(self.base.weight + self.scaling * self.lora_B @ self.lora_A)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        for p_new, p_old in zip(merged.parameters(), self.base.parameters()):
            p_new.requires_grad_(p_old.requires_grad)
        return merged


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
    wrapped = {name for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    chosen = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or any(name.startswith(w + ".") for w in wrapped):
            continue
        last = name.rsplit(".", 1)[-1]
        if (targets is None and last != "lm_head") or (targets is not None and any(t in name for t in targets)):
            chosen.append(name)
    for p in model.parameters():
        p.requires_grad_(False)
    for name in chosen:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha=alpha, dropout=dropout))
    for m in model.modules():  # LoRA factors (new and previously applied) are the only trainable tensors
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    return model


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
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.merge())
    return model


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
    head = tokenizer.encode(f"User: {prompt}\nAssistant:")
    tail = tokenizer.encode(f" {response}<|endoftext|>")
    seq = head + tail
    input_ids = seq[:-1]
    targets = [-100] * (len(head) - 1) + tail
    if max_len is not None:
        input_ids, targets = input_ids[:max_len], targets[:max_len]
    return input_ids, targets


@app.function
def collate_sft(examples, *, pad_id=0):
    """stage 4: list of (input_ids, targets) -> (B, T) int64 tensors, right-padded with pad_id and -100."""
    T = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), T), pad_id, dtype=torch.long)
    y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (ids, tg) in enumerate(examples):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[i, :len(tg)] = torch.tensor(tg, dtype=torch.long)
    return x, y


@app.function
def sft_loss(logits, targets):
    """stage 4: mean cross-entropy of logits (B, T, V) against targets (B, T), ignoring -100."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


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
    offset = 0.9677083
    pos = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1].double().numpy())
    neg = -norm.ppf(torch.linspace(offset, 0.5, 8)[:-1].double().numpy())
    levels = torch.tensor(sorted([*pos.tolist(), 0.0, *neg.tolist()]), dtype=torch.float32)
    return levels / levels.abs().max()


@app.function
def nf4_quantize(w, *, block_size=64):
    """stage 5: round each block of block_size consecutive weights to absmax * nearest NF4 level; same shape, w's dtype."""
    if w.numel() % block_size:
        raise ValueError(f"{w.numel()} weights don't split into blocks of {block_size}")
    flat = w.detach().float().reshape(-1, block_size)
    absmax = flat.abs().amax(dim=1, keepdim=True)
    scale = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    levels = nf4_levels().to(w.device)
    idx = ((flat / scale)[..., None] - levels).abs().argmin(dim=-1)
    return (levels[idx] * scale).reshape(w.shape).to(w.dtype)


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
