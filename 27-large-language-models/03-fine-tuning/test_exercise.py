"""Checks for 27 · 03 fine-tuning.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. LoRA for its initialization
(B = 0, so the wrapped layer starts equal to the frozen one), its formula, the alpha / r scaling, dropout on
the low-rank path only (train vs eval), and merging; apply_lora on a small model with Llama-style names, for
which layers get wrapped (targets, lm_head), what is frozen, r / alpha / dropout, and not wrapping twice;
merge_lora for identical outputs; SFT examples for the -100 mask on the prompt, pre-shifted targets,
truncation and padding; the loss against cross-entropy; NF4 against the published levels and its blockwise
absmax scaling.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.text import BPETokenizer  # noqa: E402

NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj, self.attn.k_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d // 2, bias=False)
        self.attn.v_proj, self.attn.o_proj = nn.Linear(d, d // 2, bias=False), nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.w_up, self.mlp.w_down = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.norm(x)
        a = self.attn.o_proj(self.attn.q_proj(h) + torch.cat([self.attn.k_proj(h), self.attn.v_proj(h)], -1))
        return x + a + self.mlp.w_down(F.gelu(self.mlp.w_up(h)))


class _TinyLM(nn.Module):
    """a stand-in with Llama-style module names: tok_emb, blocks.i.attn.{q,k,v,o}_proj, blocks.i.mlp.w_*, lm_head."""

    def __init__(self, V=11, d=8, L=2):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(_Block(d) for _ in range(L))
        self.lm_head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(x)


def _model(seed=0):
    torch.manual_seed(seed)
    return _TinyLM()


def _tok():
    return BPETokenizer(270, special_tokens=("<|endoftext|>",)).train("the king and the queen of the castle " * 20)


# ---------------------------------------------------------------- stage 1: LoRALinear


def test_stage1_init_and_formula(nb):
    torch.manual_seed(0)
    base = nn.Linear(10, 6)
    lin = nb.LoRALinear(base, r=4, alpha=2.0)
    assert lin.base is base and lin.lora_A.shape == (4, 10) and lin.lora_B.shape == (6, 4), (
        "keep the wrapped layer as .base; lora_A is (r, in_features), lora_B is (out_features, r)")
    assert not any(p.requires_grad for p in base.parameters()), "the base layer is frozen"
    assert lin.lora_A.requires_grad and lin.lora_B.requires_grad
    assert torch.all(lin.lora_B == 0), "B starts at zero, so the wrapped layer starts out equal to the base"
    bound = 1 / math.sqrt(10)
    assert lin.lora_A.abs().max() <= bound + 1e-6 and lin.lora_A.std() > 0.3 * bound, (
        "A uses nn.Linear's default (Kaiming-uniform with a=sqrt(5)): uniform in +-1/sqrt(in_features), not zero")
    x = torch.randn(3, 5, 10)
    assert torch.allclose(lin(x), base(x)), "with B = 0 the output equals base(x)"
    assert math.isclose(lin.scaling, 0.5), "scaling = alpha / r = 2 / 4"
    with torch.no_grad():
        lin.lora_B.normal_()
    want = base(x) + 0.5 * x @ lin.lora_A.T @ lin.lora_B.T
    assert torch.allclose(lin(x), want, atol=1e-5), "forward = base(x) + (alpha / r) * x A^T B^T"
    assert math.isclose(nb.LoRALinear(nn.Linear(4, 4)).scaling, 2.0), "defaults r=8, alpha=16: scaling 2"
    with pytest.raises(ValueError):
        nb.LoRALinear(nn.Linear(4, 4), r=0)
    dbl = nb.LoRALinear(nn.Linear(4, 3).double(), r=2)
    assert dbl.lora_A.dtype == torch.float64 and dbl.lora_B.dtype == torch.float64, (
        "the LoRA factors follow the base layer's dtype (and device)")


def test_stage1_dropout(nb):
    torch.manual_seed(1)
    base = nn.Linear(16, 16)
    lin = nb.LoRALinear(base, r=4, dropout=0.5)
    x = torch.randn(8, 16)
    lin.train()
    assert torch.allclose(lin(x), base(x)), "dropout applies to the LoRA path only: with B = 0 it can't change anything"
    with torch.no_grad():
        lin.lora_B.normal_()
    a, b = lin(x), lin(x)
    assert not torch.allclose(a, b), "dropout=0.5 in train mode: two calls differ"
    lin.eval()
    assert torch.allclose(lin(x), lin(x)), "eval mode: no dropout"
    assert torch.allclose(lin(x), base(x) + lin.scaling * x @ lin.lora_A.T @ lin.lora_B.T, atol=1e-5)
    assert isinstance(lin.dropout, nn.Dropout) and lin.dropout.p == 0.5, "store the dropout as an nn.Dropout"


def test_stage1_merge(nb):
    torch.manual_seed(2)
    base = nn.Linear(10, 6)
    lin = nb.LoRALinear(base, r=3, alpha=6.0, dropout=0.3)
    with torch.no_grad():
        lin.lora_B.normal_()
    lin.eval()
    merged = lin.merge()
    assert type(merged) is nn.Linear, "merge returns a plain nn.Linear"
    x = torch.randn(4, 10)
    assert torch.allclose(merged(x), lin(x), atol=1e-5), (
        "the merged weight W + (alpha / r) B A gives exactly the LoRA layer's (eval-mode) output")
    assert torch.allclose(merged.bias, base.bias), "the bias is copied"
    assert not merged.weight.requires_grad, "the merged layer keeps the base's requires_grad (frozen)"
    nob = nb.LoRALinear(nn.Linear(5, 3, bias=False), r=2)
    assert nob.merge().bias is None


# ---------------------------------------------------------------- stage 2: apply_lora


def test_stage2_default_targets(nb):
    m = _model()
    idx = torch.randint(0, 11, (2, 7), generator=torch.Generator().manual_seed(0))
    before = m(idx)
    out = nb.apply_lora(m, r=2, alpha=4.0)
    assert out is m, "apply_lora works in place and returns the model"
    wrapped = sorted(n for n, x in m.named_modules() if isinstance(x, nb.LoRALinear))
    want = sorted(f"blocks.{i}.{s}" for i in range(2) for s in
                  ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.w_up", "mlp.w_down"))
    assert wrapped == want, f"targets=None wraps every nn.Linear except lm_head: want {want}, got {wrapped}"
    assert type(m.lm_head) is nn.Linear
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable), (
        f"only LoRA factors train; everything else (embeddings, norms, biases, lm_head) is frozen. trainable: "
        f"{sorted(n for n in trainable if 'lora_' not in n)[:5]}")
    n_lora = sum(p.numel() for p in m.parameters() if p.requires_grad)
    d = 8
    shapes = [(d, d), (d // 2, d), (d // 2, d), (d, d), (2 * d, d), (d, 2 * d)]
    assert n_lora == 2 * sum(2 * (o + i) for o, i in shapes), "r * (in + out) trainable numbers per wrapped layer"
    assert torch.allclose(m(idx), before, atol=1e-6), "right after apply_lora (B = 0) the model computes the same thing"
    lay = m.blocks[0].attn.q_proj
    assert lay.r == 2 and math.isclose(lay.scaling, 2.0), "r and alpha are passed to every LoRALinear"


def test_stage2_targets_dropout_and_twice(nb):
    m = nb.apply_lora(_model(), r=3, targets=("q_proj", "v_proj"), dropout=0.25)
    wrapped = sorted(n for n, x in m.named_modules() if isinstance(x, nb.LoRALinear))
    assert wrapped == ["blocks.0.attn.q_proj", "blocks.0.attn.v_proj", "blocks.1.attn.q_proj", "blocks.1.attn.v_proj"], (
        f"targets: wrap the nn.Linear layers whose dotted name contains any of the strings; got {wrapped}")
    assert m.blocks[1].attn.v_proj.dropout.p == 0.25, "dropout is passed through"
    assert not m.blocks[0].mlp.w_up.weight.requires_grad, "untargeted layers are frozen too"
    odd = _model()
    odd.lm_head_proj = nn.Linear(8, 8)  # a name that only CONTAINS lm_head
    nb.apply_lora(odd)
    assert isinstance(odd.lm_head_proj, nb.LoRALinear) and type(odd.lm_head) is nn.Linear, (
        "targets=None skips modules whose LAST name component is 'lm_head', not every name containing it")
    m2 = nb.apply_lora(_model(), targets=("lm_head",))
    assert isinstance(m2.lm_head, nb.LoRALinear), "lm_head can be targeted explicitly"
    nb.apply_lora(m, r=3, targets=("q_proj",))
    nested = [n for n, x in m.named_modules() if isinstance(x, nb.LoRALinear) and ".base" in n]
    assert not nested, f"applying LoRA again must not wrap the frozen .base layers inside LoRALinear: {nested}"
    assert all(p.requires_grad for n, p in m.named_parameters() if "lora_" in n), "LoRA factors stay trainable"


def test_stage2_training_moves_only_lora(nb):
    m = nb.apply_lora(_model(), r=2)
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    idx = torch.randint(0, 11, (4, 6), generator=torch.Generator().manual_seed(3))
    for _ in range(5):
        loss = F.cross_entropy(m(idx).reshape(-1, 11), idx.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    for n, p in m.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), f"{n} is frozen and must not change"
    assert any(p.abs().sum() > 0 for n, p in m.named_parameters() if n.endswith("lora_B")), "training moves B"


# ---------------------------------------------------------------- stage 3: merge_lora


def test_stage3_merge_model(nb):
    m = nb.apply_lora(_model(), r=2, alpha=8.0, dropout=0.1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0, 0.1)
    m.eval()
    idx = torch.randint(0, 11, (3, 9), generator=torch.Generator().manual_seed(4))
    want = m(idx)
    out = nb.merge_lora(m)
    assert out is m, "merge_lora works in place and returns the model"
    assert not any(isinstance(x, nb.LoRALinear) for x in m.modules()), "no LoRALinear remains"
    assert type(m.blocks[1].mlp.w_down) is nn.Linear
    assert torch.allclose(m(idx), want, atol=1e-5), (
        f"the merged model must compute exactly what the LoRA model computed (max diff {(m(idx) - want).abs().max():.2e})")
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in _model().parameters()), (
        "after merging, the model has the original parameter count: LoRA costs nothing at inference")


# ---------------------------------------------------------------- stage 4: SFT data and loss


def test_stage4_sft_example(nb):
    tok = _tok()
    eot = tok.encode("<|endoftext|>")
    assert len(eot) == 1
    ids, tg = nb.build_sft_example(tok, "who is king?", "the king")
    head = tok.encode("User: who is king?\nAssistant:")
    tail = tok.encode(" the king<|endoftext|>")
    assert len(ids) == len(tg) == len(head) + len(tail) - 1, "inputs and targets are both one shorter than the sequence"
    assert ids == (head + tail)[:-1], (
        "input_ids encode 'User: {prompt}\\nAssistant:' and ' {response}<|endoftext|>' separately and concatenate them, "
        "dropping the last token")
    n_masked = sum(t == -100 for t in tg)
    assert n_masked == len(head) - 1 and tg[:n_masked] == [-100] * n_masked, (
        f"targets that are prompt tokens are -100: the first {len(head) - 1}; got {n_masked}")
    assert tg[n_masked:] == tail, "the remaining targets are the response tokens, ending with <|endoftext|>: targets are pre-shifted"
    assert tg[-1] == eot[0], "the last target is the end-of-text token: the model learns to stop"
    assert tok.decode(ids + [tg[-1]]) == "User: who is king?\nAssistant: the king<|endoftext|>"
    ids2, tg2 = nb.build_sft_example(tok, "who is king?", "the king", max_len=10)
    assert ids2 == ids[:10] and tg2 == tg[:10], "max_len truncates both lists"


class _MergingTok:
    """characters as ids, except that ': ' (colon + space) is ONE token (id 900) and the end-of-text token is 901."""

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            if text.startswith("<|endoftext|>", i):
                out.append(901)
                i += len("<|endoftext|>")
            elif text.startswith(": ", i):
                out.append(900)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_stage4_separate_encoding(nb):
    tok = _MergingTok()
    ids, tg = nb.build_sft_example(tok, "hi", "yo")
    head, tail = tok.encode("User: hi\nAssistant:"), tok.encode(" yo<|endoftext|>")
    assert ids == (head + tail)[:-1] and tg == [-100] * (len(head) - 1) + tail, (
        "encode 'User: {prompt}\\nAssistant:' and ' {response}<|endoftext|>' SEPARATELY: encoding the whole string at "
        "once lets this tokenizer merge the ':' before the response with the space after it, so the boundary between "
        "masked prompt and trained response moves")


def test_stage4_collate_and_loss(nb):
    ex = [([5, 6, 7], [-100, 7, 8]), ([1, 2], [2, 3]), ([4], [-100])]
    x, y = nb.collate_sft(ex, pad_id=9)
    assert x.dtype == torch.long and y.dtype == torch.long and x.shape == (3, 3)
    assert x.tolist() == [[5, 6, 7], [1, 2, 9], [4, 9, 9]], "right-pad inputs with pad_id"
    assert y.tolist() == [[-100, 7, 8], [2, 3, -100], [-100, -100, -100]], "right-pad targets with -100"
    assert nb.collate_sft(ex)[0][2, 1] == 0, "pad_id defaults to 0"
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 3, 10, generator=g)
    got = nb.sft_loss(logits, y)
    keep = y.reshape(-1) != -100
    want = F.cross_entropy(logits.reshape(-1, 10)[keep], y.reshape(-1)[keep])
    assert torch.allclose(got, want), (
        f"mean cross-entropy over the 4 positions whose target isn't -100: {want.item():.4f}; got {got.item():.4f}")


# ---------------------------------------------------------------- stage 5: NF4


def test_stage5_levels(nb):
    lv = nb.nf4_levels()
    assert torch.is_tensor(lv) and lv.shape == (16,), "16 levels: 4 bits"
    assert torch.allclose(lv.double(), torch.tensor(NF4, dtype=torch.float64), atol=1e-6), (
        f"the NF4 levels: normal quantiles, 8 positive and 7 negative plus an exact 0, scaled to [-1, 1] "
        f"(see the lesson for the construction); got {lv.tolist()}")


def test_stage5_hand_example(nb):
    w = torch.tensor([0.3, -1.2, 0.05, 0.9])
    got = nb.nf4_quantize(w, block_size=4)
    want = torch.tensor([NF4[10] * 1.2, -1.2, NF4[8] * 1.2, NF4[14] * 1.2])
    assert torch.allclose(got, want, atol=1e-5), (
        f"the lesson's worked example: absmax 1.2, normalized (0.25, -1, 0.042, 0.75) -> levels (0.2461, -1, 0.0796, "
        f"0.7230) -> times 1.2 = {want.tolist()}; got {got.tolist()}")


def test_stage5_quantize(nb):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 32, generator=g)
    q = nb.nf4_quantize(w, block_size=64)
    assert q.shape == w.shape and q.dtype == w.dtype
    blocks, qb = w.reshape(-1, 64), q.reshape(-1, 64)
    absmax = blocks.abs().amax(1, keepdim=True)
    codes = qb / absmax
    lv = torch.tensor(NF4)
    dist = (codes[..., None] - lv).abs().min(-1).values
    assert dist.max() < 1e-5, "each output is (block absmax) x (an NF4 level)"
    assert torch.allclose(qb.abs().amax(1), absmax.squeeze(1)), "each block's largest weight is kept exactly"
    want = lv[((blocks / absmax)[..., None] - lv).abs().argmin(-1)] * absmax
    assert torch.allclose(qb, want, atol=1e-6), "round every weight to the nearest level (after dividing by its block's absmax)"
    q16 = nb.nf4_quantize(w, block_size=16)
    assert torch.allclose(q16.reshape(-1, 16).abs().amax(1), w.reshape(-1, 16).abs().amax(1)), (
        "block_size sets the blocks: with 16, every 16 consecutive weights have their own absmax")
    wide = torch.randn(128, 128, generator=g)
    assert (nb.nf4_quantize(wide, block_size=16) - wide).pow(2).mean() < (nb.nf4_quantize(wide) - wide).pow(2).mean(), (
        "smaller blocks, smaller error")
    with pytest.raises(ValueError):
        nb.nf4_quantize(torch.randn(10, 10), block_size=64)
    half = nb.nf4_quantize(w.half())
    assert half.dtype == torch.float16, "the result has the input's dtype"
    z = torch.zeros(2, 64)
    assert torch.equal(nb.nf4_quantize(z), z), "an all-zero block stays zero (no division by zero)"
    big = torch.randn(64, 64, generator=g)
    uni = []
    for blk in big.reshape(-1, 64):
        s = blk.abs().max() / 7
        uni.append(torch.round(blk / s).clamp(-7, 7) * s)
    err_uniform = (torch.stack(uni).reshape(64, 64) - big).pow(2).mean()
    err_nf4 = (nb.nf4_quantize(big) - big).pow(2).mean()
    assert err_nf4 < err_uniform, (
        f"for normally distributed weights NF4 beats uniform 4-bit levels: {err_nf4:.5f} vs {err_uniform:.5f}")
