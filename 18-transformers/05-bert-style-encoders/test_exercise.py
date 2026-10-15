"""Checks for 18 · 05 BERT-style encoders.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the masking is checked for
its statistics (15% selected, 80/10/10) and its invariants (specials never touched, labels only where
selected); the encoder for bidirectionality, padding and a tied head; pretraining against a corpus
where every masked token is recoverable from its neighbours; fine-tuning on a task a classifier must
learn. training runs under `mlfs.testing.bounded`.
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

MASK, PAD, CLS = 0, 1, 2


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


def _warm_up():
    m = nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters())
    F.cross_entropy(m(torch.zeros(3, 2)), torch.zeros(3, dtype=torch.long)).backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()


_warm_up()


# ---------------------------------------------------------------- stage 1: masking


def test_stage1_masking_statistics(nb):
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(3, 50, (400, 100), generator=g)
    idx[:, 0] = CLS
    idx[:, -10:] = PAD
    inputs, labels = nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=50, special_ids=(CLS, PAD), generator=g)
    assert inputs.dtype == torch.int64 and labels.dtype == torch.int64 and inputs.shape == idx.shape, (
        "inputs and labels are int64 tensors shaped like idx")
    sel = labels != -100
    assert torch.equal(labels[sel], idx[sel]), "labels hold the ORIGINAL token at selected positions"
    assert not sel[:, 0].any() and not sel[:, -10:].any(), "special tokens ([CLS], [PAD]) are never selected"
    assert torch.equal(inputs[~sel], idx[~sel]), "unselected positions are left unchanged in inputs"
    frac = sel[:, 1:-10].float().mean().item()
    assert abs(frac - 0.15) < 0.01, f"each non-special position is selected with probability 0.15; selected {frac:.3f}"
    n = sel.sum().item()
    masked = (inputs[sel] == MASK).float().sum().item() / n
    kept = (inputs[sel] == idx[sel]).float().sum().item() / n
    rand = 1 - masked - kept
    assert abs(masked - 0.8) < 0.02 and abs(kept - 0.1) < 0.02 and abs(rand - 0.1) < 0.02, (
        f"of the selected positions, 80% become [MASK], 10% a random token, 10% stay unchanged; got {masked:.3f} "
        f"masked, {rand:.3f} replaced, {kept:.3f} unchanged (a random draw that equals the original counts as kept)")
    swapped = inputs[sel & (inputs != MASK) & (inputs != idx)]
    assert swapped.min() >= 0 and swapped.max() < 50 and len(swapped.unique()) > 30, (
        "random replacements are drawn uniformly from the vocabulary")


def test_stage1_reproducible_and_edge_cases(nb):
    idx = torch.randint(3, 20, (8, 30), generator=torch.Generator().manual_seed(1))
    a = nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=20, generator=torch.Generator().manual_seed(7))
    b = nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=20, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "the generator makes masking reproducible"
    inp, lab = nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=20, mask_prob=0.0)
    assert torch.equal(inp, idx) and (lab == -100).all(), "mask_prob=0 selects nothing"
    inp, lab = nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=20, mask_prob=1.0)
    assert (lab != -100).all(), "mask_prob=1 selects every non-special position"
    orig = idx.clone()
    nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=20, mask_prob=0.5)
    assert torch.equal(idx, orig), "mask_tokens must not modify its input in place (clone it)"


def test_stage1_random_tokens_cover_the_vocabulary(nb):
    idx = torch.full((200, 100), 5)
    inp, lab = nb.mask_tokens(idx, mask_token_id=99, vocab_size=10, mask_prob=1.0,
                              generator=torch.Generator().manual_seed(2))
    replaced = inp[(inp != 99)]
    counts = torch.bincount(replaced, minlength=10).double()
    others = torch.cat([counts[:5], counts[6:]])  # token 5 also includes the 10% kept unchanged
    expected = 0.1 * 20000 / 10
    chi2 = ((others - expected) ** 2 / expected).sum().item()
    assert counts[0] > 0 and counts[9] > 0 and chi2 < 30, (
        f"random replacements are drawn uniformly from ALL of 0 .. vocab_size - 1 (including 0 and vocab_size - 1); "
        f"counts per token {counts.long().tolist()}")


# ---------------------------------------------------------------- stage 2: the encoder


def test_stage2_bidirectional_and_padding(nb):
    torch.manual_seed(0)
    m = nb.BERT(30, max_len=16, n_layers=2, n_heads=2, d_model=16).eval()
    for name in ("tok_emb", "pos_emb", "blocks", "ln_f"):
        assert hasattr(m, name), f"BERT needs a submodule named {name!r}"
    assert all(b.norm_first for b in m.blocks), "pre-LN encoder blocks"
    idx = torch.randint(3, 30, (2, 10))
    h = m(idx)
    assert tuple(h.shape) == (2, 10, 16), f"forward returns hidden states (B, T, d) = (2, 10, 16); got {tuple(h.shape)}"
    idx2 = idx.clone()
    idx2[:, 9] = (idx2[:, 9] + 1) % 30
    assert not _close(m(idx2)[:, 0], h[:, 0]), (
        "BERT is bidirectional: changing the LAST token must change the hidden state of the FIRST (no causal mask)")
    am = torch.ones(2, 10, dtype=torch.bool)
    am[:, 7:] = False
    idx3 = idx.clone()
    idx3[:, 7:] = PAD
    idx4 = idx.clone()
    idx4[:, 7:] = 5
    assert _close(m(idx3, am)[:, :7], m(idx4, am)[:, :7]), (
        "with attention_mask (B, T) True = real token, the real positions must not attend to padding: changing the "
        "padded tokens must not change them. pass the mask as (B, 1, 1, T) to the blocks")
    with pytest.raises(ValueError):
        m(torch.randint(3, 30, (1, 17)))


def test_stage2_tied_mlm_head(nb):
    torch.manual_seed(1)
    m = nb.BERT(30, max_len=16, n_layers=1, n_heads=2, d_model=16).eval()
    idx = torch.randint(3, 30, (2, 8))
    logits = m.mlm_logits(m(idx))
    assert tuple(logits.shape) == (2, 8, 30), f"mlm_logits -> (B, T, V) = (2, 8, 30); got {tuple(logits.shape)}"
    vd = [n for n, p in m.named_parameters() if tuple(p.shape) == (30, 16)]
    assert vd == ["tok_emb.weight"], (
        f"the MLM head is tied to the token embedding: the only (V, d) parameter is tok_emb.weight; found {vd}")
    _, loss0 = None, F.cross_entropy(logits.reshape(-1, 30), idx.reshape(-1))
    assert abs(loss0.item() - math.log(30)) < 0.2, (
        f"with a small initialization (N(0, 0.02)) the initial MLM loss is close to ln V = {math.log(30):.2f}; "
        f"got {loss0.item():.2f}")
    h = m(idx)
    with torch.no_grad():
        m.tok_emb.weight.zero_()
    out = m.mlm_logits(h)
    assert _close(out, out[..., :1].expand_as(out)), (
        "with the token embedding zeroed, mlm_logits must give the same logit for every token (only a bias is left): "
        "the decoder must use tok_emb.weight")


def test_stage2_mlm_head_formula(nb):
    torch.manual_seed(9)
    m = nb.BERT(20, max_len=8, n_layers=1, n_heads=2, d_model=16).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        m.mlm_norm.weight.copy_(1 + 0.5 * torch.randn(16, generator=g))
        m.mlm_norm.bias.copy_(0.5 * torch.randn(16, generator=g))
        m.mlm_bias.copy_(torch.randn(20, generator=g))
    h = torch.randn(2, 5, 16, generator=g)
    want = m.mlm_norm(F.gelu(m.mlm_dense(h))) @ m.tok_emb.weight.T + m.mlm_bias
    assert _close(m.mlm_logits(h), want), (
        "mlm_logits(h) = mlm_norm(GELU(mlm_dense(h))) @ tok_emb.weight.T + mlm_bias: dense, then GELU, then LayerNorm, "
        "then the tied decoder plus a bias per token")


# ---------------------------------------------------------------- stage 3: pretraining


def _counting_corpus(n, T, V, seed):
    """each row repeats one random (non-special) token: a masked token is predictable only from its context."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, V, (n, 1), generator=g).expand(n, T).clone()


def test_stage3_mlm_loss(nb):
    torch.manual_seed(2)
    m = nb.BERT(20, max_len=12, n_layers=1, n_heads=2, d_model=16).eval()
    idx = torch.randint(3, 20, (3, 12))
    inputs, labels = nb.mask_tokens(idx, mask_token_id=MASK, vocab_size=20, mask_prob=0.5,
                                    generator=torch.Generator().manual_seed(0))
    am = torch.ones(3, 12, dtype=torch.bool)
    am[0, 9:] = False
    got = nb.mlm_loss(m, inputs, labels, am)
    want = F.cross_entropy(m.mlm_logits(m(inputs, am)).reshape(-1, 20), labels.reshape(-1), ignore_index=-100)
    assert _close(got, want), "mlm_loss is the cross-entropy over the selected positions only (labels -100 ignored)"


def test_stage3_pretraining_learns_context(nb):
    data = _counting_corpus(512, 12, 23, 0)
    torch.manual_seed(0)
    m = nb.BERT(23, max_len=12, n_layers=2, n_heads=2, d_model=32, dropout=0.0)
    losses = bounded(nb.pretrain_mlm, m, data, mask_token_id=MASK, special_ids=(CLS, PAD), steps=150, batch_size=32,
                     lr=3e-3, seed=0, budget=80_000_000, what="pretrain_mlm")
    assert isinstance(losses, list) and len(losses) == 150, "pretrain_mlm returns one loss per step"
    late = sum(losses[-20:]) / 20
    assert late < 0.5, (
        f"in this corpus every row repeats one random token, so a masked token is predictable from the rest of its row (and from nothing else: a model that ignores context can't beat ln 20 = 3.0); "
        f"after 150 steps the MLM loss should be below 0.5 (it starts near ln 23). got {late:.3f}")
    assert not m.training, "pretrain_mlm leaves the model in eval mode"
    torch.manual_seed(0)
    m2 = nb.BERT(23, max_len=12, n_layers=2, n_heads=2, d_model=32, dropout=0.0)
    again = nb.pretrain_mlm(m2, data, mask_token_id=MASK, special_ids=(CLS, PAD), steps=5, seed=0, lr=3e-3)
    assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(again, losses[:5])), "the same seed gives the same run"


class _Spy(nn.Module):
    """wraps a BERT or classifier and records every input and attention mask it receives."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.inputs, self.masks = inner, [], []
        for name in ("vocab_size", "d_model"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def forward(self, idx, attention_mask=None):
        self.inputs.append(idx.clone())
        self.masks.append(None if attention_mask is None else attention_mask.clone())
        return self.inner(idx, attention_mask)

    def mlm_logits(self, hidden):
        return self.inner.mlm_logits(hidden)


def test_stage3_pretraining_options(nb):
    torch.manual_seed(10)
    data = torch.full((1, 40), 7)  # one row, so every batch holds the same sequence
    data[0, 30:] = PAD
    am = (data != PAD)
    spy = _Spy(nb.BERT(12, max_len=40, n_layers=1, n_heads=2, d_model=16))
    nb.pretrain_mlm(spy, data, mask_token_id=MASK, special_ids=(PAD,), attention_mask=am, steps=6, batch_size=16,
                    mask_prob=0.5, seed=0)
    ins = torch.stack(spy.inputs)  # (steps, B, T)
    frac = (ins[..., :30] == MASK).float().mean().item()
    assert 0.33 < frac < 0.47, (
        f"with mask_prob=0.5, about 0.8 * 0.5 = 40% of the real tokens become [MASK]; got {frac:.2f}. pass mask_prob "
        f"to mask_tokens")
    assert not torch.equal(ins[0], ins[1]), "draw a fresh mask every step (dynamic masking)"
    assert all(mk is not None and torch.equal(mk, am.expand_as(mk)) for mk in spy.masks), (
        "pass the batch's rows of attention_mask to the model")
    orig = torch.nn.utils.clip_grad_norm_
    seen = []
    torch.nn.utils.clip_grad_norm_ = lambda params, max_norm, *a, **k: seen.append(max_norm) or orig(params, max_norm, *a, **k)
    try:
        nb.pretrain_mlm(nb.BERT(12, max_len=40, n_layers=1, n_heads=2, d_model=16), data, mask_token_id=MASK, steps=3,
                        batch_size=4)
    finally:
        torch.nn.utils.clip_grad_norm_ = orig
    assert seen == [1.0, 1.0, 1.0], f"clip the gradient norm at 1 every step with nn.utils.clip_grad_norm_; saw {seen}"


# ---------------------------------------------------------------- stage 4: fine-tuning


def test_stage4_classifier_pooling(nb):
    torch.manual_seed(3)
    enc = nb.BERT(20, max_len=10, n_layers=1, n_heads=2, d_model=16)
    clf = nb.BERTClassifier(enc, 3).eval()
    assert clf.encoder is enc and isinstance(clf.head, nn.Linear) and clf.head.out_features == 3, (
        "BERTClassifier keeps the encoder as .encoder and adds a .head Linear(d_model, num_classes)")
    idx = torch.randint(3, 20, (4, 10))
    am = torch.ones(4, 10, dtype=torch.bool)
    am[:, 6:] = False
    out = clf(idx, am)
    assert tuple(out.shape) == (4, 3), "logits (B, num_classes)"
    assert _close(out, clf.head(enc(idx, am)[:, 0])), "pool='cls' classifies from the first token's hidden state"
    mean = nb.BERTClassifier(enc, 3, pool="mean").eval()
    h = enc(idx, am)
    assert _close(mean(idx, am), mean.head(h[:, :6].mean(1))), "pool='mean' averages the hidden states of REAL tokens only"
    with pytest.raises(ValueError):
        nb.BERTClassifier(enc, 3, pool="max")


def test_stage4_finetune_learns(nb):
    g = torch.Generator().manual_seed(4)
    X = torch.randint(4, 15, (256, 8), generator=g)
    y = (X == 7).any(1).long()  # does the token 7 appear anywhere?
    X = torch.cat([torch.full((256, 1), CLS), X], 1)
    torch.manual_seed(4)
    clf = nb.BERTClassifier(nb.BERT(15, max_len=9, n_layers=1, n_heads=2, d_model=32, dropout=0.0), 2)
    hist = bounded(nb.finetune, clf, X, y, epochs=12, batch_size=32, lr=3e-3, seed=0, budget=80_000_000, what="finetune")
    assert isinstance(hist, list) and len(hist) == 12, "finetune returns one mean training loss per epoch"
    with torch.no_grad():
        acc = (clf(X).argmax(-1) == y).float().mean().item()
    assert hist[-1] < hist[0] and acc > 0.9, (
        f"fine-tuning on 'does token 7 appear?' should reach > 90% training accuracy in 12 epochs; got {acc:.2f} "
        f"(losses {[round(h, 3) for h in hist]})")
    assert not clf.training, "finetune leaves the model in eval mode"


def test_stage4_finetune_shuffles_and_masks(nb):
    torch.manual_seed(11)
    X = torch.arange(3, 3 + 64).remainder(15)[:, None].expand(64, 6).clone()
    X[:, 0] = torch.arange(64) % 15  # row identity in the first column
    am = torch.ones(64, 6, dtype=torch.bool)
    am[:, 4:] = False
    y = torch.zeros(64, dtype=torch.long)
    spy = _Spy(nb.BERTClassifier(nb.BERT(15, max_len=6, n_layers=1, n_heads=2, d_model=16), 2))
    nb.finetune(spy, X, y, attention_mask=am, epochs=2, batch_size=16, seed=0)
    firsts = [b[:, 0].tolist() for b in spy.inputs]
    assert firsts[0] != X[:16, 0].tolist() and firsts[0] != firsts[4], (
        "shuffle the training set every epoch (torch.randperm with a generator seeded by seed)")
    assert all(mk is not None and mk.shape == (16, 6) and not mk[:, 4:].any() for mk in spy.masks), (
        "pass each minibatch's rows of attention_mask to the model")
