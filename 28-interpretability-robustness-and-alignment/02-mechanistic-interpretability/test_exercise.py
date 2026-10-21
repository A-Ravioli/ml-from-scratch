"""Checks for 28 · 02 mechanistic interpretability.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. tiny GPTs and Llamas (the
reference mlfs.transformer and mlfs.llm models) and hand-built modules stand in for a real model. the cache
against manual hooks (which names, detached, tuples, hooks removed even on errors); patching against hand
computations (a clean patch restores the clean output exactly, positions only change what is downstream and
causal, broadcasting); patching sweeps against single patches, in eval mode; attention patterns against the
blocks' own saved weights; the head scores on hand-made attention patterns; head ablation against zeroed weights.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.llm import Llama  # noqa: E402
from mlfs.testing import bounded  # noqa: E402
from mlfs.transformer import GPT  # noqa: E402

V = 11


def _gpt(seed=0, dropout=0.0, n_layers=2):
    torch.manual_seed(seed)
    return GPT(V, block_size=16, n_layers=n_layers, n_heads=2, d_model=16, dropout=dropout).eval()


def _ids(seed=1, B=3, T=8):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


class _Pair(nn.Module):
    def forward(self, x):
        return x * 2, x + 1


class _Net(nn.Module):
    """x -> a (linear) -> b (tanh) -> pair (tuple) -> sum: every intermediate is known."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(3, 3)
        self.b = nn.Tanh()
        self.pair = _Pair()

    def forward(self, x):
        p, q = self.pair(self.b(self.a(x)))
        return p + q


def _no_hooks(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


# ---------------------------------------------------------------- stage 1: caching


def test_stage1_cache_contents(nb):
    net = _Net()
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(0), requires_grad=True)
    out, cache = nb.run_with_cache(net, x)
    a = net.a(x)
    assert torch.allclose(out, net(x)), "the output is the model's normal output"
    assert set(cache) == {"a", "b", "pair"}, (
        f"names=None caches every named submodule except the root (whose name is ''); got {sorted(cache)}")
    assert torch.allclose(cache["a"], a) and torch.allclose(cache["b"], torch.tanh(a)), (
        "cache[name] is the OUTPUT of that module")
    assert not cache["a"].requires_grad and not cache["b"].requires_grad, "cached tensors are detached"
    assert isinstance(cache["pair"], tuple) and len(cache["pair"]) == 2 and not cache["pair"][0].requires_grad, (
        "a module that returns a tuple is cached as a tuple of detached tensors")
    class _Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Tanh()

        def forward(self, z):
            return self.f(self.f(z) * 3)

    _, twice = nb.run_with_cache(_Twice(), x, names=["f"])
    assert torch.allclose(twice["f"], torch.tanh(3 * torch.tanh(x))), (
        "a module that runs twice in one forward pass: the cache holds its LAST output")
    _, some = nb.run_with_cache(net, x, names=["b"])
    assert set(some) == {"b"}, "names=[...] caches only those modules"
    assert _no_hooks(net), "remove every hook after the forward pass (handle.remove())"
    with pytest.raises(KeyError):
        nb.run_with_cache(net, x, names=["c"])


def test_stage1_gpt_and_cleanup(nb):
    m = _gpt()
    idx = _ids()
    (logits, loss), cache = nb.run_with_cache(m, idx, names=["blocks.0", "blocks.1", "ln_f", "lm_head"])
    assert loss is None and torch.allclose(logits, m(idx)[0]), "GPT's output is the (logits, loss) tuple, unchanged"
    assert cache["blocks.0"].shape == (3, 8, 16), "a block's output is the residual stream (B, T, d)"
    assert torch.allclose(m.lm_head(cache["ln_f"]), logits, atol=1e-5) and torch.allclose(cache["lm_head"], logits), (
        "cache['ln_f'] fed through lm_head gives the logits")

    class _Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2)

        def forward(self, x):
            self.a(x)
            raise RuntimeError("boom")

    boom = _Boom()
    with pytest.raises(RuntimeError):
        nb.run_with_cache(boom, torch.zeros(1, 2), names=["a"])
    assert _no_hooks(boom), "hooks are removed even when the forward pass raises (try/finally)"


# ---------------------------------------------------------------- stage 2: patching


def test_stage2_patch_semantics(nb):
    net = _Net()
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    val = torch.full((4, 3), 0.25)
    got = nb.patch_activation(net, x, "b", val)
    assert torch.allclose(got, 3 * val + 1), (
        "patching 'b' replaces the tanh output, so the rest of the model sees value: pair -> 2v + (v + 1)")
    scalar = nb.patch_activation(net, x, "b", torch.tensor(0.25))
    assert scalar.shape == (4, 3) and torch.allclose(scalar, got), (
        "value only needs to be broadcastable to the output: a scalar is expanded to the module's output shape")
    cols = nb.patch_activation(net, x, "b", torch.tensor(0.25), positions=[0, 2])
    tb = torch.tanh(net.a(x))
    want = 3 * torch.stack([torch.full((4,), 0.25), tb[:, 1], torch.full((4,), 0.25)], 1) + 1
    assert cols.shape == (4, 3) and torch.allclose(cols, want, atol=1e-6), (
        "with positions, a broadcastable value replaces the output only at those indices of dim 1")
    assert torch.allclose(net(x), 3 * torch.tanh(net.a(x)) + 1, atol=1e-6), "the model itself is unchanged afterwards"
    assert _no_hooks(net), "remove the hook afterwards"
    assert torch.equal(val, torch.full((4, 3), 0.25)), "don't modify value in place"
    with pytest.raises(KeyError):
        nb.patch_activation(net, x, "nope", val)


def test_stage2_clean_patch_restores_clean(nb):
    m = _gpt()
    clean, corrupt = _ids(1), _ids(2)
    clean_logits = m(clean)[0]
    _, cache = nb.run_with_cache(m, clean, names=["blocks.0", "blocks.1"])
    same = nb.patch_activation(m, clean, "blocks.0", cache["blocks.0"])[0]
    assert torch.allclose(same, clean_logits, atol=1e-5), "patching a clean activation into the clean run changes nothing"
    restored = nb.patch_activation(m, corrupt, "blocks.1", cache["blocks.1"])[0]
    assert torch.allclose(restored, clean_logits, atol=1e-5), (
        "patching the LAST block's clean output into the corrupt run gives the clean logits: everything after it "
        "depends only on that activation")


def test_stage2_positions(nb):
    m = _gpt()
    clean, corrupt = _ids(1), _ids(2)
    cl, co = m(clean)[0], m(corrupt)[0]
    _, cache = nb.run_with_cache(m, clean, names=["blocks.1", "blocks.0"])
    got = nb.patch_activation(m, corrupt, "blocks.1", cache["blocks.1"], positions=[2, 5])[0]
    assert torch.allclose(got[:, [2, 5]], cl[:, [2, 5]], atol=1e-5), (
        "at the patched positions, the last block's clean output gives the clean logits")
    keep = [0, 1, 3, 4, 6, 7]
    assert torch.allclose(got[:, keep], co[:, keep], atol=1e-5), (
        "positions=[2, 5] patches dim 1 at those positions only; every other position keeps the corrupt value")
    one = nb.patch_activation(m, corrupt, "blocks.0", cache["blocks.0"], positions=4)[0]
    assert torch.allclose(one[:, :4], co[:, :4], atol=1e-5), (
        "patching block 0 at position 4 of a causal model can't change positions 0-3")
    assert not torch.allclose(one[:, 4:], co[:, 4:], atol=1e-4), "...and does change positions 4 onwards"


# ---------------------------------------------------------------- stage 3: patching sweeps


def _metric(answer=3, other=7):
    return lambda out: (out[0][:, -1, answer] - out[0][:, -1, other]).mean()


def test_stage3_activation_patching(nb):
    m = _gpt()
    clean, corrupt = _ids(1), _ids(2)
    metric = _metric()
    names = ["blocks.0", "blocks.1", "ln_f"]
    got = bounded(nb.activation_patching, m, clean, corrupt, metric, names, budget=2_000_000,
                  what="activation_patching")
    assert set(got) == set(names) and all(isinstance(v, float) for v in got.values()), (
        "return {name: float metric value}")
    _, cache = nb.run_with_cache(m, clean, names=names)
    for n in names:
        with torch.no_grad():
            want = float(metric(nb.patch_activation(m, corrupt, n, cache[n])))
        assert np.isclose(got[n], want, atol=1e-5), f"{n}: patch the CLEAN activation into the CORRUPT run"
    assert np.isclose(got["ln_f"], float(metric(m(clean))), atol=1e-5), "patching ln_f restores the clean metric"


def test_stage3_eval_mode(nb):
    m = _gpt(dropout=0.3).train()
    clean, corrupt = _ids(1), _ids(2)
    a = bounded(nb.activation_patching, m, clean, corrupt, _metric(), ["blocks.0"], budget=2_000_000,
                what="activation_patching")
    b = bounded(nb.activation_patching, m, clean, corrupt, _metric(), ["blocks.0"], budget=2_000_000,
                what="activation_patching")
    assert a == b, "with dropout the runs disagree: patch in eval mode"
    assert m.training, "restore the model's mode afterwards"
    pp = bounded(nb.position_patching, m, clean, corrupt, _metric(), ["blocks.0"], budget=5_000_000,
                 what="position_patching")
    pp2 = bounded(nb.position_patching, m, clean, corrupt, _metric(), ["blocks.0"], budget=5_000_000,
                  what="position_patching")
    assert np.allclose(pp, pp2) and m.training, (
        "position_patching also runs in eval mode and restores the mode")


def test_stage3_position_patching(nb):
    m = _gpt()
    clean = _ids(1, B=2)
    corrupt = clean.clone()
    corrupt[:, 3] = (corrupt[:, 3] + 1) % V  # one token differs
    metric = _metric()
    names = ["tok_emb", "blocks.0", "blocks.1"]
    got = bounded(nb.position_patching, m, clean, corrupt, metric, names, budget=5_000_000, what="position_patching")
    assert isinstance(got, np.ndarray) and got.shape == (3, 8), "one row per name, one column per position"
    _, cache = nb.run_with_cache(m, clean, names=names)
    want = float(metric(nb.patch_activation(m, corrupt, "blocks.0", cache["blocks.0"], positions=[5])))
    assert np.isclose(got[1, 5], want, atol=1e-5), "entry [i, t] patches names[i] at position t only"
    corrupt_val, clean_val = float(metric(m(corrupt))), float(metric(m(clean)))
    assert np.allclose(got[:, :3], corrupt_val, atol=1e-5), (
        "before the corrupted token (positions 0-2) the clean and corrupt activations are equal: patching does nothing")
    assert np.isclose(got[0, 3], clean_val, atol=1e-5), (
        "the runs differ only in the token at position 3: patching its clean embedding there makes the whole run clean")
    assert np.isclose(got[2, 7], clean_val, atol=1e-5), (
        "the last block at the last position carries everything the last-position metric depends on")


# ---------------------------------------------------------------- stage 4: attention heads


def test_stage4_attention_patterns(nb):
    m = _gpt(dropout=0.2).train()
    idx = _ids()
    pats = nb.attention_patterns(m, idx)
    assert isinstance(pats, list) and len(pats) == 2 and pats[0].shape == (3, 2, 8, 8), (
        "a list with one (B, H, T, T) tensor per block")
    assert torch.allclose(pats[1].sum(-1), torch.ones(3, 2, 8), atol=1e-5), "rows are softmax distributions"
    assert torch.all(pats[0].triu(1) == 0), "causal: no attention to later positions"
    assert m.training and all(not b.attn.save_weights for b in m.blocks), (
        "restore the model's mode and every attn.save_weights flag afterwards")
    m.eval()
    for b in m.blocks:
        b.attn.save_weights = True
    m(idx)
    assert all(torch.allclose(p, b.attn.last_weights) for p, b in zip(pats, m.blocks)), (
        "the patterns are what each block's attention stores with save_weights=True, in eval mode")
    for b in m.blocks:
        b.attn.save_weights = False
    torch.manual_seed(1)
    llama = Llama(V, block_size=16, n_layers=2, n_heads=4, n_kv_heads=2, d_model=16).eval()
    lp = nb.attention_patterns(llama, idx)
    assert len(lp) == 2 and lp[0].shape == (3, 4, 8, 8), "works for any model with .blocks[i].attn (Llama too)"


def _pattern(T, kind, period=None, H=1, B=2):
    w = torch.zeros(B, H, T, T)
    for t in range(T):
        if kind == "prev":
            w[..., t, max(t - 1, 0)] = 1.0
        elif kind == "uniform":
            w[..., t, :t + 1] = 1.0 / (t + 1)
        elif kind == "induction":
            w[..., t, t - period + 1 if t >= period else 0] = 1.0
        elif kind == "duplicate":
            w[..., t, t - period if t >= period else 0] = 1.0
    return w


def test_stage4_worked_example(nb):
    w = torch.tensor([[1, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0], [0, 0.8, 0.2, 0, 0, 0], [0, 0.9, 0, 0.1, 0, 0],
                      [0, 0, 0.6, 0.2, 0.2, 0], [0, 0, 0, 0.7, 0.1, 0.2]])[None, None]
    prev, ind = nb.prev_token_score(w), nb.induction_score(w, 3)
    assert np.isclose(prev.item(), 0.32), f"the lesson's worked example: previous-token score 0.32, got {prev.item():.4f}"
    assert np.isclose(ind.item(), 2.2 / 3), f"the lesson's worked example: induction score 0.733, got {ind.item():.4f}"


def test_stage4_prev_token_score(nb):
    w = torch.cat([_pattern(6, "prev"), _pattern(6, "uniform")], dim=1)
    got = nb.prev_token_score(w)
    want_uniform = np.mean([1 / (t + 1) for t in range(1, 6)])
    assert got.shape == (2,) and np.isclose(got[0].item(), 1.0) and np.isclose(got[1].item(), want_uniform), (
        f"mean of weights[t, t-1] over t = 1..T-1: 1.0 for a previous-token head, {want_uniform:.4f} for uniform "
        f"attention; got {got.tolist()}. position 0 has no previous token and isn't counted")


def test_stage4_induction_score(nb):
    P, T = 4, 10
    w = torch.cat([_pattern(T, "induction", P), _pattern(T, "duplicate", P), _pattern(T, "prev")], dim=1)
    got = nb.induction_score(w, P)
    assert got.shape == (3,) and np.isclose(got[0].item(), 1.0), (
        f"an induction head at position t >= period attends to t - period + 1 (the token AFTER the previous "
        f"occurrence of the current token): score 1.0; got {got.tolist()}")
    assert np.isclose(got[1].item(), 0.0), (
        "a duplicate-token head (attending to t - period, the previous occurrence itself) scores 0")
    assert np.isclose(got[2].item(), 0.0), "a previous-token head scores 0 (period 4)"
    w2 = _pattern(T, "induction", P)
    w2[..., :P, :] = torch.rand(2, 1, P, T)  # anything in the first period is ignored
    assert np.isclose(nb.induction_score(w2, P)[0].item(), 1.0), "only positions t >= period count"
    assert np.isclose(nb.induction_score(_pattern(T, "induction", 3), 3)[0].item(), 1.0), "the period is a parameter"
    with pytest.raises(ValueError):
        nb.induction_score(w, T)


# ---------------------------------------------------------------- stage 5: ablation


def _zeroed(model, heads):
    m = copy.deepcopy(model)
    for layer, h in heads:
        attn = m.blocks[layer].attn
        proj = attn.out_proj if hasattr(attn, "out_proj") else attn.o_proj
        with torch.no_grad():
            proj.weight[:, h * attn.head_dim:(h + 1) * attn.head_dim] = 0.0
    return m


def test_stage5_ablate_heads(nb):
    m = _gpt()
    idx = _ids()
    before = copy.deepcopy(m.state_dict())
    assert torch.allclose(nb.ablate_heads(m, idx, [])[0], m(idx)[0]), "ablating nothing changes nothing"
    for heads in ([(0, 1)], [(1, 0)], [(0, 0), (1, 1)], [(1, 0), (1, 1)]):
        got = nb.ablate_heads(m, idx, heads)[0]
        want = _zeroed(m, heads)(idx)[0]
        assert torch.allclose(got, want, atol=1e-5), (
            f"ablating {heads}: zero those heads' slices (head h = features h*head_dim:(h+1)*head_dim) of the input "
            "to blocks[layer].attn.out_proj, which is what the block would get if those heads output nothing")
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items()), "don't change the weights"
    assert _no_hooks(m), "remove the hooks afterwards"
    torch.manual_seed(2)
    llama = Llama(V, block_size=16, n_layers=2, n_heads=4, n_kv_heads=2, d_model=16).eval()
    got = nb.ablate_heads(llama, idx, [(1, 3)])[0]
    assert torch.allclose(got, _zeroed(llama, [(1, 3)])(idx)[0], atol=1e-5), "a Llama's heads are read by attn.o_proj"
