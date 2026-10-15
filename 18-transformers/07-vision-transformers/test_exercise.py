"""Checks for 18 · 07 vision transformers.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. patches are checked
against explicit slicing and the patch embedding against a linear layer on patchified pixels (the two
are the same map); the ViT for its tokens, pooling, positions (without them it can't tell where a patch
came from) and for learning a task that needs them; the attention-distance probe against uniform
attention, where the answer is the mean distance between patches.
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


def _warm_up():
    m = nn.Conv2d(1, 2, 2, stride=2)
    opt = torch.optim.AdamW(m.parameters())
    m(torch.zeros(1, 1, 4, 4)).sum().backward()
    opt.step()


_warm_up()


# ---------------------------------------------------------------- stage 1: patches


def test_stage1_patchify(nb):
    x = torch.arange(2 * 3 * 8 * 12, dtype=torch.float32).reshape(2, 3, 8, 12)
    p = nb.patchify(x, 4)
    assert tuple(p.shape) == (2, 6, 48), f"(B, C, H, W) = (2, 3, 8, 12) with p = 4 -> (B, N, C*p*p) = (2, 6, 48); got {tuple(p.shape)}"
    want = x[1, :, 4:8, 8:12].reshape(-1)  # row 1, column 2 of the 2 x 3 patch grid
    assert torch.equal(p[1, 5], want), (
        "patches go in row-major order over the grid (patch 5 = grid row 1, column 2), and each is flattened in "
        "(channel, row, column) order, the same order as a Conv2d weight (d, C, p, p) flattens")
    with pytest.raises(ValueError):
        nb.patchify(torch.zeros(1, 3, 10, 10), 4)


def test_stage1_patch_embedding_is_a_linear_map(nb):
    torch.manual_seed(0)
    pe = nb.PatchEmbedding(16, 4, 3, 10)
    assert isinstance(pe.proj, nn.Conv2d) and pe.proj.kernel_size == (4, 4) and pe.proj.stride == (4, 4), (
        "proj is nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)")
    assert pe.n_patches == 16, f"a 16 x 16 image in 4 x 4 patches has 16 patches; n_patches = {pe.n_patches}"
    x = torch.randn(2, 3, 16, 16)
    out = pe(x)
    assert tuple(out.shape) == (2, 16, 10), f"(B, n_patches, d_model) = (2, 16, 10); got {tuple(out.shape)}"
    lin = nb.patchify(x, 4) @ pe.proj.weight.reshape(10, -1).T + pe.proj.bias
    assert _close(out, lin, atol=1e-5), (
        "the strided convolution is a linear layer applied to each flattened patch: tokens must come out in the "
        "same row-major order as patchify (flatten the spatial grid, then move channels last)")
    with pytest.raises(ValueError):
        nb.PatchEmbedding(30, 4, 3, 8)


# ---------------------------------------------------------------- stage 2: the ViT


def test_stage2_tokens_and_pooling(nb):
    torch.manual_seed(1)
    m = nb.ViT(16, 4, 3, 5, d_model=24, n_layers=2, n_heads=2).eval()
    assert tuple(m.cls_token.shape) == (1, 1, 24) and tuple(m.pos_emb.shape) == (1, 17, 24), (
        "cls_token (1, 1, d) and pos_emb (1, N + 1, d): a learned position for [CLS] and each of the 16 patches")
    assert len(m.blocks) == 2 and all(b.norm_first for b in m.blocks), "n_layers pre-LN TransformerBlocks"
    x = torch.randn(3, 3, 16, 16)
    f = m.forward_features(x)
    assert tuple(f.shape) == (3, 17, 24), f"forward_features -> (B, N + 1, d), [CLS] first; got {tuple(f.shape)}"
    assert _close(f.mean(-1), torch.zeros(3, 17), atol=1e-4) and _close(f.std(-1, unbiased=False), torch.ones(3, 17), atol=1e-2), (
        "forward_features returns the tokens AFTER the final LayerNorm (at initialization each token has mean 0, std 1)")
    out = m(x)
    assert tuple(out.shape) == (3, 5) and _close(out, m.head(f[:, 0])), "pool='cls': logits = head(token 0 after the final norm)"
    mean = nb.ViT(16, 4, 3, 5, d_model=24, n_layers=2, n_heads=2, pool="mean").eval()
    assert mean.cls_token is None and tuple(mean.pos_emb.shape) == (1, 16, 24), "pool='mean' has no [CLS] token"
    fm = mean.forward_features(x)
    assert tuple(fm.shape) == (3, 16, 24) and _close(mean(x), mean.head(fm.mean(1))), "pool='mean' averages the patch tokens"
    feats = nb.ViT(16, 4, 3, 0, d_model=24, n_layers=1, n_heads=2).eval()
    assert tuple(feats(x).shape) == (3, 24), "num_classes=0 returns the pooled features (B, d_model)"
    with pytest.raises(ValueError):
        nb.ViT(16, 4, 3, 5, pool="max")
    n = sum(p.numel() for p in nb.ViT().parameters())
    assert n == 2_693_578, f"worked example: the default ViT() has 2,693,578 parameters; got {n:,}"


def _permute_patches(x, perm, p):
    B, C, H, W = x.shape
    g = H // p
    t = x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C, p, p)[:, perm]
    return t.reshape(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def test_stage2_positions_matter(nb):
    torch.manual_seed(2)
    m = nb.ViT(16, 4, 3, 4, d_model=16, n_layers=2, n_heads=2).eval()
    x = torch.randn(2, 3, 16, 16)
    perm = torch.randperm(16)
    xs = _permute_patches(x, perm, 4)
    assert not _close(m(x), m(xs)), "with learned positions, rearranging the patches changes the prediction"
    with torch.no_grad():
        m.pos_emb.zero_()
    assert _close(m(x), m(xs), atol=1e-5), (
        "with the position embeddings zeroed, a ViT is a set model: rearranging the patches must not change the "
        "output. if it does, something other than the position embedding depends on patch order")


def test_stage2_learns_where(nb):
    """class = which quadrant holds a bright square; the patches' content alone doesn't say where they are."""
    def make(n, g):
        y = torch.randint(0, 4, (n,), generator=g)
        x = 0.3 * torch.randn(n, 1, 16, 16, generator=g)
        for i in range(n):
            r, c = divmod(int(y[i]), 2)
            x[i, :, 8 * r + 2:8 * r + 6, 8 * c + 2:8 * c + 6] += 2.0
        return x, y

    def run(seed):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        m = nb.ViT(16, 4, 1, 4, d_model=32, n_layers=2, n_heads=2)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for _ in range(150):
            x, y = make(32, g)
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        x, y = make(300, g)
        with torch.no_grad():
            return (m(x).argmax(-1) == y).float().mean().item()

    accs = [bounded(run, seed, budget=60_000_000, what="training the ViT") for seed in range(3)]
    assert sorted(accs)[1] > 0.9, (
        f"a 2-layer ViT should learn which quadrant holds a bright square within 150 steps (median accuracy over 3 "
        f"seeds > 0.9); got {[round(a, 2) for a in accs]}. check that the positions are added to the tokens and that "
        f"the head reads the pooled token")


def test_stage2_embedding_dropout(nb):
    torch.manual_seed(7)
    m = nb.ViT(16, 4, 3, 4, d_model=16, n_layers=1, n_heads=2, dropout=0.5).train()
    for mod in m.blocks.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0
    x = torch.randn(2, 3, 16, 16)
    assert not _close(m(x), m(x)), "the ViT applies dropout to the tokens (patches + positions) before the blocks"
    m.eval()
    assert _close(m(x), m(x)), "and not in eval mode"


def test_stage3_runs_in_eval_mode(nb):
    torch.manual_seed(8)
    m = nb.ViT(16, 4, 3, 4, d_model=16, n_layers=2, n_heads=2, dropout=0.5).train()
    x = torch.randn(3, 3, 16, 16)
    a, b = nb.attention_distance(m, x), nb.attention_distance(m, x)
    m.eval()
    c = nb.attention_distance(m, x)
    assert _close(a, b) and _close(a, c), (
        "attention_distance must put the model in eval mode (no dropout on the attention weights) while it measures")
    m.train()


# ---------------------------------------------------------------- stage 3: attention distance


def test_stage3_uniform_attention(nb):
    torch.manual_seed(3)
    m = nb.ViT(16, 4, 3, 4, d_model=16, n_layers=2, n_heads=2).train()
    with torch.no_grad():
        for b in m.blocks:
            for lin in (b.attn.q_proj, b.attn.k_proj):
                lin.weight.zero_()
                lin.bias.zero_()
    d = nb.attention_distance(m, torch.randn(4, 3, 16, 16))
    assert tuple(d.shape) == (2, 2), f"one number per layer and head: (n_layers, n_heads) = (2, 2); got {tuple(d.shape)}"
    rows, cols = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    coords = torch.stack([rows.flatten(), cols.flatten()], 1).float() * 4
    want = torch.cdist(coords, coords).mean()
    assert _close(d, torch.full((2, 2), want.item()), atol=1e-4), (
        f"with zeroed query/key projections every query attends uniformly, so the mean attention distance is the mean "
        f"Euclidean distance between patch centres in pixels ({want:.3f} for a 4 x 4 grid of 4-pixel patches). leave "
        f"out the [CLS] token (as query and as key, renormalizing the weights over patches); got {d}")
    assert m.training, "attention_distance restores the model's mode"
    assert all(not b.attn.save_weights for b in m.blocks), "and turns save_weights back off"


def test_stage3_matches_definition_on_peaked_attention(nb):
    torch.manual_seed(4)
    m = nb.ViT(16, 4, 3, 4, d_model=16, n_layers=2, n_heads=2).eval()
    with torch.no_grad():
        for b in m.blocks:
            b.attn.q_proj.weight.mul_(8.0)  # sharp, position-dependent attention
    x = torch.randn(3, 3, 16, 16)
    got = nb.attention_distance(m, x)
    for b in m.blocks:
        b.attn.save_weights = True
    with torch.no_grad():
        m(x)
    rows, cols = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    coords = torch.stack([rows.flatten(), cols.flatten()], 1).float() * 4
    dist = torch.cdist(coords, coords)
    want = []
    for b in m.blocks:
        w = b.attn.last_weights[:, :, 1:, 1:]
        w = w / w.sum(-1, keepdim=True)
        want.append((w * dist).sum(-1).mean(dim=(0, 2)))
        b.attn.save_weights = False
    want = torch.stack(want)
    assert _close(got, want, atol=1e-4), (
        f"with sharp attention: drop the [CLS] query row AND key column (index 0), renormalize each row over the 16 "
        f"patches, weight the pixel distances between patch positions, sum over keys, average over images and "
        f"queries.\nexpected\n{want}\ngot\n{got}")
