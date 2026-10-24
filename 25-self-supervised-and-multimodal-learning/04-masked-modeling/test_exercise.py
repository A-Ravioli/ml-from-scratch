"""Checks for exercise.py (masked modeling).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Patchify is checked
against an explicit loop; random masking against the exact draws of a seeded generator; the MAE's
encoder and full forward pass against step-by-step recomputations from its own submodules (with
hooks confirming the encoder sees only the visible patches); the training loop against an exact
replay of the specified AdamW loop.
"""
import math

import numpy as np
import pytest
import torch
from torch import nn

from mlfs.transformer import TransformerBlock, sinusoidal_positions

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


# ---------------------------------------------------------------- stage 1: patchify, unpatchify

def _loop_patchify(x, p):
    B, C, H, W = x.shape
    out = torch.zeros(B, (H // p) * (W // p), p * p * C)
    for b in range(B):
        for gi in range(H // p):
            for gj in range(W // p):
                k = gi * (W // p) + gj
                col = 0
                for i in range(p):
                    for j in range(p):
                        for c in range(C):
                            out[b, k, col] = x[b, c, gi * p + i, gj * p + j]
                            col += 1
    return out


def test_stage1_matches_loop(nb):
    x = torch.arange(2 * 2 * 4 * 6, dtype=torch.float32).reshape(2, 2, 4, 6)
    got = nb.patchify(x, 2)
    assert got.shape == (2, 6, 8), f"(B, C, H, W) = (2, 2, 4, 6), p=2 -> (B, N, p*p*C) = (2, 6, 8), got {tuple(got.shape)}"
    want = _loop_patchify(x, 2)
    assert torch.equal(got, want), (
        "patch order or within-patch order is wrong. patches go row by row over the image grid; inside a patch the "
        "values are ordered (row i, column j, channel c) with the channel varying fastest (MAE's order). note that "
        "ch 18's patchify used (c, i, j): here the channel comes LAST")


def test_stage1_roundtrip_and_channels(nb):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 3, 8, 8, generator=g)
    got = nb.unpatchify(nb.patchify(x, 4), 4)
    assert torch.equal(got, x), "unpatchify(patchify(x)) must give x back exactly"
    x1 = torch.rand(2, 1, 12, 12, generator=g)
    p = nb.patchify(x1, 3)
    assert p.shape == (2, 16, 9)
    assert torch.equal(nb.unpatchify(p, 3, channels=1), x1), "unpatchify must honour `channels` (MNIST: channels=1)"
    assert torch.equal(nb.unpatchify(_loop_patchify(x, 2), 2), x), "unpatchify must invert the MAE order"
    with pytest.raises(ValueError):
        nb.patchify(torch.zeros(1, 3, 10, 10), 4)


# ---------------------------------------------------------------- stage 2: random_masking

def _expected_masking(x, ratio, noise):
    B, N, D = x.shape
    n_keep = math.floor(N * (1 - ratio) + 0.5)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    keep = ids_shuffle[:, :n_keep]
    vis = torch.gather(x, 1, keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N)
    mask[:, :n_keep] = 0
    return vis, torch.gather(mask, 1, ids_restore), ids_restore


def test_stage2_exact_draws(nb):
    x = torch.randn(3, 10, 4, generator=torch.Generator().manual_seed(1))
    for ratio in (0.75, 0.3, 0.0):
        vis, mask, ids = nb.random_masking(x, ratio, generator=torch.Generator().manual_seed(7))
        noise = torch.rand(3, 10, generator=torch.Generator().manual_seed(7))
        w_vis, w_mask, w_ids = _expected_masking(x, ratio, noise)
        assert vis.shape == w_vis.shape, (
            f"mask_ratio={ratio}, N=10: x_visible should be {tuple(w_vis.shape)}, got {tuple(vis.shape)}")
        assert ids.dtype == torch.int64 and torch.equal(ids, w_ids), (
            "ids_restore is wrong: draw noise = torch.rand(B, N, generator=generator), shuffle by argsort(noise) "
            "(ascending: the smallest noise values are KEPT), and ids_restore = argsort(ids_shuffle)")
        assert torch.equal(vis, w_vis), "x_visible must be the kept tokens, in shuffled order"
        assert mask.dtype.is_floating_point and torch.equal(mask, w_mask), "mask: float, 1 = masked, 0 = visible"
    torch.manual_seed(3)
    vis, mask, ids = nb.random_masking(x, 0.5)
    torch.manual_seed(3)
    w = _expected_masking(x, 0.5, torch.rand(3, 10))
    assert torch.equal(ids, w[2]), "with generator=None, draw the noise from torch's global generator"


def test_stage2_rounding_and_errors(nb):
    for N, ratio, keep in ((10, 0.9, 1), (64, 0.75, 16), (10, 1 / 3, 7), (49, 0.75, 12), (16, 0.6, 6)):
        vis, mask, _ = nb.random_masking(torch.zeros(2, N, 3), ratio, generator=torch.Generator().manual_seed(0))
        assert vis.shape[1] == keep and (mask.sum(1) == N - keep).all(), (
            f"N={N}, mask_ratio={ratio:.3f}: keep round(N * (1 - mask_ratio)) = {keep} patches, kept {vis.shape[1]}. "
            "careful: int(10 * (1 - 0.9)) is 0, because 1 - 0.9 is 0.0999... in floating point")
    with pytest.raises(ValueError):
        nb.random_masking(torch.zeros(1, 8, 2), 1.0)


def test_stage2_restore_and_independence(nb):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(64, 16, 5, generator=g)
    vis, mask, ids = nb.random_masking(x, 0.75, generator=g)
    full = torch.cat([vis, torch.zeros(64, 12, 5)], dim=1)
    restored = torch.gather(full, 1, ids.unsqueeze(-1).expand(-1, -1, 5))
    keep = mask == 0
    assert torch.equal(restored[keep], x[keep]), (
        "appending placeholders to x_visible and gathering with ids_restore must put every visible token back at its "
        "original position, where mask == 0")
    assert (restored[~keep] == 0).all()
    assert len({tuple(m.tolist()) for m in mask}) > 50, "every sample needs its own random mask"
    freq = mask.mean(0)
    assert freq.min() > 0.4 and freq.max() < 1.0, "every position should be masked sometimes, and visible sometimes"


def test_stage2_device(nb):
    x = torch.empty(2, 8, 3, device="meta")  # a second device without needing a GPU
    vis, mask, ids = nb.random_masking(x, 0.5, generator=torch.Generator().manual_seed(0))
    assert vis.device.type == mask.device.type == ids.device.type == "meta", (
        "draw the noise on the CPU (where the generator lives) and move it to x's device; build the mask there too")
    assert vis.shape == (2, 4, 3)


# ---------------------------------------------------------------- stage 3: MAE encoder

def _tiny(nb, seed=0, **kw):
    torch.manual_seed(seed)
    args = dict(encoder_dim=16, encoder_depth=2, encoder_heads=2, decoder_dim=12, decoder_depth=1, decoder_heads=3)
    args.update(kw)
    return nb.MAE(8, 2, 3, **args)


def test_stage3_structure(nb):
    m = _tiny(nb)
    assert isinstance(m.patch_embed, nn.Linear) and (m.patch_embed.in_features, m.patch_embed.out_features) == (12, 16), (
        "patch_embed: nn.Linear(p*p*C, encoder_dim), applied to patchify(x)")
    assert isinstance(m.blocks, nn.ModuleList) and len(m.blocks) == 2 and all(isinstance(b, TransformerBlock) for b in m.blocks), (
        "blocks: an nn.ModuleList of encoder_depth TransformerBlocks")
    assert isinstance(m.norm, nn.LayerNorm)
    names = dict(m.named_buffers())
    assert "pos_embed" in names and "pos_embed" not in dict(m.named_parameters()), "pos_embed is a fixed buffer"
    torch.testing.assert_close(m.pos_embed, sinusoidal_positions(16, 16).unsqueeze(0),
                               msg="pos_embed = sinusoidal_positions(N, encoder_dim), shape (1, N, encoder_dim)")
    torch.testing.assert_close(m.decoder_pos_embed, sinusoidal_positions(16, 12).unsqueeze(0))
    assert m.mask_token.shape == (1, 1, 12) and isinstance(m.mask_token, nn.Parameter), "mask_token: (1, 1, decoder_dim)"
    assert len(m.decoder_blocks) == 1 and (m.decoder_embed.in_features, m.decoder_embed.out_features) == (16, 12)
    assert (m.decoder_pred.in_features, m.decoder_pred.out_features) == (12, 12), "decoder_pred: decoder_dim -> p*p*C"
    assert isinstance(m.decoder_norm, nn.LayerNorm)
    big = _tiny(nb, 0, decoder_dim=2048, decoder_heads=4)
    sd = big.mask_token.detach().std().item()
    assert 0.017 < sd < 0.023 and abs(big.mask_token.detach().mean().item()) < 0.003, (
        f"mask_token's values have std {sd:.4f}: initialise it from a normal distribution with std 0.02")
    with pytest.raises(ValueError):
        nb.MAE(10, 4, 3)


def test_stage3_encode(nb):
    m = _tiny(nb, 1)
    x = torch.rand(5, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    m.blocks[0].attn.save_weights = True
    got = m.encode(x)
    assert got.shape == (5, 16), f"encode returns (B, encoder_dim) = (5, 16), got {tuple(got.shape)}"
    assert m.blocks[0].attn.last_weights.shape == (5, 2, 16, 16), (
        "encode runs the encoder on ALL 16 patches (no masking), with encoder_heads=2 heads")
    t = m.patch_embed(nb.patchify(x, 2)) + m.pos_embed
    for b in m.blocks:
        t = b(t)
    want = m.norm(t).mean(1)
    torch.testing.assert_close(got, want, msg="encode = mean over tokens of norm(blocks(patch_embed(patchify(x)) + pos_embed))")
    torch.testing.assert_close(m.encode(x), got, msg="encode must be deterministic (no masking)")


# ---------------------------------------------------------------- stage 4: MAE forward

def _replay_forward(nb, m, x, seed):
    torch.manual_seed(seed)
    patches = nb.patchify(x, m.patch_size)
    t = m.patch_embed(patches) + m.pos_embed
    t, mask, ids = nb.random_masking(t, m.mask_ratio)
    for b in m.blocks:
        t = b(t)
    y = m.decoder_embed(m.norm(t))
    B, k, D = y.shape
    y = torch.cat([y, m.mask_token.expand(B, patches.shape[1] - k, D)], 1)
    y = torch.gather(y, 1, ids.unsqueeze(-1).expand(-1, -1, D)) + m.decoder_pos_embed
    for b in m.decoder_blocks:
        y = b(y)
    return m.decoder_pred(m.decoder_norm(y)), mask, patches


def test_stage4_forward_matches_replay(nb):
    m = _tiny(nb, 2, mask_ratio=0.5)
    x = torch.rand(4, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(11)
    loss, pred, mask = m(x)
    assert pred.shape == (4, 16, 12) and mask.shape == (4, 16), "forward returns (loss, pred (B, N, p*p*C), mask (B, N))"
    assert torch.equal(mask.sum(1), torch.full((4,), 8.0)), "mask_ratio=0.5 of 16 patches: 8 masked per image"
    want_pred, want_mask, patches = _replay_forward(nb, m, x, 11)
    assert torch.equal(mask, want_mask), (
        "the mask should come from random_masking(tokens, mask_ratio) with torch's global generator, the only random "
        "draw in forward")
    torch.testing.assert_close(pred, want_pred, atol=1e-5, rtol=1e-4, msg=(
        "pred differs from the specified pipeline: patch_embed + pos_embed -> random_masking -> blocks -> norm -> "
        "decoder_embed -> append mask tokens -> UNSHUFFLE with ids_restore -> + decoder_pos_embed -> decoder_blocks -> "
        "decoder_norm -> decoder_pred"))
    want_loss = (((pred - patches) ** 2).mean(-1) * mask).sum() / mask.sum()
    assert abs(loss.item() - want_loss.item()) < 1e-6, (
        f"loss {loss.item():.6f}, expected {want_loss.item():.6f}: the mean squared error per patch, averaged over "
        "the MASKED patches only")


def test_stage4_encoder_sees_only_visible_patches(nb):
    m = _tiny(nb, 3)  # mask_ratio 0.75: 4 of 16 patches visible
    shapes = {}

    def record(name):
        def hook(mod, inp, out):
            shapes.setdefault(name, tuple(inp[0].shape))
        return hook

    m.blocks[0].register_forward_hook(record("enc"))
    m.decoder_blocks[0].register_forward_hook(record("dec"))
    x = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(2))
    m.decoder_blocks[0].attn.save_weights = True
    torch.manual_seed(5)
    _, pred, mask = m(x)
    assert m.decoder_blocks[0].attn.last_weights.shape == (2, 3, 16, 16), "the decoder blocks use decoder_heads=3 heads"
    m.decoder_blocks[0].attn.save_weights = False
    assert shapes["enc"] == (2, 4, 16), (
        f"the encoder's blocks should see only the 4 visible tokens, (B, 4, encoder_dim); they saw {shapes['enc']}. "
        "drop the masked tokens before the encoder, don't replace them")
    assert shapes["dec"] == (2, 16, 12), f"the decoder sees all N tokens, (B, 16, decoder_dim); it saw {shapes['dec']}"
    hidden = mask.bool().reshape(2, 4, 4)  # (B, gh, gw) patch grid, row-major like patchify
    pix = hidden.repeat_interleave(2, 1).repeat_interleave(2, 2).unsqueeze(1).expand(-1, 3, -1, -1)
    x2 = torch.where(pix, torch.rand_like(x), x)  # change only pixels in masked patches
    torch.manual_seed(5)
    _, pred2, mask2 = m(x2)
    assert torch.equal(mask, mask2)
    torch.testing.assert_close(pred2, pred, msg="changing only the masked patches must not change any prediction")


def test_stage4_norm_pix_loss_and_ratio(nb):
    m = _tiny(nb, 4, norm_pix_loss=True, mask_ratio=0.25)
    x = torch.rand(3, 3, 8, 8, generator=torch.Generator().manual_seed(3)) * 5
    torch.manual_seed(1)
    loss, pred, mask = m(x)
    assert torch.equal(mask.sum(1), torch.full((3,), 4.0)), "mask_ratio=0.25 of 16 patches: 4 masked"
    t = nb.patchify(x, 2)
    t = (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True) + 1e-6) ** 0.5
    want = (((pred - t) ** 2).mean(-1) * mask).sum() / mask.sum()
    assert abs(loss.item() - want.item()) < 1e-5, (
        f"norm_pix_loss=True: loss {loss.item():.5f}, expected {want.item():.5f}. normalise each TARGET patch by its "
        "own mean and (unbiased) variance: (t - mean) / sqrt(var + 1e-6)")
    loss.backward()
    assert m.mask_token.grad is not None and m.patch_embed.weight.grad is not None


# ---------------------------------------------------------------- stage 5: train_mae

def _images(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 3, 8, 8), dtype=np.float32)
    for i in range(n):  # smooth blobs: predictable from their neighbourhood
        cx, cy = rng.uniform(1, 7, size=2)
        yy, xx = np.mgrid[0:8, 0:8]
        X[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 4.0)[None] * rng.uniform(0.3, 1, size=(3, 1, 1))
    return X


def test_stage5_matches_replay(nb):
    X = _images(40, 0)
    m = _tiny(nb, 5, mask_ratio=0.5)
    m.eval()
    h = nb.train_mae(m, X, epochs=2, batch_size=16, lr=3e-3, weight_decay=0.1, seed=9)
    assert m.training, "train_mae must put the model in train mode (model.train() each epoch)"
    assert len(h["loss"]) == 4, f"2 epochs of 40 images in batches of 16 (full batches only) = 4 steps, got {len(h['loss'])}"
    ref = _tiny(nb, 5, mask_ratio=0.5)
    torch.manual_seed(9)
    rng = np.random.default_rng(9)
    opt = torch.optim.AdamW(ref.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    Xt = torch.as_tensor(X)
    losses = []
    for _ in range(2):
        ref.train()
        perm = rng.permutation(40)
        for s in range(0, 40 - 16 + 1, 16):
            loss = ref(Xt[perm[s:s + 16]])[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert np.allclose(h["loss"], losses, rtol=1e-4, atol=1e-6), (
        f"losses {np.round(h['loss'], 5)} differ from the specified loop {np.round(losses, 5)}: torch.manual_seed(seed) "
        "first, AdamW(lr, betas=(0.9, 0.95), weight_decay), a fresh permutation per epoch from "
        "np.random.default_rng(seed), full batches only")
    for a, b in zip(m.parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4, msg="the trained weights differ from the specified loop")


def test_stage5_learns(nb):
    drops = []
    for seed in range(3):
        m = _tiny(nb, seed, mask_ratio=0.5)
        h = nb.train_mae(m, _images(128, seed), epochs=6, batch_size=32, lr=3e-3, seed=seed)
        drops.append(np.mean(h["loss"][:3]) / np.mean(h["loss"][-3:]))
    assert np.median(drops) > 2, f"the reconstruction loss fell by a factor of {np.round(drops, 2)} (3 seeds); expected > 2"


class _DeviceSpy(nn.Module):
    """records where batches and the model are sent; computes on the CPU whatever the inputs' device."""

    def __init__(self, out=4):
        super().__init__()
        self.lin = nn.Linear(3, out)
        self.moved_to, self.seen = [], []

    def to(self, *args, **kwargs):
        self.moved_to.append(str(args[0] if args else kwargs.get("device")))
        return self

    def _out(self, x):
        self.seen.append(x.device.type)
        return self.lin(torch.ones(x.shape[0], 3))


def test_stage5_device(nb):
    class Spy(_DeviceSpy):
        def forward(self, x):
            out = self._out(x)
            return out.pow(2).mean(), out, None

    spy = Spy()
    nb.train_mae(spy, _images(32, 1), epochs=1, batch_size=16, device="meta")
    assert spy.moved_to == ["meta"], f"move the model to `device` with model.to(device); saw {spy.moved_to}"
    assert spy.seen == ["meta", "meta"], "move every batch to `device`"
