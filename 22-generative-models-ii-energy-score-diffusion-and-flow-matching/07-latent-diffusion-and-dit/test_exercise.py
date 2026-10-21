"""Checks for exercise.py (latent diffusion and DiT).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Patchify is checked element by
element on a hand-made tensor and for being exactly inverted by unpatchify; the 2-D positions against chapter 18's
sinusoidal_positions. The DiT block is checked for starting as the identity (adaLN-Zero), for being steered by the
condition once its modulation weights are nonzero, and for treating tokens as a set (permutation equivariance).
The full DiT is checked for a zero output at initialization, its block count and token count, its use of t, y and
the null label, and for knowing where each patch is. The latent helpers are checked against a recording VAE (means,
not samples; eval mode, restored) and against chapter 21's ConvVAE.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randomize(module, scale=0.2, seed=0):
    g = _gen(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return module


# ---------------------------------------------------------------- stage 1: patches and positions


def test_stage1_patchify_by_hand(nb):
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    tok = nb.patchify(x, 2)
    assert tok.shape == (1, 4, 8), "(B, C, H, W) -> (B, (H/p)(W/p), p*p*C)"
    want = torch.tensor([x[0, c, 0 + i, 2 + j].item() for i in range(2) for j in range(2) for c in range(2)])
    assert torch.equal(tok[0, 1], want), (
        f"patch 1 (row 0, column 1 of the patch grid) should be {want.tolist()} (ordered row-in-patch, column-in-patch, "
        f"channel); got {tok[0, 1].tolist()}")
    want3 = torch.tensor([x[0, c, 2 + i, 2 + j].item() for i in range(2) for j in range(2) for c in range(2)])
    assert torch.equal(tok[0, 3], want3), "patches in raster order: patch 3 is the bottom-right one"
    y = torch.randn(3, 3, 8, 8, generator=_gen(0))
    for p in (1, 2, 4):
        assert torch.equal(nb.unpatchify(nb.patchify(y, p), p, 3), y), f"unpatchify must invert patchify exactly (p={p})"
    with pytest.raises(ValueError):
        nb.patchify(torch.zeros(1, 1, 6, 6), 4)


def test_stage1_pos_embed(nb):
    from mlfs.transformer import sinusoidal_positions

    pe = nb.pos_embed_2d(3, 8)
    one = sinusoidal_positions(3, 4)
    assert pe.shape == (9, 8), "(grid_size^2, d_model)"
    assert torch.allclose(pe[5], torch.cat([one[1], one[2]]), atol=1e-6), (
        "token 5 is row 1, column 2 of a 3 x 3 grid: [positions(3, d/2)[1], positions(3, d/2)[2]]")
    assert len({tuple(r.round(decimals=5).tolist()) for r in pe}) == 9, "every grid position gets its own embedding"


# ---------------------------------------------------------------- stage 2: the adaLN-Zero block


def test_stage2_block_identity_at_init(nb):
    torch.manual_seed(1)
    blk = nb.DiTBlock(16, 2)
    x = torch.randn(2, 5, 16)
    c = torch.randn(2, 16)
    assert torch.allclose(blk(x, c), x, atol=1e-6), (
        "adaLN-Zero: the modulation layer starts at zero, so every gate is 0 and the block is the identity")
    zero_init = [m for m in blk.modules() if isinstance(m, torch.nn.Linear) and m.out_features == 6 * 16]
    assert len(zero_init) == 1 and zero_init[0].weight.abs().sum() == 0, "a Linear(d, 6d) producing (shift, scale, gate) x 2"


def test_stage2_block_conditioning_and_equivariance(nb):
    torch.manual_seed(2)
    blk = _randomize(nb.DiTBlock(16, 2), seed=3)
    x = torch.randn(2, 6, 16)
    c = torch.randn(2, 16)
    out = blk(x, c)
    assert out.shape == x.shape and not torch.allclose(out, x, atol=1e-4)
    assert not torch.allclose(out, blk(x, c + 1.0), atol=1e-4), "the condition must steer the block"
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    assert torch.allclose(blk(x[:, perm], c), out[:, perm], atol=1e-5), (
        "attention + per-token MLP + per-example modulation: permuting tokens permutes the output (no positions inside)")
    c2 = c.clone()
    c2[1] += 1.0
    out2 = blk(x, c2)
    assert torch.allclose(out2[0], out[0], atol=1e-5) and not torch.allclose(out2[1], out[1], atol=1e-4), (
        "each example is modulated by its own condition: changing c[1] changes only example 1")
    ada = [m for m in blk.modules() if isinstance(m, torch.nn.Linear) and m.out_features == 6 * 16][0]
    with torch.no_grad():
        ada.weight.zero_()
        ada.bias.copy_(torch.cat([torch.zeros(16), -torch.ones(16), torch.ones(16)] * 2))  # (shift, scale, gate) x 2
    xa, xb = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    da, db = blk(xa, c) - xa, blk(xb, c) - xb
    assert torch.allclose(da, da[:, :1].expand_as(da), atol=1e-5) and torch.allclose(da, db, atol=1e-5), (
        "with scale = -1, shift = 0 and gate = 1 the modulated input LN(h) (1 + scale) + shift is 0, so each branch adds "
        "the same vector to every token whatever x is: modulate as LN(h) * (1 + scale) + shift, chunks ordered (shift, "
        "scale, gate) for attention, then for the MLP")
    ln = [m for m in blk.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(ln) == 2 and all(not m.elementwise_affine for m in ln), (
        "two LayerNorms without their own scale and shift (adaLN supplies them)")


# ---------------------------------------------------------------- stage 3: DiT


def test_stage3_dit_structure(nb):
    torch.manual_seed(4)
    m = nb.DiT(8, 2, 4, d_model=32, n_layers=3, n_heads=2)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    out = m(x, t)
    assert out.shape == x.shape, "the output has the input's shape"
    assert torch.all(out == 0), "adaLN-Zero and a zero-initialized output layer: a fresh DiT outputs exactly zero"
    assert isinstance(m.blocks, torch.nn.ModuleList) and len(m.blocks) == 3, "self.blocks: n_layers DiTBlocks"
    seen = []
    m.blocks[0].register_forward_pre_hook(lambda mod, inp: seen.append(tuple(inp[0].shape)))
    m(x, t)
    nb.DiT(8, 4, 4, d_model=32, n_layers=1, n_heads=2).blocks[0].register_forward_pre_hook(
        lambda mod, inp: seen.append(tuple(inp[0].shape)))
    assert seen[0] == (2, 16, 32), "8 x 8 inputs with patch_size 2 are 16 tokens of width d_model"
    small = sum(p.numel() for p in nb.DiT(8, 2, 4, d_model=32, n_layers=1, n_heads=2).parameters())
    big = sum(p.numel() for p in nb.DiT(8, 2, 4, d_model=32, n_layers=2, n_heads=2).parameters())
    assert big > small, "n_layers must add blocks"
    assert nb.DiT(4, 1, 2, d_model=16, n_layers=1, n_heads=2)(torch.randn(1, 2, 4, 4), torch.rand(1)).shape == (1, 2, 4, 4)
    pos = [b for b in m.buffers() if tuple(b.shape) == (16, 32)]
    assert pos and not any(p.shape == pos[0].shape and torch.equal(p, pos[0]) for p in m.parameters()), (
        "the 2-D positional embedding (16 tokens x 32) is a fixed buffer (register_buffer), not a trained parameter")


def test_stage3_dit_final_layer(nb):
    torch.manual_seed(9)
    m = nb.DiT(8, 2, 4, d_model=32, n_layers=2, n_heads=2)
    x = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        for p in m.out.parameters():
            p.normal_(0, 0.2, generator=_gen(10))
    a, b = m(x, torch.full((2,), 0.1)), m(x, torch.full((2,), 0.9))
    assert a.abs().sum() > 0 and torch.allclose(a, b, atol=1e-6), (
        "with only `out` randomized, the blocks (zero gates) and `ada_out` (zero-initialized) ignore t, so the output must "
        "be nonzero but the same for every t: zero-initialize ada_out")
    with torch.no_grad():
        for p in m.ada_out.parameters():
            p.normal_(0, 0.2, generator=_gen(11))
    assert not torch.allclose(m(x, torch.full((2,), 0.1)), m(x, torch.full((2,), 0.9)), atol=1e-5), (
        "once ada_out is nonzero the output must depend on t through the final layer: modulate the final LayerNorm with "
        "ada_out's (shift, scale)")


def test_stage3_dit_conditioning_and_positions(nb):
    torch.manual_seed(5)
    m = _randomize(nb.DiT(8, 2, 4, d_model=32, n_layers=2, n_heads=2, n_classes=3), seed=6)
    x = torch.randn(3, 4, 8, 8)
    t = torch.full((3,), 0.3)
    y = torch.tensor([0, 1, 2])
    out = m(x, t, y)
    assert not torch.allclose(out, m(x, torch.full((3,), 0.7), y), atol=1e-4), "the output must depend on t"
    assert not torch.allclose(out, m(x, t, torch.tensor([1, 2, 0])), atol=1e-4), "the output must depend on y"
    assert torch.allclose(m(x, t), m(x, t, torch.full((3,), 3)), atol=1e-5), "y=None means the null label n_classes"
    tables = [mod for mod in m.modules() if isinstance(mod, torch.nn.Embedding)]
    assert len(tables) == 1 and tables[0].num_embeddings == 4, "one class table with n_classes + 1 rows"
    with pytest.raises(ValueError):
        nb.DiT(8, 2, 4, d_model=32, n_layers=1, n_heads=2)(x, t, y)
    swapped = x.clone()
    swapped[:, :, 0:2, 0:2], swapped[:, :, 6:8, 6:8] = x[:, :, 6:8, 6:8], x[:, :, 0:2, 0:2]
    o2 = m(swapped, t, y)
    o2_back = o2.clone()
    o2_back[:, :, 0:2, 0:2], o2_back[:, :, 6:8, 6:8] = o2[:, :, 6:8, 6:8], o2[:, :, 0:2, 0:2]
    assert not torch.allclose(o2_back, out, atol=1e-4), (
        "swapping two patches must not just swap the outputs: the DiT needs to know where each patch is (add the "
        "2-D positional embedding to the patch tokens)")


# ---------------------------------------------------------------- stage 4: latents


class _RecVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.modes = []

    def encode(self, x):
        self.modes.append(self.training)
        mu = x[:, :, ::4, ::4] * 2.0
        return mu, torch.full_like(mu, -1.0)

    def decode(self, z):
        self.modes.append(self.training)
        return z.repeat_interleave(4, -1).repeat_interleave(4, -2) / 2.0


def test_stage4_encode_latents(nb):
    from mlfs.torchnn import make_loader

    X = np.random.default_rng(7).normal(size=(10, 1, 8, 8)).astype(np.float32)
    y = np.arange(10)
    vae = _RecVAE()
    z, labels = bounded(nb.encode_latents, vae, make_loader(X, y, batch_size=4, shuffle=False), budget=500_000,
                        what="encode_latents", hint="one pass over the loader")
    assert z.shape == (10, 1, 2, 2) and torch.allclose(z, torch.tensor(X)[:, :, ::4, ::4] * 2), (
        "latents are the encoder MEANS mu, in loader order (no sampling)")
    assert labels.dtype == torch.int64 and labels.tolist() == list(range(10)), "labels (N,) int64, in loader order"
    assert vae.modes and not any(vae.modes) and vae.training, "eval mode while encoding, then restore the mode"
    assert not z.requires_grad
    vae.modes.clear()
    back = bounded(nb.decode_latents, vae, z, scale=0.5, batch_size=3, budget=500_000, what="decode_latents")
    assert torch.allclose(back, (z * 0.5).repeat_interleave(4, -1).repeat_interleave(4, -2) / 2), (
        "decode_latents = vae.decode(z * scale), batched")
    assert len(vae.modes) == 4 and not any(vae.modes) and vae.training, "batches of batch_size, eval mode, restored"


def test_stage4_with_conv_vae(nb):
    from mlfs.generative import ConvVAE
    from mlfs.torchnn import make_loader

    torch.manual_seed(8)
    vae = ConvVAE(3, latent_channels=4, base_channels=8)
    X = np.random.default_rng(9).random((6, 3, 16, 16)).astype(np.float32)
    z, labels = bounded(nb.encode_latents, vae, make_loader(X, np.zeros(6, dtype=np.int64), batch_size=4, shuffle=False),
                        budget=3_000_000, what="encode_latents")
    assert not z.requires_grad, "encode under torch.no_grad(): the latents carry no graph"
    assert z.shape == (6, 4, 4, 4), "a ConvVAE's latents are (N, latent_channels, H/4, W/4)"
    assert bounded(nb.decode_latents, vae, z, budget=3_000_000, what="decode_latents").shape == (6, 3, 16, 16)
