"""Checks for exercise.py (multimodal learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The CLIP loss is checked
by hand, against an independent formula on asymmetric data and with gradcheck; the text encoder for
padding invariance, order sensitivity and against a recomputation from its submodules; MiniCLIP's
projections and temperature exactly; zero-shot prediction and prompt ensembling with hand-built
embeddings whose norms would mislead a dot product; the training loop against an exact replay.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mlfs.transformer import TransformerBlock, sinusoidal_positions

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


def _ref_clip(img, txt, scale):
    img, txt = F.normalize(img, dim=1), F.normalize(txt, dim=1)
    logits = scale * img @ txt.T
    t = torch.arange(len(img))
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2


# ---------------------------------------------------------------- stage 1: clip_loss

def test_stage1_worked_example(nb):
    e = torch.eye(2)
    got = nb.clip_loss(e, e, 1.0)
    want = math.log(1 + math.exp(-1))
    assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor"
    assert abs(got.item() - want) < 1e-6, (
        f"lesson worked example: two orthogonal matching pairs, logit_scale 1: loss {got.item():.4f}, expected "
        f"log(1 + e^-1) = {want:.4f}")
    got = nb.clip_loss(e, e, 10.0).item()
    assert abs(got - math.log(1 + math.exp(-10))) < 1e-6, "logit_scale multiplies the cosine similarities"
    img = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    txt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    want = _ref_clip(img, txt, 1.0).item()
    assert abs(nb.clip_loss(img, txt, 1.0).item() - want) < 1e-6, (
        f"two identical images with different captions: expected {want:.4f}. average the image->text cross-entropy "
        "(over rows) and the text->image one (over columns)")


def test_stage1_matches_formula(nb):
    g = torch.Generator().manual_seed(0)
    for b, d, s in ((5, 4, 2.0), (7, 3, 14.3), (3, 6, 50.0)):
        img, txt = torch.randn(b, d, generator=g) * 3, torch.randn(b, d, generator=g)
        got = nb.clip_loss(img, txt, s)
        want = _ref_clip(img, txt, s)
        assert abs(got.item() - want.item()) < 1e-5, (
            f"B={b}, logit_scale={s}: loss {got.item():.5f}, expected {want.item():.5f}: normalise both, logits = "
            "logit_scale * img @ txt.T, targets on the diagonal, mean of the row-wise and column-wise cross-entropies")
    img, txt = torch.randn(6, 4, generator=g), torch.randn(6, 4, generator=g)
    a = nb.clip_loss(img, txt, 5.0).item()
    assert abs(nb.clip_loss(txt, img, 5.0).item() - a) < 1e-6, "the loss is symmetric in images and texts"
    assert abs(nb.clip_loss(img * 7, txt * 0.1, 5.0).item() - a) < 1e-5, "use cosine similarities: scale must not matter"


def test_stage1_gradients(nb):
    g = torch.Generator().manual_seed(1)
    img = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    txt = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, s: nb.clip_loss(a, b, s), (img, txt, scale), eps=1e-6, atol=1e-5), (
        "the gradient doesn't match finite differences")
    nb.clip_loss(img, txt, scale).backward()
    assert scale.grad is not None and scale.grad.abs() > 0, "a tensor logit_scale must receive a gradient"


# ---------------------------------------------------------------- stage 2: TextTransformer

def test_stage2_structure_and_recompute(nb):
    torch.manual_seed(0)
    enc = nb.TextTransformer(20, max_len=8, d_model=16, n_layers=2, n_heads=4)
    assert isinstance(enc.tok_emb, nn.Embedding) and enc.tok_emb.num_embeddings == 20
    assert "pos_emb" in dict(enc.named_buffers()), "pos_emb is a fixed buffer"
    torch.testing.assert_close(enc.pos_emb, sinusoidal_positions(8, 16), msg="pos_emb = sinusoidal_positions(max_len, d_model)")
    assert len(enc.blocks) == 2 and all(isinstance(b, TransformerBlock) for b in enc.blocks)
    assert isinstance(enc.ln_f, nn.LayerNorm)
    tok = torch.tensor([[3, 4, 5, 0, 0], [7, 8, 9, 10, 11]])
    enc.blocks[0].attn.save_weights = True
    got = enc(tok)
    assert got.shape == (2, 16), f"output (B, d_model) = (2, 16), got {tuple(got.shape)}"
    w = enc.blocks[0].attn.last_weights
    assert w.shape == (2, 4, 5, 5), "n_heads=4 heads of self-attention over the T tokens"
    assert w[0, :, :, 3:].abs().max() < 1e-6, "no token may attend to padding: pass a key mask (B, 1, 1, T) to the blocks"
    real = tok != 0
    x = enc.tok_emb(tok) + enc.pos_emb[:5]
    for b in enc.blocks:
        x = b(x, mask=real[:, None, None, :])
    x = enc.ln_f(x)
    want = (x * real[..., None]).sum(1) / real.sum(1, keepdim=True)
    torch.testing.assert_close(got, want, msg="output = mean over REAL tokens of ln_f(blocks(tok_emb + pos_emb))")


def test_stage2_padding_and_order(nb):
    torch.manual_seed(1)
    enc = nb.TextTransformer(12, max_len=6, d_model=8, n_layers=1, n_heads=2)
    short = enc(torch.tensor([[5, 6, 7]]))
    padded = enc(torch.tensor([[5, 6, 7, 0, 0, 0]]))
    torch.testing.assert_close(short, padded, msg=(
        "padding changed the output: pads must be ignored by attention (key mask) AND by the mean pooling"))
    assert not torch.allclose(enc(torch.tensor([[5, 6]])), enc(torch.tensor([[6, 5]]))), (
        "'red seven' and 'seven red' must differ: add the positional embeddings")
    enc3 = nb.TextTransformer(12, max_len=6, d_model=8, n_layers=1, n_heads=2, pad_id=3)
    torch.testing.assert_close(enc3(torch.tensor([[5, 6, 3, 3]])), enc3(torch.tensor([[5, 6]])),
                               msg="use pad_id (here 3) to find the padding")
    with pytest.raises(ValueError):
        enc(torch.zeros(1, 7, dtype=torch.int64) + 2)


# ---------------------------------------------------------------- stage 3: MiniCLIP

class _Lin(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.lin(x.float().flatten(1)))


def test_stage3_projections_and_temperature(nb):
    torch.manual_seed(2)
    ienc, tenc = _Lin(6, 5), _Lin(4, 3)
    m = nb.MiniCLIP(ienc, tenc, 5, 3, embed_dim=7)
    assert m.image_proj.weight.shape == (7, 5) and m.image_proj.bias is None, "image_proj: Linear(image_dim, embed_dim, bias=False)"
    assert m.text_proj.weight.shape == (7, 3) and m.text_proj.bias is None, "text_proj: Linear(text_dim, embed_dim, bias=False)"
    assert isinstance(m.log_scale, nn.Parameter) and abs(m.log_scale.item() - math.log(1 / 0.07)) < 1e-6, (
        "log_scale is a Parameter initialised to log(1 / init_temperature)")
    x, t = torch.randn(4, 6), torch.randn(4, 4)
    img, txt, scale = m(x, t)
    torch.testing.assert_close(img, F.normalize(m.image_proj(ienc(x)), dim=1))
    torch.testing.assert_close(txt, F.normalize(m.text_proj(tenc(t)), dim=1))
    torch.testing.assert_close(m.encode_image(x), img)
    torch.testing.assert_close(m.encode_text(t), txt)
    assert abs(scale.item() - 1 / 0.07) < 1e-4, "logit_scale = exp(log_scale)"
    m2 = nb.MiniCLIP(ienc, tenc, 5, 3, init_temperature=0.5)
    assert abs(m2.log_scale.item() - math.log(2)) < 1e-6 and m2.image_proj.weight.shape[0] == 128, (
        "init_temperature and the default embed_dim=128 must be used")
    with torch.no_grad():
        m.log_scale.fill_(math.log(1000.0))
    assert abs(m(x, t)[2].item() - 100.0) < 1e-4, "the logit scale is clamped to at most 100"


def test_stage3_temperature_learns(nb):
    torch.manual_seed(3)
    m = nb.MiniCLIP(_Lin(6, 5), _Lin(4, 3), 5, 3, embed_dim=7)
    img, txt, scale = m(torch.randn(8, 6), torch.randn(8, 4))
    _ref_clip(img, txt, scale).backward()
    assert m.log_scale.grad is not None and m.log_scale.grad.abs() > 0, (
        "the temperature must be learnable: compute the logit scale from log_scale inside forward, with autograd")
    assert m.image_proj.weight.grad is not None and m.text_proj.weight.grad is not None


# ---------------------------------------------------------------- stage 4: zero_shot_predict, class_embeddings

def test_stage4_zero_shot_uses_cosine(nb):
    img = torch.tensor([[1.0, 0.1], [0.1, 1.0], [1.0, 1.05]])
    cls = torch.tensor([[10.0, 10.0], [1.0, 0.0], [0.0, 1.0]])  # class 0 has a huge norm
    got = nb.zero_shot_predict(img, cls)
    assert got.dtype == torch.int64 and got.shape == (3,), "return int64 class indices, shape (B,)"
    assert got.tolist() == [1, 2, 0], (
        f"predictions {got.tolist()}, expected [1, 2, 0]: compare by COSINE similarity (normalise both sides); a dot "
        "product lets the class embedding with the largest norm win everything")


class _FakeText(nn.Module):
    """encode_text looks up one fixed vector per sentence id; records the mode it was called in."""

    def __init__(self, table):
        super().__init__()
        self.table = nn.Parameter(table)
        self.modes = []

    def encode_text(self, ids):
        self.modes.append((self.training, torch.is_grad_enabled()))
        return self.table[ids]


def test_stage4_prompt_ensembles(nb):
    sentences = {"a cat": 0, "the cat": 1, "a dog": 2, "the dog": 3}
    table = torch.tensor([[2.0, 0.0], [0.0, 0.5], [1.0, 1.0], [0.0, 3.0]])
    model = _FakeText(table)
    model.train()
    seen = []

    def tokenize(texts):
        seen.append(list(texts))
        return torch.tensor([sentences[t] for t in texts])

    got = nb.class_embeddings(model, ["cat", "dog"], ["a {}", "the {}"], tokenize)
    assert got.shape == (2, 2), "one embedding per class: (n_classes, d)"
    assert ["a cat", "the cat"] in seen and ["a dog", "the dog"] in seen, (
        f"format every template with the class name and tokenize the list; tokenize saw {seen}")
    n = F.normalize(table, dim=1)
    want = torch.stack([F.normalize((n[0] + n[1]) / 2, dim=0), F.normalize((n[2] + n[3]) / 2, dim=0)])
    torch.testing.assert_close(got, want, msg=(
        "normalise each prompt's embedding, average over the templates, and renormalise. averaging the raw "
        "embeddings lets long vectors dominate: for 'cat', (2, 0) would swamp (0, 0.5)"))
    assert all(not tr and not g for tr, g in model.modes), "encode the prompts in eval mode under torch.no_grad()"
    assert model.training, "restore the model's mode afterwards"
    assert not got.requires_grad
    model.eval()
    nb.class_embeddings(model, ["cat"], ["a {}"], tokenize)
    assert not model.training, "a model that arrived in eval mode must stay in eval mode"


# ---------------------------------------------------------------- stage 5: train_clip

def _paired(n, seed, k=4):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    centres = rng.normal(size=(k, 6))
    X = (centres[labels] + 0.3 * rng.normal(size=(n, 6))).astype(np.float32)
    T = np.stack([labels + 1, np.zeros(n, dtype=int)], axis=1)  # token = label + 1, then a pad
    return X, T, labels


class _Tok(nn.Module):
    def __init__(self, k=5, d=8):
        super().__init__()
        self.emb = nn.Embedding(k, d)

    def forward(self, t):
        return self.emb(t[:, 0])


def _model(nb, seed, dropout=0.0):
    torch.manual_seed(seed)
    return nb.MiniCLIP(_Lin(6, 8, dropout), _Tok(), 8, 8, embed_dim=8)


def test_stage5_matches_replay(nb):
    X, T, _ = _paired(40, 0)
    m = _model(nb, 1, dropout=0.2)  # dropout makes torch's RNG matter too
    m.eval()
    h = nb.train_clip(m, X, T, epochs=2, batch_size=16, lr=1e-2, seed=4)
    assert m.training, "train_clip must put the model in train mode"
    assert len(h["loss"]) == 4 and len(h["logit_scale"]) == 4, "2 epochs x (40 // 16) full batches = 4 steps"
    ref = _model(nb, 1, dropout=0.2)
    ref.train()
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    Xt, Tt = torch.as_tensor(X), torch.as_tensor(T)
    losses, scales = [], []
    for _ in range(2):
        perm = rng.permutation(40)
        for s in range(0, 40 - 16 + 1, 16):
            idx = perm[s:s + 16]
            img, txt, scale = ref(Xt[idx], Tt[idx])
            loss = _ref_clip(img, txt, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            scales.append(scale.item())
    assert np.allclose(h["loss"], losses, rtol=1e-5, atol=1e-6), (
        f"losses {np.round(h['loss'], 4)} differ from the specified loop {np.round(losses, 4)}: torch.manual_seed(seed) "
        "first, Adam(lr), a fresh permutation per epoch from np.random.default_rng(seed), full batches, clip_loss")
    assert np.allclose(h["logit_scale"], scales, rtol=1e-5), "log the logit scale used at each step"
    torch.testing.assert_close(m.log_scale.detach(), ref.log_scale.detach(), msg="the temperature must be trained too")


def test_stage5_learns_to_match(nb):
    accs = []
    for seed in range(3):
        X, T, labels = _paired(256, seed)
        m = _model(nb, seed)
        nb.train_clip(m, X, T, epochs=10, batch_size=32, lr=1e-2, seed=seed)
        with torch.no_grad():
            img = m.encode_image(torch.as_tensor(X))
            cls = m.encode_text(torch.tensor([[c + 1, 0] for c in range(4)]))
        accs.append(float((nb.zero_shot_predict(img, cls).numpy() == labels).mean()))
    assert np.median(accs) > 0.9, f"zero-shot accuracy after training, 3 seeds: {np.round(accs, 3)}; expected > 0.9"


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
        def forward(self, images, texts):
            img = self._out(images)
            self.seen.append(texts.device.type)
            return F.normalize(img, dim=1), F.normalize(img + 1.0, dim=1), torch.tensor(2.0)

    X, T, _ = _paired(32, 5)
    spy = Spy()
    nb.train_clip(spy, X, T, epochs=1, batch_size=16, device="meta")
    assert spy.moved_to == ["meta"], f"move the model to `device` once, with model.to(device); saw {spy.moved_to}"
    assert spy.seen and set(spy.seen) == {"meta"}, "move every batch of images and texts to `device`"
