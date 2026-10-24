"""Checks for capstone.py (reverse-engineer a circuit).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. everything runs on the CPU with
small p and tiny models; nothing is downloaded, and no test needs a model to grok. the data split for coverage,
labels and the seed; the model for its shape and untied head; the training loop for what it logs, when, its
optimizer settings (lr and weight decay both matter), checkpoints and modes; the Fourier tools on planted
embeddings; the logit fit and the progress measures on logits built from known cosines; the neuron tools against
a manual hook and planted activations.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _grid(p):
    a, b = np.divmod(np.arange(p * p), p)
    return a, b


# ---------------------------------------------------------------- stage 1: data


def test_stage1_data(nb):
    d = nb.modular_addition_data(7, train_frac=0.3, seed=0)
    assert d["all_x"].shape == (49, 3) and d["all_x"].dtype == torch.int64, "all p^2 inputs, int64 token ids"
    a, b = _grid(7)
    assert torch.equal(d["all_x"][:, 0], torch.tensor(a)) and torch.equal(d["all_x"][:, 1], torch.tensor(b)), (
        "grid order: row a * p + b holds (a, b)")
    assert (d["all_x"][:, 2] == 7).all(), "the third token is '=', id p"
    assert torch.equal(d["all_y"], torch.tensor((a + b) % 7)), "labels (a + b) mod p"
    tr, te = d["train_idx"].tolist(), d["test_idx"].tolist()
    assert tr == sorted(tr) and te == sorted(te), "train_idx and test_idx are sorted"
    assert len(tr) == round(0.3 * 49) and sorted(tr + te) == list(range(49)), (
        "train and test partition all p^2 pairs, round(train_frac * p^2) of them for training")
    assert torch.equal(d["train_x"], d["all_x"][d["train_idx"]]) and torch.equal(d["test_y"], d["all_y"][d["test_idx"]])
    again = nb.modular_addition_data(7, train_frac=0.3, seed=0)
    other = nb.modular_addition_data(7, train_frac=0.3, seed=1)
    assert torch.equal(again["train_idx"], d["train_idx"]) and not torch.equal(other["train_idx"], d["train_idx"]), (
        "the split is a permutation drawn from the seed")
    assert len(nb.modular_addition_data(7, train_frac=0.6, seed=0)["train_idx"]) == round(0.6 * 49), "train_frac is used"


# ---------------------------------------------------------------- stage 2: model and training


def test_stage2_model(nb):
    m = nb.make_model(7, d_model=16, n_heads=2, d_mlp=24, seed=0)
    assert m.tok_emb.weight.shape == (8, 16) and m.lm_head.weight.shape == (8, 16), "p + 1 tokens, width d_model"
    assert m.lm_head.weight.data_ptr() != m.tok_emb.weight.data_ptr(), "untied: the unembedding is its own matrix"
    assert len(m.blocks) == 1 and m.blocks[0].mlp.fc1.out_features == 24 and m.block_size == 3, (
        "one block, d_mlp MLP neurons, block size 3")
    same = nb.make_model(7, d_model=16, n_heads=2, d_mlp=24, seed=0)
    assert torch.equal(same.tok_emb.weight, m.tok_emb.weight), "the seed fixes the initial weights"
    x = nb.modular_addition_data(7)["all_x"][:5]
    out = nb.answer_logits(m, x)
    assert out.shape == (5, 7) and torch.allclose(out, m(x)[0][:, -1, :7]), (
        "answer_logits: the logits at the last position ('='), only the p number classes")


def _train(nb, p=7, frac=0.6, steps=60, **kw):
    d = nb.modular_addition_data(p, train_frac=frac, seed=0)
    m = nb.make_model(p, d_model=32, n_heads=2, d_mlp=64, seed=0)
    h = bounded(nb.train_grokking, m, d, budget=30_000_000, what="train_grokking", steps=steps, **kw)
    return m, d, h


def test_stage2_training_log(nb):
    m, d, h = _train(nb, steps=50, log_every=20, checkpoint_every=25)
    assert h["step"] == [0, 20, 40, 50], f"log at 0, every log_every updates, and after the last update: {h['step']}"
    for k in ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm"):
        assert len(h[k]) == 4, f"history[{k!r}] has one entry per logged step"
    fresh = nb.make_model(7, d_model=32, n_heads=2, d_mlp=64, seed=0)
    fresh.eval()
    with torch.no_grad():
        want0 = F.cross_entropy(nb.answer_logits(fresh, d["train_x"]), d["train_y"]).item()
    assert np.isclose(h["train_loss"][0], want0, atol=1e-5), "entry 0 is the untrained model, logged before any update"
    with torch.no_grad():
        lt = nb.answer_logits(m, d["test_x"])
    assert np.isclose(h["test_loss"][-1], F.cross_entropy(lt, d["test_y"]).item(), atol=1e-5) and np.isclose(
        h["test_acc"][-1], (lt.argmax(1) == d["test_y"]).float().mean().item()), "the last entry is the final model"
    assert np.isclose(h["weight_norm"][-1], math.sqrt(sum(p.pow(2).sum().item() for p in m.parameters())), rtol=1e-5), (
        "weight_norm is the 2-norm of all parameters together")
    assert [t for t, _ in h["checkpoints"]] == [0, 25, 50], "checkpoints at 0, every checkpoint_every and the end"
    assert torch.equal(h["checkpoints"][-1][1]["tok_emb.weight"], m.tok_emb.weight.detach().cpu()), (
        "a checkpoint is a copy of the state_dict at that step")
    assert not torch.equal(h["checkpoints"][0][1]["tok_emb.weight"], m.tok_emb.weight.detach().cpu()), (
        "copies, not references to the live tensors")
    assert not m.training, "the model is left in eval mode"
    _, _, h2 = _train(nb, steps=10, log_every=5)
    assert "checkpoints" not in h2 or not h2["checkpoints"], "no checkpoints unless checkpoint_every is given"


def test_stage2_optimizer_and_modes(nb):
    d = nb.modular_addition_data(7, train_frac=0.6, seed=0)
    m = nb.make_model(7, d_model=32, n_heads=2, d_mlp=64, seed=0)
    ref = nb.make_model(7, d_model=32, n_heads=2, d_mlp=64, seed=0)
    bounded(nb.train_grokking, m, d, budget=30_000_000, what="train_grokking", steps=5, lr=2e-3, weight_decay=0.5,
            log_every=2)
    opt = torch.optim.AdamW(ref.parameters(), lr=2e-3, weight_decay=0.5, betas=(0.9, 0.98))
    for _ in range(5):
        ref.train()
        loss = F.cross_entropy(ref(d["train_x"])[0][:, -1, :7], d["train_y"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for (k, p), q in zip(m.named_parameters(), ref.parameters()):
        assert torch.allclose(p, q, atol=1e-6), (
            f"{k}: after 5 steps the weights differ from full-batch AdamW(lr, weight_decay, betas=(0.9, 0.98)) on "
            "the whole training set (check the betas, the batch and that logging doesn't step the optimizer)")
    from mlfs.transformer import GPT
    torch.manual_seed(0)
    drop = GPT(8, block_size=3, n_layers=1, n_heads=2, d_model=32, d_ff=64, dropout=0.5, tie_weights=False)
    h = bounded(nb.train_grokking, drop, d, budget=5_000_000, what="train_grokking", steps=0, log_every=1)
    drop.eval()
    with torch.no_grad():
        want = F.cross_entropy(nb.answer_logits(drop, d["train_x"]), d["train_y"]).item()
    assert np.isclose(h["train_loss"][0], want, atol=1e-6), (
        "the history is measured in eval mode (dropout off), so it's a deterministic function of the weights")


def test_stage2_training_learns_and_options(nb):
    _, _, h = _train(nb, steps=300, weight_decay=0.0, log_every=100)
    assert h["train_acc"][-1] >= 0.95 and h["train_loss"][-1] < 0.1 * h["train_loss"][0], (
        f"300 full-batch AdamW steps (no weight decay) should (almost) memorise 29 training pairs: train accuracy "
        f"{h['train_acc'][-1]:.2f}")
    _, _, slow = _train(nb, steps=40, lr=1e-5, log_every=40)
    _, _, fast = _train(nb, steps=40, lr=1e-3, log_every=40)
    assert slow["train_loss"][-1] > fast["train_loss"][-1] + 0.1, "lr is the optimizer's learning rate"
    _, _, wd = _train(nb, steps=300, lr=3e-3, weight_decay=3.0, log_every=300)
    _, _, nowd = _train(nb, steps=300, lr=3e-3, weight_decay=0.0, log_every=300)
    assert wd["weight_norm"][-1] < 0.8 * nowd["weight_norm"][-1], (
        f"weight_decay is AdamW's decoupled decay: after 300 steps, decay 3.0 should leave much smaller weights than 0 "
        f"(got norms {wd['weight_norm'][-1]:.2f} vs {nowd['weight_norm'][-1]:.2f})")
    _, _, h1 = _train(nb, steps=30, log_every=30)
    _, _, h2 = _train(nb, steps=30, log_every=30)
    assert h1["train_loss"] == h2["train_loss"], "full-batch training is deterministic"


# ---------------------------------------------------------------- stage 3: the Fourier basis


def test_stage3_basis(nb):
    p = 11
    Fb = nb.fourier_basis(p)
    assert Fb.shape == (p, p) and np.allclose(Fb @ Fb.T, np.eye(p), atol=1e-10), "orthonormal rows"
    x = np.arange(p)
    assert np.allclose(Fb[0], 1 / math.sqrt(p)), "row 0 is the constant"
    for k in (1, 3, 5):
        assert np.allclose(Fb[2 * k - 1], np.cos(2 * math.pi * k * x / p) * math.sqrt(2 / p)), f"row {2 * k - 1}: cos k={k}"
        assert np.allclose(Fb[2 * k], np.sin(2 * math.pi * k * x / p) * math.sqrt(2 / p)), f"row {2 * k}: sin k={k}"
    with pytest.raises(ValueError):
        nb.fourier_basis(10)


def test_stage3_spectrum_and_keys(nb):
    p = 13
    x = np.arange(p)
    rng = np.random.default_rng(0)
    u, v, w = rng.normal(size=(3, 6))
    W = (np.outer(np.cos(2 * math.pi * 3 * x / p), u) + np.outer(np.sin(2 * math.pi * 3 * x / p), v)
         + 0.5 * np.outer(np.cos(2 * math.pi * 5 * x / p), w) + 4.0)  # plus a constant offset, which doesn't count
    W = np.vstack([W, rng.normal(size=(1, 6)) * 100])  # an extra row for '=' (token p), which is ignored
    fr = nb.embedding_frequency_norms(W, p)
    e3 = (np.sum(u ** 2) + np.sum(v ** 2)) * p / 2
    e5 = 0.25 * np.sum(w ** 2) * p / 2
    assert fr.shape == (6,) and np.isclose(fr.sum(), 1.0), "one fraction per frequency k = 1..(p-1)/2, summing to 1"
    assert np.allclose(fr[[2, 4]], [e3 / (e3 + e5), e5 / (e3 + e5)], atol=1e-8) and np.allclose(fr[[0, 1, 3, 5]], 0, atol=1e-10), (
        f"all the energy is at k = 3 and k = 5 (cos and sin parts together), in proportion to their squared norms; "
        f"the constant and the '=' row don't count; got {np.round(fr, 4)}")
    assert nb.key_frequencies(np.array([0.05, 0.5, 0.05, 0.3, 0.1])) == [2, 4, 5], (
        "the fewest frequencies reaching 90%: 0.5 + 0.3 + 0.1, returned as sorted 1-based frequencies")
    assert nb.key_frequencies(np.array([0.05, 0.5, 0.05, 0.3, 0.1]), frac=0.75) == [2, 4], "frac is the target"
    assert nb.key_frequencies(np.array([0.25, 0.25, 0.25, 0.25]), frac=1.0) == [1, 2, 3, 4]


# ---------------------------------------------------------------- stage 4: the algorithm in the logits


def _cos_logits(p, alphas, noise=0.0, offset=True, seed=0):
    a, b = _grid(p)
    c = np.arange(p)
    L = sum(al * np.cos(2 * math.pi * k * (a[:, None] + b[:, None] - c[None]) / p) for k, al in alphas.items())
    rng = np.random.default_rng(seed)
    if offset:
        L = L + rng.normal(size=(p * p, 1)) * 3  # a per-input offset the softmax ignores
    return L + noise * rng.normal(size=L.shape)


def test_stage4_cosine_fit(nb):
    p = 13
    L = _cos_logits(p, {2: 3.0, 5: -1.5})
    alphas, fve = nb.logit_cosine_fit(L, p, [2, 5])
    assert np.allclose(alphas, [3.0, -1.5], atol=1e-8) and np.isclose(fve, 1.0), (
        f"logits that ARE 3 cos(w2 (a+b-c)) - 1.5 cos(w5 (a+b-c)) plus per-input offsets: alphas (3, -1.5), FVE 1; "
        f"got {alphas}, {fve:.4f}. remember to remove each input's mean over c first")
    _, part = nb.logit_cosine_fit(L, p, [2])
    assert np.isclose(part, 9 / (9 + 2.25)), "FVE with only k = 2: its share of the energy, 9 / 11.25"
    _, wrong = nb.logit_cosine_fit(L, p, [1, 3])
    assert wrong < 1e-10, "frequencies that aren't there explain nothing"
    noisy = _cos_logits(p, {4: 2.0}, noise=1.0)
    _, fv = nb.logit_cosine_fit(noisy, p, [4])
    assert 0.5 < fv < 0.8, f"noise of the same size as the signal: FVE about 2/3; got {fv:.3f}"


def test_stage4_progress_measures(nb):
    p = 11
    d = nb.modular_addition_data(p, train_frac=0.5, seed=0)
    y = d["all_y"].numpy()
    a, b = _grid(p)
    good = 8.0 * np.cos(2 * math.pi * 3 * (a[:, None] + b[:, None] - np.arange(p)[None]) / p)
    memo = np.zeros((p * p, p))
    tr = d["train_idx"].numpy()
    memo[tr, y[tr]] = 6.0  # a lookup table for the training pairs
    pm = nb.progress_measures(good + memo, y, tr, p, [3])
    assert set(pm) == {"restricted", "excluded"}

    def ce(L, t):
        z = L - L.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(t)), t]))

    key = np.zeros_like(good)
    for fn in (np.cos, np.sin):
        psi = fn(2 * math.pi * 3 * (a + b) / p)
        key += np.outer(psi, psi @ (good + memo)) / (p * p / 2)
    want_r = ce((good + memo).mean(0, keepdims=True) + key, y)
    want_e = ce((good + memo - key)[tr], y[tr])
    assert np.isclose(pm["restricted"], want_r, atol=1e-8) and np.isclose(pm["excluded"], want_e, atol=1e-8), (
        f"restricted: keep each output's mean over inputs plus the projection onto cos and sin of w_k (a + b), scored "
        f"on ALL inputs; excluded: subtract that projection, scored on the TRAINING inputs. expected "
        f"({want_r:.4f}, {want_e:.4f}), got ({pm['restricted']:.4f}, {pm['excluded']:.4f})")
    clean = nb.progress_measures(good * 4, y, tr, p, [3])
    assert clean["restricted"] < 0.05 and clean["excluded"] > math.log(p) - 0.2, (
        "for a pure cosine circuit the key frequencies carry everything: restricted loss ~ 0, excluded loss ~ log p")
    lookup = nb.progress_measures(memo, y, tr, p, [3])
    assert lookup["excluded"] < lookup["restricted"], "a pure lookup table survives exclusion but not restriction"


# ---------------------------------------------------------------- stage 5: neurons


def test_stage5_activations(nb):
    m = nb.make_model(7, d_model=16, n_heads=2, d_mlp=24, seed=1)
    m.train()
    x = nb.modular_addition_data(7)["all_x"]
    got = nb.neuron_activations(m, x)
    assert m.training, "restore the model's mode"
    store = {}
    h = m.blocks[0].mlp.act.register_forward_hook(lambda mod, i, o: store.setdefault("a", o.detach()))
    m.eval()
    with torch.no_grad():
        m(x)
    h.remove()
    assert isinstance(got, np.ndarray) and got.shape == (49, 24), "an ndarray (n, d_mlp)"
    assert np.allclose(got, store["a"][:, -1].numpy(), atol=1e-6), (
        "the output of blocks.0.mlp.act (after the nonlinearity) at the LAST position, in eval mode")


def test_stage5_frequency_fit(nb):
    p = 13
    a, b = _grid(p)
    w3, w5 = 2 * math.pi * 3 / p, 2 * math.pi * 5 / p
    rng = np.random.default_rng(0)
    acts = np.stack([
        2 * np.cos(w3 * a) * np.cos(w3 * b) + 0.5,  # = cos(w3(a+b)) + cos(w3(a-b)), plus a constant
        np.sin(w5 * a) + 0.3 * np.cos(w5 * (a + b)),
        np.cos(w3 * (a - b)) + 0.5 * rng.normal(size=p * p),
        rng.normal(size=p * p),
        np.zeros(p * p),
        np.sin(w5 * b),
        np.sin(w3 * (a - b)) - 2.0,
    ], axis=1)
    k, fve = nb.neuron_frequency_fit(acts, p)
    assert k.shape == fve.shape == (7,)
    assert k[5] == 5 and np.isclose(fve[5], 1.0) and k[6] == 3 and np.isclose(fve[6], 1.0), (
        f"sin(w5 b) and sin(w3 (a - b)) are among the 8 terms: FVE 1 at frequencies 5 and 3; got {k[5:]}, "
        f"{np.round(fve[5:], 4)}")
    assert k[0] == 3 and np.isclose(fve[0], 1.0) and k[1] == 5 and np.isclose(fve[1], 1.0), (
        f"products and sums of cos/sin of one frequency are exactly the 8 terms cos/sin(w a), (w b), (w(a+b)), "
        f"(w(a-b)): best frequency 3 and 5 with FVE 1; got {k[:2]}, {np.round(fve[:2], 4)}")
    assert k[2] == 3 and 0.6 < fve[2] < 0.8, f"a noisy frequency-3 neuron: FVE about 2/3; got {fve[2]:.3f}"
    assert fve[3] < 0.2, "pure noise isn't explained by any one frequency (8 of 168 dimensions)"
    assert fve[4] == 0 and not np.isnan(fve).any(), "a dead neuron has FVE 0, not NaN"
