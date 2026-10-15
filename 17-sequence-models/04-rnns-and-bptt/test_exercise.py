"""Checks for exercise.py (RNNs & BPTT).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The NumPy forward pass
is checked against a hand example and torch's nn.RNN, the hand-written BPTT against torch autograd
in float64, the PyTorch module against nn.RNN with copied weights, the gradient norms against their
exact last value and their decay or growth for small and large recurrent weights, and the truncated
BPTT loop with a spy model (state carried, detached, reset per epoch) and a learnable sequence.
Training runs under `mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _warm_up():
    """torch imports parts of itself on first use (millions of lines); do that here, outside any line budget."""
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p])
    (p ** 2).sum().backward()
    nn.utils.clip_grad_norm_([p], 1.0)
    opt.step()
    nn.functional.cross_entropy(torch.zeros(2, 3, requires_grad=True), torch.tensor([0, 1])).backward()


_warm_up()


def _params(seed, D=3, H=4, scale=0.6):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(H, D)) * scale, rng.normal(size=(H, H)) * scale / math.sqrt(H), rng.normal(size=H) * 0.1)


def _torch_rnn(W_xh, W_hh, b):
    rnn = nn.RNN(W_xh.shape[1], W_xh.shape[0], batch_first=True).double()
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(torch.tensor(W_xh))
        rnn.weight_hh_l0.copy_(torch.tensor(W_hh))
        rnn.bias_ih_l0.copy_(torch.tensor(b))
        rnn.bias_hh_l0.zero_()
    return rnn


# ---------------------------------------------------------------- stage 1: forward


def test_stage1_hand_example(nb):
    hs = nb.rnn_forward(np.array([[[1.0], [0.0]]]), np.zeros((1, 1)), np.array([[1.0]]), np.array([[0.5]]),
                        np.zeros(1))
    assert np.shape(hs) == (1, 2, 1), f"hs has shape (B, T, H) = (1, 2, 1); got {np.shape(hs)}"
    assert np.allclose(hs.ravel(), [math.tanh(1.0), math.tanh(0.5 * math.tanh(1.0))]), (
        f"h1 = tanh(1 * 1 + 0.5 * 0) = 0.7616, h2 = tanh(1 * 0 + 0.5 * 0.7616) = 0.3634; got {hs.ravel()}")


def test_stage1_matches_torch(nb):
    W_xh, W_hh, b = _params(0)
    rng = np.random.default_rng(1)
    x, h0 = rng.normal(size=(5, 7, 3)), rng.normal(size=(5, 4))
    x0 = x.copy()
    hs = nb.rnn_forward(x, h0, W_xh, W_hh, b)
    want, _ = _torch_rnn(W_xh, W_hh, b)(torch.tensor(x), torch.tensor(h0)[None])
    assert np.allclose(hs, want.detach().numpy(), atol=1e-12), (
        "your hidden states differ from torch's nn.RNN. W_xh is (H, D) like nn.Linear's weight, so the input term is "
        "x_t @ W_xh.T; and every step starts from the previous h, beginning with h0")
    assert np.array_equal(x, x0), "rnn_forward must not modify its input"


# ---------------------------------------------------------------- stage 2: BPTT by hand


def test_stage2_matches_autograd(nb):
    for seed, (B, T, D, H) in enumerate([(2, 5, 3, 4), (3, 9, 2, 6), (1, 1, 2, 3)]):
        W_xh, W_hh, b = _params(seed + 10, D, H)
        rng = np.random.default_rng(seed + 20)
        x, h0, dhs = rng.normal(size=(B, T, D)), rng.normal(size=(B, H)), rng.normal(size=(B, T, H))
        hs = nb.rnn_forward(x, h0, W_xh, W_hh, b)
        g = nb.rnn_backward(x, h0, hs, W_xh, W_hh, dhs)
        tw = [torch.tensor(a, requires_grad=True) for a in (W_xh, W_hh, b, h0, x)]
        h, out = tw[3], []
        for t in range(T):
            h = torch.tanh(tw[4][:, t] @ tw[0].T + h @ tw[1].T + tw[2])
            out.append(h)
        (torch.stack(out, 1) * torch.tensor(dhs)).sum().backward()
        for name, ref in zip(["dW_xh", "dW_hh", "db", "dh0", "dx"], tw):
            assert name in g, f"rnn_backward returns a dict with key {name!r}"
            assert np.shape(g[name]) == tuple(ref.shape), f"{name} has the shape of what it differentiates"
            assert np.allclose(g[name], ref.grad.numpy(), atol=1e-10), (
                f"(B, T, D, H) = {(B, T, D, H)}: {name} differs from autograd. walk t = T-1 .. 0: add dhs[:, t] to the "
                f"gradient arriving from the future, multiply by 1 - h_t^2, accumulate the weight gradients, then pass "
                f"da @ W_hh back to h_{{t-1}}")


def test_stage2_no_future_no_gradient(nb):
    W_xh, W_hh, b = _params(3)
    rng = np.random.default_rng(4)
    x, h0 = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 4))
    dhs = np.zeros((2, 6, 4))
    dhs[:, 2] = 1.0  # only step 2's loss
    g = nb.rnn_backward(x, h0, nb.rnn_forward(x, h0, W_xh, W_hh, b), W_xh, W_hh, dhs)
    assert np.allclose(g["dx"][:, 3:], 0), "a loss at step 2 can't depend on inputs from steps 3-5: dx must be 0 there"
    assert np.abs(g["dx"][:, :3]).sum() > 0 and np.abs(g["dh0"]).sum() > 0, "but it reaches every earlier step and h0"


# ---------------------------------------------------------------- stage 3: the PyTorch module


def _copy_from_torch(mine, ref):
    with torch.no_grad():
        for k in range(ref.num_layers):
            mine.ih[k].weight.copy_(getattr(ref, f"weight_ih_l{k}"))
            mine.ih[k].bias.copy_(getattr(ref, f"bias_ih_l{k}"))
            mine.hh[k].weight.copy_(getattr(ref, f"weight_hh_l{k}"))
            mine.hh[k].bias.copy_(getattr(ref, f"bias_hh_l{k}"))


def test_stage3_structure(nb):
    m = nb.RNN(3, 5, num_layers=2)
    assert len(m.ih) == 2 and len(m.hh) == 2, "ih and hh are ModuleLists with one nn.Linear per layer"
    assert tuple(m.ih[0].weight.shape) == (5, 3) and tuple(m.ih[1].weight.shape) == (5, 5), (
        "layer 0 maps the input (D = 3) to H = 5; layer 1 maps H to H")
    assert tuple(m.hh[1].weight.shape) == (5, 5)
    assert not any(isinstance(mod, nn.RNNBase) for mod in m.modules()), "build it from nn.Linear, not nn.RNN"
    with pytest.raises(ValueError):
        nb.RNN(3, 5, nonlinearity="sigmoid")


@pytest.mark.parametrize("layers,act", [(1, "tanh"), (2, "tanh"), (3, "relu")])
def test_stage3_matches_nn_rnn(nb, layers, act):
    torch.manual_seed(layers)
    ref = nn.RNN(4, 6, num_layers=layers, nonlinearity=act, batch_first=True)
    m = nb.RNN(4, 6, num_layers=layers, nonlinearity=act)
    _copy_from_torch(m, ref)
    x = torch.randn(3, 8, 4)
    h0 = torch.randn(layers, 3, 6)
    out, hn = m(x, h0)
    rout, rhn = ref(x, h0)
    assert tuple(out.shape) == (3, 8, 6) and tuple(hn.shape) == (layers, 3, 6), (
        f"outputs (B, T, H) of the top layer and h_n (num_layers, B, H); got {tuple(out.shape)}, {tuple(hn.shape)}")
    assert torch.allclose(out, rout, atol=1e-5) and torch.allclose(hn, rhn, atol=1e-5), (
        f"{layers} layer(s), {act}: outputs differ from nn.RNN with the same weights. each layer runs over the whole "
        f"sequence; its outputs are the next layer's inputs; h_n stacks every layer's last state")
    out0, _ = m(x)
    rout0, _ = ref(x)
    assert torch.allclose(out0, rout0, atol=1e-5), "h0=None means zeros"
    out.sum().backward()
    assert m.hh[0].weight.grad is not None, "gradients reach the recurrent weights"


# ---------------------------------------------------------------- stage 4: gradient norms through time


def test_stage4_exact_values(nb):
    torch.manual_seed(0)
    m = nb.RNN(3, 5)
    x = torch.randn(4, 6, 3)
    norms = np.asarray(nb.hidden_grad_norms(m, x))
    assert norms.shape == (6,), f"one norm per time step; got shape {norms.shape}"
    assert np.isclose(norms[-1], math.sqrt(5)), (
        f"dL/dh_T for L = sum of h_T is a vector of ones, norm sqrt(H) = {math.sqrt(5):.4f}; got {norms[-1]:.4f}")
    # oracle: explicit unroll with retain_grad
    h, hs = torch.zeros(4, 5), []
    for t in range(6):
        h = torch.tanh(m.ih[0](x[:, t]) + m.hh[0](h))
        h.retain_grad()
        hs.append(h)
    hs[-1].sum().backward()
    want = np.array([hh.grad.norm(dim=1).mean().item() for hh in hs])
    assert np.allclose(norms, want, rtol=1e-5), (
        f"expected {np.round(want, 4)}, got {np.round(norms, 4)}: the gradient at h_t must include every path "
        f"through the later steps (use torch.autograd.grad on the list of hidden states)")


def test_stage4_vanishing_and_exploding(nb):
    torch.manual_seed(1)
    small = nb.RNN(2, 8)
    with torch.no_grad():
        small.hh[0].weight.copy_(0.2 * torch.eye(8))
    n = np.asarray(nb.hidden_grad_norms(small, torch.randn(2, 20, 2)))
    assert n[0] / n[-1] < 1e-8, f"with W_hh = 0.2 I each step multiplies the gradient by <= 0.2: {n[0] / n[-1]:.2e}"
    big = nb.RNN(2, 8, nonlinearity="relu")
    with torch.no_grad():
        big.hh[0].weight.copy_(1.3 * torch.eye(8))
        big.ih[0].bias.fill_(1.0)
    n = np.asarray(nb.hidden_grad_norms(big, torch.rand(2, 20, 2)))
    assert n[0] / n[-1] > 100, (
        f"relu with W_hh = 1.3 I and active units: each step multiplies the gradient by 1.3, so after 19 steps it "
        f"is 1.3^19 = 146 times larger; got a ratio of {n[0] / n[-1]:.3g}")


# ---------------------------------------------------------------- stage 5: truncated BPTT


def test_stage5_batchify_and_chunks(nb):
    data = nb.batchify(np.arange(23), 4)
    assert torch.is_tensor(data) and data.dtype == torch.int64 and tuple(data.shape) == (4, 5), (
        f"23 ids in 4 streams: 5 each, the last 3 dropped, int64 tensor (4, 5); got {getattr(data, 'shape', data)}")
    assert data[1].tolist() == [5, 6, 7, 8, 9], f"each row is a CONTIGUOUS stretch of the text; got row 1 {data[1].tolist()}"
    chunks = nb.tbptt_batches(data, 2)
    assert [tuple(x.shape) for x, _ in chunks] == [(4, 2), (4, 2)], (
        f"5 columns give 4 (input, target) steps: chunks of 2 and 2; got {[tuple(x.shape) for x, _ in chunks]}")
    xs = torch.cat([x for x, _ in chunks], 1)
    ys = torch.cat([y for _, y in chunks], 1)
    assert torch.equal(xs, data[:, :-1]) and torch.equal(ys, data[:, 1:]), (
        "consecutive chunks cover the streams in order; the targets are the inputs shifted by one")
    assert [tuple(x.shape) for x, _ in nb.tbptt_batches(nb.batchify(np.arange(40), 2), 7)][-1] == (2, 5), (
        "the last chunk is shorter when the length doesn't divide evenly")


def test_stage5_char_rnn(nb):
    m = nb.CharRNN(11, 16, num_layers=2)
    assert isinstance(m.embed, nn.Embedding) and isinstance(m.head, nn.Linear) and len(m.rnn.ih) == 2
    logits, h = m(torch.randint(0, 11, (3, 7)))
    assert tuple(logits.shape) == (3, 7, 11) and tuple(h.shape) == (2, 3, 16), (
        f"logits (B, T, V) and the RNN's h_n; got {tuple(logits.shape)}, {tuple(h.shape)}")


def test_stage5_state_is_carried_and_detached(nb):
    seen = []

    class Spy(nb.CharRNN):
        def forward(self, idx, h=None):
            seen.append(h)
            return super().forward(idx, h)

    torch.manual_seed(0)
    m = Spy(5, 8)
    ids = np.random.default_rng(0).integers(0, 5, size=200)
    losses = bounded(nb.train_tbptt, m, ids, batch_size=4, seq_len=10, epochs=2, budget=20_000_000,
                     what="train_tbptt")
    per_epoch = len(nb.tbptt_batches(nb.batchify(ids, 4), 10))
    assert len(losses) == 2 * per_epoch, f"one loss per chunk: {2 * per_epoch} expected, got {len(losses)}"
    assert seen[0] is None and seen[per_epoch] is None, "each epoch starts with h = None (a fresh state)"
    mid = seen[1:per_epoch]
    assert all(torch.is_tensor(h) for h in mid), (
        "between chunks, pass the previous chunk's final state in: the streams are contiguous, so the state carries "
        "real information across the boundary")
    assert all(not h.requires_grad for h in mid), (
        "detach the carried state (h = h.detach()): otherwise backward() runs through every earlier chunk, or fails "
        "because their graphs were freed")


def test_stage5_learns_a_sequence(nb):
    pattern = [0, 1, 2, 3, 1, 2, 0, 3, 2]  # period 9; each symbol needs 2 symbols of context to predict
    finals = []
    for s in range(5):
        torch.manual_seed(s)
        m = nb.CharRNN(4, 32)
        ids = np.array(pattern * 60)
        losses = bounded(nb.train_tbptt, m, ids, batch_size=8, seq_len=16, epochs=25, lr=1e-2, budget=60_000_000,
                         what="train_tbptt")
        finals.append(np.mean(losses[-3:]))
    assert np.median(finals) < 0.15, (
        f"the sequence is deterministic given 2 previous symbols, so the loss should approach 0; final losses over 5 "
        f"seeds: {np.round(finals, 3)}")
