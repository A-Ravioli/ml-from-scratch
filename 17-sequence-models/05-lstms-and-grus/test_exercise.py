"""Checks for exercise.py (LSTMs & GRUs).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Cells and sequence
modules are checked against torch's nn.LSTMCell / nn.GRUCell / nn.LSTM / nn.GRU with copied
weights, against hand examples with zero weights, and against the defining property of the cell
state (a saturated forget gate keeps it unchanged). The copy-task tools are checked directly and by
training an LSTM tagger on a short delay (aggregated over seeds), under `mlfs.testing.bounded`.
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
    opt.step()
    nn.functional.cross_entropy(torch.zeros(2, 3, requires_grad=True), torch.tensor([0, -100]),
                                ignore_index=-100).backward()


_warm_up()


def _copy_cell(mine, ref, suffix=""):
    with torch.no_grad():
        mine.ih.weight.copy_(getattr(ref, "weight_ih" + suffix))
        mine.ih.bias.copy_(getattr(ref, "bias_ih" + suffix))
        mine.hh.weight.copy_(getattr(ref, "weight_hh" + suffix))
        mine.hh.bias.copy_(getattr(ref, "bias_hh" + suffix))


def _zero(cell):
    with torch.no_grad():
        for p in cell.parameters():
            p.zero_()


# ---------------------------------------------------------------- stage 1: LSTMCell


def test_stage1_structure_and_forget_bias(nb):
    cell = nb.LSTMCell(3, 5)
    assert tuple(cell.ih.weight.shape) == (20, 3) and tuple(cell.hh.weight.shape) == (20, 5), (
        "ih = Linear(D, 4H) and hh = Linear(H, 4H): all four gates in one matrix multiply each")
    assert torch.all(cell.ih.bias[5:10] == 1.0), (
        "the forget gate's slice of ih.bias (positions H..2H, gate order i, f, g, o) starts at 1, so the cell "
        "remembers by default (Jozefowicz et al., 2015)")


def test_stage1_matches_nn_lstmcell(nb):
    torch.manual_seed(0)
    ref = nn.LSTMCell(4, 6)
    cell = nb.LSTMCell(4, 6)
    _copy_cell(cell, ref)
    x, h, c = torch.randn(3, 4), torch.randn(3, 6), torch.randn(3, 6)
    got = cell(x, (h, c))
    want = ref(x, (h, c))
    assert isinstance(got, tuple) and len(got) == 2, "forward returns the pair (h, c)"
    assert torch.allclose(got[0], want[0], atol=1e-6) and torch.allclose(got[1], want[1], atol=1e-6), (
        "your LSTMCell differs from nn.LSTMCell with the same weights. split ih(x) + hh(h) into four chunks in the "
        "order i, f, g, o; i, f, o go through a sigmoid, g through tanh; c' = f c + i g; h' = o tanh(c')")
    got0, want0 = cell(x), ref(x)
    assert torch.allclose(got0[0], want0[0], atol=1e-6), "state=None means h = c = 0"


def test_stage1_hand_example_and_memory(nb):
    cell = nb.LSTMCell(2, 3)
    _zero(cell)
    h, c = cell(torch.zeros(1, 2), (torch.zeros(1, 3), torch.ones(1, 3)))
    assert torch.allclose(c, torch.full((1, 3), 0.5)) and torch.allclose(h, torch.full((1, 3), 0.5 * math.tanh(0.5))), (
        f"with all weights and biases 0, every gate is sigmoid(0) = 0.5 and g = tanh(0) = 0: from c = 1, c' = 0.5 and "
        f"h' = 0.5 tanh(0.5) = 0.2311; got c' = {c.ravel().tolist()}, h' = {h.ravel().tolist()}")
    with torch.no_grad():
        cell.ih.bias[3:6] = 20.0  # forget gate fully open
        cell.ih.bias[0:3] = -20.0  # input gate shut
    torch.manual_seed(1)
    state = (torch.zeros(2, 3), torch.tensor([[0.3, -1.0, 2.0], [1.0, 0.0, -0.5]]))
    c0 = state[1].clone()
    for _ in range(50):
        state = cell(torch.randn(2, 2), state)
    assert torch.allclose(state[1], c0, atol=1e-6), (
        "with the forget gate at 1 and the input gate at 0, the cell state must pass through 50 steps unchanged: "
        "c' = f * c + i * g")


# ---------------------------------------------------------------- stage 2: GRUCell


def test_stage2_matches_nn_grucell(nb):
    torch.manual_seed(2)
    ref = nn.GRUCell(4, 6)
    cell = nb.GRUCell(4, 6)
    assert tuple(cell.ih.weight.shape) == (18, 4) and tuple(cell.hh.weight.shape) == (18, 6), (
        "ih = Linear(D, 3H), hh = Linear(H, 3H): chunks r, z, n")
    _copy_cell(cell, ref)
    x, h = torch.randn(3, 4), torch.randn(3, 6)
    assert torch.allclose(cell(x, h), ref(x, h), atol=1e-6), (
        "your GRUCell differs from nn.GRUCell with the same weights. r and z are sigmoids of the sums of their "
        "chunks; n = tanh(ih_n(x) + r * hh_n(h)), with the reset gate multiplying hh's chunk INCLUDING its bias; "
        "h' = (1 - z) * n + z * h")
    assert torch.allclose(cell(x), ref(x), atol=1e-6), "h=None means zeros"


def test_stage2_hand_example(nb):
    cell = nb.GRUCell(2, 3)
    _zero(cell)
    h = torch.tensor([[1.0, -2.0, 4.0]])
    assert torch.allclose(cell(torch.zeros(1, 2), h), 0.5 * h), (
        "with all weights 0: r = z = 0.5 and n = tanh(0) = 0, so h' = 0.5 * 0 + 0.5 * h")
    with torch.no_grad():
        cell.ih.bias[3:6] = 20.0  # update gate z = 1: copy the old state
    assert torch.allclose(cell(torch.randn(1, 2), h), h, atol=1e-6), "z = 1 copies the previous state exactly"


# ---------------------------------------------------------------- stage 3: sequence modules


@pytest.mark.parametrize("layers", [1, 2])
def test_stage3_lstm_matches_nn_lstm(nb, layers):
    torch.manual_seed(layers)
    ref = nn.LSTM(4, 5, num_layers=layers, batch_first=True)
    m = nb.LSTM(4, 5, num_layers=layers)
    assert len(m.cells) == layers and isinstance(m.cells[0], nb.LSTMCell)
    assert not any(isinstance(mod, (nn.RNNBase, nn.RNNCellBase)) for mod in m.modules()), "build it from your cells"
    for k in range(layers):
        _copy_cell(m.cells[k], ref, f"_l{k}")
    x = torch.randn(3, 7, 4)
    state = (torch.randn(layers, 3, 5), torch.randn(layers, 3, 5))
    out, (h, c) = m(x, state)
    rout, (rh, rc) = ref(x, state)
    assert tuple(out.shape) == (3, 7, 5) and tuple(h.shape) == (layers, 3, 5) and tuple(c.shape) == (layers, 3, 5)
    assert torch.allclose(out, rout, atol=1e-5) and torch.allclose(h, rh, atol=1e-5) and torch.allclose(c, rc, atol=1e-5), (
        f"{layers} layer(s): outputs, h_n or c_n differ from nn.LSTM with the same weights. layer k starts from "
        f"state[0][k], state[1][k] and reads the outputs of layer k - 1")
    assert torch.allclose(m(x)[0], ref(x)[0], atol=1e-5), "state=None means zeros"


@pytest.mark.parametrize("layers", [1, 3])
def test_stage3_gru_matches_nn_gru(nb, layers):
    torch.manual_seed(10 + layers)
    ref = nn.GRU(3, 4, num_layers=layers, batch_first=True)
    m = nb.GRU(3, 4, num_layers=layers)
    assert len(m.cells) == layers and isinstance(m.cells[0], nb.GRUCell)
    for k in range(layers):
        _copy_cell(m.cells[k], ref, f"_l{k}")
    x, h0 = torch.randn(2, 6, 3), torch.randn(layers, 2, 4)
    out, h = m(x, h0)
    rout, rh = ref(x, h0)
    assert torch.allclose(out, rout, atol=1e-5) and torch.allclose(h, rh, atol=1e-5), (
        f"{layers} layer(s): your GRU differs from nn.GRU with the same weights")


# ---------------------------------------------------------------- stage 4: the copy task


def test_stage4_batch(nb):
    g = torch.Generator().manual_seed(0)
    x, y = nb.recall_batch(64, 9, 4, generator=g)
    assert x.dtype == torch.int64 and tuple(x.shape) == (64, 10) and tuple(y.shape) == (64, 10), (
        f"x and y have shape (B, delay + 1) = (64, 10); got {tuple(x.shape)}, {tuple(y.shape)}")
    assert int(x[:, 0].min()) >= 0 and int(x[:, 0].max()) < 4, "the symbol to remember is in [0, K)"
    assert int(x[:, 1:].min()) >= 4 and int(x[:, 1:].max()) < 8, "the distractors are in [K, 2K): never a symbol"
    assert len(torch.unique(x[:, 0])) == 4 and len(torch.unique(x[:, 1:])) == 4, "both drawn uniformly"
    assert torch.all(y[:, :-1] == -100) and torch.equal(y[:, -1], x[:, 0]), (
        "only the last step is scored: y[:, -1] = x[:, 0], everything else -100 (ignored by cross_entropy)")
    x2, _ = nb.recall_batch(64, 9, 4, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x, x2), "the generator makes batches reproducible"


def test_stage4_tagger_and_accuracy(nb):
    for cell, cls in [("rnn", "RNN"), ("gru", "GRU"), ("lstm", "LSTM")]:
        m = nb.SeqTagger(6, 8, cell=cell)
        assert type(m.rnn).__name__ == cls, f"cell={cell!r} uses your {cls}"
        assert tuple(m(torch.randint(0, 6, (2, 5))).shape) == (2, 5, 6), "logits (B, T, V)"
    with pytest.raises(ValueError):
        nb.SeqTagger(6, 8, cell="transformer")

    class Oracle(nn.Module):
        def __init__(self, y):
            super().__init__()
            self.y = y

        def forward(self, x):
            return nn.functional.one_hot(self.y.clamp(min=0), 5).float()

    x, y = nb.recall_batch(8, 5, 5, generator=torch.Generator().manual_seed(1))
    oracle = Oracle(y).train()
    assert nb.copy_accuracy(oracle, x, y) == 1.0, "a perfect predictor scores 1.0 (on the positions with y != -100)"
    assert oracle.training, "copy_accuracy restores the model's training mode"
    y2 = y.clone()
    y2[:3, -1] = (y2[:3, -1] + 1) % 5
    assert math.isclose(nb.copy_accuracy(oracle, x, y2), 5 / 8, abs_tol=1e-6), (
        "wrong on 3 of the 8 scored positions: 5/8. score only the positions where y != -100")


def _train_recall(nb, cell, delay, seed, steps=150):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = nb.SeqTagger(8, 32, cell=cell)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = nb.recall_batch(32, delay, 4, generator=g)
        loss = nn.functional.cross_entropy(m(x).reshape(-1, 8), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = nb.recall_batch(512, delay, 4, generator=g)
    return nb.copy_accuracy(m, x, y)


def test_stage4_lstm_learns_to_recall(nb):
    accs = [bounded(_train_recall, nb, "lstm", 8, s, budget=80_000_000, what="training your LSTM tagger")
            for s in range(5)]
    assert np.median(accs) > 0.9, (
        f"an LSTM tagger (forget bias 1) should learn to recall the first symbol across 8 distractors in 150 steps; "
        f"accuracies over 5 seeds: {np.round(accs, 2)} (chance is 0.25)")
