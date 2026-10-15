import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    N_KEYS, N_PAIRS = 64, 16  # the recall task: 16 (key, value) pairs per example, keys and values from 64 symbols


@app.class_definition
class Recall(nn.Module):
    """one attention layer that must find the pair whose key matches the query and read out its value."""

    def __init__(self, d, *, scaled=True):
        super().__init__()
        self.ek, self.ev, self.eq = (nn.Embedding(N_KEYS, d) for _ in range(3))
        self.attn = ex.MultiHeadAttention(d, 1)
        self.head = nn.Linear(d, N_KEYS)
        if not scaled:  # multiply the queries by sqrt(d): the scores become q . k, unscaled
            self.attn.q_proj.register_forward_hook(lambda mod, inp, out: out * math.sqrt(d))

    def forward(self, keys, values, query):
        memory = self.ek(keys) + self.ev(values)  # (B, N, d): one token per pair
        return self.head(self.attn(self.eq(query)[:, None], memory)[:, 0])


@app.function
def recall_batch(B, generator):
    """B examples: distinct keys, random values, and a query that repeats one of the keys."""
    keys = torch.stack([torch.randperm(N_KEYS, generator=generator)[:N_PAIRS] for _ in range(B)])
    values = torch.randint(0, N_KEYS, (B, N_PAIRS), generator=generator)
    i = torch.randint(0, N_PAIRS, (B,), generator=generator)
    rows = torch.arange(B)
    return keys, values, keys[rows, i], values[rows, i]


@app.function
def train_recall(d, scaled, *, steps=600, seed=0, lr=1e-3):
    """Adam on batches of 64; returns the training loss every 25 steps and the final accuracy on 1,000 examples."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Recall(d, scaled=scaled)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for step in range(1, steps + 1):
        k, v, q, y = recall_batch(64, g)
        loss = nn.functional.cross_entropy(model(k, v, q), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            curve.append(loss.item())
    k, v, q, y = recall_batch(1000, g)
    with torch.no_grad():
        acc = (model(k, v, q).argmax(-1) == y).float().mean().item()
    return curve, acc


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # self-attention · lab

    every experiment here runs **your** `scaled_dot_product_attention` and
    `MultiHeadAttention` from `exercise.py`. for each one: write down a prediction, lock it in,
    run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes about a minute on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what the $\sqrt{d}$ does at initialization

    draw queries and keys with independent standard-normal entries (what a well-initialized
    layer produces, roughly), and look at the attention of 64 queries over 64 keys, with your
    scaling and without it (the queries multiplied by $\sqrt d$, which turns your scores
    back into raw dot products). three numbers per width $d$: the standard deviation of the
    scores, the average largest weight in a row (1/64 = 0.016 is uniform, 1 is one-hot), and
    the size of the gradient that reaches the scores from a random loss on the output.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at d = 1024, without the scaling, what is the typical largest attention weight in a row of 64? and how does the "
        "gradient reaching the scores compare with the scaled version: 2x smaller, 10x, 1000x?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _g = torch.Generator().manual_seed(0)
    _rows = []
    for _d in (4, 16, 64, 256, 1024):
        _q = torch.randn(32, 64, _d, generator=_g)
        _k = torch.randn(32, 64, _d, generator=_g)
        _v = torch.randn(32, 64, _d, generator=_g)
        _upstream = torch.randn(32, 64, _d, generator=_g) / math.sqrt(_d)  # so that dL/d(weights) is O(1) at every d
        _stats = []
        for _scale in (1.0, math.sqrt(_d)):
            _out, _w = ex.scaled_dot_product_attention(_q * _scale, _k, _v)
            _scores = (_q * _scale) @ _k.transpose(-2, -1) / math.sqrt(_d)
            # backprop of L = sum(out * upstream) to the scores S, through your weights a = softmax(S):
            # dL/da = upstream v^T, and the softmax's Jacobian gives dL/dS = a * (dL/da - sum(a * dL/da))
            _da = _upstream @ _v.transpose(-2, -1)
            _dS = _w * (_da - (_w * _da).sum(-1, keepdim=True))
            _stats.append((_scores.std().item(), _w.max(-1).values.mean().item(), _dS.abs().mean().item()))
        _rows.append((_d, *_stats[0], *_stats[1]))
    _t = ("| $d$ | score std (scaled) | max weight | mean \\|dL/dS\\| | score std (unscaled) | max weight | mean \\|dL/dS\\| |\n"
          "|---|---|---|---|---|---|---|\n")
    _t += "\n".join(f"| {r[0]} | {r[1]:.2f} | {r[2]:.3f} | {r[3]:.2e} | {r[4]:.1f} | {r[5]:.3f} | {r[6]:.2e} |" for r in _rows)
    mo.md(_t)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the score's standard deviation grow like $\sqrt d$ without the
    scaling (on paper 1 in the lesson)? the softmax's Jacobian is
    $\operatorname{diag}(a) - aa^\top$: what does it look like when one weight is 1 and the
    rest are 0, and what does that do to the gradient? why does the *scaled* gradient not
    depend on $d$ at all?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does it matter once you train?

    numbers at initialization are one thing; Adam rescales gradients, so maybe it recovers. here
    is a task that *needs* sharp attention: each example is 16 (key, value) pairs and a query
    that repeats one of the keys; the model (embeddings, one single-head attention layer from
    your `MultiHeadAttention`, a linear readout) must output that key's value. chance is
    1/64, a loss of $\ln 64 = 4.16$. the same model and seed, scaled or not, at three widths, 600
    Adam steps at learning rate $10^{-3}$. the projections use PyTorch's default initialization,
    which is *not* standard normal: it already shrinks the scores somewhat.
    """)
    return


@app.cell
def _():
    width = mo.ui.dropdown(options=["32", "128", "512"], value="512", label="width shown in the table")
    p2 = ui.predict(
        "will the unscaled model still learn the task at d = 32? at d = 512? (Adam normalizes gradient sizes, so does "
        "a tiny gradient even matter?)"
    )
    mo.vstack([p2, width])
    return p2, width


@app.cell
def _(p2):
    ui.gate(p2)
    _t0 = time.perf_counter()
    runs = {(d, s): train_recall(d, s) for d in (32, 128, 512) for s in (True, False)}
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for _ax, _d in zip(_axes, (32, 128, 512)):
        for _s, _c in ((True, "C0"), (False, "C3")):
            _curve, _acc = runs[(_d, _s)]
            _ax.plot(np.arange(1, len(_curve) + 1) * 25, _curve, color=_c,
                     label=f"{'scaled' if _s else 'unscaled'} (acc {_acc:.2f})")
        _ax.axhline(math.log(N_KEYS), color="gray", ls="--", lw=0.8)
        _ax.set(title=f"d = {_d}", xlabel="step")
        _ax.legend(fontsize=7)
    _axes[0].set_ylabel("training loss")
    mo.vstack([_fig, mo.md(f"{time.perf_counter() - _t0:.0f} s")])
    return (runs,)


@app.cell
def _(runs, width):
    _d = int(width.value)
    mo.md(f"d = {_d}: final accuracy scaled **{runs[(_d, True)][1]:.3f}**, unscaled **{runs[(_d, False)][1]:.3f}**; "
          f"final loss {runs[(_d, True)][0][-1]:.3f} vs {runs[(_d, False)][0][-1]:.3f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the unscaled model's problem gets worse with width even though Adam
    divides every gradient by its own running size. which parameters are stuck, and why can't
    Adam's normalization rescue a gradient whose *direction* is uninformative? why does the
    small model barely care?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what attention knows about order

    take your `MultiHeadAttention` (random weights, no positions anywhere), feed it 6 tokens,
    then feed the same tokens shuffled. compare the outputs, without a mask and with
    `is_causal=True`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "without a mask, how different are the outputs for the shuffled tokens (after un-shuffling them)? with "
        "is_causal=True, does the model have any way to tell positions apart?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    torch.manual_seed(0)
    _m = ex.MultiHeadAttention(16, 4).eval()
    _x = torch.randn(1, 6, 16)
    _perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        _plain = (_m(_x)[:, _perm] - _m(_x[:, _perm])).abs().max().item()
        _causal = (_m(_x, is_causal=True)[:, _perm] - _m(_x[:, _perm], is_causal=True)).abs().max().item()
        # the same token everywhere: is the output the same at every position?
        _same = _x[:, :1].expand(1, 6, 16)
        _o_plain = _m(_same)
        _o_causal = _m(_same, is_causal=True)
    mo.md(f"""
    | | largest \\|f(x)[perm] − f(x[perm])\\| |
    |---|---|
    | no mask | {_plain:.1e} |
    | causal | {_causal:.2f} |

    six copies of the **same** token: spread of the outputs across positions, no mask
    {(_o_plain - _o_plain[:, :1]).abs().max().item():.1e}, causal
    {(_o_causal - _o_causal[:, :1]).abs().max().item():.1e}.

    now a sequence of *distinct* random tokens. the causal output at position 0 is always
    `out_proj(v_proj(x_0))` exactly, whatever x_0 is: a fact about position, delivered by the mask
    alone. max difference: {(_m(_x, is_causal=True)[0, 0] - _m.out_proj(_m.v_proj(_x[0, 0]))).abs().max().item():.1e}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** prove (lesson, on paper 3) that unmasked self-attention is
    permutation-equivariant. the causal mask breaks the symmetry, but the six identical tokens
    still give identical outputs: why? what *can* a causal model learn about position without
    any positional encoding (think of how many keys position $t$ averages over), and why is
    that not enough for a bidirectional model like BERT? topic 02 adds positions properly.
    """)
    return


if __name__ == "__main__":
    app.run()
