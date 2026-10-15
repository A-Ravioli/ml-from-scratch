import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _n_train = min(20000, len(_d["X_train"]))
    X_train = (_d["X_train"][:_n_train, None].astype(np.float64) / 255.0 - 0.13) / 0.31
    y_train = _d["y_train"][:_n_train]
    X_test = (_d["X_test"][:5000, None].astype(np.float64) / 255.0 - 0.13) / 0.31
    y_test = _d["y_test"][:5000]


@app.function
def hand_made_filters():
    """eight 5x5 filters from chapter 00's capstone: blur, gaussian, sobel x/y, two diagonals, laplacian, sharpen."""
    def embed(k3):
        k = np.zeros((5, 5))
        k[1:4, 1:4] = k3
        return k

    ax = np.arange(5) - 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / 2.0)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    diag = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=float)
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    sharpen = -np.ones((3, 3)) / 9
    sharpen[1, 1] += 2.0
    bank = [np.ones((5, 5)) / 25, g / g.sum(), embed(sobel_x), embed(sobel_x.T), embed(diag), embed(diag[:, ::-1]),
            embed(lap), embed(sharpen)]
    names = ["box blur", "gaussian", "sobel x", "sobel y", "diagonal /", "diagonal \\", "laplacian", "sharpen"]
    bank = np.stack([k / np.linalg.norm(k) for k in bank])[:, None]  # unit norm, (8, 1, 5, 5)
    return bank, names


@app.function
def train_tiny_cnn(w0, *, train_filters=True, epochs=2, lr=0.05, batch_size=64, seed=0):
    """conv(1 -> 8, 5x5, pad 2) -> ReLU -> 4x4 average pool -> Linear(392 -> 10), trained with YOUR conv2d and
    conv2d_backward and plain SGD with momentum. returns (filters, test accuracy, seconds)."""
    rng = np.random.default_rng(seed)
    w = w0.copy()
    b = np.zeros(w.shape[0])
    k = w.shape[0] * 7 * 7
    V = rng.normal(scale=np.sqrt(1.0 / k), size=(k, 10))
    c = np.zeros(10)
    params = [w, b, V, c]
    vel = [np.zeros_like(p) for p in params]

    def forward(x):
        z = ex.conv2d(x, w, b, padding=2)  # (B, 8, 28, 28)
        h = np.maximum(z, 0.0)
        p = h.reshape(len(x), w.shape[0], 7, 4, 7, 4).mean(axis=(3, 5))  # (B, 8, 7, 7)
        return z, h, p, p.reshape(len(x), -1) @ V + c

    t0 = time.perf_counter()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(X_train)), max(1, len(X_train) // batch_size)):
            x, y = X_train[idx], y_train[idx]
            z, h, p, logits = forward(x)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            g = probs
            g[np.arange(len(y)), y] -= 1.0
            g /= len(y)
            dV, dc = p.reshape(len(x), -1).T @ g, g.sum(axis=0)
            dp = (g @ V.T).reshape(p.shape)
            dh = np.repeat(np.repeat(dp, 4, axis=2), 4, axis=3) / 16.0
            dz = dh * (z > 0)
            _, dw, db = ex.conv2d_backward(dz, x, w, padding=2)
            grads = [dw, db, dV, dc] if train_filters else [None, None, dV, dc]
            for p_, v_, g_ in zip(params, vel, grads):
                if g_ is not None:
                    v_ *= 0.9
                    v_ -= lr * g_
                    p_ += v_
    seconds = time.perf_counter() - t0
    acc = float(np.mean(np.concatenate([forward(X_test[i:i + 500])[3].argmax(axis=1)
                                        for i in range(0, len(X_test), 500)]) == y_test))
    return w, acc, seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # convolution · lab

    every experiment here uses **your** `conv2d_naive`, `im2col`, `conv2d` and
    `conv2d_backward` from `exercise.py`, on MNIST digits (20,000 for training, 5,000 for
    testing; offline, a synthetic stand-in with the same shapes). for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: loops, a matrix product, and PyTorch

    one forward pass of a $3 \times 3$ convolution, 16 → 16 channels with padding 1, on a
    batch of 8 images, for image sizes from 8 to 64 pixels: your `conv2d_naive` (loops over
    output pixels), your `conv2d` (`im2col` + one matmul) and `torch.nn.functional.conv2d`
    in float64 on the CPU. we also record how much memory `im2col`'s matrix takes compared
    with the input itself. each time is the best of 3.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at 64x64 pixels, how many times faster is your im2col conv2d than your loops? and how many times more memory "
        "does im2col's matrix take than the image it was built from?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    def _best(fn, reps=3):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    _rng = np.random.default_rng(0)
    _w = _rng.normal(size=(16, 16, 3, 3))
    _sizes, _rows, _t = [8, 16, 32, 64], [], {"naive": [], "im2col": [], "torch": []}
    for _s in _sizes:
        _x = _rng.normal(size=(8, 16, _s, _s))
        _t["naive"].append(_best(lambda: ex.conv2d_naive(_x, _w, padding=1)))
        _t["im2col"].append(_best(lambda: ex.conv2d(_x, _w, padding=1)))
        _xt, _wt = torch.tensor(_x), torch.tensor(_w)
        _t["torch"].append(_best(lambda: torch.nn.functional.conv2d(_xt, _wt, padding=1)))
        _mem = ex.im2col(_x, 3, padding=1).nbytes / _x.nbytes
        _rows.append(f"| {_s} | {_t['naive'][-1] * 1e3:.1f} | {_t['im2col'][-1] * 1e3:.2f} | "
                     f"{_t['torch'][-1] * 1e3:.2f} | {_t['naive'][-1] / _t['im2col'][-1]:.0f}× | {_mem:.1f}× |")
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    for _k, _v in _t.items():
        _ax.loglog(_sizes, np.array(_v) * 1e3, "o-", label=_k, base=2)
    _ax.set(xlabel="image size (pixels per side)", ylabel="ms per forward pass")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| size | naive (ms) | im2col (ms) | torch (ms) | naive / im2col | im2col memory / input |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both versions do exactly the same multiplications. where does the naive
    version's time go (count the Python-level operations per forward pass)? why is the
    memory ratio almost exactly $k^2 = 9$, and what would it be for a $7 \times 7$ kernel?
    PyTorch is faster still: what might it do that avoids building the whole matrix?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what does a convolutional layer learn?

    a tiny network, conv($1 \to 8$, $5 \times 5$) → ReLU → $4 \times 4$ average pooling →
    linear → 10 classes, trained for 2 epochs with **your** `conv2d` and `conv2d_backward`
    (and a hand-written backward for the rest) from small random filters. then we put its 8
    learned filters next to the 8 filters you built by hand in chapter 00 (blur, gaussian,
    sobel, diagonal edges, laplacian, sharpen), and for each learned filter find the
    hand-made one it resembles most (largest $|\cos|$ between the flattened kernels).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "will the learned filters look like any of your hand-made ones (edges, blurs)? or like noise? what's the "
        "typical best |cosine| with the hand-made bank: 0.2, 0.5, 0.9?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    bank, bank_names = hand_made_filters()
    _w0 = np.random.default_rng(1).normal(scale=0.1, size=(8, 1, 5, 5))
    learned, acc_learned, _sec = train_tiny_cnn(_w0, train_filters=True)
    _flat_l = learned.reshape(8, -1) / np.linalg.norm(learned.reshape(8, -1), axis=1, keepdims=True)
    _cos = _flat_l @ bank.reshape(8, -1).T  # (learned, hand-made)
    _best = np.abs(_cos).argmax(axis=1)
    _fig, _axes = plt.subplots(2, 8, figsize=(11, 3.2))
    for _i in range(8):
        _axes[0, _i].imshow(learned[_i, 0], cmap="RdBu_r", vmin=-np.abs(learned).max(), vmax=np.abs(learned).max())
        _axes[0, _i].set_title(f"learned {_i}\n|cos| {np.abs(_cos[_i, _best[_i]]):.2f}", fontsize=8)
        _axes[1, _i].imshow(bank[_best[_i], 0] * np.sign(_cos[_i, _best[_i]]), cmap="RdBu_r",
                            vmin=-np.abs(bank).max(), vmax=np.abs(bank).max())
        _axes[1, _i].set_title(f"nearest: {bank_names[_best[_i]]}", fontsize=8)
    for _a in _axes.ravel():
        _a.set(xticks=[], yticks=[])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"test accuracy of the trained network: **{acc_learned:.3f}** (trained in {_sec:.1f} s). "
                           f"best |cos| per learned filter: "
                           + ", ".join(f"{v:.2f}" for v in np.abs(_cos).max(axis=1))
                           + f" · median **{np.median(np.abs(_cos).max(axis=1)):.2f}**. "
                             "(two random 5×5 filters have |cos| ≈ 0.16 on average; the best of 8 random "
                             "matches is typically about 0.3.)")])
    return acc_learned, bank


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the second row shows each learned filter's nearest hand-made filter,
    with its sign matched (red positive, blue negative). which kinds of hand-made filter
    does the network rediscover, and which never show up? why might a filter feeding into a
    ReLU and an average pool want a large mean (a "blur" component) or a strong
    positive/negative split? nothing told the network about edges: where did they come from?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: do the filters need to be learned at all?

    the same network, three ways: the filters learned (experiment 2's run), the filters
    **frozen** at your 8 hand-made ones, and the filters **frozen at random** (unit-norm
    Gaussian noise). in the frozen runs only the linear layer on top learns. use the
    dropdown to choose how many filters the frozen banks get (the learned network keeps 8).
    """)
    return


@app.cell
def _():
    n_filters = mo.ui.dropdown(options=["8", "32"], value="8", label="filters in the frozen banks")
    p3 = ui.predict(
        "rank the three: learned, hand-made frozen, random frozen. how far behind the learned filters is a bank of "
        "pure random noise?"
    )
    mo.vstack([p3, n_filters])
    return n_filters, p3


@app.cell
def _(acc_learned, bank, n_filters, p3):
    ui.gate(p3)
    _k = int(n_filters.value)
    _rng = np.random.default_rng(2)
    _rand = _rng.normal(size=(_k, 1, 5, 5))
    _rand /= np.linalg.norm(_rand.reshape(_k, -1), axis=1).reshape(-1, 1, 1, 1)
    _hand = bank if _k == 8 else np.concatenate([bank, _rng.normal(size=(_k - 8, 1, 5, 5)) * 0.2])
    _, _acc_hand, _ = train_tiny_cnn(_hand, train_filters=False)
    _, _acc_rand, _ = train_tiny_cnn(_rand, train_filters=False)
    _base = float(np.mean(y_test == np.bincount(y_train).argmax()))
    _se = lambda a: np.sqrt(a * (1 - a) / len(y_test))  # noqa: E731
    mo.md(f"""
    | filters | trained? | test accuracy (± 1 s.e.) |
    |---|---|---|
    | 8 learned | yes | **{acc_learned:.3f}** ± {_se(acc_learned):.3f} |
    | {'8 hand-made' if _k == 8 else f'8 hand-made + {_k - 8} small random'} | frozen | **{_acc_hand:.3f}** ± {_se(_acc_hand):.3f} |
    | {_k} random | frozen | **{_acc_rand:.3f}** ± {_se(_acc_rand):.3f} |
    | majority class | | {_base:.3f} |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how much of the learned network's accuracy does a bank of random noise
    already deliver? what does that say about where the work in this network is done (the
    filters, the ReLU and pooling, or the linear layer)? random filters are still *local*
    and *shared across positions*: which of those two properties do you think matters
    more, and how would you test it? why would this gap grow in a deeper network on harder
    images?
    """)
    return


if __name__ == "__main__":
    app.run()
