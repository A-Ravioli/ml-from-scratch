import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # federated averaging · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `get_weights`, `set_weights`, `fedavg_aggregate`: models as vectors, and the server's weighted average |
    | 2 | `client_update`: local SGD on one client's data |
    | 3 | `partition_iid`, `partition_shards`: splitting a dataset across clients |
    | 4 | `fedavg`: the simulator, with byte-level communication accounting |
    | 5 | `stochastic_quantize`: unbiased low-bit compression of an update |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def get_weights(model):
    """stage 1: every parameter of `model`, flattened in model.parameters() order into one detached copy."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


@app.function
def set_weights(model, weights):
    """stage 1: copy the flat vector `weights` into model's parameters in place (ValueError on a size mismatch)."""
    total = sum(p.numel() for p in model.parameters())
    if weights.numel() != total:
        raise ValueError(f"weights has {weights.numel()} entries but the model has {total} parameters")
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(weights[offset:offset + n].reshape(p.shape))
            offset += n
    return model


@app.function
def fedavg_aggregate(client_weights, num_samples):
    """stage 1: the average of the client vectors, each weighted by its number of samples."""
    if len(client_weights) == 0:
        raise ValueError("no client vectors to aggregate")
    n = torch.as_tensor(np.asarray(num_samples, dtype=np.float64))
    if (n < 0).any() or n.sum() <= 0:
        raise ValueError("num_samples must be non-negative with a positive total")
    stacked = torch.stack(list(client_weights))
    coef = (n / n.sum()).to(stacked.dtype)
    return coef @ stacked


@app.cell(hide_code=True)
def _():
    checks.run(1, get_weights=get_weights, set_weights=set_weights, fedavg_aggregate=fedavg_aggregate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def client_update(model, X, y, *, epochs=1, batch_size=10, lr=0.1, loss_fn=None, seed=None):
    """stage 2: `epochs` passes of minibatch SGD on (X, y), in place; returns (get_weights(model), n_steps)."""
    loss_fn = F.cross_entropy if loss_fn is None else loss_fn
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    n = len(X)
    bs = n if batch_size is None else int(batch_size)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    steps = 0
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(n))
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            model.zero_grad(set_to_none=True)
            loss_fn(model(X[idx]), y[idx]).backward()
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
            steps += 1
    return get_weights(model), steps


@app.cell(hide_code=True)
def _():
    checks.run(2, client_update=client_update, get_weights=get_weights)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def partition_iid(n, n_clients, *, seed=None):
    """stage 3: a random split of range(n) into n_clients index arrays whose sizes differ by at most 1."""
    perm = np.random.default_rng(seed).permutation(n)
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(perm, n_clients)]


@app.function
def partition_shards(y, n_clients, shards_per_client=2, *, seed=None):
    """stage 3: McMahan et al.'s pathological split: sort by label, cut into shards, deal shards_per_client to each."""
    y = np.asarray(y)
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, n_clients * shards_per_client)
    deal = np.random.default_rng(seed).permutation(len(shards))
    return [np.concatenate([shards[s] for s in deal[k * shards_per_client:(k + 1) * shards_per_client]]).astype(np.int64)
            for k in range(n_clients)]


@app.cell(hide_code=True)
def _():
    checks.run(3, partition_iid=partition_iid, partition_shards=partition_shards)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fedavg(model, client_data, *, rounds, client_fraction=0.1, local_epochs=1, batch_size=10, lr=0.1,
           local_update=None, compress=None, eval_fn=None, eval_every=1, seed=None):
    """stage 4: FedAvg on client_data = [(X_k, y_k), ...], training `model` in place; returns a history dict."""
    local_update = client_update if local_update is None else local_update
    rng = np.random.default_rng(seed)
    N = len(client_data)
    m = max(1, int(round(client_fraction * N)))
    w = get_weights(model)
    msg_bytes = w.numel() * w.element_size()
    hist = {"round": [], "bytes_up": [], "bytes_down": [], "client_steps": [], "clients": [],
            "eval_round": [], "eval": []}
    up = down = steps = 0
    if eval_fn is not None:
        hist["eval_round"].append(0)
        hist["eval"].append(eval_fn(model))
    for r in range(1, rounds + 1):
        chosen = rng.choice(N, size=m, replace=False)
        w = get_weights(model)
        deltas, sizes = [], []
        for k in chosen:
            X, y = client_data[k]
            set_weights(model, w)
            down += msg_bytes
            w_k, s = local_update(model, X, y, epochs=local_epochs, batch_size=batch_size, lr=lr,
                                  seed=int(rng.integers(2**31)))
            delta = w_k - w
            if compress is None:
                up += msg_bytes
            else:
                delta, nb = compress(delta, rng)
                up += int(nb)
            deltas.append(delta)
            sizes.append(len(y))
            steps += s
        set_weights(model, w + fedavg_aggregate(deltas, sizes))
        hist["round"].append(r)
        hist["bytes_up"].append(up)
        hist["bytes_down"].append(down)
        hist["client_steps"].append(steps)
        hist["clients"].append(np.asarray(chosen))
        if eval_fn is not None and (r % eval_every == 0 or r == rounds):
            hist["eval_round"].append(r)
            hist["eval"].append(eval_fn(model))
    return hist


@app.cell(hide_code=True)
def _():
    checks.run(4, fedavg=fedavg, get_weights=get_weights, set_weights=set_weights, client_update=client_update)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def stochastic_quantize(v, bits, *, rng=None):
    """stage 5: unbiased stochastic rounding of v onto 2**bits evenly spaced levels in [min v, max v]; (v_hat, n_bytes)."""
    if not (isinstance(bits, (int, np.integer)) and 1 <= bits <= 16):
        raise ValueError(f"bits must be an integer in 1..16, got {bits!r}")
    rng = np.random.default_rng() if rng is None else rng
    n_bytes = math.ceil(v.numel() * bits / 8) + 8
    lo, hi = v.min(), v.max()
    if hi == lo:
        return v.clone(), n_bytes
    levels = 2 ** bits - 1
    x = (v - lo) / (hi - lo) * levels
    low = torch.floor(x)
    u = torch.as_tensor(rng.random(v.shape), dtype=v.dtype)
    q = torch.clamp(low + (u < (x - low)).to(v.dtype), 0, levels)
    return lo + q * (hi - lo) / levels, n_bytes


@app.cell(hide_code=True)
def _():
    checks.run(5, stochastic_quantize=stochastic_quantize)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, you have a federated-learning simulator. open the lab
    (`marimo edit lab.py`) to find out how much local work buys you in communication rounds.
    topics 28 and 29 plug new client updates and new aggregators into your `fedavg`.
    """)
    return


if __name__ == "__main__":
    app.run()
