import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # heterogeneity · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `partition_dirichlet`, `label_distribution`, `heterogeneity`: tunable label skew, and a number for it |
    | 2 | `global_optimum`, `quadratic_fedavg`, `quadratic_fixed_point`: client drift, exactly, on quadratics |
    | 3 | `quadratic_scaffold`: control variates that remove the drift |
    | 4 | `fedprox_client_update`: local SGD with a proximal term, ready for your `fedavg` |
    | 5 | `fine_tune`, `per_client_accuracy`: personalization by local fine-tuning |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def partition_dirichlet(y, n_clients, alpha, *, seed=None, min_size=1, max_tries=100):
    """stage 1: per class, split its (shuffled) indices across clients in Dirichlet(alpha) proportions."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(y == c))
            p = rng.dirichlet(np.full(n_clients, float(alpha)))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        parts = [np.concatenate(ps).astype(np.int64) for ps in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"no split with every client holding >= {min_size} examples after {max_tries} tries; "
                       "lower min_size or raise alpha")


@app.function
def label_distribution(parts, y, n_classes):
    """stage 1: (n_clients, n_classes) array; row k is client k's label histogram, normalized to sum to 1."""
    y = np.asarray(y)
    out = np.zeros((len(parts), n_classes))
    for k, p in enumerate(parts):
        counts = np.bincount(y[np.asarray(p, dtype=np.int64)], minlength=n_classes)
        out[k] = counts / max(counts.sum(), 1)
    return out


@app.function
def heterogeneity(parts, y, n_classes):
    """stage 1: sum_k (n_k / n) * TV(client k's label distribution, the global one), a float in [0, 1]."""
    dist = label_distribution(parts, y, n_classes)
    sizes = np.array([len(p) for p in parts], dtype=float)
    glob = sizes @ dist / sizes.sum()
    tv = 0.5 * np.abs(dist - glob).sum(axis=1)
    return float(sizes @ tv / sizes.sum())


@app.cell(hide_code=True)
def _():
    checks.run(1, partition_dirichlet=partition_dirichlet, label_distribution=label_distribution,
               heterogeneity=heterogeneity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def global_optimum(As, cs, weights=None):
    """stage 2: the minimizer of sum_k p_k * 0.5 (w - c_k)^T A_k (w - c_k)."""
    As, cs = np.asarray(As, dtype=float), np.asarray(cs, dtype=float)
    p = np.full(len(As), 1 / len(As)) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    H = np.einsum("k,kij->ij", p, As)
    return np.linalg.solve(H, np.einsum("k,kij,kj->i", p, As, cs))


@app.function
def quadratic_fedavg(As, cs, w0, *, rounds, local_steps, lr, mu=0.0, weights=None):
    """stage 2: FedAvg (or FedProx if mu > 0) with full participation and local GD; (rounds + 1, d) iterates."""
    As, cs = np.asarray(As, dtype=float), np.asarray(cs, dtype=float)
    p = np.full(len(As), 1 / len(As)) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    w = np.asarray(w0, dtype=float).copy()
    out = [w.copy()]
    for _ in range(rounds):
        new = np.zeros_like(w)
        for pk, A, c in zip(p, As, cs):
            v = w.copy()
            for _ in range(local_steps):
                v = v - lr * (A @ (v - c) + mu * (v - w))
            new += pk * v
        w = new
        out.append(w.copy())
    return np.array(out)


@app.function
def quadratic_fixed_point(As, cs, *, local_steps, lr, mu=0.0, weights=None):
    """stage 2: the point quadratic_fedavg converges to (when it converges), in closed form."""
    As, cs = np.asarray(As, dtype=float), np.asarray(cs, dtype=float)
    p = np.full(len(As), 1 / len(As)) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    d = As.shape[1]
    I = np.eye(d)
    T_bar, s_bar = np.zeros((d, d)), np.zeros(d)
    for pk, A, c in zip(p, As, cs):
        G = I - lr * (A + mu * I)
        S, Gj = np.zeros((d, d)), I.copy()
        for _ in range(local_steps):
            S += Gj
            Gj = Gj @ G
        T_bar += pk * (Gj + lr * mu * S)  # Gj is now G^K
        s_bar += pk * lr * S @ A @ c
    return np.linalg.solve(I - T_bar, s_bar)


@app.cell(hide_code=True)
def _():
    checks.run(2, global_optimum=global_optimum, quadratic_fedavg=quadratic_fedavg,
               quadratic_fixed_point=quadratic_fixed_point)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def quadratic_scaffold(As, cs, w0, *, rounds, local_steps, lr, weights=None):
    """stage 3: SCAFFOLD (full participation, option II control variates, server lr 1); (rounds + 1, d) iterates."""
    As, cs = np.asarray(As, dtype=float), np.asarray(cs, dtype=float)
    p = np.full(len(As), 1 / len(As)) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    x = np.asarray(w0, dtype=float).copy()
    ci = np.zeros((len(As), len(x)))
    c = np.zeros(len(x))
    out = [x.copy()]
    for _ in range(rounds):
        dx, dc = np.zeros_like(x), np.zeros_like(x)
        for k, (A, ck) in enumerate(zip(As, cs)):
            v = x.copy()
            for _ in range(local_steps):
                v = v - lr * (A @ (v - ck) - ci[k] + c)
            new_ci = ci[k] - c + (x - v) / (local_steps * lr)
            dx += p[k] * (v - x)
            dc += p[k] * (new_ci - ci[k])
            ci[k] = new_ci
        x = x + dx
        c = c + dc
        out.append(x.copy())
    return np.array(out)


@app.cell(hide_code=True)
def _():
    checks.run(3, quadratic_scaffold=quadratic_scaffold, quadratic_fedavg=quadratic_fedavg,
               global_optimum=global_optimum, quadratic_fixed_point=quadratic_fixed_point)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fedprox_client_update(model, X, y, *, mu=0.01, epochs=1, batch_size=10, lr=0.1, loss_fn=None, seed=None):
    """stage 4: client_update on loss + (mu / 2) ||w - w_global||^2, w_global = the weights at entry; (weights, n_steps)."""
    loss_fn = F.cross_entropy if loss_fn is None else loss_fn
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    n = len(X)
    bs = n if batch_size is None else int(batch_size)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    anchor = [p.detach().clone() for p in params]
    model.train()
    steps = 0
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(n))
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[idx]), y[idx])
            loss = loss + 0.5 * mu * sum(((p - a) ** 2).sum() for p, a in zip(params, anchor))
            loss.backward()
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
            steps += 1
    weights = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    return weights, steps


@app.cell(hide_code=True)
def _():
    checks.run(4, fedprox_client_update=fedprox_client_update)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def fine_tune(model, X, y, *, steps=10, lr=0.05, batch_size=10, head_only=False, loss_fn=None, seed=None):
    """stage 5: a fine-tuned deep copy of model after `steps` SGD steps on (X, y); the original is untouched."""
    loss_fn = F.cross_entropy if loss_fn is None else loss_fn
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    local = copy.deepcopy(model)
    if head_only:
        head = [m for m in local.modules() if isinstance(m, nn.Linear)][-1]
        params = list(head.parameters())
    else:
        params = list(local.parameters())
    n = len(X)
    bs = n if batch_size is None else min(int(batch_size), n)
    rng = np.random.default_rng(seed)
    local.train()
    order, pos = rng.permutation(n), 0
    for _ in range(steps):
        if pos + bs > n:
            order, pos = rng.permutation(n), 0
        idx = torch.as_tensor(order[pos:pos + bs])
        pos += bs
        local.zero_grad(set_to_none=True)
        loss_fn(local(X[idx]), y[idx]).backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
    return local


@app.function
def per_client_accuracy(models, client_data):
    """stage 5: accuracy of model k (or of one shared model) on client k's (X_k, y_k); a NumPy array."""
    if isinstance(models, nn.Module):
        models = [models] * len(client_data)
    accs = []
    for m, (X, y) in zip(models, client_data):
        m.eval()
        with torch.no_grad():
            accs.append((m(torch.as_tensor(X)).argmax(1) == torch.as_tensor(y)).float().mean().item())
    return np.array(accs)


@app.cell(hide_code=True)
def _():
    checks.run(5, fine_tune=fine_tune, per_client_accuracy=per_client_accuracy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it runs your topic 27
    `fedavg` on increasingly skewed clients, plugs your `fedprox_client_update` into it, and
    measures what personalization buys each client.
    """)
    return


if __name__ == "__main__":
    app.run()
