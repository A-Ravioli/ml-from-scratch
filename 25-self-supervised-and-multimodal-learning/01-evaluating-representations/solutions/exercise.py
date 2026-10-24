import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.linear import SoftmaxRegression  # yours, from ch 08 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # evaluating representations · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `extract_features`: run a frozen encoder over a loader, in eval mode → `mlfs.ssl` |
    | 2 | `linear_probe`: standardise, fit a softmax regression, report test accuracy → `mlfs.ssl` |
    | 3 | `knn_eval`: cosine-similarity weighted kNN, in memory-bounded chunks → `mlfs.ssl` |
    | 4 | `fine_tune`: a linear head on top of the encoder, trained end to end or frozen |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def extract_features(encoder, loader, *, device="cpu"):
    """stage 1: (feats float32 (n, d), labels (n,)) ndarrays from encoder over (x, y) batches; eval mode + no_grad, modes restored."""
    modes = {m: m.training for m in encoder.modules()}
    encoder.eval()
    feats, labels = [], []
    try:
        with torch.no_grad():
            for x, y in loader:
                out = encoder(x.to(device))
                feats.append(out.reshape(out.shape[0], -1).float().cpu().numpy())
                labels.append(np.asarray(y.cpu() if torch.is_tensor(y) else y))
    finally:
        for m, mode in modes.items():
            m.training = mode
    if not feats:
        raise ValueError("the loader yielded no batches")
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


@app.cell(hide_code=True)
def _():
    checks.run(1, extract_features=extract_features)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def linear_probe(train_feats, train_labels, test_feats, test_labels, *, alpha=0.0001, max_iters=1000):
    """stage 2: standardise with TRAIN statistics, fit SoftmaxRegression(alpha, max_iters), return test accuracy (float)."""
    Xtr = np.asarray(train_feats, dtype=np.float64)
    Xte = np.asarray(test_feats, dtype=np.float64)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    clf = SoftmaxRegression(alpha=alpha, max_iters=max_iters).fit((Xtr - mu) / sd, np.asarray(train_labels))
    pred = clf.predict((Xte - mu) / sd)
    return float(np.mean(pred == np.asarray(test_labels)))


@app.cell(hide_code=True)
def _():
    checks.run(2, linear_probe=linear_probe)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def knn_eval(train_feats, train_labels, test_feats, test_labels, *, k=20, temperature=0.07):
    """stage 3: cosine-similarity kNN, neighbour votes weighted exp(sim / temperature); test accuracy (float)."""
    def unit(a):
        a = np.asarray(a, dtype=np.float32)
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    tr, te = unit(train_feats), unit(test_feats)
    classes, y_idx = np.unique(np.asarray(train_labels), return_inverse=True)
    n_train, n_classes = len(tr), len(classes)
    k = min(int(k), n_train)
    chunk = max(1, (1 << 21) // max(n_train, 1))  # keeps each similarity block near 2M entries
    preds = []
    for start in range(0, len(te), chunk):
        sims = te[start:start + chunk] @ tr.T  # (c, n_train)
        nn_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        nn_sim = np.take_along_axis(sims, nn_idx, axis=1).astype(np.float64)
        w = np.exp((nn_sim - nn_sim.max(axis=1, keepdims=True)) / temperature)  # stable: same argmax
        votes = np.zeros((len(nn_idx), n_classes))
        np.add.at(votes, (np.arange(len(nn_idx))[:, None], y_idx[nn_idx]), w)
        preds.append(classes[votes.argmax(axis=1)])
    pred = np.concatenate(preds)
    return float(np.mean(pred == np.asarray(test_labels)))


@app.cell(hide_code=True)
def _():
    checks.run(3, knn_eval=knn_eval)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fine_tune(encoder, feature_dim, n_classes, loader, *, epochs=1, lr=1e-3, freeze=False, device="cpu", seed=None):
    """stage 4: nn.Sequential(encoder, Linear(feature_dim, n_classes)) trained with Adam(lr) for `epochs`; freeze keeps the encoder untouched."""
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        head = nn.Linear(feature_dim, n_classes)
    model = nn.Sequential(encoder, head).to(device)
    params = head.parameters() if freeze else model.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        if freeze:
            encoder.eval()  # frozen: BatchNorm statistics must not move either
            head.train()
        else:
            model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if freeze:
                with torch.no_grad():
                    h = encoder(x)
                logits = head(h)
            else:
                logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@app.cell(hide_code=True)
def _():
    checks.run(4, fine_tune=fine_tune)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `extract_features`, `linear_probe` and `knn_eval` are part of
    `mlfs.ssl`: every later topic in this chapter (and chapter 28) evaluates its
    representations with them. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
