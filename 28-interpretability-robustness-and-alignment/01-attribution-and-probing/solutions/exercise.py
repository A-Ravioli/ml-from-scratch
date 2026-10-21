import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    from scipy import stats as sps

    from mlfs import checks, lesson
    from mlfs.ssl import linear_probe  # yours, from ch 25 · 01
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # attribution & probing · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `saliency`: the gradient of a class logit with respect to the input → `mlfs.safety` |
    | 2 | `integrated_gradients`: the path integral that makes attributions add up → `mlfs.safety` |
    | 3 | `control_labels`, `probe_selectivity`: linear probes, and a control task that keeps them honest |
    | 4 | `rank_correlation`, `randomize_layers`, `cascading_randomization`: the sanity checks for saliency maps |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def saliency(model, x, target):
    """stage 1: |d model(x)[b, target_b] / d x|, same shape as x; eval mode (restored after); parameter grads untouched."""
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():  # works inside torch.no_grad() too
            x = x.detach().clone().requires_grad_(True)
            out = model(x)
            tgt = torch.as_tensor(target, device=out.device).long().expand(out.shape[0])
            picked = out.gather(1, tgt[:, None]).sum()
            (grad,) = torch.autograd.grad(picked, x)
    finally:
        model.train(was_training)
    return grad.abs()


@app.cell(hide_code=True)
def _():
    checks.run(1, saliency=saliency)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def integrated_gradients(model, x, target, *, baseline=None, n_steps=50):
    """stage 2: (x - baseline) * trapezoid average of the gradient at baseline + k/n_steps (x - baseline), k = 0..n_steps."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    x = x.detach()
    base = torch.zeros_like(x) if baseline is None else torch.as_tensor(baseline).to(x).expand_as(x)
    was_training = model.training
    model.eval()
    try:
        tgt = None
        total = torch.zeros_like(x)
        for k in range(n_steps + 1):
            weight = (0.5 if k in (0, n_steps) else 1.0) / n_steps
            with torch.enable_grad():
                xk = (base + (k / n_steps) * (x - base)).requires_grad_(True)
                out = model(xk)
                if tgt is None:
                    tgt = torch.as_tensor(target, device=out.device).long().expand(out.shape[0])
                (grad,) = torch.autograd.grad(out.gather(1, tgt[:, None]).sum(), xk)
            total = total + weight * grad
    finally:
        model.train(was_training)
    return (x - base) * total


@app.cell(hide_code=True)
def _():
    checks.run(2, integrated_gradients=integrated_gradients)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def control_labels(types, n_classes, *, rng=None):
    """stage 3: one uniformly random label in range(n_classes) per distinct type (Hewitt & Liang); (n,) int64."""
    rng = as_rng(rng)
    types = np.asarray(types)
    uniq, inverse = np.unique(types, return_inverse=True)
    table = rng.integers(0, n_classes, size=len(uniq))
    return table[inverse.reshape(-1)].astype(np.int64)


@app.function
def probe_selectivity(train_feats, train_labels, test_feats, test_labels, train_types, test_types, *, rng=None):
    """stage 3: linear-probe accuracy on the task and on a control task (one label table for both splits); dict."""
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    n_classes = int(max(train_labels.max(), test_labels.max())) + 1
    types = np.concatenate([np.asarray(train_types), np.asarray(test_types)])
    control = control_labels(types, n_classes, rng=rng)
    c_train, c_test = control[:len(train_labels)], control[len(train_labels):]
    task = linear_probe(train_feats, train_labels, test_feats, test_labels)
    ctrl = linear_probe(train_feats, c_train, test_feats, c_test)
    return {"task": float(task), "control": float(ctrl), "selectivity": float(task - ctrl)}


@app.cell(hide_code=True)
def _():
    checks.run(3, control_labels=control_labels, probe_selectivity=probe_selectivity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def rank_correlation(a, b):
    """stage 4: Spearman correlation of two equal-size arrays (flattened, ties get average ranks)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"arrays have {a.size} and {b.size} elements")
    ra, rb = sps.rankdata(a), sps.rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


@app.function
def randomize_layers(model, n, *, generator=None):
    """stage 4: deep copy with the LAST n parameterised modules re-initialised (reset_parameters, seeded by generator);
    the original and the global RNG state are left unchanged."""
    clone = copy.deepcopy(model)
    layers = [m for m in clone.modules() if any(True for _ in m.parameters(recurse=False))]
    if not 0 <= n <= len(layers):
        raise ValueError(f"the model has {len(layers)} parameterised layers, can't randomize {n}")
    for m in layers[len(layers) - n:]:
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator).item())
        fresh = copy.deepcopy(m).to("cpu")  # re-initialise on the CPU: only the CPU generator is touched
        with torch.random.fork_rng(devices=[]):  # ...and restored afterwards
            torch.random.default_generator.manual_seed(seed)
            fresh.reset_parameters()
        with torch.no_grad():
            for p, q in zip(m.parameters(recurse=False), fresh.parameters(recurse=False)):
                p.copy_(q.to(device=p.device, dtype=p.dtype))
    return clone


@app.function
def cascading_randomization(model, attribute, x, target, *, generator=None):
    """stage 4: Spearman(|attr| of model, |attr| after randomizing the top k layers) for k = 1..L; ndarray (L,)."""
    reference = attribute(model, x, target).detach().abs().cpu().numpy()
    n_layers = sum(1 for m in model.modules() if any(True for _ in m.parameters(recurse=False)))
    out = []
    for k in range(1, n_layers + 1):
        rand = randomize_layers(model, k, generator=generator)
        attr = attribute(rand, x, target).detach().abs().cpu().numpy()
        out.append(rank_correlation(reference, attr))
    return np.array(out)


@app.cell(hide_code=True)
def _():
    checks.run(4, rank_correlation=rank_correlation, randomize_layers=randomize_layers,
               cascading_randomization=cascading_randomization, saliency=saliency)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import saliency, integrated_gradients` gives you both attribution
    methods (Captum's `Saliency` and `IntegratedGradients` are unlocked). topic 02 moves from
    the input to the inside of the model: caching and patching its activations.
    """)
    return


if __name__ == "__main__":
    app.run()
