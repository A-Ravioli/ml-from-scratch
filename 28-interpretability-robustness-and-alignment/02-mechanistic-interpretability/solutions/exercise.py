import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # mechanistic interpretability · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `run_with_cache`: read every activation of a forward pass with hooks → `mlfs.safety` |
    | 2 | `patch_activation`: run the model with one activation replaced → `mlfs.safety` |
    | 3 | `activation_patching`, `position_patching`: clean-into-corrupt patching, per layer and per position → `mlfs.safety` |
    | 4 | `attention_patterns`, `prev_token_score`, `induction_score`: read attention heads and score what they do |
    | 5 | `ablate_heads`: knock heads out to test what they cause |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def run_with_cache(model, idx, *, names=None):
    """stage 1: model(idx) with forward hooks on named submodules (None: all but the root); (output, {name: detached output})."""
    modules = dict(model.named_modules())
    if names is None:
        wanted = [n for n in modules if n]
    else:
        wanted = list(names)
        unknown = [n for n in wanted if n not in modules]
        if unknown:
            raise KeyError(f"no submodule named {unknown}; see dict(model.named_modules()) for the names")

    def detach(o):
        if torch.is_tensor(o):
            return o.detach()
        if isinstance(o, (tuple, list)):
            return type(o)(detach(v) for v in o)
        return o

    cache, handles = {}, []
    try:
        for n in wanted:
            handles.append(modules[n].register_forward_hook(
                lambda _m, _inp, out, n=n: cache.__setitem__(n, detach(out))))
        output = model(idx)
    finally:
        for h in handles:
            h.remove()
    return output, cache


@app.cell(hide_code=True)
def _():
    checks.run(1, run_with_cache=run_with_cache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def patch_activation(model, idx, name, value, *, positions=None):
    """stage 2: model(idx) with module `name`'s output replaced by value (broadcastable), only at positions (dim 1) if given."""
    module = dict(model.named_modules()).get(name)
    if module is None:
        raise KeyError(f"no submodule named {name!r}")

    def hook(_m, _inp, out):
        v = torch.as_tensor(value).to(device=out.device, dtype=out.dtype).expand_as(out)
        if positions is None:
            return v.clone()
        new = out.clone()
        new[:, positions] = v[:, positions]
        return new

    handle = module.register_forward_hook(hook)
    try:
        return model(idx)
    finally:
        handle.remove()


@app.cell(hide_code=True)
def _():
    checks.run(2, patch_activation=patch_activation, run_with_cache=run_with_cache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def activation_patching(model, clean_idx, corrupt_idx, metric, names):
    """stage 3: {name: metric(corrupt run with name's clean output patched in)}; eval mode (restored), no grad."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, cache = run_with_cache(model, clean_idx, names=names)
            return {n: float(metric(patch_activation(model, corrupt_idx, n, cache[n]))) for n in names}
    finally:
        model.train(was_training)


@app.function
def position_patching(model, clean_idx, corrupt_idx, metric, names):
    """stage 3: ndarray (len(names), T): metric after patching one name's clean output at one position t."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, cache = run_with_cache(model, clean_idx, names=names)
            T = clean_idx.shape[1]
            out = np.zeros((len(names), T))
            for i, n in enumerate(names):
                for t in range(T):
                    out[i, t] = float(metric(patch_activation(model, corrupt_idx, n, cache[n], positions=[t])))
            return out
    finally:
        model.train(was_training)


@app.cell(hide_code=True)
def _():
    checks.run(3, activation_patching=activation_patching, position_patching=position_patching,
               patch_activation=patch_activation, run_with_cache=run_with_cache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def attention_patterns(model, idx):
    """stage 4: [(B, H, T, T) attention weights of each block in model.blocks], via attn.save_weights; eval mode, flags restored."""
    attns = [block.attn for block in model.blocks]
    flags = [a.save_weights for a in attns]
    was_training = model.training
    model.eval()
    try:
        for a in attns:
            a.save_weights, a.last_weights = True, None
        with torch.no_grad():
            model(idx)
        return [a.last_weights for a in attns]
    finally:
        for a, f in zip(attns, flags):
            a.save_weights = f
        model.train(was_training)


@app.function
def prev_token_score(weights):
    """stage 4: (H,) mean attention from position t to t - 1, over t >= 1 and the batch; weights (B, H, T, T)."""
    T = weights.shape[-1]
    t = torch.arange(1, T, device=weights.device)
    return weights[..., t, t - 1].mean(dim=(0, 2))


@app.function
def induction_score(weights, period):
    """stage 4: (H,) mean attention from t to t - period + 1 over t >= period (sequences repeating every `period`)."""
    T = weights.shape[-1]
    if not 1 <= period < T:
        raise ValueError(f"period must be in [1, T) = [1, {T}), got {period}")
    t = torch.arange(period, T, device=weights.device)
    return weights[..., t, t - period + 1].mean(dim=(0, 2))


@app.cell(hide_code=True)
def _():
    checks.run(4, attention_patterns=attention_patterns, prev_token_score=prev_token_score,
               induction_score=induction_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def ablate_heads(model, idx, heads):
    """stage 5: model(idx) with each (layer, head) in heads zeroed at the input of blocks[layer].attn.out_proj (or o_proj)."""
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    handles = []
    try:
        for layer, hs in by_layer.items():
            attn = model.blocks[layer].attn
            proj = attn.out_proj if hasattr(attn, "out_proj") else attn.o_proj
            dh = attn.head_dim

            def pre_hook(_m, args, hs=hs, dh=dh):
                x = args[0].clone()
                for h in hs:
                    x[..., h * dh:(h + 1) * dh] = 0.0
                return (x,) + tuple(args[1:])

            handles.append(proj.register_forward_pre_hook(pre_hook))
        return model(idx)
    finally:
        for h in handles:
            h.remove()


@app.cell(hide_code=True)
def _():
    checks.run(5, ablate_heads=ablate_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import run_with_cache, patch_activation, activation_patching` lets you
    read and edit any model's activations (TransformerLens-style hooks are unlocked). topic 03
    asks what the directions in those activations mean when there are more features than
    dimensions.
    """)
    return


if __name__ == "__main__":
    app.run()
