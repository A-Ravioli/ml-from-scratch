# mechanistic interpretability · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`modules = dict(model.named_modules())`. register `modules[n].register_forward_hook(lambda m, inp, out, n=n: cache.__setitem__(n, detach(out)))` for each name (the `n=n` default freezes the loop variable), keep the handles, run the model inside `try`, and call `h.remove()` on every handle in `finally`. `detach` is a small recursive helper: tensors → `.detach()`, tuples and lists → the same type of their detached elements, anything else unchanged.

</details>

## stage 2

<details><summary>hint</summary>

a forward hook that *returns* a value replaces the module's output. inside it: `v = torch.as_tensor(value).to(out).expand_as(out)`; without positions return `v.clone()`; with positions, `new = out.clone(); new[:, positions] = v[:, positions]; return new`. register, run inside `try`, remove in `finally`.

</details>

## stage 3

<details><summary>hint</summary>

remember the mode, `model.eval()`, and under `torch.no_grad()`: `_, cache = run_with_cache(model, clean_idx, names=names)`, then one `patch_activation(model, corrupt_idx, n, cache[n])` per name (with `positions=[t]` inside a loop over `t` for the grid). wrap the metric in `float(...)`. restore the mode in `finally`.

</details>

## stage 4

<details><summary>hint</summary>

`attns = [b.attn for b in model.blocks]`; save their `save_weights` flags, set them all to True, run under `no_grad` in eval mode, collect `a.last_weights`, and restore everything in `finally`. for the scores, advanced indexing does it in one line: `t = torch.arange(1, T)`, then `weights[..., t, t - 1]` has shape `(B, H, T - 1)`; average over dims 0 and 2. the induction version uses `t = torch.arange(period, T)` and `t - period + 1`.

</details>

## stage 5

<details><summary>hint</summary>

group the heads by layer. for each layer, `proj = attn.out_proj if hasattr(attn, "out_proj") else attn.o_proj` and register a `register_forward_pre_hook(fn)`: `fn(module, args)` gets the input tuple and may return a new one: clone `args[0]`, set `x[..., h * dh:(h + 1) * dh] = 0` for each head, return `(x,)`. bind the layer's head list and `dh` as default arguments; remove the hooks in `finally`.

</details>
