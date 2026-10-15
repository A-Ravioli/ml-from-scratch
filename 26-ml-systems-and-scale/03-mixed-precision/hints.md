# mixed precision · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
bias = 2 ** (exp_bits - 1) - 1
a = np.abs(x)
_, e = np.frexp(a)                       # a = f * 2**e, f in [0.5, 1)
e = np.maximum(e - 1, 1 - bias)          # exponent of the leading bit, floored at the subnormal range
q = np.ldexp(1.0, e - man_bits)          # spacing of the format around a
r = np.round(a / q) * q                  # round half to even
r = np.where(r > fmt_max, np.inf, r)
r = np.where(np.isfinite(a), r, a)       # pass inf and nan through
return np.copysign(r, x)
```

wrap it in `np.errstate(...)` to silence the warnings that `frexp` and the division give for 0, inf and nan. `np.copysign` keeps the sign of $-0$.

</details>

## stage 2

<details><summary>hint</summary>

round every gradient with `round_to_format` first. without a master copy, round `w0` at the start and `w - lr * g` after every update; with one, keep `w` in float64 (`np.asarray(w0, dtype=np.float64).copy()`), update it without rounding, and round only the value you return.

</details>

## stage 3

<details><summary>hint</summary>

keep four pieces of state: the scale, a clean-step counter, a `found_inf` flag and an `unscaled` flag. `unscale_` raises if `unscaled` is already set, multiplies each `p.grad` by `1 / scale` (skip `None`), sets `found_inf` if any `not torch.isfinite(p.grad).all()`, and sets `unscaled`. `step` unscales if needed and returns `False` (no `optimizer.step()`) when `found_inf`. `update` backs off and zeroes the counter on `found_inf`, else increments it and grows the scale (zeroing the counter) when it reaches `growth_interval`; then clears both flags.

</details>

## stage 4

<details><summary>hint</summary>

the lesson's six-line step is nearly the whole function. return `float(loss.detach())` and the value of `scaler.step(optimizer)` (or `True` without a scaler). without a scaler, call `loss.float().backward()`, clip if asked, and `optimizer.step()`.

</details>
