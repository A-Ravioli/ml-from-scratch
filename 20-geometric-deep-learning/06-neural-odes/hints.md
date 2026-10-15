# neural ODEs · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

translate the formulas literally; `t` is a tensor and `h` a number or tensor, so `t + h / 2` works. RK4: `k1 = f(t, y)`, `k2 = f(t + h/2, y + h/2 * k1)`, `k3 = f(t + h/2, y + h/2 * k2)`, `k4 = f(t + h, y + h * k3)`, then `y + h/6 * (k1 + 2*k2 + 2*k3 + k4)`. no in-place updates of `y` (autograd needs the old values).

</details>

## stage 2

<details><summary>hint</summary>

`steps_per_interval`: `[1] * (len(t) - 1)` without a step size, else `max(1, math.ceil(abs(dt) / step_size - 1e-9))` per interval. `odeint`: convert `t` to a tensor of `y0`'s dtype, look up the step function in a dict (raise `ValueError` for unknown names), and for each interval `k` with `n` substeps use `h = (t[k+1] - t[k]) / n` (negative when time decreases) and `for i in range(n): y = step(f, t[k] + i*h, y, h)`. collect `y0` and the value at each `t[k+1]`, then `torch.stack`.

</details>

## stage 3

<details><summary>hint</summary>

define a `torch.autograd.Function` inside `odeint_adjoint` so it can close over `f`, `method` and `step_size`; pass `(y0, t, *params)` to `.apply`. its `forward` runs `odeint` (autograd is already off there) and saves `t`, the outputs and the parameters. its `backward` integrates one flat vector `[y, a, a_theta]` from `t[k]` back to `t[k-1]` with your own `odeint`, where the vector field evaluates `fy = f(t, y)` under `torch.enable_grad()` on a detached `y` and gets both vector-Jacobian products from one `torch.autograd.grad(fy, (y, *params), -a)`. after each interval, add `grad_ys[k-1]` to `a`. return one gradient per input of `apply` (`None` for `t`).

</details>

## stage 4

<details><summary>hint</summary>

store `func` as an attribute (an `nn.Module` attribute is registered automatically) along with the settings. `forward`: `t = torch.tensor([0.0, t1], dtype=y0.dtype)`, choose `odeint_adjoint` or `odeint`, call it with `step_size = t1 / n_steps`, return the last row.

</details>
