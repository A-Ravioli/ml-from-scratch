# scaling laws · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`a * np.asarray(x, dtype=float) ** (-b) + c`. for the local exponents, `lx = np.log(x)`, `ly = np.log(y - c)`, and return `-np.diff(ly) / np.diff(lx)`.

</details>

## stage 2

<details><summary>hint</summary>

write `loglinear(c)`: `slope, intercept = np.polyfit(np.log(x), np.log(y - c), 1)`, returning `(exp(intercept), -slope)`. that's the whole answer for `offset=False`. with the offset, score a candidate by `sum((log y - log(a x^-b + c))^2)`, try `loglinear(c)` for a grid of `c` from 0 up to just below `min(y)` (spacing the grid geometrically towards `min(y)` helps), keep the best, then polish with `scipy.optimize.least_squares` on `(log a, b, c)` with bounds `[0, min(y))` on `c`, and keep the polished result only if it scores better.

</details>

## stage 3

<details><summary>hint</summary>

from $C = 6ND$ and $D = rN$: $C = 6rN^2$, so `n = np.sqrt(compute / (6 * tokens_per_param))` and `d = tokens_per_param * n`.

</details>

## stage 4

<details><summary>hint</summary>

`G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))`, `n = G * (C / 6) ** (beta / (alpha + beta))`, `d = (C / 6) ** (alpha / (alpha + beta)) / G`. check that `6 * n * d == C`.

</details>

## stage 5

<details><summary>hint</summary>

collect `id(model.tok_emb.weight)` and `id(model.pos_emb.weight)` for the modules that exist (`getattr(model, "pos_emb", None)` may be `None`), then sum `p.numel()` over `model.parameters()` (which already yields a tied tensor only once), skipping those ids.

</details>
