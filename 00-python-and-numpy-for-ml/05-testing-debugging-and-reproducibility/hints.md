# testing, debugging & reproducibility · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`as_rng` is a chain of `isinstance` checks, one per accepted type, plus a `TypeError` at the end. the order matters: Python's `bool` is a subclass of `int` (`isinstance(True, int)` is `True`), so rule bools out explicitly. `set_seed` is three seeding calls and a return.

</details>

<details><summary>hint 2: the approach</summary>

`as_rng`:
- `rng is None` → `np.random.default_rng()`
- `isinstance(rng, np.random.Generator)` → `return rng` (the same object)
- `isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_))` → `np.random.default_rng(int(rng))`
- otherwise `raise TypeError(...)`

`set_seed`: `random.seed(seed)`, `np.random.seed(seed)`, and, if `importlib.util.find_spec("torch") is not None`, `import torch` and `torch.manual_seed(seed)`. return `np.random.default_rng(seed)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if rng is None:
    return np.random.default_rng()
if isinstance(rng, np.random.Generator):
    return rng
if isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)):
    return np.random.default_rng(int(rng))
raise TypeError(f"rng must be None, an int seed or a Generator, got {type(rng).__name__}")
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

work in three phases: (1) convert and compare shapes; (2) compute a boolean array `ok` saying which elements pass; (3) if any fail, find the worst one and build the message. phase 2 is one line of elementwise operations; phase 3 is `np.argmax` plus `np.unravel_index`.

</details>

<details><summary>hint 2: the approach</summary>

- `a = np.asarray(actual, dtype=float)`, same for `e`; if `a.shape != e.shape`, raise `AssertionError(f"{name}: shape mismatch: actual {a.shape}, expected {e.shape}")`.
- `diff = np.abs(a - e)`, `tol = atol + rtol * np.abs(e)`, and `ok = (np.isfinite(e) & (diff <= tol)) | (a == e) | (np.isnan(a) & np.isnan(e))`. (`a == e` handles matching infinities; `inf - inf` is `nan`, so wrap this in `np.errstate(invalid="ignore")`.)
- if `ok.all()`, return `None`.
- `badness = diff / tol` for the failing elements and `-1` for passing ones; replace `nan` (and a zero `tol`) by `np.inf`. then `worst = np.unravel_index(np.argmax(badness), a.shape)`, converted to a tuple of Python ints.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
with np.errstate(divide="ignore", invalid="ignore"):
    badness = np.where(ok, -1.0, diff / tol)
badness = np.where(~ok & ~(badness >= 0), np.inf, badness)   # nan comparisons are False
worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(badness)), a.shape))
av, ev = float(a[worst]), float(e[worst])
```

then format the three lines of the message exactly as the docstring shows, with `{av:.10g}`, `{abs_err:.3g}` and so on.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

make the generator once, before the loop, with `as_rng(rng)`. every trial calls `generate` with that same generator, so the whole sequence of inputs is determined by the seed.

</details>

<details><summary>hint 2: the approach</summary>

loop `n_trials` times: `x = generate(rng)`; then `try: passed = prop(x)` / `except Exception: return x`; then `if not passed: return x`. after the loop, `return None`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = as_rng(rng)
for _ in range(n_trials):
    x = generate(rng)
    try:
        passed = prop(x)
    except Exception:
        return x
    if not passed:
        return x
return None
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

keep the invariant "the first bad version is somewhere in `[lo, hi]`". it's true at the start, `lo = 0` and `hi = n - 1`, because version `n - 1` is bad. each test must shrink the interval while keeping the invariant true.

</details>

<details><summary>hint 2: the approach</summary>

while `lo < hi`: `mid = (lo + hi) // 2`. if `is_bad(mid)`, the first bad version is `mid` or earlier, so `hi = mid` (not `mid - 1`: `mid` might be the answer). otherwise it's after `mid`, so `lo = mid + 1`. since `mid < hi` whenever `lo < hi`, the interval always shrinks. return `lo`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if not is_bad(n - 1):
    raise ValueError("the newest version must be bad")
lo, hi = 0, n - 1
while lo < hi:
    mid = (lo + hi) // 2
    if is_bad(mid):
        hi = mid
    else:
        lo = mid + 1
return lo
```

</details>
