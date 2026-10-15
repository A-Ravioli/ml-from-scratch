# your first ML model · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

with NumPy, `w * x + b` computes the prediction for every input at once. the loss is "how far off, squared, on average".

</details>

<details><summary>hint 2: the approach</summary>

`predict`: convert `x` to a float array and return `w * x + b`. `mse_loss`: residuals `r = predict(x, w, b) - y` (with `y` as a float array), then `float(np.mean(r ** 2))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
r = predict(x, w, b) - np.asarray(y, dtype=float)
return float(np.mean(r**2))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

do the derivative on paper first, for a single term $(wx_i + b - y_i)^2$. the chain rule gives "2 times the inside, times the derivative of the inside". the inside's derivative is $x_i$ for $w$ and $1$ for $b$.

</details>

<details><summary>hint 2: the approach</summary>

compute the residuals once, `r = predict(x, w, b) - y`. then `dw = 2 * np.sum(r * x) / n` and `db = 2 * np.sum(r) / n`. return them as two floats.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)
n = len(x)
r = predict(x, w, b) - y
return float(2 * np.sum(r * x) / n), float(2 * np.sum(r) / n)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

it's a loop of `n_steps` iterations. each iteration does three things: compute the gradient, take the step, record the loss. the loss list starts with one entry, the loss at the starting point.

</details>

<details><summary>hint 2: the approach</summary>

start with `w, b = float(w0), float(b0)` and `losses = [mse_loss(x, y, w, b)]`. in the loop, `dw, db = mse_gradient(x, y, w, b)`, then update both in one line, `w, b = w - lr * dw, b - lr * db` (Python evaluates the right side completely before assigning, so both use the old values), then append the new loss. return `w, b, np.array(losses)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
w, b = float(w0), float(b0)
losses = [mse_loss(x, y, w, b)]
for _ in range(n_steps):
    dw, db = mse_gradient(x, y, w, b)
    w, b = w - lr * dw, b - lr * db
    losses.append(mse_loss(x, y, w, b))
return w, b, np.array(losses)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

two formulas from idea 4. compute the deviations from the means first; both sums use them.

</details>

<details><summary>hint 2: the approach</summary>

`dx = x - x.mean()` and `dy = y - y.mean()`. the denominator `sxx = np.sum(dx * dx)` is zero only when every x is the same, so check it and raise `ValueError`. then `w = np.sum(dx * dy) / sxx` and `b = y.mean() - w * x.mean()`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
dx, dy = x - x.mean(), y - y.mean()
sxx = np.sum(dx * dx)
if sxx == 0:
    raise ValueError("all x values are equal")
w = np.sum(dx * dy) / sxx
return float(w), float(y.mean() - w * x.mean())
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`standardize` is two NumPy calls and one line of arithmetic. for `unstandardize_params`, write $\hat y = w_z \cdot \frac{x - m}{s} + b_z$ and multiply it out: what multiplies $x$, and what's left over?

</details>

<details><summary>hint 2: the approach</summary>

`standardize`: `mean, std = x.mean(), x.std()`, raise `ValueError` if `std == 0`, and return `(x - mean) / std, float(mean), float(std)`. `unstandardize_params`: $\hat y = \frac{w_z}{s}x + b_z - \frac{w_zm}{s}$, so `w = w_z / std` and `b = b_z - w_z * mean / std`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
return float(w_z / std), float(b_z - w_z * mean / std)
```

</details>
