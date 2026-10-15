# initialization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`calculate_fan`: turn the shape into a tuple of ints and raise `ValueError` if it has fewer than 2 entries. the receptive field is the product of everything after the first two entries (1 if there's nothing); fan_in is `shape[1] * receptive`, fan_out `shape[0] * receptive`. `calculate_gain` is a chain of `if`s; for `'leaky_relu'`, use `param` as the slope if it's given, else 0.01.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
shape = tuple(int(s) for s in shape)
if len(shape) < 2:
    raise ValueError(...)
receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
return shape[1] * receptive, shape[0] * receptive
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

three lines each: `arr = weight_array(param)`, `fan_in, fan_out = calculate_fan(arr.shape)`, then fill `arr[...]` with one draw of the full shape from `as_rng(rng)` and `return param`. uniform bound $a = \text{gain}\sqrt{6/(\text{fan\_in} + \text{fan\_out})}$; normal std $\text{gain}\sqrt{2/(\text{fan\_in} + \text{fan\_out})}$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
arr = weight_array(param)
fan_in, fan_out = calculate_fan(arr.shape)
a = gain * np.sqrt(6.0 / (fan_in + fan_out))
arr[...] = as_rng(rng).uniform(-a, a, size=arr.shape)
return param
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`kaiming_std(arr.shape, nonlinearity, mode)` already gives you $\sigma = \text{gain}/\sqrt{\text{fan}}$ (and raises for a bad `mode`). the normal version draws `normal(0, std)`; the uniform one draws `uniform(-b, b)` with `b = np.sqrt(3) * std`, because a uniform's variance is a third of its bound squared.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
arr = weight_array(param)
bound = np.sqrt(3.0) * kaiming_std(arr.shape, nonlinearity, mode)
arr[...] = as_rng(rng).uniform(-bound, bound, size=arr.shape)
return param
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`rng = as_rng(rng)` once. loop over consecutive pairs `zip(sizes[:-1], sizes[1:])`: build `Linear(n_in, n_out, rng=rng)`; unless `init == 'default'`, overwrite its weight (`xavier_normal_(layer.weight, rng=rng)`, `kaiming_normal_(layer.weight, nonlinearity=activation, rng=rng)`, or `rng.normal(0, init, size=...)` for a float) and set `layer.bias.data[...] = 0`. append an activation module after every layer except the last, unless the activation is `'linear'`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
    layer = Linear(n_in, n_out, rng=rng)
    if init == "xavier":
        xavier_normal_(layer.weight, rng=rng)
    elif init == "kaiming":
        kaiming_normal_(layer.weight, nonlinearity=activation, rng=rng)
    elif init != "default":
        layer.weight.data[...] = rng.normal(0.0, float(init), size=layer.weight.shape)
    if init != "default":
        layer.bias.data[...] = 0.0
    layers.append(layer)
    if i < len(sizes) - 2 and activation != "linear":
        layers.append(ReLU() if activation == "relu" else Tanh())
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

wrap `X` as a `Tensor`, then inside `with no_grad():` loop `for i in range(len(model))`, set `x = model[i](x)`, and append a dict built from `x.data`: `float(a.mean())`, `float(a.std())`, `float(np.mean(a == 0))`, and the name `f"{i}.{type(model[i]).__name__}"`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
out, x = [], Tensor(X)
with no_grad():
    for i in range(len(model)):
        x = model[i](x)
        a = x.data
        out.append({"name": f"{i}.{type(model[i]).__name__}", "mean": float(a.mean()),
                    "std": float(a.std()), "frac_zero": float(np.mean(a == 0))})
return out
```

</details>
