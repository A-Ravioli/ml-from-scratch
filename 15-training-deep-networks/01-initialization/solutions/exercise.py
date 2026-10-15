import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.nn import Linear, ReLU, Sequential, Tanh  # yours, from ch 14 · topic 04
    from mlfs.tensor import Tensor, no_grad  # yours, from ch 14 · topic 03
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # initialization · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a deep network's starting point decides
    whether its signal survives the trip through fifty layers or dies on the way, and the
    fix is two lines of arithmetic you're about to derive. everything here becomes the first
    part of **`mlfs.train`**, with the same names and conventions as `torch.nn.init`.

    | stage | you build |
    |---|---|
    | 1 | `calculate_fan` and `calculate_gain`: the two numbers every initializer needs |
    | 2 | `xavier_uniform_` and `xavier_normal_` (Glorot) |
    | 3 | `kaiming_uniform_` and `kaiming_normal_` (He) |
    | 4 | `deep_mlp`: a deep network of your `mlfs.nn` layers with a chosen init |
    | 5 | `activation_stats`: watch the signal layer by layer |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def calculate_fan(shape):
    """stage 1: (fan_in, fan_out) ints for a torch-layout weight shape: (out, in) or (out, in, *kernel)."""
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise ValueError(f"fans need a weight with at least 2 dimensions (out, in, ...), got shape {shape}")
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive


@app.function
def calculate_gain(nonlinearity, param=None):
    """stage 1: the recommended gain for an activation: linear/sigmoid 1, tanh 5/3, relu sqrt(2), leaky_relu sqrt(2/(1+a^2))."""
    if nonlinearity in ("linear", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return float(np.sqrt(2.0))
    if nonlinearity == "leaky_relu":
        a = 0.01 if param is None else float(param)
        return float(np.sqrt(2.0 / (1.0 + a * a)))
    raise ValueError(f"unknown nonlinearity {nonlinearity!r}: use 'linear', 'sigmoid', 'tanh', 'relu' or 'leaky_relu'")


@app.cell(hide_code=True)
def _():
    checks.run(1, calculate_fan=calculate_fan, calculate_gain=calculate_gain)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def weight_array(param):
    """the ndarray an initializer fills in place: param.data for a Parameter/Tensor, the array itself for an ndarray (given)."""
    arr = param if isinstance(param, np.ndarray) else param.data
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"expected a Parameter/Tensor or an ndarray, got {type(param).__name__}")
    return arr


@app.function
def xavier_uniform_(param, *, gain=1.0, rng=None):
    """stage 2: fill param in place from U(-a, a), a = gain * sqrt(6 / (fan_in + fan_out)), with ONE rng.uniform call; return param."""
    arr = weight_array(param)
    fan_in, fan_out = calculate_fan(arr.shape)
    a = gain * np.sqrt(6.0 / (fan_in + fan_out))
    arr[...] = as_rng(rng).uniform(-a, a, size=arr.shape)
    return param


@app.function
def xavier_normal_(param, *, gain=1.0, rng=None):
    """stage 2: fill param in place from N(0, std^2), std = gain * sqrt(2 / (fan_in + fan_out)), with ONE rng.normal call; return param."""
    arr = weight_array(param)
    fan_in, fan_out = calculate_fan(arr.shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    arr[...] = as_rng(rng).normal(0.0, std, size=arr.shape)
    return param


@app.cell(hide_code=True)
def _():
    checks.run(2, xavier_uniform_=xavier_uniform_, xavier_normal_=xavier_normal_, calculate_fan=calculate_fan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def kaiming_std(shape, nonlinearity, mode):
    """the He standard deviation gain / sqrt(fan) for a weight shape (shared by both kaiming initializers, given)."""
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan_in, fan_out = calculate_fan(shape)
    fan = fan_in if mode == "fan_in" else fan_out
    return calculate_gain(nonlinearity) / np.sqrt(fan)


@app.function
def kaiming_uniform_(param, *, nonlinearity="relu", mode="fan_in", rng=None):
    """stage 3: fill param in place from U(-b, b) with b = sqrt(3) * gain / sqrt(fan), ONE rng.uniform call; return param."""
    arr = weight_array(param)
    bound = np.sqrt(3.0) * kaiming_std(arr.shape, nonlinearity, mode)
    arr[...] = as_rng(rng).uniform(-bound, bound, size=arr.shape)
    return param


@app.function
def kaiming_normal_(param, *, nonlinearity="relu", mode="fan_in", rng=None):
    """stage 3: fill param in place from N(0, std^2) with std = gain / sqrt(fan), ONE rng.normal call; return param."""
    arr = weight_array(param)
    std = kaiming_std(arr.shape, nonlinearity, mode)
    arr[...] = as_rng(rng).normal(0.0, std, size=arr.shape)
    return param


@app.cell(hide_code=True)
def _():
    checks.run(3, kaiming_uniform_=kaiming_uniform_, kaiming_normal_=kaiming_normal_)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def deep_mlp(sizes, *, activation="relu", init="kaiming", rng=None):
    """stage 4: Sequential of Linear layers with the activation between them; init 'default' | 'xavier' | 'kaiming' | a float std, zero biases."""
    if activation not in ("relu", "tanh", "linear"):
        raise ValueError(f"activation must be 'relu', 'tanh' or 'linear', got {activation!r}")
    if not (init in ("default", "xavier", "kaiming") or isinstance(init, (int, float))):
        raise ValueError(f"init must be 'default', 'xavier', 'kaiming' or a float std, got {init!r}")
    rng = as_rng(rng)
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = Linear(n_in, n_out, rng=rng)
        if init != "default":
            if init == "xavier":
                xavier_normal_(layer.weight, rng=rng)
            elif init == "kaiming":
                kaiming_normal_(layer.weight, nonlinearity=activation, rng=rng)
            else:
                layer.weight.data[...] = rng.normal(0.0, float(init), size=layer.weight.shape)
            layer.bias.data[...] = 0.0
        layers.append(layer)
        if i < len(sizes) - 2 and activation != "linear":
            layers.append(ReLU() if activation == "relu" else Tanh())
    return Sequential(*layers)


@app.cell(hide_code=True)
def _():
    checks.run(4, deep_mlp=deep_mlp)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def activation_stats(model, X):
    """stage 5: run X through each layer of a Sequential under no_grad; a list of dict(name, mean, std, frac_zero), one per layer output."""
    out = []
    x = X if isinstance(X, Tensor) else Tensor(X)
    with no_grad():
        for i in range(len(model)):
            layer = model[i]
            x = layer(x)
            a = x.data
            out.append({"name": f"{i}.{type(layer).__name__}", "mean": float(a.mean()),
                        "std": float(a.std()), "frac_zero": float(np.mean(a == 0))})
    return out


@app.cell(hide_code=True)
def _():
    checks.run(5, activation_stats=activation_stats, deep_mlp=deep_mlp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.train import kaiming_normal_` gives you these initializers.
    """)
    return


if __name__ == "__main__":
    app.run()
