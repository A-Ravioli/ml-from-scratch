import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # stochastic optimization · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    classes and functions to write, and checks them live as you edit. this is the most-used
    code you'll write in the course: from chapter 08 on, every model that's trained by
    gradients (logistic regression, neural networks, transformers) is trained by the
    `Optimizer` classes in this notebook, through `mlfs.optim`.

    | stage | you build |
    |---|---|
    | 1 | `Param`, `Optimizer`, `minibatch_indices`: the parameter protocol, the base class, and epochs of minibatches → `mlfs.optim` |
    | 2 | `SGD`: stochastic gradient descent with momentum, Nesterov and weight decay → `mlfs.optim` |
    | 3 | `Adagrad`, `RMSProp`: per-coordinate step sizes → `mlfs.optim` |
    | 4 | `Adam`, `AdamW`: momentum plus adaptive scaling, with bias correction → `mlfs.optim` |
    | 5 | `minibatch_gradient_variance`, `sgd_noise_floor`: how noisy a minibatch gradient is, and where constant-step SGD stalls |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Param:
    """a minimal trainable parameter: .data (float64 ndarray, updated in place by optimizers)
    and .grad (ndarray of the same shape, or None when there is no gradient).

    any object with these two attributes works with the optimizers (chapter 14's Tensor and
    nn.Parameter are others).
    """

    def __init__(self, data):
        """stage 1: store a float64 COPY of data as self.data (np.array(data, dtype=np.float64)),
        so the caller's array is never changed by training, and set self.grad = None.
        """
        raise NotImplementedError("stage1: implement Param.__init__")

    def __repr__(self):
        return f"Param(shape={self.data.shape}, grad={'None' if self.grad is None else 'set'})"


@app.class_definition
class Optimizer:
    """base class of every stateful optimizer in mlfs.optim.

    the protocol: parameters are objects with .data (float ndarray) and .grad (same shape, or
    None). step() updates every p.data IN PLACE (the array object stays the same, so anything
    holding a reference sees the new values) and skips parameters whose .grad is None.
    zero_grad() sets every .grad to None. .lr is a plain attribute that schedulers may change
    between steps. per-parameter state lives in self.state[i], keyed by the parameter's index
    in self.params.
    """

    def __init__(self, params, lr):
        """stage 1: self.params = list(params) (params may be any iterable, even a generator, so
        materialize it once); self.lr = lr; self.state = {} (filled lazily by step())."""
        raise NotImplementedError("stage1: implement Optimizer.__init__")

    def step(self):
        """update every parameter in place. each subclass implements its own rule."""
        raise TypeError(f"{type(self).__name__} is the abstract base: use a subclass (SGD, Adam, ...) or override step()")

    def zero_grad(self):
        """stage 1: set p.grad = None for every parameter (not zeros: None means 'no gradient')."""
        raise NotImplementedError("stage1: implement Optimizer.zero_grad")


@app.function
def minibatch_indices(n, batch_size, *, shuffle=True, drop_last=False, rng=None):
    """stage 1: one epoch of minibatches: a generator of int index arrays covering range(n).

    shuffle=True: a random permutation (rng = as_rng(rng), from mlfs.utils); otherwise 0..n-1 in
    order. yield consecutive chunks of batch_size indices; the last chunk is smaller if
    batch_size doesn't divide n, unless drop_last=True, which skips it. every index appears
    at most once per call (exactly once unless drop_last drops some).
    """
    raise NotImplementedError("stage1: implement minibatch_indices")


@app.cell(hide_code=True)
def _():
    checks.run(1, Param=Param, Optimizer=Optimizer, minibatch_indices=minibatch_indices)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SGD(Optimizer):
    """stochastic gradient descent with optional momentum, Nesterov momentum and L2 weight
    decay, with PyTorch's semantics (so results match torch.optim.SGD to rounding)."""

    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
        """stage 2: call Optimizer.__init__(params, lr) and store momentum, nesterov and
        weight_decay as same-named attributes. nesterov=True with momentum <= 0 raises
        ValueError (Nesterov momentum needs a momentum)."""
        raise NotImplementedError("stage2: implement SGD.__init__")

    def step(self):
        """stage 2: for each parameter i with p.grad not None:
            g = p.grad + weight_decay * p.data            (a NEW array: never modify p.grad)
            if momentum != 0:
                b = g (a copy) on this parameter's first step, else b = momentum * b + g
                (the buffer b lives in self.state[i]["momentum_buffer"])
                g = g + momentum * b   if nesterov   else   g = b
            p.data -= lr * g                              (in place)
        uses the CURRENT self.lr, so a scheduler can change it between steps.
        """
        raise NotImplementedError("stage2: implement SGD.step")


@app.cell(hide_code=True)
def _():
    checks.run(2, Param=Param, SGD=SGD)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class Adagrad(Optimizer):
    """Adagrad (Duchi, Hazan & Singer, 2011): each coordinate's step shrinks with the total
    squared gradient it has seen."""

    def __init__(self, params, lr=0.01, eps=1e-10):
        """stage 3: Optimizer.__init__(params, lr); store eps."""
        raise NotImplementedError("stage3: implement Adagrad.__init__")

    def step(self):
        """stage 3: for each parameter i with a gradient g:
            s += g * g                         (s starts at zeros; self.state[i]["sum"])
            p.data -= lr * g / (sqrt(s) + eps)  (in place, elementwise)
        """
        raise NotImplementedError("stage3: implement Adagrad.step")


@app.class_definition
class RMSProp(Optimizer):
    """RMSProp (Tieleman & Hinton, 2012): Adagrad with an exponential moving average instead
    of a sum, so old gradients are forgotten."""

    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-08):
        """stage 3: Optimizer.__init__(params, lr); store alpha and eps."""
        raise NotImplementedError("stage3: implement RMSProp.__init__")

    def step(self):
        """stage 3: for each parameter i with a gradient g:
            v = alpha * v + (1 - alpha) * g * g   (v starts at zeros; self.state[i]["square_avg"])
            p.data -= lr * g / (sqrt(v) + eps)     (in place)
        """
        raise NotImplementedError("stage3: implement RMSProp.step")


@app.cell(hide_code=True)
def _():
    checks.run(3, Param=Param, Adagrad=Adagrad, RMSProp=RMSProp)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015): momentum on the gradient, RMSProp-style scaling, and bias
    correction for both moving averages. weight_decay is COUPLED L2: it is added to the
    gradient, and then scaled like the rest of the gradient."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0):
        """stage 4: Optimizer.__init__(params, lr); store betas (a tuple), eps, weight_decay."""
        raise NotImplementedError("stage4: implement Adam.__init__")

    def step(self):
        """stage 4: for each parameter i with p.grad not None (self.state[i] holds "step",
        "exp_avg" and "exp_avg_sq", zeros at first; "step" counts only the steps where THIS
        parameter had a gradient):
            t += 1
            g = p.grad + weight_decay * p.data           (new array)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** t);  v_hat = v / (1 - beta2 ** t)
            p.data -= lr * m_hat / (sqrt(v_hat) + eps)    (in place)
        """
        raise NotImplementedError("stage4: implement Adam.step")


@app.class_definition
class AdamW(Optimizer):
    """AdamW (Loshchilov & Hutter, 2019): Adam with DECOUPLED weight decay, applied directly
    to the weights instead of through the gradient."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        """stage 4: Optimizer.__init__(params, lr); store betas (a tuple), eps, weight_decay."""
        raise NotImplementedError("stage4: implement AdamW.__init__")

    def step(self):
        """stage 4: for each parameter with a gradient: first shrink the weights,
            p.data *= 1 - lr * weight_decay                (in place)
        then the Adam update of stage 4 with g = p.grad (no weight decay in g), using the
        same state keys ("step", "exp_avg", "exp_avg_sq").
        """
        raise NotImplementedError("stage4: implement AdamW.step")


@app.cell(hide_code=True)
def _():
    checks.run(4, Param=Param, Adam=Adam, AdamW=AdamW, minibatch_indices=minibatch_indices)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def minibatch_gradient_variance(per_example_grads: np.ndarray, batch_size: int) -> float:
    """stage 5: the exact mean squared error E ||g_B - g_full||^2 of a minibatch gradient.

    per_example_grads (n, d): row i is the gradient of example i's loss; g_full is their mean
    and g_B the mean over a batch of batch_size examples drawn uniformly WITHOUT replacement.
    returns  (n - B) / (B (n - 1)) * (1/n) sum_i ||g_i - g_full||^2  as a float
    (0 when B = n: the full batch has no noise). requires 1 <= B <= n and n >= 2.
    """
    raise NotImplementedError("stage5: implement minibatch_gradient_variance")


@app.function
def sgd_noise_floor(lr: float, curvatures, noise_vars) -> float:
    """stage 5: the stationary expected excess loss E[f(x) - f*] of constant-step SGD on a
    quadratic f(x) = sum_i a_i x_i^2 / 2 whose gradient noise along coordinate i is
    independent with variance noise_vars[i]:
        sum_i  lr * s_i / (2 (2 - lr * a_i))
    curvatures a (k,), noise_vars s (k,); requires 0 < lr * a_i < 2 for every i. returns a float.
    """
    raise NotImplementedError("stage5: implement sgd_noise_floor")


@app.cell(hide_code=True)
def _():
    checks.run(5, minibatch_gradient_variance=minibatch_gradient_variance, sgd_noise_floor=sgd_noise_floor,
               Param=Param, SGD=SGD)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your optimizers are `mlfs.optim.SGD`, `Adam`, `AdamW`, ...
    the training loop you'll write in every later chapter looks like this:

    ```python
    params = [Param(w0)]
    opt = Adam(params, lr=1e-2)
    for epoch in range(n_epochs):
        for idx in minibatch_indices(n, 32, rng=rng):
            opt.zero_grad()
            params[0].grad = grad_of_loss(params[0].data, X[idx], y[idx])
            opt.step()
    ```

    then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
