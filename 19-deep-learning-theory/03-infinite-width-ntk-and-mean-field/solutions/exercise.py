import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # infinite width: NTK & mean field · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. stages 1–2 join **`mlfs.dltheory`**.

    | stage | you build |
    |---|---|
    | 1 | `empirical_ntk`: the neural tangent kernel of any scalar-output model → `mlfs.dltheory` |
    | 2 | `kernel_regression_predict`: (ridge) kernel regression from Gram matrices → `mlfs.dltheory` |
    | 3 | `NTKNet`, `relu_ntk`: a two-layer network in NTK or mean-field parameterization, and its infinite-width kernel |
    | 4 | `ntk_gd_predictions`: what gradient descent on the linearized network predicts, in closed form |
    | 5 | `gd_train`, `relative_change`: full-batch gradient descent, and how far the weights moved |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def empirical_ntk(model, x1, x2=None):
    """stage 1: (n1, n2) tensor K[i, j] = <d f(x1_i)/d theta, d f(x2_j)/d theta> over params with requires_grad; x2=None means x1."""
    params = [p for p in model.parameters() if p.requires_grad]

    def jacobian(x):
        out = model(x).reshape(-1)
        rows = []
        for i in range(out.shape[0]):
            g = torch.autograd.grad(out[i], params, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                                   for gi, p in zip(g, params)]))
        return torch.stack(rows)

    J1 = jacobian(x1)
    J2 = J1 if x2 is None else jacobian(x2)
    return (J1 @ J2.T).detach()


@app.cell(hide_code=True)
def _():
    checks.run(1, empirical_ntk=empirical_ntk)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def kernel_regression_predict(K_train, y_train, K_test_train, *, ridge=0.0):
    """stage 2: K_test_train @ solve(K_train + ridge I, y_train); returns a tensor for tensor K_test_train (on its
    device; K_train and y_train are moved there), else an ndarray."""
    if torch.is_tensor(K_test_train):
        dev = K_test_train.device  # everything on K_test_train's device (CPU, CUDA or MPS)
        K = torch.as_tensor(K_train, dtype=K_test_train.dtype, device=dev)
        y = torch.as_tensor(y_train, dtype=K_test_train.dtype, device=dev)
        A = K + ridge * torch.eye(K.shape[0], dtype=K.dtype, device=dev)
        return K_test_train @ torch.linalg.solve(A, y)
    K = np.asarray(K_train.detach() if torch.is_tensor(K_train) else K_train, dtype=float)
    y = np.asarray(y_train.detach() if torch.is_tensor(y_train) else y_train, dtype=float)
    A = K + ridge * np.eye(K.shape[0])
    return np.asarray(K_test_train, dtype=float) @ np.linalg.solve(A, y)


@app.cell(hide_code=True)
def _():
    checks.run(2, kernel_regression_predict=kernel_regression_predict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class NTKNet(nn.Module):
    """f(x) = scale * a . relu(W x): W (width, in_dim) and a (width,) ~ N(0, 1); scale 1/sqrt(width) ('ntk') or 1/width ('mf')."""

    def __init__(self, in_dim, width, *, parameterization="ntk"):
        """stage 3: parameters W (width, in_dim) and a (width,), standard normal; attributes width, parameterization."""
        super().__init__()
        if parameterization not in ("ntk", "mf"):
            raise ValueError(f"parameterization must be 'ntk' or 'mf', got {parameterization!r}")
        self.width, self.parameterization = width, parameterization
        self.W = nn.Parameter(torch.randn(width, in_dim))
        self.a = nn.Parameter(torch.randn(width))

    def forward(self, x):
        """stage 3: x (n, in_dim) -> (n,)."""
        scale = self.width ** -0.5 if self.parameterization == "ntk" else 1.0 / self.width
        return scale * (torch.relu(x @ self.W.T) @ self.a)


@app.function
def relu_ntk(X1, X2=None):
    """stage 3: the infinite-width NTK of NTKNet ('ntk'): |x||x'|/(2 pi) (sin t + (pi - t) cos t) + x.x' (pi - t)/(2 pi)."""
    X1 = torch.as_tensor(X1, dtype=torch.float64)
    X2 = X1 if X2 is None else torch.as_tensor(X2, dtype=torch.float64)
    dots = X1 @ X2.T
    norms = X1.norm(dim=1)[:, None] * X2.norm(dim=1)[None, :]
    cos = torch.where(norms > 0, dots / norms.clamp_min(1e-300), torch.zeros_like(dots)).clamp(-1.0, 1.0)
    theta = torch.arccos(cos)
    nngp = norms / (2 * math.pi) * (torch.sin(theta) + (math.pi - theta) * cos)
    return nngp + dots * (math.pi - theta) / (2 * math.pi)


@app.cell(hide_code=True)
def _():
    checks.run(3, NTKNet=NTKNet, relu_ntk=relu_ntk, empirical_ntk=empirical_ntk)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ntk_gd_predictions(K_train, K_test_train, y_train, f0_train, f0_test, *, lr, steps):
    """stage 4: (f_train, f_test) after `steps` GD steps of size lr on 1/2 sum (f - y)^2 for the model linearized at init."""
    K = torch.as_tensor(K_train, dtype=torch.float64)
    Kx = torch.as_tensor(K_test_train, dtype=torch.float64)
    y = torch.as_tensor(y_train, dtype=torch.float64)
    f0 = torch.as_tensor(f0_train, dtype=torch.float64)
    f0x = torch.as_tensor(f0_test, dtype=torch.float64)
    lam, V = torch.linalg.eigh((K + K.T) / 2)
    r0 = V.T @ (y - f0)
    decay = (1 - lr * lam) ** steps
    small = lam.abs() <= 1e-12 * max(1.0, float(lam.abs().max()))
    # (1 - (1 - lr lam)^t) / lam, with its limit lr * t as lam -> 0
    gain = torch.where(small, torch.full_like(lam, lr * steps), (1 - decay) / torch.where(small, torch.ones_like(lam), lam))
    f_train = y - V @ (decay * r0)
    f_test = f0x + Kx @ (V @ (gain * r0))
    return f_train, f_test


@app.cell(hide_code=True)
def _():
    checks.run(4, ntk_gd_predictions=ntk_gd_predictions)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def gd_train(model, x, y, *, lr, steps):
    """stage 5: `steps` plain full-batch GD steps on 1/2 sum (f(x) - y)^2, in place; returns the steps + 1 losses (before each step, then final)."""
    params = [p for p in model.parameters() if p.requires_grad]
    y = torch.as_tensor(y, dtype=next(iter(params)).dtype).reshape(-1)
    losses = []
    for _ in range(steps):
        loss = 0.5 * ((model(x).reshape(-1) - y) ** 2).sum()
        losses.append(float(loss.detach()))
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, g in zip(params, grads):
                p -= lr * g
    with torch.no_grad():
        losses.append(float(0.5 * ((model(x).reshape(-1) - y) ** 2).sum()))
    return losses


@app.function
def relative_change(model, initial_state):
    """stage 5: ||theta - theta_0|| / ||theta_0|| over model.named_parameters() (one global norm); initial_state maps names to tensors."""
    num, den = 0.0, 0.0
    for name, p in model.named_parameters():
        p0 = initial_state[name].to(p)
        num += float(((p.detach() - p0) ** 2).sum())
        den += float((p0 ** 2).sum())
    return math.sqrt(num) / math.sqrt(den)


@app.cell(hide_code=True)
def _():
    checks.run(5, gd_train=gd_train, relative_change=relative_change, NTKNet=NTKNet)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.dltheory import empirical_ntk, kernel_regression_predict` gives you the NTK of any
    scalar-output PyTorch model, and the capstone's NTK option builds on all five stages.
    """)
    return


if __name__ == "__main__":
    app.run()
