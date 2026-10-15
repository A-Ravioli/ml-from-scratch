import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # K-FAC · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. every function works in the dtype of its inputs (the
    checks use float64, the lab float32).

    | stage | you build |
    |---|---|
    | 1 | `damped_inverses`, `kron_precondition`: inverting a Kronecker product one factor at a time |
    | 2 | `kfac_factors`: the two factors of every linear layer's Fisher block, from activations and backpropagated gradients |
    | 3 | `kfac_step`: one K-FAC step from given factors, with KL clipping |
    | 4 | `KFAC`: the optimizer, with running-average factors, sampled labels, periodic inverses and momentum |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def damped_inverses(A, G, damping):
    """stage 1: ((A + pi sqrt(damping) I)^-1, (G + sqrt(damping)/pi I)^-1), pi = sqrt((tr A / dim A) / (tr G / dim G))."""
    eye_a = torch.eye(A.shape[0], dtype=A.dtype)
    eye_g = torch.eye(G.shape[0], dtype=G.dtype)
    if damping == 0:
        return torch.linalg.inv(A), torch.linalg.inv(G)
    pi = torch.sqrt((torch.trace(A) / A.shape[0]) / (torch.trace(G) / G.shape[0]))
    root = math.sqrt(damping)
    return torch.linalg.inv(A + (pi * root) * eye_a), torch.linalg.inv(G + (root / pi) * eye_g)


@app.function
def kron_precondition(grad, A_inv, G_inv):
    """stage 1: (G kron A)^-1 vec(grad) as a matrix, G_inv @ grad @ A_inv; grad is (out, in + 1)."""
    return G_inv @ grad @ A_inv


@app.cell(hide_code=True)
def _():
    checks.run(1, damped_inverses=damped_inverses, kron_precondition=kron_precondition)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def linear_layers(model):
    """helper: the nn.Linear modules of `model`, in model.modules() order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


@app.function
def kfac_factors(model, X, labels=None):
    """stage 2: [(A, G) per nn.Linear]: A = mean of [a, 1][a, 1]^T, G = mean of g g^T, g = d log p(y|x) / d(layer output)."""
    layers = linear_layers(model)
    inputs, outputs = [], []

    def hook(module, inp, out):
        inputs.append(inp[0].detach())
        outputs.append(out)

    handles = [layer.register_forward_hook(hook) for layer in layers]
    try:
        logp = torch.log_softmax(model(X), dim=-1)
    finally:
        for h in handles:
            h.remove()
    N, K = logp.shape
    A = []
    for a in inputs:
        a_aug = torch.cat([a, torch.ones(N, 1, dtype=a.dtype)], dim=1)
        A.append(a_aug.T @ a_aug / N)
    G = [torch.zeros(s.shape[1], s.shape[1], dtype=s.dtype) for s in outputs]
    if labels is None:  # exact expectation over y ~ p(. | x_n): one backward pass per class
        probs = logp.detach().exp()
        for k in range(K):
            grads = torch.autograd.grad(logp[:, k].sum(), outputs, retain_graph=True)
            w = probs[:, k].sqrt().unsqueeze(1)
            for i, g in enumerate(grads):
                G[i] += (w * g).T @ (w * g) / N
    else:
        picked = logp.gather(1, labels.reshape(-1, 1)).sum()
        grads = torch.autograd.grad(picked, outputs)
        for i, g in enumerate(grads):
            G[i] += g.T @ g / N
    return list(zip(A, G))


@app.cell(hide_code=True)
def _():
    checks.run(2, kfac_factors=kfac_factors)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def layer_grad(layer):
    """helper: [dW | db] of an nn.Linear from its .grad fields, shape (out, in + 1)."""
    return torch.cat([layer.weight.grad, layer.bias.grad.unsqueeze(1)], dim=1)


@app.function
def kfac_step(model, factors, *, lr, damping, kl_clip=None):
    """stage 3: precondition each Linear's .grad by its factors and update in place; returns the KL-clip scale nu."""
    layers = linear_layers(model)
    grads = [layer_grad(layer) for layer in layers]
    updates = [kron_precondition(g, *damped_inverses(A, G, damping)) for g, (A, G) in zip(grads, factors)]
    nu = 1.0
    if kl_clip is not None:
        vg = sum(float((u * g).sum()) for u, g in zip(updates, grads))
        if vg > 0:
            nu = min(1.0, math.sqrt(kl_clip / (lr**2 * vg)))
    with torch.no_grad():
        for layer, u in zip(layers, updates):
            layer.weight -= lr * nu * u[:, :-1]
            layer.bias -= lr * nu * u[:, -1]
    return nu


@app.cell(hide_code=True)
def _():
    checks.run(3, kfac_step=kfac_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class KFAC:
    """K-FAC for MLPs built from nn.Linear layers (other layers must have no parameters)."""

    def __init__(self, model, lr=0.1, damping=1e-2, *, momentum=0.9, ema_decay=0.95, stat_every=1, inv_every=10,
                 kl_clip=1e-3, fisher="sample", seed=None):
        """stage 4: store the settings; factors, inverses and momentum buffers start empty, self.steps = 0."""
        if fisher not in ("sample", "exact", "empirical"):
            raise ValueError(f"fisher must be 'sample', 'exact' or 'empirical', got {fisher!r}")
        self.model, self.lr, self.damping = model, lr, damping
        self.momentum, self.ema_decay, self.kl_clip, self.fisher = momentum, ema_decay, kl_clip, fisher
        self.stat_every, self.inv_every = stat_every, inv_every
        self.layers = linear_layers(model)
        self.factors, self.inverses, self.velocity = None, None, None
        self.steps = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(0 if seed is None else seed)

    def step(self, X, y):
        """stage 4: one K-FAC step on the minibatch (X, y): mean cross-entropy, factors, inverses, update; returns the loss."""
        model = self.model
        if self.steps % self.stat_every == 0:
            if self.fisher == "exact":
                labels = None
            elif self.fisher == "empirical":
                labels = y
            else:
                with torch.no_grad():
                    probs = torch.softmax(model(X), dim=-1)
                labels = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
            new = kfac_factors(model, X, labels)
            if self.factors is None:
                self.factors = new
            else:
                e = self.ema_decay
                self.factors = [(e * A + (1 - e) * An, e * G + (1 - e) * Gn)
                                for (A, G), (An, Gn) in zip(self.factors, new)]
        if self.inverses is None or self.steps % self.inv_every == 0:
            self.inverses = [damped_inverses(A, G, self.damping) for A, G in self.factors]
        model.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        grads = [layer_grad(layer) for layer in self.layers]
        updates = [kron_precondition(g, A_inv, G_inv) for g, (A_inv, G_inv) in zip(grads, self.inverses)]
        nu = 1.0
        if self.kl_clip is not None:
            vg = sum(float((u * g).sum()) for u, g in zip(updates, grads))
            if vg > 0:
                nu = min(1.0, math.sqrt(self.kl_clip / (self.lr**2 * vg)))
        if self.velocity is None:
            self.velocity = [torch.zeros_like(u) for u in updates]
        with torch.no_grad():
            for i, (layer, u) in enumerate(zip(self.layers, updates)):
                self.velocity[i] = self.momentum * self.velocity[i] + nu * u
                layer.weight -= self.lr * self.velocity[i][:, :-1]
                layer.bias -= self.lr * self.velocity[i][:, -1]
        self.steps += 1
        return loss.item()


@app.cell(hide_code=True)
def _():
    checks.run(4, KFAC=KFAC)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it ends with the block's
    project, K-FAC against a tuned Adam on chapter 15's MNIST network.
    """)
    return


if __name__ == "__main__":
    app.run()
