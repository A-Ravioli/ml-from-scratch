import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.transformer import TransformerBlock  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # in-context learning as meta-learning · build

    spec-driven (L4): the **spec** boxes in the lesson define every function and class, the
    checks enforce them, and the internals are yours. a batch of $B$ regression tasks is
    `xs (B, n, d)`, `ys (B, n)`; every predictor returns `(B, n)`, where entry $i$ predicts
    $y_i$ from the first $i$ pairs (the **context**) and $x_i$, so entry 0 has no context at all.

    | stage | you build |
    |---|---|
    | 1 | `sample_linear_tasks`, `least_squares_predictions`, `position_mse`: the task family and its classical answer |
    | 2 | `gd_predictions`, `lsa_predict`, `gd_lsa_weights`: one step of gradient descent, written as linear self-attention |
    | 3 | `interleave`, `ICLTransformer`: a causal transformer that reads $(x_1, y_1, \dots, x_i)$ and predicts $y_i$ |
    | 4 | `train_icl`: meta-training on a stream of fresh tasks |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sample_linear_tasks(n_tasks, n_points, dim, *, noise_std=0.0, x_scale=1.0, generator=None):
    """stage 1: dict xs (B, n, d) ~ N(0, x_scale^2 I), w (B, d) ~ N(0, I), ys (B, n) = xs @ w + noise_std * N(0, 1)."""
    w = torch.randn(n_tasks, dim, generator=generator)
    xs = x_scale * torch.randn(n_tasks, n_points, dim, generator=generator)
    ys = (xs @ w.unsqueeze(-1)).squeeze(-1) + noise_std * torch.randn(n_tasks, n_points, generator=generator)
    return {"xs": xs, "ys": ys, "w": w}


@app.function
def least_squares_predictions(xs, ys, *, ridge=0.0):
    """stage 1: (B, n): entry i = x_i . w_hat from the first i pairs, minimum-norm least squares (ridge=0) or ridge; entry 0 is 0."""
    B, n, d = xs.shape
    X, Y = xs.double(), ys.double()
    out = torch.zeros(B, n, dtype=torch.float64)
    for i in range(1, n):
        Xi, Yi = X[:, :i], Y[:, :i, None]
        if ridge > 0:
            A = Xi.transpose(1, 2) @ Xi + ridge * torch.eye(d, dtype=torch.float64)
            w = torch.linalg.solve(A, Xi.transpose(1, 2) @ Yi)
        else:
            w = torch.linalg.pinv(Xi) @ Yi
        out[:, i] = (X[:, i:i + 1] @ w)[:, 0, 0]
    return out.to(xs.dtype)


@app.function
def position_mse(preds, ys):
    """stage 1: (n,) mean over tasks of the squared error at each position."""
    return ((preds - ys) ** 2).mean(0)


@app.cell(hide_code=True)
def _():
    checks.run(1, sample_linear_tasks=sample_linear_tasks, least_squares_predictions=least_squares_predictions,
               position_mse=position_mse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gd_predictions(xs, ys, *, lr, n_steps=1):
    """stage 2: (B, n): entry i = x_i . w after n_steps of GD from w = 0 on 1/2 sum_{j<i} (w . x_j - y_j)^2."""
    B, n, d = xs.shape
    out = torch.zeros(B, n, dtype=xs.dtype)
    for i in range(1, n):
        Xi, Yi = xs[:, :i], ys[:, :i]
        w = torch.zeros(B, d, dtype=xs.dtype)
        for _ in range(n_steps):
            resid = (Xi @ w.unsqueeze(-1)).squeeze(-1) - Yi  # (B, i)
            w = w - lr * (Xi.transpose(1, 2) @ resid.unsqueeze(-1)).squeeze(-1)
        out[:, i] = (xs[:, i] * w).sum(-1)
    return out


@app.function
def lsa_predict(xs, ys, x_query, W_kq, W_pv):
    """stage 2: (B,) last coordinate of e_q + W_pv sum_j e_j (e_j^T W_kq e_q), with e_j = (x_j, y_j) and e_q = (x_q, 0)."""
    E = torch.cat([xs, ys.unsqueeze(-1)], dim=-1)  # (B, n, d+1)
    e_q = torch.cat([x_query, torch.zeros_like(x_query[:, :1])], dim=-1)  # (B, d+1)
    scores = E @ (W_kq @ e_q.unsqueeze(-1))  # (B, n, 1): e_j^T W_kq e_q
    update = (scores * (E @ W_pv.T)).sum(1)  # sum_j (W_pv e_j) * score_j
    return (e_q + update)[:, -1]


@app.function
def gd_lsa_weights(dim, lr):
    """stage 2: (W_kq, W_pv), each (dim+1, dim+1), for which lsa_predict equals one GD step from w = 0 with step size lr."""
    W_kq = torch.zeros(dim + 1, dim + 1)
    W_kq[:dim, :dim] = torch.eye(dim)  # e_j^T W_kq e_q = x_j . x_q
    W_pv = torch.zeros(dim + 1, dim + 1)
    W_pv[dim, dim] = lr  # W_pv e_j = (0, ..., 0, lr * y_j)
    return W_kq, W_pv


@app.cell(hide_code=True)
def _():
    checks.run(2, gd_predictions=gd_predictions, lsa_predict=lsa_predict, gd_lsa_weights=gd_lsa_weights)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def interleave(xs, ys):
    """stage 3: (B, 2n, d+1) tokens: position 2i is (x_i, 0), position 2i+1 is (0, ..., 0, y_i)."""
    B, n, d = xs.shape
    z = torch.zeros(B, 2 * n, d + 1, dtype=xs.dtype, device=xs.device)
    z[:, 0::2, :d] = xs
    z[:, 1::2, d] = ys
    return z


@app.class_definition
class ICLTransformer(nn.Module):
    """a causal transformer over interleave(xs, ys) that predicts y_i at the token of x_i."""

    def __init__(self, dim, *, max_points=40, d_model=64, n_layers=4, n_heads=4):
        """stage 3: read_in (d+1 -> d_model), learned positions pos (2*max_points, d_model), blocks, ln_f, read_out (d_model -> 1)."""
        super().__init__()
        self.dim, self.max_points = dim, max_points
        self.read_in = nn.Linear(dim + 1, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(2 * max_points, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.read_out = nn.Linear(d_model, 1)

    def forward(self, xs, ys):
        """stage 3: (B, n, d), (B, n) -> (B, n) predictions; prediction i sees x_1, y_1, ..., x_i only."""
        B, n, d = xs.shape
        if d != self.dim or n > self.max_points:
            raise ValueError(f"expected dim {self.dim} and at most {self.max_points} points, got ({n}, {d})")
        h = self.read_in(interleave(xs, ys)) + self.pos[:2 * n]
        for block in self.blocks:
            h = block(h, is_causal=True)
        return self.read_out(self.ln_f(h))[:, 0::2, 0]


@app.cell(hide_code=True)
def _():
    checks.run(3, interleave=interleave, ICLTransformer=ICLTransformer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_icl(model, *, n_points, n_steps, batch_size=64, lr=1e-3, noise_std=0.0, seed=0):
    """stage 4: Adam on the mean squared error over fresh tasks each step (train mode); returns the list of losses, model left in eval mode."""
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(n_steps):
        batch = sample_linear_tasks(batch_size, n_points, model.dim, noise_std=noise_std, generator=generator)
        loss = ((model(batch["xs"], batch["ys"]) - batch["ys"]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    model.eval()
    return history


@app.cell(hide_code=True)
def _():
    checks.run(4, sample_linear_tasks=sample_linear_tasks, least_squares_predictions=least_squares_predictions,
               position_mse=position_mse, ICLTransformer=ICLTransformer, train_icl=train_icl)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it trains your transformer,
    looks inside a one-layer attention model, and ends with the block's project: in-context
    learning against least squares and your MAML from topic 09.
    """)
    return


if __name__ == "__main__":
    app.run()
