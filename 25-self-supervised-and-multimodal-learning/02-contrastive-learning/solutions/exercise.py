import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # contrastive learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `info_nce_loss`: the NT-Xent loss over 2B views → `mlfs.ssl` |
    | 2 | `SimCLRAugment`: batched random crops, flips, colour jitter and grayscale → `mlfs.ssl` |
    | 3 | `ProjectionHead`: the MLP the loss is computed on → `mlfs.ssl` |
    | 4 | `train_simclr`: the SimCLR training loop |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def info_nce_loss(z1, z2, *, temperature=0.1):
    """stage 1: NT-Xent: z1[i], z2[i] (B, d) are positives, the other 2B-2 views negatives; cosine / temperature; mean."""
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    n = z.shape[0]
    b = n // 2
    logits = z @ z.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    targets = (torch.arange(n, device=z.device) + b) % n
    return F.cross_entropy(logits, targets)


@app.cell(hide_code=True)
def _():
    checks.run(1, info_nce_loss=info_nce_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SimCLRAugment:
    """SimCLR's augmentations, batched on tensors: every image in the batch gets its own random draw."""

    LUMA = (0.299, 0.587, 0.114)

    def __init__(self, size=32, *, strength=0.5):
        """stage 2: output size (square) and colour-jitter strength s (factors in [1 - 0.8s, 1 + 0.8s])."""
        self.size = size
        self.strength = strength

    def _gray(self, x):
        if x.shape[1] == 1:
            return x
        w = torch.tensor(self.LUMA, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    def __call__(self, x):
        """stage 2: x (B, C, H, W) floats in [0, 1] -> (B, C, size, size) in [0, 1]: crop, flip, jitter (p=0.8), grayscale (p=0.2)."""
        B, C = x.shape[:2]
        dev, dt = x.device, x.dtype

        def u(lo, hi):  # (B,) uniform draws from torch's global generator
            return lo + (hi - lo) * torch.rand(B, device=dev, dtype=dt)

        # random resized crop: area fraction in [0.2, 1], aspect ratio log-uniform in [3/4, 4/3]
        area = u(0.2, 1.0)
        ratio = torch.exp(u(math.log(3 / 4), math.log(4 / 3)))
        w = torch.sqrt(area * ratio).clamp(max=1.0)  # width and height as fractions of the image
        h = torch.sqrt(area / ratio).clamp(max=1.0)
        cx = w / 2 + (1 - w) * torch.rand(B, device=dev, dtype=dt)
        cy = h / 2 + (1 - h) * torch.rand(B, device=dev, dtype=dt)
        flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0).to(dt)
        theta = torch.zeros(B, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = w * flip
        theta[:, 0, 2] = 2 * cx - 1
        theta[:, 1, 1] = h
        theta[:, 1, 2] = 2 * cy - 1
        grid = F.affine_grid(theta, (B, C, self.size, self.size), align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        # colour jitter, applied to each image with probability 0.8: brightness, contrast, saturation
        s = 0.8 * self.strength
        apply = (torch.rand(B, device=dev) < 0.8).view(B, 1, 1, 1)
        jit = out * u(1 - s, 1 + s).view(B, 1, 1, 1)
        jit = jit.clamp(0, 1)
        mean = self._gray(jit).mean(dim=(1, 2, 3), keepdim=True)
        jit = ((jit - mean) * u(1 - s, 1 + s).view(B, 1, 1, 1) + mean).clamp(0, 1)
        g = self._gray(jit)
        jit = ((jit - g) * u(1 - s, 1 + s).view(B, 1, 1, 1) + g).clamp(0, 1)
        out = torch.where(apply, jit, out)

        # random grayscale with probability 0.2 (keeps C channels)
        gray = (torch.rand(B, device=dev) < 0.2).view(B, 1, 1, 1)
        out = torch.where(gray, self._gray(out).expand(-1, C, -1, -1), out)
        return out.clamp(0, 1)


@app.cell(hide_code=True)
def _():
    checks.run(2, SimCLRAugment=SimCLRAugment)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class ProjectionHead(nn.Module):
    """g(h): the MLP between the representation h and the embedding z the loss sees."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        """stage 3: self.net = Linear(in_dim, hidden_dim) -> BatchNorm1d -> ReLU -> Linear(hidden_dim, out_dim)."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        """stage 3: (B, in_dim) -> (B, out_dim)."""
        return self.net(x)


@app.cell(hide_code=True)
def _():
    checks.run(3, ProjectionHead=ProjectionHead)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_simclr(encoder, feature_dim, images, *, epochs=1, batch_size=256, temperature=0.1, lr=1e-3, strength=0.5,
                 hidden_dim=512, proj_dim=128, seed=0, device="cpu"):
    """stage 4: SimCLR on unlabelled images (N, C, H, W) in [0, 1]; trains encoder in place; returns (head, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    n = len(X)
    bs = min(batch_size, n)
    head = ProjectionHead(feature_dim, hidden_dim, proj_dim)
    encoder.to(device)
    head.to(device)
    augment = SimCLRAugment(X.shape[-1], strength=strength)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    history = {"loss": []}
    for _ in range(epochs):
        encoder.train()
        head.train()
        perm = rng.permutation(n)
        for start in range(0, n - bs + 1, bs):
            xb = X[perm[start:start + bs]].to(device)
            z1 = head(encoder(augment(xb)))
            z2 = head(encoder(augment(xb)))
            loss = info_nce_loss(z1, z2, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            history["loss"].append(loss.item())
    return head, history


@app.cell(hide_code=True)
def _():
    checks.run(4, train_simclr=train_simclr, info_nce_loss=info_nce_loss, ProjectionHead=ProjectionHead,
               SimCLRAugment=SimCLRAugment)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `info_nce_loss`, `SimCLRAugment` and `ProjectionHead` are part
    of `mlfs.ssl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
