import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # autoregressive models · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. an autoregressive model writes a
    joint distribution as a chain of conditionals, $p(x) = \prod_i p(x_i \mid x_{<i})$, and
    computes all of them in one forward pass by hiding the future with **masks**. you'll
    build the masked layers, MADE for vectors, and PixelCNN for images, with exact
    likelihoods and (slow) exact sampling.

    | stage | you build |
    |---|---|
    | 1 | `MaskedLinear`: a linear layer whose weight is multiplied by a fixed 0/1 mask → `mlfs.generative` |
    | 2 | `MADE`: masks from degrees, so output $i$ sees only inputs $< i$ → `mlfs.generative` |
    | 3 | `made_nll`, `made_sample`: exact likelihood of binary vectors, and sampling one dimension at a time |
    | 4 | `MaskedConv2d`: type-A and type-B masked convolutions → `mlfs.generative` |
    | 5 | `PixelCNN`, `pixelcnn_nll`: a stack of masked convolutions with a categorical output per pixel → `mlfs.generative` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class MaskedLinear(nn.Linear):
    """a linear layer computing x @ (weight * mask).T + bias, with mask a registered (out, in) buffer."""

    def __init__(self, in_features, out_features, bias=True):
        """stage 1: an nn.Linear plus a buffer `mask` of ones, shape (out_features, in_features)."""
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        """stage 1: copy a 0/1 mask (tensor or ndarray, shape (out, in)) into the buffer."""
        self.mask.copy_(torch.as_tensor(np.asarray(mask), dtype=self.mask.dtype))

    def forward(self, x):
        """stage 1: F.linear with the masked weight, so masked entries get no output and no gradient."""
        return F.linear(x, self.weight * self.mask, self.bias)


@app.cell(hide_code=True)
def _():
    checks.run(1, MaskedLinear=MaskedLinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class MADE(nn.Module):
    """masked autoencoder for distribution estimation, natural ordering: output [:, :, i] depends only on x[:, :i]."""

    def __init__(self, in_dim, hidden_dims, *, n_outputs_per_dim=1, seed=0):
        """stage 2: input degrees 1..D; hidden degrees drawn with np.random.default_rng(seed) from [min of the previous
        layer's degrees, D - 1]; masks hidden >= previous, output d (degree d+1) > hidden; MaskedLinear + ReLU layers."""
        super().__init__()
        self.in_dim, self.n_out = in_dim, n_outputs_per_dim
        rng = np.random.default_rng(seed)
        degrees = [np.arange(1, in_dim + 1)]
        for h in hidden_dims:
            lo = min(int(degrees[-1].min()), in_dim - 1)
            degrees.append(rng.integers(lo, in_dim - 1, size=h, endpoint=True))
        sizes = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(hidden_dims)):
            lin = MaskedLinear(sizes[i], sizes[i + 1])
            lin.set_mask(degrees[i + 1][:, None] >= degrees[i][None, :])
            layers += [lin, nn.ReLU()]
        out = MaskedLinear(sizes[-1], in_dim * n_outputs_per_dim)
        out_deg = np.tile(np.arange(1, in_dim + 1), n_outputs_per_dim)
        out.set_mask(out_deg[:, None] > degrees[-1][None, :])
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """stage 2: x (B, D) -> (B, n_outputs_per_dim, D); output unit o * D + d is entry [:, o, d]."""
        return self.net(x).view(x.shape[0], self.n_out, self.in_dim)


@app.cell(hide_code=True)
def _():
    checks.run(2, MADE=MADE, MaskedLinear=MaskedLinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def made_nll(model, x):
    """stage 3: exact nll in nats (B,) of binary x (B, D) under a MADE with one output (a Bernoulli logit) per dim."""
    logits = model(x)[:, 0, :]
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)


@app.function
def made_sample(model, n, *, generator=None):
    """stage 3: n binary samples (n, D) float, one dimension per forward pass (D passes), x_i ~ Bernoulli(sigmoid)."""
    x = torch.zeros(n, model.in_dim)
    with torch.no_grad():
        for i in range(model.in_dim):
            p = torch.sigmoid(model(x)[:, 0, i])
            x[:, i] = torch.bernoulli(p, generator=generator)
    return x


@app.cell(hide_code=True)
def _():
    checks.run(3, made_nll=made_nll, made_sample=made_sample, MADE=MADE)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class MaskedConv2d(nn.Conv2d):
    """a raster-scan causal convolution: 'A' hides the centre pixel and everything after it, 'B' only what's after."""

    def __init__(self, mask_type, in_channels, out_channels, kernel_size):
        """stage 4: nn.Conv2d with padding kernel_size // 2 and a buffer `mask` (out, in, k, k); ValueError for other types."""
        if mask_type not in ("A", "B"):
            raise ValueError(f"mask_type must be 'A' or 'B', got {mask_type!r}")
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        k, c = kernel_size, kernel_size // 2
        mask = torch.ones(out_channels, in_channels, k, k)
        mask[:, :, c, c + (mask_type == "B"):] = 0
        mask[:, :, c + 1:, :] = 0
        self.mask_type = mask_type
        self.register_buffer("mask", mask)

    def forward(self, x):
        """stage 4: convolution with weight * mask."""
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)


@app.cell(hide_code=True)
def _():
    checks.run(4, MaskedConv2d=MaskedConv2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class PixelCNN(nn.Module):
    """x (B, C, H, W) in [0, 1] -> logits (B, n_levels, C, H, W); channels of a pixel are independent given the past."""

    def __init__(self, in_channels=1, *, n_levels=256, hidden_channels=64, n_layers=7):
        """stage 5: a 7x7 type-A masked conv, n_layers - 1 residual 3x3 type-B masked convs, two 1x1 convs to n_levels * C."""
        super().__init__()
        self.in_channels, self.n_levels = in_channels, n_levels
        h = hidden_channels
        self.first = MaskedConv2d("A", in_channels, h, 7)
        self.blocks = nn.ModuleList(MaskedConv2d("B", h, h, 3) for _ in range(n_layers - 1))
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(h, h, 1), nn.ReLU(), nn.Conv2d(h, n_levels * in_channels, 1))

    def forward(self, x):
        """stage 5: logits (B, n_levels, C, H, W); the input is rescaled to [-1, 1] first."""
        h = self.first(2 * x - 1)
        for block in self.blocks:
            h = h + block(torch.relu(h))
        out = self.head(h)
        b, _, hh, ww = out.shape
        return out.view(b, self.n_levels, self.in_channels, hh, ww)

    @torch.no_grad()
    def sample(self, n, img_size, *, generator=None, device="cpu"):
        """stage 5: n images (n, C, H, W) in [0, 1], one forward pass per pixel in raster order (H * W passes);
        img_size is an int or (H, W); levels drawn with torch.multinomial(..., generator=generator) on the CPU."""
        hh, ww = (img_size, img_size) if isinstance(img_size, int) else img_size
        was = self.training
        self.eval()
        x = torch.zeros(n, self.in_channels, hh, ww, device=device)
        for i in range(hh):
            for j in range(ww):
                logits = self(x)[:, :, :, i, j]  # (n, L, C)
                probs = torch.softmax(logits.float(), dim=1).permute(0, 2, 1).reshape(-1, self.n_levels).cpu()
                lev = torch.multinomial(probs, 1, generator=generator).view(n, self.in_channels)
                x[:, :, i, j] = (lev.float() / (self.n_levels - 1)).to(device)
        self.train(was)
        return x


@app.function
def pixelcnn_nll(model, x):
    """stage 5: exact nll in nats per image (B,) of x (B, C, H, W) in [0, 1] with levels round(x * (n_levels - 1))."""
    levels = torch.round(x * (model.n_levels - 1)).long()
    return F.cross_entropy(model(x), levels, reduction="none").sum(dim=(1, 2, 3))


@app.cell(hide_code=True)
def _():
    checks.run(5, PixelCNN=PixelCNN, pixelcnn_nll=pixelcnn_nll, MaskedConv2d=MaskedConv2d)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import MADE, PixelCNN` gives you these models; the capstone
    trains your PixelCNN as the autoregressive family.
    """)
    return


if __name__ == "__main__":
    app.run()
