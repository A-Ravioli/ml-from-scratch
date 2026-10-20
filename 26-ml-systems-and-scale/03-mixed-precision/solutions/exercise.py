import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # mixed precision · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `float_format`, `round_to_format`: any floating-point format (fp16, bf16, fp8) simulated in NumPy |
    | 2 | `low_precision_sgd`: why mixed precision keeps an fp32 master copy of the weights |
    | 3 | `GradScaler`: dynamic loss scaling → `mlfs.systems` |
    | 4 | `amp_step`: one mixed-precision training step with autocast and loss scaling |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def float_format(exp_bits, man_bits):
    """stage 1: {'max', 'min_normal', 'min_subnormal', 'eps'} of an IEEE-style format (top exponent = inf/nan)."""
    bias = 2 ** (exp_bits - 1) - 1
    return {
        "max": (2.0 - 2.0**-man_bits) * 2.0**bias,
        "min_normal": 2.0 ** (1 - bias),
        "min_subnormal": 2.0 ** (1 - bias - man_bits),
        "eps": 2.0**-man_bits,
    }


@app.function
def round_to_format(x, exp_bits, man_bits):
    """stage 1: round float array x to the nearest value of the format (ties to even); overflow -> +-inf; float64 out."""
    x = np.asarray(x, dtype=np.float64)
    bias = 2 ** (exp_bits - 1) - 1
    fmt = float_format(exp_bits, man_bits)
    a = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        _, e = np.frexp(a)  # a = f * 2**e with f in [0.5, 1): the leading bit is 2**(e - 1)
        e = np.maximum(e - 1, 1 - bias)  # below the normal range, the spacing stops shrinking: subnormals
        q = np.ldexp(1.0, e - man_bits)  # spacing between neighbouring representable values
        r = np.round(a / q) * q  # np.round rounds halves to even
        r = np.where(r > fmt["max"], np.inf, r)
        r = np.where(np.isfinite(a), r, a)  # inf and nan pass through
    return np.copysign(r, x)


@app.cell(hide_code=True)
def _():
    checks.run(1, float_format=float_format, round_to_format=round_to_format)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def low_precision_sgd(w0, grads, lr, exp_bits, man_bits, *, master=False):
    """stage 2: w <- w - lr * g for each g in grads; weights stored in the format, or in float64 if master; returns rounded w."""
    w = round_to_format(w0, exp_bits, man_bits) if not master else np.asarray(w0, dtype=np.float64).copy()
    for g in grads:
        g = round_to_format(g, exp_bits, man_bits)  # gradients arrive in low precision either way
        if master:
            w = w - lr * g  # the update accumulates in high precision
        else:
            w = round_to_format(w - lr * g, exp_bits, man_bits)  # every update is rounded back into the format
    return round_to_format(w, exp_bits, man_bits)


@app.cell(hide_code=True)
def _():
    checks.run(2, low_precision_sgd=low_precision_sgd, round_to_format=round_to_format)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class GradScaler:
    """dynamic loss scaling for any device (torch.amp.GradScaler semantics, step returns whether it stepped)."""

    def __init__(self, init_scale=65536.0, *, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        """stage 3: store the settings; scale starts at init_scale; no steps counted yet."""
        self._scale = float(init_scale)
        self.growth_factor, self.backoff_factor, self.growth_interval = growth_factor, backoff_factor, growth_interval
        self._good_steps = 0
        self._found_inf = False
        self._unscaled = False

    def get_scale(self):
        """stage 3: the current loss scale (float)."""
        return self._scale

    def scale(self, loss):
        """stage 3: loss (float32) times the current scale, to call .backward() on."""
        return loss.float() * self._scale

    def unscale_(self, optimizer):
        """stage 3: divide every .grad in optimizer.param_groups by the scale in place; record any inf/nan."""
        if self._unscaled:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update()")
        inv = 1.0 / self._scale
        bad = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.mul_(inv)
                bad.append(~torch.isfinite(p.grad).all())  # stays on the device: no sync per parameter
        self._found_inf = bool(torch.stack(bad).any()) if bad else False  # one sync for the whole model
        self._unscaled = True

    def step(self, optimizer):
        """stage 3: unscale_ if not done yet; skip optimizer.step() and return False if any grad is inf/nan, else step, True."""
        if not self._unscaled:
            self.unscale_(optimizer)
        if self._found_inf:
            return False
        optimizer.step()
        return True

    def update(self):
        """stage 3: back off the scale after an overflow, grow it after growth_interval clean steps; reset per-step state."""
        if self._found_inf:
            self._scale *= self.backoff_factor
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps == self.growth_interval:
                self._scale *= self.growth_factor
                self._good_steps = 0
        self._found_inf = self._unscaled = False


@app.cell(hide_code=True)
def _():
    checks.run(3, GradScaler=GradScaler)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def amp_step(model, loss_fn, optimizer, x, y, *, dtype=torch.float16, scaler=None, grad_clip=None):
    """stage 4: zero grads, forward under autocast(x's device, dtype), (scaled) backward, unscale, clip, step, update; (loss, stepped)."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=x.device.type, dtype=dtype):
        loss = loss_fn(model(x), y)
    if scaler is None:
        loss.float().backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach()), True
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    stepped = scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), stepped


@app.cell(hide_code=True)
def _():
    checks.run(4, amp_step=amp_step, GradScaler=GradScaler)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.systems import GradScaler` gives you loss scaling for any device, and the
    capstone trains your GPT in fp16 with it.
    """)
    return


if __name__ == "__main__":
    app.run()
