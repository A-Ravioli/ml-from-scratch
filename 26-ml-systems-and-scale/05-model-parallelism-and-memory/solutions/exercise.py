import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # model parallelism & memory · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `training_memory`: bytes per device for weights, gradients and optimizer state, with ZeRO sharding → `mlfs.systems` |
    | 2 | `saved_tensor_bytes`, `checkpoint`: measuring activation memory, and trading it for recomputation |
    | 3 | `split_mlp`, `tp_mlp_forward`: a tensor-parallel MLP (column- then row-parallel) |
    | 4 | `gpipe_schedule`, `bubble_fraction`: pipeline parallelism and its idle time |
    | 5 | `ZeroAdamW`: AdamW with its state sharded across ranks (ZeRO-1) → `mlfs.systems` |

    stage 5's multi-process check starts two gloo processes that load your notebook from disk;
    in the notebook it's skipped, so run `pytest -k stage5` in a terminal.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def training_memory(n_params, *, bytes_per_param=2, optimizer="adamw", zero_stage=0, world_size=1):
    """stage 1: per-device bytes {'params','grads','optimizer','total'}; fp32 master copy if bytes_per_param < 4; ZeRO shards."""
    states = {"adamw": 8, "momentum": 4, "sgd": 0}  # float32 optimizer state per parameter
    if optimizer not in states:
        raise ValueError(f"optimizer must be one of {sorted(states)}, got {optimizer!r}")
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2 or 3, got {zero_stage!r}")
    opt_per = states[optimizer] + (4 if bytes_per_param < 4 else 0)  # plus a float32 master copy in mixed precision
    shard = -(-n_params // world_size)  # each rank's slice of the flattened parameters

    def part(per_param, sharded):
        return (shard if sharded else n_params) * per_param

    out = {
        "params": part(bytes_per_param, zero_stage >= 3),
        "grads": part(bytes_per_param, zero_stage >= 2),
        "optimizer": part(opt_per, zero_stage >= 1),
    }
    out["total"] = out["params"] + out["grads"] + out["optimizer"]
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, training_memory=training_memory)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def saved_tensor_bytes(fn, *args, exclude=()):
    """stage 2: (fn(*args), bytes of the distinct storages autograd saves for backward during the call, minus `exclude`'s)."""
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in skip:
            seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(*args)
    return out, sum(seen.values())


@app.function
def checkpoint(fn, *args):
    """stage 2: fn(*args) (one tensor output) that saves only args and recomputes fn during backward, same RNG."""

    class Recompute(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            ctx.rng = torch.get_rng_state()
            ctx.save_for_backward(*inputs)
            with torch.no_grad():  # nothing inside fn is recorded: its activations are freed right away
                return fn(*inputs)

        @staticmethod
        def backward(ctx, grad_out):
            inputs = [a.detach().requires_grad_(a.requires_grad) for a in ctx.saved_tensors]
            with torch.random.fork_rng(devices=[]):  # leave the global RNG where the forward left it
                torch.set_rng_state(ctx.rng)  # replay the forward's random numbers (dropout masks)
                with torch.enable_grad():
                    out = fn(*inputs)
            torch.autograd.backward(out, grad_out)  # also accumulates into the parameters used by fn
            return tuple(a.grad if a.requires_grad else None for a in inputs)

    return Recompute.apply(*args)


@app.cell(hide_code=True)
def _():
    checks.run(2, saved_tensor_bytes=saved_tensor_bytes, checkpoint=checkpoint)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def split_mlp(fc1, fc2, world_size):
    """stage 3: per-rank (W1 rows, b1 slice, W2 columns) for hidden units [r h/p, (r+1) h/p); ValueError if p doesn't divide h."""
    h = fc1.out_features
    if h % world_size:
        raise ValueError(f"hidden width {h} isn't divisible by world_size {world_size}")
    k = h // world_size
    return [(fc1.weight[r * k:(r + 1) * k].detach().clone(), fc1.bias[r * k:(r + 1) * k].detach().clone(),
             fc2.weight[:, r * k:(r + 1) * k].detach().clone()) for r in range(world_size)]


@app.function
def tp_mlp_forward(x, shards, b2):
    """stage 3: (out, partials): rank r's partial gelu(x W1_r^T + b1_r) W2_r^T; out = their sum (the all-reduce) + b2."""
    partials = [F.gelu(x @ w1.T + b1) @ w2.T for w1, b1, w2 in shards]
    out = partials[0]
    for p in partials[1:]:
        out = out + p
    return out + b2, partials


@app.cell(hide_code=True)
def _():
    checks.run(3, split_mlp=split_mlp, tp_mlp_forward=tp_mlp_forward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def gpipe_schedule(n_stages, n_micro):
    """stage 4: grid[t][s] = ('F', m), ('B', m) or None: all forwards as a wavefront, then all backwards in reverse."""
    T = 2 * (n_micro + n_stages - 1)
    grid = [[None] * n_stages for _ in range(T)]
    for s in range(n_stages):
        for m in range(n_micro):
            grid[s + m][s] = ("F", m)
            grid[(n_micro + n_stages - 1) + (n_stages - 1 - s) + m][s] = ("B", m)
    return grid


@app.function
def bubble_fraction(n_stages, n_micro):
    """stage 4: idle fraction of a GPipe schedule: (p - 1) / (m + p - 1)."""
    return (n_stages - 1) / (n_micro + n_stages - 1)


@app.cell(hide_code=True)
def _():
    checks.run(4, gpipe_schedule=gpipe_schedule, bubble_fraction=bubble_fraction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class ZeroAdamW:
    """ZeRO-1 AdamW: each rank keeps AdamW state for its 1/world_size shard of the flattened parameters."""

    def __init__(self, params, *, rank, world_size, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        """stage 5: flatten sizes; this rank's shard = [rank S, (rank+1) S) with S = ceil(N / world_size); zero state."""
        self.params = list(params)
        self.rank, self.world_size = rank, world_size
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.numel = sum(p.numel() for p in self.params)
        self.shard_size = -(-self.numel // world_size)
        start = rank * self.shard_size
        self.shard = (min(start, self.numel), min(start + self.shard_size, self.numel))
        ref = self.params[0]
        self.exp_avg = torch.zeros(self.shard_size, dtype=torch.float32, device=ref.device)
        self.exp_avg_sq = torch.zeros(self.shard_size, dtype=torch.float32, device=ref.device)
        self.steps = torch.zeros(self.shard_size, dtype=torch.float32, device=ref.device)  # per element, like AdamW's per-parameter step

    def _flat(self, tensors):
        flat = torch.zeros(self.shard_size * self.world_size, dtype=torch.float32, device=self.params[0].device)
        flat[:self.numel] = torch.cat([t.detach().reshape(-1).float() for t in tensors])
        return flat

    @torch.no_grad()
    def step(self):
        """stage 5: average flat grads over ranks (all_reduce), AdamW on this rank's shard, all_gather the new params."""
        grads = self._flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params])
        has = self._flat([torch.full_like(p, float(p.grad is not None)) for p in self.params]) > 0
        comm = self.world_size > 1
        if comm:
            dist.all_reduce(grads)
            grads /= self.world_size
        flat = self._flat(self.params)
        lo, hi = self.rank * self.shard_size, (self.rank + 1) * self.shard_size
        w, g, h = flat[lo:hi], grads[lo:hi], has[lo:hi]  # like AdamW, parameters without a .grad are left alone
        self.steps += h.float()
        t = self.steps.clamp(min=1)
        b1, b2 = self.betas
        m = torch.where(h, b1 * self.exp_avg + (1 - b1) * g, self.exp_avg)
        v = torch.where(h, b2 * self.exp_avg_sq + (1 - b2) * g * g, self.exp_avg_sq)
        w_new = w * (1 - self.lr * self.weight_decay)  # decoupled weight decay
        w_new = w_new - self.lr / (1 - b1**t) * m / ((v / (1 - b2**t)).sqrt() + self.eps)
        w = torch.where(h, w_new, w)
        self.exp_avg, self.exp_avg_sq = m, v
        if comm:
            pieces = [torch.empty_like(w) for _ in range(self.world_size)]
            dist.all_gather(pieces, w)
            flat = torch.cat(pieces)
        else:
            flat[lo:hi] = w
        offset = 0
        for p in self.params:
            p.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()

    def zero_grad(self):
        """stage 5: set every parameter's .grad to None."""
        for p in self.params:
            p.grad = None


@app.cell(hide_code=True)
def _():
    checks.run(5, ZeroAdamW=ZeroAdamW)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green (including `pytest -k stage5` in a terminal), open the lab:
    `marimo edit lab.py`. from now on `from mlfs.systems import training_memory, ZeroAdamW`
    gives you the memory model and a sharded optimizer; chapter 27 uses the memory model to
    size its Llama, and the capstone uses `saved_tensor_bytes`'s idea to explain its memory.
    """)
    return


if __name__ == "__main__":
    app.run()
