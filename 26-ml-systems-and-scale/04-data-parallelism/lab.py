import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    import textwrap

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    N_CORES = os.cpu_count() or 4

    plt.rcParams["figure.autolayout"] = True

    WORKER = textwrap.dedent(r"""
        import json, os, pickle, sys, time
        import numpy as np
        import torch, torch.distributed as dist
        from torch import nn
        nb_path, rank, world, store, out, task, args = sys.argv[1:8]
        rank, world, args = int(rank), int(world), json.loads(args)
        torch.set_num_threads(args.get("threads", 1))
        dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=world)
        res = {}
        try:
            if task == "burn":  # a fixed amount of single-threaded work: 512^3 matmuls, or elementwise passes
                a = torch.randn(512, 512) if args["kind"] == "matmul" else torch.randn(2**16)
                b = torch.randn_like(a)
                fn = (lambda: a @ b) if args["kind"] == "matmul" else (lambda: torch.exp(a).mul_(b))
                fn()
                dist.barrier()  # every process starts its work at the same moment
                t0 = time.perf_counter()
                for _ in range(args["iters"]):
                    fn()
                dist.barrier()  # ... and the clock stops when the slowest has finished
                res = {"time": time.perf_counter() - t0}
            elif task == "allreduce":
                for nbytes in args["sizes"]:
                    t = torch.ones(nbytes // 4)
                    times = []
                    for rep in range(args["reps"] + 2):
                        dist.barrier()
                        t0 = time.perf_counter()
                        dist.all_reduce(t)
                        times.append(time.perf_counter() - t0)
                    res[nbytes] = min(times[2:])
                small = [torch.ones(1024) for _ in range(100)]
                big = torch.ones(100 * 1024)
                for name, fn in (("100 x 4 KB", lambda: [dist.all_reduce(s) for s in small]),
                                 ("1 x 400 KB", lambda: dist.all_reduce(big))):
                    times = []
                    for rep in range(args["reps"] + 2):
                        dist.barrier()
                        t0 = time.perf_counter()
                        fn()
                        times.append(time.perf_counter() - t0)
                    res[name] = min(times[2:])
            else:
                from mlfs import notebooks
                nb = notebooks.load(nb_path)
                g = torch.Generator().manual_seed(0)
                X = torch.randn(args["n"], 256, generator=g)
                y = (X[:, :10].sum(1) > 0).long()
                torch.manual_seed(0)
                m = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 2))
                comm = []
                real = dist.all_reduce
                def timed(*a, **k):
                    t0 = time.perf_counter()
                    r = real(*a, **k)
                    comm.append(time.perf_counter() - t0)
                    return r
                dist.all_reduce = timed
                if args.get("no_comm"):  # the same per-rank work, with every collective of ddp_train switched off
                    dist.all_reduce = lambda *a, **k: None
                    dist.broadcast = lambda *a, **k: None
                dist.barrier()
                t0 = time.perf_counter()
                losses = nb.ddp_train(m, X, y, nn.functional.cross_entropy, rank=rank, world_size=world, epochs=1,
                                      batch_size=args["batch"], lr=0.05, seed=0)
                dist.barrier()
                res = {"time": time.perf_counter() - t0, "comm": sum(comm), "steps": len(losses), "loss": losses[-1]}
            with open(out, "wb") as f:
                pickle.dump(res, f)
        finally:
            dist.destroy_process_group()
    """)


@app.function
def run_ranks(world, task, args, timeout=150):
    """start `world` gloo processes running WORKER's `task`; return rank 0's result (kills everything on timeout)."""
    with tempfile.TemporaryDirectory() as d:
        outs = [os.path.join(d, f"r{r}.pkl") for r in range(world)]
        procs = [subprocess.Popen([sys.executable, "-c", WORKER, ex.__file__, str(r), str(world),
                                   os.path.join(d, "store"), outs[r], task, json.dumps(args)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env={**os.environ, "MLFS_ROOT": str(notebooks.repo_root())}) for r in range(world)]
        try:
            errs = [p.communicate(timeout=timeout)[1] for p in procs]
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            raise RuntimeError(f"{world} processes timed out after {timeout} s")
        for p, e in zip(procs, errs):
            if p.returncode != 0:
                raise RuntimeError(e[-2000:])
        with open(outs[0], "rb") as f:
            return pickle.load(f)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # data parallelism · lab

    the experiments start real processes on your machine that talk through `torch.distributed`'s
    gloo backend, and run **your** `ddp_train` (loaded from `exercise.py` on disk, so save it
    first) next to your cost model. for each one: write down a prediction, lock it in, run it,
    then explain what happened. compare with `solutions/lab_notes.md` afterwards.

    your machine reports **{N_CORES}** cores. every rank here is a process on that one
    machine, so "the network" is shared memory and the ranks compete for the same cores and
    memory bandwidth: the shape of the results is what carries over to a cluster, not the
    numbers. about 1–2 minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what an all-reduce costs

    gloo all-reduces of 1 KB to 16 MB between 2 and 4 processes, best of 5 each. the line is
    your `allreduce_time` with a latency $\alpha$ and bandwidth $\beta$ fitted to the
    measurements by least squares on relative error ($T$ is linear in $\alpha$ and $1/\beta$). then the same
    400 KB reduced as one bucket or as 100 separate 4 KB tensors.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("all-reducing 100 tensors of 4 KB one by one vs one 400 KB bucket: how many times slower?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    sizes = [2**k for k in range(10, 25, 2)]
    ar = {w: run_ranks(w, "allreduce", {"sizes": sizes, "reps": 5}) for w in (2, 4)}
    fits = {}
    for _w, _res in ar.items():
        _n = np.array(sizes, dtype=float)
        _t = np.array([_res[s] for s in sizes])
        _A = np.stack([2 * (_w - 1) * np.ones_like(_n), 2 * (_w - 1) / _w * _n], axis=1)
        # weighted by 1 / T: fit relative errors, so the small (latency-bound) messages count as much as the big ones
        (_alpha, _inv_beta), *_ = np.linalg.lstsq(_A / _t[:, None], np.ones_like(_t), rcond=None)
        fits[_w] = (max(_alpha, 0.0), 1 / _inv_beta)
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _w, _res in ar.items():
        _a, _b = fits[_w]
        _ax.loglog(sizes, [_res[s] for s in sizes], "o", label=f"measured, p = {_w}")
        _ax.loglog(sizes, [ex.allreduce_time(s, _w, bandwidth=_b, latency=_a) for s in sizes], "-", color=_ax.lines[-1].get_color(),
                   label=f"model: α = {1e6 * _a:.0f} µs, β = {_b / 1e9:.1f} GB/s")
    _ax.set(xlabel="message size (bytes)", ylabel="all-reduce time (s)", title="gloo on one machine")
    _ax.legend(fontsize=7)
    _rows = "\n".join(f"| {w} | {1e3 * ar[w]['100 x 4 KB']:.2f} ms | {1e3 * ar[w]['1 x 400 KB']:.2f} ms | "
                      f"**{ar[w]['100 x 4 KB'] / ar[w]['1 x 400 KB']:.0f}×** |" for w in ar)
    _bytes = "\n".join(f"| {p} | {ex.allreduce_bytes(1.0, p):.3f} n | {2 * (p - 1)} n |" for p in (2, 4, 8, 64, 1024))
    mo.vstack([_fig, mo.md("| ranks | 100 × 4 KB | 1 × 400 KB | ratio |\n|---|---|---|---|\n" + _rows +
                           "\n\nbytes through the busiest rank, from your `allreduce_bytes`:\n\n"
                           "| ranks | ring (per rank) | parameter server (root) |\n|---|---|---|\n" + _bytes)])
    return (ar,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** below what size is an all-reduce latency-bound on your machine, according
    to the fit? the bucket comparison moves the same bytes: account for the ratio with the
    model. what does that imply for a framework that reduces each parameter tensor of a GPT
    separately? does going from 2 to 4 ranks change the time for large messages much, and
    does the ring model say it should?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does your DDP scale?

    your `ddp_train` on 16,384 synthetic examples (one epoch, local batch 64, an MLP with
    1.3M parameters, so a 5 MB gradient per step) with 1, 2 and 4 processes. more processes
    means fewer steps each, so ideal **strong scaling** halves the time with every doubling.
    first with one thread per process (as if each rank were its own small machine), then with
    the machine's cores split evenly between the ranks.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("with one thread per rank, what speedup will 4 ranks give over 1? with the cores split between the "
                    "ranks?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    scaling = {}
    for _mode in ("1 thread per rank", "cores split between ranks"):
        for _w in (1, 2, 4):
            _threads = 1 if _mode.startswith("1") else max(1, N_CORES // _w)
            scaling[(_mode, _w)] = run_ranks(_w, "ddp", {"n": 16384, "batch": 64, "threads": _threads})
    # control: the same per-rank work as independent 1-process jobs started together (no communication at all)
    control = {}
    for _w in (2, 4):
        control[_w] = run_ranks(_w, "ddp", {"n": 16384, "batch": 64, "threads": 1, "no_comm": True})["time"]
    # second control: is it the matmuls? the same fixed work as 1 or 4 concurrent processes, matmul vs elementwise
    burn = {}
    for _kind, _iters in (("matmul", 1500), ("elementwise", 3000)):
        for _w in (1, 4):  # no communication inside the timed loop: the process group only lines up the start
            burn[(_kind, _w)] = run_ranks(_w, "burn", {"kind": _kind, "iters": _iters, "threads": 1})["time"]
    _rows = []
    for (_mode, _w), _r in scaling.items():
        _t1 = scaling[(_mode, 1)]["time"]
        _rows.append(f"| {_mode} | {_w} | {_r['steps']} | {_r['time']:.2f} s | {_t1 / _r['time']:.2f}× | "
                     f"**{100 * _t1 / _r['time'] / _w:.0f}%** | {100 * _r['comm'] / _r['time']:.0f}% |")
    _fig, _ax = plt.subplots(figsize=(5, 3))
    for _mode in ("1 thread per rank", "cores split between ranks"):
        _t1 = scaling[(_mode, 1)]["time"]
        _ax.plot([1, 2, 4], [_t1 / scaling[(_mode, w)]["time"] for w in (1, 2, 4)], "o-", label=_mode)
    _ax.plot([1, 4], [1, 4], "k:", label="ideal")
    _ax.set(xlabel="ranks", ylabel="speedup over 1 rank", xticks=[1, 2, 4])
    _ax.legend(fontsize=7)
    _ctrl = "\n".join(f"| {w} 1-thread ranks with all_reduce and broadcast switched off, {16384 // w:,} examples each | "
                       f"{control[w]:.2f} s (slowest) | vs {scaling[('1 thread per rank', 1)]['time'] / w:.2f} s if they "
                       "didn't slow each other down |" for w in control)
    _burn = "\n".join(f"| {k} | {burn[(k, 1)]:.2f} s | {burn[(k, 4)]:.2f} s | **{4 * burn[(k, 1)] / burn[(k, 4)]:.1f}×** |"
                       for k in ("matmul", "elementwise"))
    mo.vstack([_fig, mo.md("| setting | ranks | steps per rank | time | speedup | efficiency | time in all_reduce |\n"
                           "|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
                           + "\n\n**control 1** (no communication):\n\n| run | time | ideal |\n|---|---|---|\n" + _ctrl
                           + "\n\n**control 2** (single-threaded work, one process vs four at once; throughput = "
                           "4 × time(1) / time(4), ideal 4×):\n\n| work | 1 process | 4 processes, started together | "
                           "total throughput of 4 |\n|---|---|---|---|\n" + _burn)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with one thread per rank, how far from ideal is the speedup, and how much
    of each rank's time goes to all-reduce? use experiment 1's fitted model to check whether
    the communication time is what a 5 MB all-reduce should cost. why is the
    "cores split" setting so much worse at scaling, even though its 1-rank run is the fastest
    of all? the control runs the same per-rank work with no communication at all: what does it
    say about where the lost speedup goes? what would change on a cluster where each rank has its own machine and a network
    between them?
    """)
    return


if __name__ == "__main__":
    app.run()
