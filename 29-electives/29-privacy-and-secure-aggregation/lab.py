import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    # and your topic 27 code (electives own no mlfs module, so load it directly)
    def _smoke(m):  # raises NotImplementedError if a stage you need here isn't done yet
        m.partition_iid(10, 2, seed=0)
        m.client_update(torch.nn.Linear(2, 2), torch.randn(4, 2), torch.tensor([0, 1, 0, 1]), batch_size=None, seed=0)

    prev = notebooks.load_topic("27-federated-averaging", key="fedavg", smoke=_smoke)
    fl = prev.module
    torch.set_num_threads(1)  # small models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # privacy & secure aggregation · lab

    every experiment here runs **your** code: secure aggregation, the DP aggregate and the
    accountant from this topic, and `client_update` and `partition_iid` from topic 27. for
    each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the lab takes about
    two and a half minutes on one CPU core.

    experiments 1 and 2 look at the two tools separately. experiment 3 is the block's
    **project**: the privacy–accuracy–communication triangle.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: secure aggregation at scale

    $n$ clients each hold a random update the size of experiment 3's model (6,634 numbers).
    your `secure_aggregate` runs the whole protocol (keys, Shamir shares, pairwise masks,
    and recovery when 20% of the clients drop out after the masks are agreed), and we
    compare its output with the plain sum of the survivors' updates. we also look at what
    one masked upload looks like, and at the upload overhead your `secagg_upload_bytes`
    predicts for larger cohorts.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how close is the secure sum to the plain sum (the largest error, as a number)? and at what cohort size does "
        "the per-client key-and-share overhead exceed the masked update itself for a 6,634-parameter model?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _dim = 6_634
    _g = np.random.default_rng(0)
    _rows = []
    for _n in [5, 20, 50]:
        _ups = {i: _g.normal(scale=0.05, size=_dim) for i in range(_n)}
        _drop = list(range(0, _n, 5))  # every fifth client drops out
        _t0 = time.perf_counter()
        _sec = ex.secure_aggregate(_ups, threshold=int(np.ceil(0.6 * _n)), dropped=_drop, seed=_n)
        _dt = time.perf_counter() - _t0
        _plain = sum(v for i, v in _ups.items() if i not in _drop)
        _rows.append(f"| {_n} | {len(_drop)} | {np.abs(_sec - _plain).max():.2e} | {(_n - len(_drop)) * 0.5 / 2**16:.2e} | "
                     f"{_dt:.2f} |")
    _secrets = {i: int(_g.integers(2, 2**61 - 2)) for i in range(5)}
    _publics = {i: ex.public_key(s) for i, s in _secrets.items()}
    _z = ex.encode(_g.normal(scale=0.05, size=_dim))
    _y = ex.mask(0, _z, _secrets[0], _publics)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.4))
    _axes[0].hist(_z / 2**32, bins=50)
    _axes[0].set(title="one client's encoded update / 2^32", xlabel="value", ylabel="count")
    _axes[1].hist(_y / 2**32, bins=50)
    _axes[1].set(title="the same update after masking / 2^32", xlabel="value")
    _fig.tight_layout()
    _ns = [10, 100, 1_000, 10_000]
    _over = "\n".join(f"| {P:,} | " + " | ".join(f"{(ex.secagg_upload_bytes(n, P) - 4 * P) / (4 * P):.1%}" for n in _ns)
                      + " |" for P in (6_634, 50_890, 1_000_000))
    mo.vstack([
        mo.md("| clients | dropped | largest error vs plain sum | rounding bound n·0.5/scale | seconds |\n|---|---|---|---|---|\n"
              + "\n".join(_rows)),
        _fig,
        mo.md("**upload overhead per client** (keys and shares, relative to the 4-byte-per-entry masked update):\n\n"
              "| parameters | " + " | ".join(f"n = {n:,}" for n in _ns) + " |\n|---|---|---|---|---|\n" + _over),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the error exactly at the level of fixed-point rounding and not
    larger, even though every masked number is a huge random integer? the left histogram is
    an update encoded around 0 (small positive values near 0, negative ones near $2^{32}$);
    why does the masked version carry no visible trace of it? the overhead per client grows
    linearly with $n$: what does that make the total traffic of a round, and what would you
    change to run secure aggregation over a million phones (hint: Bell et al., 2020, mask
    with a sparse graph of neighbours instead of everyone)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what does the accountant buy?

    DP-FedAvg with noise multiplier $\sigma = 1$ and $\delta = 10^{-5}$, for up to 100,000
    rounds, sampling each client with probability $q$. three ways to compute the privacy
    cost, all from your stage 5 functions:

    - **your accountant**: RDP of the subsampled Gaussian, composed over rounds, converted once;
    - **no amplification**: the same, but pretending every client is used every round ($q = 1$);
    - **(ε, δ) composition**: convert *one* round to $(\varepsilon_1, \delta/T)$ with your
      accountant, then add up $T$ copies (basic composition: $T\varepsilon_1$, total $\delta$).
    """)
    return


@app.cell
def _():
    q_choice = mo.ui.dropdown(["0.01", "0.03", "0.1"], value="0.01", label="sampling rate q")
    p2 = ui.predict("with q = 0.01, σ = 1, after 1,000 rounds: what ε does your accountant report? and the other two?")
    mo.vstack([q_choice, p2])
    return p2, q_choice


@app.cell
def _(p2, q_choice):
    ui.gate(p2)
    _q = float(q_choice.value)
    _orders = tuple(range(2, 257))
    _rdp_q = ex.rdp_sampled_gaussian(_q, 1.0, _orders)
    _rdp_1 = ex.rdp_sampled_gaussian(1.0, 1.0, _orders)
    _Ts = np.unique(np.logspace(0, 5, 31).astype(int))
    _ours = [ex.rdp_to_dp(T * _rdp_q, _orders, 1e-5)[0] for T in _Ts]
    _noamp = [ex.rdp_to_dp(T * _rdp_1, _orders, 1e-5)[0] for T in _Ts]
    _basic = [T * ex.rdp_to_dp(_rdp_q, _orders, 1e-5 / T)[0] for T in _Ts]
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_Ts, _ours, "o-", ms=3, label="your accountant (subsampled RDP)")
    _ax.plot(_Ts, _noamp, "s-", ms=3, label="no amplification (q = 1)")
    _ax.plot(_Ts, _basic, "^-", ms=3, label="(ε, δ) basic composition")
    _ax.set(xscale="log", yscale="log", xlabel="rounds T", ylabel="ε at δ = 1e-5", title=f"σ = 1, q = {_q}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _pick = [list(_Ts).index(T) for T in (1, 10, 100, 1000, 10_000, 100_000) if T in _Ts]
    _best = [ex.rdp_to_dp(T * _rdp_q, _orders, 1e-5)[1] for T in _Ts]
    _rows = "\n".join(f"| {_Ts[i]:,} | {_ours[i]:.2f} (order {_best[i]}) | {_noamp[i]:.1f} | {_basic[i]:.1f} |" for i in _pick)
    mo.vstack([_fig, mo.md("| rounds | your accountant (best order) | no amplification | (ε, δ) composition |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does sampling a client in only 1% of rounds reduce ε by so much more
    than 100×-fewer-uses would suggest at first sight (or does it)? your accountant's curve
    is nearly flat for the first ~100 rounds and only then starts to climb: look at the best
    order column and at your RDP values around orders 10–15 (print
    `ex.rdp_sampled_gaussian(0.01, 1.0, range(2, 20))`) and explain the plateau. why does
    adding up per-round ε's grow like $T$ while RDP composition grows far slower? the accountant gives
    an upper bound: what would a tighter accountant change about these curves, and what
    would it *not* change?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3 (project): the privacy–accuracy–communication triangle

    a population of 1,000 users, each holding 30 MNIST images (downsampled to 14 × 14), and
    an MLP 196 → 32 → 10 (6,634 parameters). each round samples every user with probability
    $q$, runs one local epoch of your `client_update` (batch 10, lr 0.1), and aggregates with
    your `dp_fedavg` (clip 0.5, server lr 1) for 40 rounds. for each privacy target
    $\varepsilon \in \{\infty, 1, 0.25, 0.1\}$ at $\delta = 10^{-4}$ (user-level: it protects
    everything a user holds), we find the noise multiplier $\sigma$ that your accountant says
    reaches it after 40 rounds, by bisection, and train with it. three cohort sizes,
    $q \in \{0.025, 0.1, 0.4\}$ (about 25, 100 or 400 users per round), two seeds each.
    communication is counted by `dp_fedavg` as secure-aggregation uploads.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    _per = 30
    n_users = min(1_000, len(_d["y_train"]) // _per)

    def _prep(x):
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        return F.avg_pool2d(x[:, None], 2).reshape(len(x), -1)

    _X = _prep(_d["X_train"][:n_users * _per])
    _y = torch.tensor(_d["y_train"][:n_users * _per])
    users = [(_X[p], _y[p]) for p in fl.partition_iid(n_users * _per, n_users, seed=0)]
    _nt = min(2_000, len(_d["y_test"]))
    _Xt, _yt = _prep(_d["X_test"][:_nt]), torch.tensor(_d["y_test"][:_nt])

    def test_accuracy(model):
        with torch.no_grad():
            return (model(_Xt).argmax(1) == _yt).float().mean().item()

    def sigma_for(eps, q, rounds, delta):
        """the smallest noise multiplier (to ~1%) whose epsilon after `rounds` is at most eps."""
        if not np.isfinite(eps):
            return 0.0
        lo, hi = 0.1, 1e4
        for _ in range(30):
            mid = (lo * hi) ** 0.5
            if ex.compute_epsilon(q, mid, rounds, delta) > eps:
                lo = mid
            else:
                hi = mid
        return hi

    mo.md(f"{n_users:,} users × {_per} images; {_nt:,} test images.")
    return n_users, sigma_for, test_accuracy, users


@app.cell
def _():
    p3 = ui.predict(
        "with ~25 users per round, how many points of accuracy does ε = 0.1 cost compared with no noise? with ~400 users "
        "per round? which corner of the triangle would you give up first?"
    )
    p3
    return (p3,)


@app.cell
def _(n_users, p3, sigma_for, test_accuracy, users):
    ui.gate(p3)
    _R, _delta = 40, 1e-4
    _eps_list = [np.inf, 1.0, 0.25, 0.1]
    _qs = [0.025, 0.1, 0.4]
    results = {}
    for _eps in _eps_list:
        for _q in _qs:
            _sig = sigma_for(_eps, _q, _R, _delta)
            _accs, _mb = [], None
            for _s in range(2):
                torch.manual_seed(_s)
                _m = nn.Sequential(nn.Linear(196, 32), nn.ReLU(), nn.Linear(32, 10))
                _h = ex.dp_fedavg(_m, users, fl.client_update, rounds=_R, sampling_rate=_q, clip=0.5,
                                  noise_multiplier=_sig, delta=_delta, server_lr=1.0,
                                  local_kwargs=dict(epochs=1, batch_size=10, lr=0.1), seed=_s)
                _accs.append(test_accuracy(_m))
                _mb = _h["bytes_up"][-1] / 1e6
            results[_eps, _q] = dict(sigma=_sig, accs=_accs, mb=_mb,
                                     eps=_h["epsilon"][-1] if _sig > 0 else np.inf)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _eps in _eps_list:
        _axes[0].plot([results[_eps, q]["mb"] for q in _qs], [np.mean(results[_eps, q]["accs"]) for q in _qs], "o-",
                      label="no noise (ε = ∞)" if not np.isfinite(_eps) else f"ε = {_eps:g}")
    _axes[0].set(xscale="log", xlabel="megabytes uploaded in 40 rounds (log)", ylabel="test accuracy (mean of 2 seeds)")
    _mbs = [results[_eps_list[0], q]["mb"] for q in _qs]
    _axes[0].set_xticks(_mbs, [f"{v:.0f}" for v in _mbs])
    _axes[0].minorticks_off()
    _axes[0].legend(fontsize=8)
    for _q in _qs:
        _axes[1].plot([1 / e if np.isfinite(e) else 0 for e in _eps_list],
                      [np.mean(results[e, _q]["accs"]) for e in _eps_list], "o-",
                      label=f"q = {_q} (~{round(_q * n_users)} users/round)")
    _axes[1].set(xlabel="1 / ε  (0 = no noise; right = more private)", ylabel="test accuracy")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {'∞' if not np.isfinite(e) else f'{e:g}'} | {q} | {r['sigma']:.2f} | "
        f"{np.mean(r['accs']):.3f} ({', '.join(f'{a:.3f}' for a in r['accs'])}) | {r['mb']:.1f} |"
        for (e, q), r in results.items())
    mo.vstack([_fig, mo.md("| ε | q | σ | accuracy (per seed) | MB uploaded |\n|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### experiment 3b: the same number of client updates

    in experiment 3 a bigger cohort also meant more data per round: 400 users × 40 rounds is
    16× the client updates of 25 × 40. here every configuration gets about 16,000 client
    updates (and so about the same upload volume): $q = 0.025$ for 640 rounds, $q = 0.1$ for
    160, $q = 0.4$ for 40, with no noise and at ε = 0.1 (σ recalibrated for each number of
    rounds). one seed each.
    """)
    return


@app.cell
def _():
    p3b = ui.predict(
        "at equal uploads and no noise, does cohort size still matter? and at ε = 0.1?"
    )
    p3b
    return (p3b,)


@app.cell
def _(p3b, sigma_for, test_accuracy, users):
    ui.gate(p3b)
    _rows = []
    _res = {}
    for _eps in (np.inf, 0.1):
        for _q, _R in ((0.025, 640), (0.1, 160), (0.4, 40)):
            _sig = sigma_for(_eps, _q, _R, 1e-4)
            torch.manual_seed(0)
            _m = nn.Sequential(nn.Linear(196, 32), nn.ReLU(), nn.Linear(32, 10))
            _h = ex.dp_fedavg(_m, users, fl.client_update, rounds=_R, sampling_rate=_q, clip=0.5,
                              noise_multiplier=_sig, delta=1e-4, local_kwargs=dict(epochs=1, batch_size=10, lr=0.1),
                              seed=0)
            _res[_eps, _q] = test_accuracy(_m)
            _rows.append(f"| {'∞' if not np.isfinite(_eps) else f'{_eps:g}'} | {_q} | {_R} | {_sig:.1f} | "
                         f"{_sig / _q:.0f} | {sum(_h['clients']):,} | {_res[_eps, _q]:.3f} | {_h['bytes_up'][-1] / 1e6:.0f} |")
    mo.md("| ε | q | rounds | σ | σ / q | client updates | accuracy | MB uploaded |\n|---|---|---|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain** (this is the project write-up; a paragraph each):

    1. **the mechanism.** the noise added to the *average* update has standard deviation
       $\sigma S/(qN)$. read σ off the table: how does the σ needed for a fixed ε grow with
       $q$, and what happens to $\sigma/q$? use that to explain why large cohorts make
       privacy cheap.
    2. **the triangle.** for each corner, say what you pay to improve it while holding a
       second fixed. which configuration would you deploy if uploads were the binding
       constraint, and which if privacy were? where is privacy nearly free?
    3. **the baseline.** without any noise, the cohort size still matters in experiment 3.
       use experiment 3b to separate "more data per round" from "less noise per round": at
       equal client updates, what does cohort size do without noise, and at ε = 0.1? relate
       the answer to the σ / q column.
    4. **honesty.** list three things this experiment holds fixed that a real deployment
       would tune (and that could change the conclusions), and what "user-level ε = 1 with
       1,000 users" does and does not promise a user.
    """)
    return


if __name__ == "__main__":
    app.run()
