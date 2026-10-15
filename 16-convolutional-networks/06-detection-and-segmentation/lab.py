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

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def detection_scene(n_objects, crowd, rng):
    """ground-truth boxes on a 256x256 image and a simulated detector's output: 3-6 jittered, scored boxes per
    object plus low-scoring false positives. crowd=True places the objects in overlapping pairs."""
    gts = []
    while len(gts) < n_objects:
        w, h = rng.uniform(24, 48, size=2)
        x, y = rng.uniform(0, 256 - w), rng.uniform(0, 256 - h)
        gts.append([x, y, x + w, y + h])
        if crowd and len(gts) < n_objects:  # a neighbour shifted by 15-30% of the box's width
            dx = rng.choice([-1, 1]) * rng.uniform(0.15, 0.3) * w
            gts.append([x + dx, y, x + dx + w, y + h])
    gts = np.array(gts)
    boxes, scores = [], []
    for g in gts:
        w, h = g[2] - g[0], g[3] - g[1]
        conf = rng.uniform(0.6, 1.0)
        for _ in range(rng.integers(3, 7)):
            boxes.append(g + rng.normal(scale=0.07, size=4) * [w, h, w, h])
            scores.append(conf * rng.uniform(0.7, 1.0))
    for _ in range(n_objects):
        x, y = rng.uniform(0, 220, size=2)
        boxes.append([x, y, x + 30, y + 30])
        scores.append(rng.uniform(0.0, 0.4))
    return torch.tensor(gts), torch.tensor(np.array(boxes)), torch.tensor(np.array(scores))


@app.function
def precision_recall(pred, gts, iou_threshold=0.5):
    """greedy matching (predictions in the given order): precision and recall at IoU >= iou_threshold."""
    if len(pred) == 0:
        return 1.0, 0.0
    iou = ex.box_iou(pred, gts)
    used, tp = set(), 0
    for i in range(len(pred)):
        j = int(torch.argmax(iou[i]))
        if iou[i, j] >= iou_threshold and j not in used:
            used.add(j)
            tp += 1
    return tp / len(pred), tp / len(gts)


@app.function
def shapes_batch(n, size, rng):
    """images (n, 1, size, size) of 1-2 discs and squares on a textured, noisy background, and their masks.
    radii are drawn log-uniformly from 1 to 5 pixels, so most objects are tiny."""
    yy, xx = np.mgrid[:size, :size]
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    masks = np.zeros_like(imgs)
    radii = []
    for i in range(n):
        texture = np.sin(xx / rng.uniform(2, 6) + rng.uniform(0, 6)) * np.cos(yy / rng.uniform(2, 6))
        img = 0.4 * texture + 0.5 * rng.normal(size=(size, size))
        m = np.zeros((size, size), dtype=bool)
        for _ in range(rng.integers(1, 3)):
            r = float(np.exp(rng.uniform(np.log(1.0), np.log(5.0))))
            cx, cy = rng.uniform(r, size - r, size=2)
            shape = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2) if rng.random() < 0.5 else (
                (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r))
            m |= shape
            radii.append(r)
        img[m] += 1.2
        imgs[i, 0], masks[i, 0] = img, m
    return torch.tensor(imgs), torch.tensor(masks), radii


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # detection & segmentation · lab

    every experiment here uses **your** `box_iou`, `nms`, `make_anchors`, `match_anchors`,
    `dice_loss` and `UNet` from `exercise.py`, on synthetic scenes generated in the
    notebook (so the lab is the same offline). for each experiment: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: choosing the NMS threshold

    a simulated detector fires 3–6 slightly jittered boxes on each of 10 objects (all
    with similar scores) plus 10 low-scoring false alarms, in 200 random scenes. boxes
    scoring below 0.45 are dropped first (the usual score threshold), then your `nms`
    cleans up the rest at IoU thresholds from 0.1 to 0.9, and we measure precision and recall
    (a detection is correct if it overlaps an unmatched object with IoU ≥ 0.5). in
    **sparse** scenes the objects are placed independently; in **crowded** scenes they come
    in pairs that overlap by 70–85% of their width (people in a crowd, cars in a car
    park).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "in crowded scenes, is there an NMS threshold that keeps recall above 0.95 AND precision above 0.9?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _thresholds = np.round(np.arange(0.1, 0.95, 0.1), 2)
    _res = {}
    for _crowd in (False, True):
        _rng = np.random.default_rng(0)
        _scenes = [detection_scene(10, _crowd, _rng) for _ in range(200)]
        _p, _r = [], []
        for _t in _thresholds:
            _pr = [precision_recall(b[s >= 0.45][ex.nms(b[s >= 0.45], s[s >= 0.45], float(_t))], g)
                   for g, b, s in _scenes]
            _p.append(np.mean([a for a, _ in _pr]))
            _r.append(np.mean([b for _, b in _pr]))
        _res["crowded" if _crowd else "sparse"] = (_p, _r)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for _ax, (_name, (_p, _r)) in zip(_axes, _res.items()):
        _ax.plot(_thresholds, _p, "o-", label="precision")
        _ax.plot(_thresholds, _r, "s-", label="recall")
        _ax.set(title=f"{_name} scenes", xlabel="NMS IoU threshold")
        _ax.legend(fontsize=8)
    _axes[0].set_ylabel("mean over 200 scenes")
    _fig.tight_layout()
    _rows = "\n".join(f"| {t:.1f} | {_res['sparse'][0][i]:.3f} | {_res['sparse'][1][i]:.3f} | "
                      f"{_res['crowded'][0][i]:.3f} | {_res['crowded'][1][i]:.3f} |" for i, t in enumerate(_thresholds))
    mo.vstack([_fig, mo.md("| threshold | sparse precision | sparse recall | crowded precision | crowded recall |\n"
                           "|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what limits recall at low thresholds, and what limits precision at high
    ones? in the crowded scenes, two neighbours' boxes have IoU between 0.54 and 0.74, and a
    duplicate detection of one object typically has IoU 0.6–0.9 with the best one: what does
    greedy NMS do with each at threshold 0.5, and at 0.8? why can no single threshold fix
    both (and what have detectors done about it: soft-NMS, learned NMS, DETR's set
    prediction)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: which objects do the anchors cover?

    a typical anchor grid: stride 16 on a $256 \times 256$ image, sizes 32, 64 and 128,
    ratios $\frac12$, 1 and 2 (9 anchors per position, 2,304 in all), from your
    `make_anchors`. 3,000 ground-truth boxes with random sizes from 4 to 200 pixels and
    random aspect ratios from $\frac13$ to 3. for each, the best IoU of any anchor
    (`box_iou`), and whether `match_anchors` finds an anchor with IoU ≥ 0.5 for it, by
    object size.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what fraction of objects around 16 pixels across get an anchor with IoU >= 0.5? and objects around 64 "
        "pixels?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _anchors = ex.make_anchors(16, 16, (32, 64, 128), (0.5, 1.0, 2.0)).double()
    _rng = np.random.default_rng(1)
    _side = np.exp(_rng.uniform(np.log(4), np.log(200), 3000))
    _ratio = np.exp(_rng.uniform(np.log(1 / 3), np.log(3), 3000))
    _w, _h = _side / np.sqrt(_ratio), _side * np.sqrt(_ratio)
    _x = _rng.uniform(0, 1, 3000) * np.maximum(256 - _w, 1)
    _y = _rng.uniform(0, 1, 3000) * np.maximum(256 - _h, 1)
    _gts = torch.tensor(np.stack([_x, _y, _x + _w, _y + _h], axis=1))
    _best = ex.box_iou(_anchors, _gts).max(dim=0).values.numpy()
    _bins = [4, 8, 16, 32, 64, 128, 200]
    _rows, _cov, _mid = [], [], []
    for _lo, _hi in zip(_bins[:-1], _bins[1:]):
        _sel = (_side >= _lo) & (_side < _hi)
        _cov.append(float(np.mean(_best[_sel] >= 0.5)))
        _mid.append(np.sqrt(_lo * _hi))
        _rows.append(f"| {_lo}–{_hi} | {_sel.sum()} | {np.median(_best[_sel]):.2f} | {_cov[-1]:.1%} |")
    _labels, _ = ex.match_anchors(_anchors, _gts[:50])
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.semilogx(_side, _best, ".", ms=2, alpha=0.4)
    _ax.axhline(0.5, color="C3", lw=1, label="IoU 0.5")
    for _s in (32, 64, 128):
        _ax.axvline(_s, color="gray", lw=0.8, ls=":")
    _ax.set(xlabel="object size (geometric mean of width and height, pixels)", ylabel="best anchor IoU")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| object size | objects | median best IoU | covered at IoU ≥ 0.5 |\n|---|---|---|---|\n"
                           + "\n".join(_rows)
                           + f"\n\nfor a scene with the first 50 of these objects, `match_anchors` labels "
                             f"{int((_labels == 1).sum())} anchors positive, {int((_labels == -1).sum())} ignored and "
                             f"{int((_labels == 0).sum())} negative (background).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the IoU of a small box with even the best-placed 32-pixel
    anchor stay low (compute it for a $16 \times 16$ box centred in a $32 \times 32$
    anchor)? what does `match_anchors` do for such an object, and what does that mean for
    how many positive training examples small objects provide? why does the positive /
    negative count make detection training a class-imbalance problem, and how do
    multi-scale feature maps (FPN) and focal loss address the two issues?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: segmenting small shapes

    your `UNet` (base 8 channels, depth 3) learns to segment one or two discs or squares
    (radius 1 to 5 pixels, most of them tiny) on textured, noisy $32 \times 32$ images. three runs
    with identical networks and data order, differing only in the loss: binary
    cross-entropy, your `dice_loss`, and their sum (300 Adam steps, batch 16, learning
    rate $3 \cdot 10^{-3}$). we report the test IoU (threshold 0.5) over 200 fresh images,
    and the fraction of small objects (radius below 3) that are found at all (at least
    half their pixels predicted).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "objects cover about 3% of the pixels. which loss gives the best IoU, and which finds the most small objects?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _train = [shapes_batch(16, 32, _rng)[:2] for _ in range(300)]
    _Xte, _Mte, _ = shapes_batch(200, 32, np.random.default_rng(3))
    _rng_small = np.random.default_rng(4)
    _small = []  # single-object test images with a small object, to measure detection rate
    while len(_small) < 150:
        _x, _m, _r = shapes_batch(1, 32, _rng_small)
        if len(_r) == 1 and _r[0] < 3:
            _small.append((_x, _m))
    _Xs, _Ms = torch.cat([a for a, _ in _small]), torch.cat([b for _, b in _small])
    _losses = {"BCE": lambda o, t: F.binary_cross_entropy_with_logits(o, t),
               "dice": lambda o, t: ex.dice_loss(o, t),
               "BCE + dice": lambda o, t: F.binary_cross_entropy_with_logits(o, t) + ex.dice_loss(o, t)}
    _rows, _preds = [], {}
    for _name, _loss_fn in _losses.items():
        torch.manual_seed(0)
        _m = ex.UNet(1, 1, base_channels=8, depth=3)
        _opt = torch.optim.Adam(_m.parameters(), lr=3e-3)
        _t0 = time.perf_counter()
        _m.train()
        for _xb, _mb in _train:
            _opt.zero_grad()
            _loss_fn(_m(_xb), _mb).backward()
            _opt.step()
        _m.eval()
        with torch.no_grad():
            _p = (_m(_Xte) > 0).float()
            _ps = (_m(_Xs) > 0).float()
        _iou = float(((_p * _Mte).sum() / ((_p + _Mte) > 0).float().sum()))
        _found = float(((_ps * _Ms).sum(dim=(1, 2, 3)) >= 0.5 * _Ms.sum(dim=(1, 2, 3))).float().mean())
        _fg = float(_p.mean())
        _preds[_name] = _p
        _rows.append(f"| {_name} | **{_iou:.3f}** | {_found:.1%} | {_fg:.3f} | {time.perf_counter() - _t0:.0f} s |")
    _fig, _axes = plt.subplots(3, 5, figsize=(9, 5.6))
    for _j in range(3):
        _axes[_j, 0].imshow(_Xte[_j, 0], cmap="gray")
        _axes[_j, 1].imshow(_Mte[_j, 0], cmap="gray")
        for _k, _name in enumerate(_losses):
            _axes[_j, 2 + _k].imshow(_preds[_name][_j, 0], cmap="gray")
    for _k, _title in enumerate(["image", "true mask", *_losses]):
        _axes[0, _k].set_title(_title, fontsize=9)
    for _a in _axes.ravel():
        _a.set(xticks=[], yticks=[])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"true foreground fraction: {float(_Mte.mean()):.3f}\n\n"
                           "| loss | test IoU | small objects found | predicted foreground fraction | time |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with 97% background pixels, what does a network that predicts
    "background everywhere" score under BCE, and under dice? how does each loss weigh one
    missed pixel of a tiny object against one wrong pixel of the background? look at the
    predicted foreground fractions: which loss is biased towards predicting too little,
    and which too much? why is the sum a common default?
    """)
    return


if __name__ == "__main__":
    app.run()
