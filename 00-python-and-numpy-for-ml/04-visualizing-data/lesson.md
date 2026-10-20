# visualizing data

## why this matters

in 1973 the statistician Francis Anscombe published four small datasets. each has 11 points, and all four have the same mean of $x$, the same mean of $y$, the same variances, the same correlation (0.82) and the same best-fit line, to two or three decimal places. plot them, and they couldn't be more different: one is a noisy line, one is a perfect curve, one is a perfect line with a single outlier, and one is a vertical stack of points with one far-away point that creates the whole "correlation" by itself.

that's the case for this topic in one example: **numbers summarize, pictures reveal.** in ML you'll look at pictures constantly. training curves tell you whether a model is learning, overfitting or diverging, long before the final number does. a grid of images tells you whether your data loader is feeding the network what you think, or upside-down, mislabeled or all black. a decision boundary tells you what a classifier actually learned, which is often not what you hoped. the most expensive bugs in ML are the ones where every number looks fine and nobody looked at a picture.

but pictures can lie too: the wrong bin width hides structure in a histogram, a linear axis hides the improvement in a loss curve, a colormap exaggerates a small difference. you'll learn how matplotlib is organized and build the three plotting helpers the course uses from here on: `plot_curves`, `image_grid` and `plot_decision_boundary`, all part of `mlfs.utils`.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. a loss falls from 1.0 to 0.01 over 1,000 steps, then to 0.001 over the next 1,000. on an ordinary (linear) y axis, what does the second half of the curve look like?
2. you draw a histogram of 100 numbers with 50 bins, then with 5 bins. which picture is more trustworthy? is either?
3. a grayscale image is stored as a 2-D array of shape `(28, 28)`. a color one as `(32, 32, 3)`. PyTorch stores a batch of color images as `(N, 3, 32, 32)`. what does `plt.imshow` do if you give it one image in PyTorch's layout, `(3, 32, 32)`?
4. to color the plane by a classifier's prediction at 200 × 200 grid points, how many predictions do you need? if you call the classifier once per point and each call takes 50 µs, how long does that take?
5. in `fig, ax = plt.subplots()`, what's the difference between `fig` and `ax`?

## prerequisites check

- [ ] you can build and reshape NumPy arrays, and know what `(N, H, W)` means as a shape (→ ch 00 · topic 02)
- [ ] you can use broadcasting and fancy indexing, and `np.searchsorted` won't scare you once you've read its docstring (→ ch 00 · topic 02)
- [ ] you know what a mean and a standard deviation are (high school; chapter 04 does them properly)
- [ ] you can read a scatter plot and a line plot

## core ideas

### idea 1: the anatomy of a figure

matplotlib has two ways to use it, and mixing them up is the root of most confusion. the quick way is the **pyplot state machine**: `plt.plot(x, y)`, `plt.xlabel("step")`, `plt.show()`. each call acts on "the current plot", whatever that happens to be. it's fine for a one-off plot at the prompt, and a mess as soon as you have two plots, or a function that draws into a plot someone else made.

the way this course uses is **object-oriented**. a plot is a tree of Python objects, and you call methods on the one you mean:

- the **`Figure`** is the whole canvas: the window or image file. it holds one or more Axes.
- an **`Axes`** (plural and singular are both "axes") is one plotting area with its own coordinate system: the box with the x axis, the y axis, the data, a title and a legend. "plot", in everyday speech, usually means an Axes.
- **artists** are everything drawn on an Axes: `Line2D` objects for lines, `Text` objects for labels, a `PathCollection` for scatter points, an `AxesImage` for images, a `Legend`.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.5))     # one Figure containing one Axes
ax.plot(steps, losses, label="train")       # adds a Line2D to ax
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_yscale("log")
ax.legend()                                  # builds a Legend from the labeled artists
```

that answers warm-up 5: `fig` is the canvas, `ax` is the plotting area you draw into. `plt.subplots(2, 3)` gives one figure and a $2 \times 3$ array of Axes, for side-by-side comparisons.

every helper you write in this topic follows one convention: **take an optional `ax` argument, draw into it, and return it.** if the caller passes `ax`, the plot goes exactly where they want it (say, the third panel of a grid); if not, the function makes its own figure. this makes helpers composable. you can build a figure of four panels from four calls, and keep adding to any panel afterwards. it's also what makes the tests possible: they hand your function an Axes and then inspect what you drew on it with methods like `ax.get_lines()`, `ax.get_xlabel()` and `ax.get_yscale()`.

**log scales.** training losses typically fall fast at first and then slowly for a long time. on a linear y axis everything after the first few percent of training looks flat: a drop from 0.01 to 0.001 is a tenfold improvement, but it's 1% of the axis height when the curve starts at 1.0 (warm-up 1). on a **logarithmic** y axis, equal *ratios* get equal distances: 1 → 0.1 → 0.01 → 0.001 are evenly spaced, so the slow phase is as visible as the fast one. if a quantity shrinks by a constant factor per step ($L_t = L_0 r^t$), it's a straight line on a log y axis, since $\log L_t = \log L_0 + t\log r$. **plot losses on a log scale** unless you have a reason not to.

> **▶ build: stage 1.** implement `plot_curves(curves, *, ax=None, xlabel="step", ylabel="value", logy=False, title=None)` (open the notebook with `marimo edit exercise.py`). each entry of `curves` is a label and either a sequence of y values or an `(x, y)` pair.

### idea 2: histograms, and the bin-width dilemma

a **histogram** answers "how are these numbers distributed?": split the range into **bins** and count how many values land in each. to build one:

1. choose the range: from the smallest value to the largest.
2. choose the number of bins $k$, and split the range into $k$ equal-width intervals. their boundaries are the $k + 1$ **edges** $e_0 < e_1 < \dots < e_k$, `np.linspace(lo, hi, k + 1)`.
3. count. bin $i$ holds the values $v$ with $e_i \le v < e_{i+1}$: closed on the left, open on the right, so a value sitting exactly on an edge goes to the bin on its right. the **last** bin is closed on both ends, $e_{k-1} \le v \le e_k$, so the largest value is counted. (NumPy's convention.)

vectorized, step 3 is a search problem. `np.searchsorted(edges, x, side="right")` finds, for every value at once, how many edges are $\le$ it; subtract 1 and you have the bin index, with only the values equal to the last edge needing a nudge back into the last bin. then `np.bincount(indices, minlength=k)` counts how many times each index appears. two calls, no loops, a million values in milliseconds.

one degenerate case: if every value is the same, the range has zero width. NumPy widens it to $[v - 0.5, v + 0.5]$, and so will you.

**the bin-width dilemma** (warm-up 2). with too few bins, you average away real structure: two separate peaks merge into one lump. with too many, each bin holds only a few values, and the counts are dominated by randomness: a bin expected to hold 4 values will often hold 1 or 7, so the histogram looks jagged even when the true distribution is smooth. roughly, a bin expecting $c$ values has random fluctuations of about $\sqrt{c}$, a relative noise of $1/\sqrt{c}$. so with 100 numbers in 50 bins ($c \approx 2$), the noise is about 70% of the signal, and the picture is mostly noise. with 5 bins ($c \approx 20$, about 22% noise), the counts are reliable, but each bin is so wide that you can only see the coarsest shape. neither is "the" right picture. rules of thumb exist (Sturges' $k \approx 1 + \log_2 n$; Freedman–Diaconis' bin width $2\,\text{IQR}/n^{1/3}$), but the honest practice is to **try several bin counts**, and to be suspicious of any feature that appears at only one of them. the lab shows a dataset whose story changes completely with the bin count.

once you have counts and edges, `ax.stairs(counts, edges, fill=True)` draws the histogram. (`ax.hist(x, bins)` does the counting and the drawing in one call, and from now on you've earned it.)

> **▶ build: stage 2.** implement `histogram(x, bins=10)`, returning `(counts, edges)` that match `np.histogram(x, bins)` exactly.

> **🔮 predict:** you draw 1,000 samples from a smooth bell-shaped distribution and plot a histogram with 200 bins. how many "peaks" will you see? (try it in a scratch cell with your `histogram` and `ax.stairs`.)

### idea 3: images are arrays

to a computer, a grayscale image is just a 2-D array of brightness values, shape `(H, W)`: row 0 is the top of the picture, column 0 the left edge. a color image adds a third axis for the red, green and blue **channels**, shape `(H, W, 3)`. pixel values come in two common conventions: integers from 0 to 255 (dtype `uint8`, one byte per channel, how images are stored on disk) or floats from 0 to 1 (convenient for arithmetic). mixing them up gives either an all-white image (treating 0–255 as 0–1) or an almost-black one.

`ax.imshow(img)` draws an array as an image. for 2-D arrays, it maps values to colors through a **colormap** (`cmap="gray"` for photographs and digits; `"viridis"` for heat maps), stretching the smallest value to the bottom of the colormap and the largest to the top unless you fix the range with `vmin`, `vmax`. for `(H, W, 3)` arrays it uses the values directly as colors.

**channel order.** NumPy, matplotlib and most image files put the channels last, `(H, W, C)` ("channels-last", and a batch is `NHWC`). PyTorch puts them first, `(C, H, W)` (a batch is `NCHW`), because that layout is faster for convolutions. handing `imshow` a `(3, 32, 32)` array makes it think you have an image 3 pixels tall and 32 wide, with 32 channels, and it raises an error (warm-up 3). `img.transpose(1, 2, 0)` moves the channel axis to the end; for a batch, `.transpose(0, 2, 3, 1)`. the course's convention (see `CONTRACTS.md`) is that images inside models are NCHW, so you'll do this conversion every time you look at one.

**looking at many images at once.** in ML you rarely want one image; you want 64 of them, to see what a batch looks like. one option is a grid of subplots, `plt.subplots(8, 8)`, with 64 Axes, which is slow and cluttered. the better option is to **tile** the images into one big array and show it with a single `imshow`. for $N$ images of size $H \times W$, `nrow` per row, with `padding` pixels of border around and between tiles:

$$n_{\text{cols}} = \min(\texttt{nrow}, N), \qquad n_{\text{rows}} = \lceil N / n_{\text{cols}} \rceil,$$

$$\text{grid shape} = \big(n_{\text{rows}}H + (n_{\text{rows}} + 1)\,p,\;\; n_{\text{cols}}W + (n_{\text{cols}} + 1)\,p\big),$$

and image $k$ goes into tile $(r, c) = (\lfloor k / n_{\text{cols}} \rfloor,\ k \bmod n_{\text{cols}})$, which is Python's `divmod(k, ncols)`, with its top-left corner at $(p + r(H + p),\ p + c(W + p))$. filling the canvas with the pad value first and copying each image into its slot is a loop over images (not pixels), which is fine: each iteration copies a whole block. this is exactly what `torchvision.utils.make_grid` does.

> **▶ build: stage 3.** implement `make_grid(images, *, nrow=8, padding=1, pad_value=None)`, which handles `(N, H, W)`, `NHWC` and `NCHW` input, and `image_grid(images, *, nrow=8, ax=None, cmap="gray", titles=None)`, which shows the grid.

### idea 4: scatter plots and decision boundaries

a **scatter plot** draws one marker per data point. with two features per point, `ax.scatter(X[:, 0], X[:, 1], c=labels)` shows a classification dataset at a glance: `c` assigns each point a number, and the colormap turns numbers into colors. labels can be strings ("cat", "dog"), which `c` can't take, so map them to integers first: `classes, idx = np.unique(y, return_inverse=True)` returns the sorted distinct labels and, for every point, the index of its label in that list.

a classifier is a **function on the plane**: it assigns a class to every point, not just the training points. its **decision boundary** is where that answer changes. to see it, evaluate the classifier on a fine grid of points covering the data and color each grid point by its prediction. NumPy builds the grid with `np.meshgrid`:

```python
xs = np.linspace(x_min, x_max, 200)          # 200 x positions
ys = np.linspace(y_min, y_max, 200)          # 200 y positions
xx, yy = np.meshgrid(xs, ys)                 # both (200, 200): xx[i, j] = xs[j], yy[i, j] = ys[i]
points = np.column_stack([xx.ravel(), yy.ravel()])   # (40000, 2): every grid point as a row
```

`xx` varies along each row and `yy` down each column, so `(xx[i, j], yy[i, j])` is the grid point in row $i$, column $j$. stacking the flattened arrays gives a $(40{,}000, 2)$ array of points, the same layout as a dataset `X`.

that's warm-up 4: a $200 \times 200$ grid needs 40,000 predictions. at 50 µs per call, calling the classifier once per point takes 2 seconds. calling it **once on all 40,000 points**, as a batch, takes a few milliseconds if the classifier is vectorized (and every one you'll build is). so `plot_decision_boundary` calls `predict` exactly once, reshapes the 40,000 answers back to `(200, 200)`, and draws them with `ax.contourf(xx, yy, Z, ...)`, which fills the regions between contour levels with color. with class indices $0, \dots, k-1$, levels at $-0.5, 0.5, \dots, k - 0.5$ give each class its own band, and coloring the scatter with the same colormap makes each point's color match its region. a point whose color differs from the region it sits in is a training error, visible at a glance.

a margin of `padding` around the data matters: the interesting behaviour of a classifier is often just outside the data (does the boundary curve back? does a region extend forever?).

some classifiers return **scores** (a probability, or a signed distance from a boundary) instead of labels. if the predictions aren't all labels from `y`, shade them continuously instead, and the boundary appears as the color transition.

> **▶ build: stage 4.** implement `make_meshgrid(X, *, resolution=200, padding=0.5)` and `plot_decision_boundary(predict, X, y, *, ax=None, resolution=200, padding=0.5)`.

> **🔮 predict:** a 1-nearest-neighbour classifier predicts the label of the closest training point. it gets 100% accuracy on its own training data. what do you expect its decision boundary to look like on two overlapping, noisy classes? (the lab draws it.)

## worked example

**a histogram by hand.** take $x = (0, 1, 1, 2, 2, 2, 3)$ with 3 bins. the range is $[0, 3]$, so the edges are `np.linspace(0, 3, 4)` $= (0, 1, 2, 3)$. now place each value:

| value | `searchsorted(edges, v, side="right")` | minus 1 | bin |
|---|---|---|---|
| 0 | 1 (one edge, 0, is $\le 0$) | 0 | $[0, 1)$ |
| 1, 1 | 2 | 1 | $[1, 2)$ |
| 2, 2, 2 | 3 | 2 | $[2, 3]$ |
| 3 | 4 | 3 → clamped to 2 | $[2, 3]$ (the last bin is closed) |

so the counts are $(1, 2, 4)$. notice the two edge cases: the 1s sit exactly on an edge and go to the bin on their right, and the 3 sits on the last edge and stays in the last bin. `np.histogram([0, 1, 1, 2, 2, 2, 3], 3)` agrees.

**a grid by hand.** five images of $2 \times 3$ pixels, `nrow=3`, `padding=1`. then $n_{\text{cols}} = \min(3, 5) = 3$ and $n_{\text{rows}} = \lceil 5/3 \rceil = 2$, so the grid is $(2\cdot 2 + 3\cdot 1,\ 3 \cdot 3 + 4 \cdot 1) = (7, 13)$. image 0 goes to tile $(0, 0)$ with its corner at $(1, 1)$, so it occupies rows 1–2 and columns 1–3. image 1 is in tile $(0, 1)$, corner $(1, 1 + 1\cdot(3 + 1)) = (1, 5)$: columns 5–7. image 3 is `divmod(3, 3)` $= (1, 0)$, tile row 1: corner $(1 + 1\cdot(2 + 1), 1) = (4, 1)$. the sixth tile, $(1, 2)$, stays empty and keeps the pad value.

the tests check both: your `histogram` must return `[1, 2, 4]` with edges `[0, 1, 2, 3]`, and your `make_grid` must produce a $(7, 13)$ grid with the images exactly where they're placed above.

## on paper

1. **straight lines on log axes.** show that $L_t = L_0 r^t$ (with $0 < r < 1$) is a straight line on a plot with a log y axis and a linear x axis, and find its slope (in units of $\log_{10}$ per step). what kind of curve is a straight line on a log–log plot (both axes logarithmic)?

<details><summary>solution</summary>

$\log_{10}L_t = \log_{10}L_0 + t\log_{10}r$: linear in $t$, with slope $\log_{10}r < 0$. for example $r = 0.99$ gives slope $-0.0044$, one decade every ~230 steps. on log–log axes, $\log L = \log a - b\log t$ is a straight line, which is a **power law** $L = a\,t^{-b}$. many losses decay like power laws late in training, which is why "scaling law" plots (chapter 19) are drawn on log–log axes.

</details>

2. **histogram noise.** you have $n$ values and $k$ bins of roughly equal expected counts. if the count in a bin fluctuates by about $\sqrt{c}$ around its expected value $c$, what relative noise do you get with $n = 10{,}000$ and $k = 100$? how many bins can you afford if you want at most 10% noise?

<details><summary>solution</summary>

$c = n/k = 100$, so the relative noise is $\sqrt{100}/100 = 10\%$. at most 10% means $c \ge 100$, so $k \le n/100 = 100$ bins. the relative noise is $\sqrt{k/n}$: four times the data buys twice as many bins at the same noise level.

</details>

3. **grid sizes.** 64 MNIST digits ($28 \times 28$) are tiled with `nrow=8` and `padding=2`. what's the shape of the grid? what fraction of its pixels is padding?

<details><summary>solution</summary>

$8 \times 8$ tiles: $8 \cdot 28 + 9 \cdot 2 = 242$ in each direction, so $(242, 242)$. the images cover $64 \cdot 784 = 50{,}176$ of $58{,}564$ pixels, so about 14% is padding.

</details>

## experiment

`lab.py` takes the title of this topic's lab literally: **one dataset, four plots, four different stories.**

1. a histogram of one sample of measurements, with the number of bins on a slider.
2. two models' training curves on a linear and on a log y axis: which model is better?
3. four datasets with identical summary statistics, plotted.
4. the decision boundary of a classifier that is 100% accurate on its training data.

record your predictions first.

## check your understanding

1. a colleague shows a training curve that looks flat after step 2,000 and concludes training has converged. what do you ask for?

<details><summary>answer</summary>

the same curve on a log y axis. a flat-looking tail on a linear axis can hide tenfold improvements. (and the validation curve: a falling training loss with a rising validation loss means overfitting, not convergence.)

</details>

2. why must the last bin of a histogram be closed on the right?

<details><summary>answer</summary>

the largest value sits exactly on the last edge, $e_k = \max x$. with every bin half-open, it would fall outside all of them, and the counts wouldn't add up to $n$.

</details>

3. you show a colour image with plain `ax.imshow` (not your `image_grid`) and see a solid white square. name two likely causes, and say why `image_grid` doesn't have this problem.

<details><summary>answer</summary>

pixel values in 0–255 stored as *floats*: matplotlib reads float RGB as 0–1, so everything above 1 clips to white. or the image really is blank or saturated, from a bug in the data loader, which is exactly why you look. `image_grid` rescales float images before drawing them (min–max for colour, imshow's autoscaling for grayscale), so 0–255 floats come out fine there. the flip side is that a perfectly constant grayscale image has no range to rescale, so it draws as one flat colour, not as its true brightness.

</details>

4. (the predict callout in idea 2.) how many peaks does a 200-bin histogram of 1,000 smooth samples show?

<details><summary>answer</summary>

dozens. each bin expects about 5 values, so the counts fluctuate by about $\sqrt 5 \approx 2.2$, nearly half their size, and the histogram is a jagged comb of spurious peaks. none of them are real. this is how people "discover" structure that's only noise.

</details>

5. (the predict callout in idea 4.) what does the boundary of a 1-nearest-neighbour classifier look like on noisy, overlapping classes?

<details><summary>answer</summary>

fragmented: every training point carves out its own little island of its class, including the noisy points deep inside the other class's territory. 100% training accuracy, a boundary nobody would believe generalizes. chapter 09 shows how averaging over more neighbours smooths it.

</details>

## common bugs

- **pyplot state instead of the given `ax`.** symptom: your function draws on the wrong subplot, or into a figure the caller never sees. use `ax.plot`, `ax.set_xlabel` and friends, never `plt.plot` inside a helper.
- **forgetting the legend, or the labels.** symptom: a plot with three unlabeled lines that no one, including you next week, can read. every line gets a `label=`, every axis a name.
- **off-by-one at bin edges.** symptom: counts that add up to $n - 1$ (the maximum fell out), or values on an edge landing in the left bin. right-open bins, closed last bin.
- **NCHW images in `imshow`.** symptom: `TypeError: Invalid shape (3, 32, 32) for image data`. transpose the channels to the end.
- **calling `predict` per grid point.** symptom: a decision-boundary plot that takes seconds or minutes. stack the grid into one `(m, 2)` array and make one call.
- **`meshgrid` axes swapped.** symptom: a decision boundary that's mirrored along the diagonal relative to the points. `np.meshgrid(xs, ys)` gives arrays whose rows follow `ys`; keep `xx` paired with `X[:, 0]`.

## going deeper

- Anscombe (1973), "Graphs in statistical analysis", *The American Statistician* 27(1): 17–21. the four datasets, and the original argument for plotting before summarizing. [doi:10.1080/00031305.1973.10478966](https://doi.org/10.1080/00031305.1973.10478966)
- Matejka & Fitzmaurice (2017), "Same stats, different graphs: generating datasets with varied appearance and identical statistics through simulated annealing", *CHI 2017*. Anscombe's idea taken to extremes, including the "datasaurus". [doi:10.1145/3025453.3025912](https://doi.org/10.1145/3025453.3025912)
- Hunter (2007), "Matplotlib: a 2D graphics environment", *Computing in Science & Engineering* 9(3): 90–95, and the official [matplotlib quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html), which explains the Figure/Axes/artist tree and the two interfaces. [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
- Freedman & Diaconis (1981), "On the histogram as a density estimator: $L_2$ theory", *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete* 57: 453–476. where the bin-width rule comes from. [doi:10.1007/BF01025868](https://doi.org/10.1007/BF01025868)
- Wilke (2019), *Fundamentals of Data Visualization*, O'Reilly: a practical guide to choosing plots, axes and colors. [free online](https://clauswilke.com/dataviz/)
- Crameri, Shephard & Heron (2020), "The misuse of colour in science communication", *Nature Communications* 11: 5444. why rainbow colormaps distort data, and what to use instead. [doi:10.1038/s41467-020-19160-7](https://doi.org/10.1038/s41467-020-19160-7)

## connections

- ← builds on: arrays, shapes, broadcasting and fancy indexing (topic 02); the cost of per-element Python calls (topic 01).
- → used in: `plot_curves` for every loss and learning curve from chapter 06 on; `image_grid` for datasets and samples (ch 07, ch 16, ch 21–22); `plot_decision_boundary` for linear models, kernels and trees (ch 08–10); histograms as density estimators, and kernel density estimation as their smooth successor (ch 09); assertions that check what a plot shows (topic 05).
