# k-nearest neighbors · lab notes

the numbers below are from the reference solution on the real MNIST data in `~/.cache/mlfs`, with the lab's fixed seeds (1,000 points per dataset in experiment 1). offline (`MLFS_OFFLINE=1`) the digits are a synthetic stand-in and experiments 1 and 3 give different, meaningless numbers for the "real data" parts.

## experiment 1: do distances concentrate?

**what people usually predict:** that the contrast shrinks "a bit" with dimension, and that 784-pixel images behave like 784-dimensional points, so $k$-NN on images should be hopeless.

**what happens:** for uniform data the median relative contrast falls from **93.7** at $d = 2$ to **0.439** at $d = 100$ and **0.133** at $d = 784$: there, the farthest of 1,000 random points is only 13% farther away than the nearest. above $d \approx 20$ the curve runs parallel to $1/\sqrt d$. real MNIST digits, with the same 784 coordinates, have a median contrast of **1.458**, eleven times larger: a digit's nearest neighbour is less than half as far away as its farthest.

**why:** a squared distance between uniform points is a sum of $d$ independent terms, so its mean grows like $d$ and its spread like $\sqrt d$ (on paper 2); the contrast measures spread over size. at low $d$ the contrast is huge because the *nearest* distance goes to zero with 1,000 points in a square. digits are nothing like uniform noise: most of the 784 pixels are always black, neighbouring pixels are strongly correlated, and the images vary along a few dozen directions (stroke thickness, slant, shape of the loop). their *intrinsic* dimension is low, so near and far still mean something, which is why $k$-NN works on MNIST at all (experiment 3). the curse is about intrinsic dimension, not the number of columns.

## experiment 2: how much does your KD-tree skip?

**what people usually predict:** that a tree is always faster than brute force, or that it degrades slowly and only in hundreds of dimensions.

**what happens:** the fraction of distances computed is **0.5%** at $d = 1$, **1.1%** at $d = 2$, **6.1%** at $d = 4$, **22%** at $d = 6$, **56%** at $d = 8$, **85%** at $d = 10$ and **100%** from $d = 16$ on. the tree is exact throughout; it just stops skipping anything.

**why:** with 4,000 points and leaves of 16, the tree is only 8 levels deep, so each coordinate is split at most once or twice along any path: in 10-D some coordinates are never split at all. and the pruning test compares the gap in *one* coordinate, $|q_{\text{dim}} - s|$, with the current $k$-th best distance $r$, which sums over all $d$ coordinates and (by experiment 1) is barely smaller than the distance to everything else. in high dimension the gap almost never exceeds $r$, so the far child is almost never skipped. the rule of thumb $n \gg 2^d$ predicts the collapse around $d \approx \log_2(4000/16) \approx 8$, where the dashed line is. in practice a tree in 20-D is *slower* than brute force, because it does all the same distance computations plus the recursion.

## experiment 3: k-NN on handwritten digits

**what people usually predict:** somewhere between 50% and 80% for a model with no training on raw pixels, and that more neighbours means a better, smoother vote.

**what happens:** 1-NN reaches **93.1%** test accuracy (standard error 0.8 points) with 5,000 stored training digits, against a majority-class baseline of **10.4%**. uniform voting peaks at **93.5%** at $k = 5$; distance weighting peaks at **93.9%** at $k = 3$. then accuracy falls: at $k = 50$ it's **88.9%** (uniform) and **89.3%** (distance). distance weighting is a little better at every $k > 1$.

**why:** digits of the same class form tight, curved clusters in pixel space (experiment 1), so the closest training image is usually the same digit written slightly differently. the errors are the genuinely ambiguous ones (4 vs 9, 3 vs 5 vs 8). as $k$ grows, the neighbourhood reaches past the thin cluster into neighbouring classes, which are close in pixel distance: the bias of a wide neighbourhood beats the variance reduction. distance weighting lets the closest few images dominate, which softens that bias without giving up the extra votes for the ambiguous cases, but $1/d$ decays slowly, so the far neighbours still count. with all 60,000 training digits every cluster is denser and the neighbourhoods shrink: 1-NN on raw pixels then scores **96.8%** on the same 1,000 test digits (measured with a plain NumPy 1-NN, not in the lab); the price is 60,000 distance computations of 784 floats per query, which is why the lab uses 5,000.
