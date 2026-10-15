# symmetry & equivariance · lab notes

numbers are for real MNIST (6,000 training digits, 2,000 test digits, standardized pixels), measured with the reference solution, seed 0, on a laptop CPU (about two minutes for the whole lab). ranges in brackets are over seeds 0, 1 and 2 (different initializations and data orders). offline, `verify.py` uses a synthetic stand-in; experiment 1 behaves identically there, and experiments 2 and 3 run but their accuracies mean nothing.

## experiment 1: is a CNN translation-invariant?

**what people usually predict:** that a CNN with a global pool is invariant to shifts (0 everywhere), or, from people who've read about aliasing, that it's "approximately" invariant with small errors that grow with the shift.

**what happens** (max output change / mean output size, 200 digits):

| stride | 0 px | 1 px | 2 px | 3 px | 4 px | 5 px | 6 px | 7 px | 8 px |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1.4 | 1.7 | 1.4 | 0 | 1.4 | 1.7 | 1.4 | 0 |

with stride 1 the network is exactly invariant (errors of $10^{-15}$, printed as 0) to every shift. with two stride-2 layers it's exactly invariant to shifts by 0, 4 and 8 pixels and badly non-invariant to everything else: a 1-pixel shift changes the output by more than its typical size. the error is periodic with period 4 and doesn't grow with the shift; 2 pixels is worse than 1. choose stride 3 in the dropdown and even multiples of 9 don't give exact invariance, because 28 isn't divisible by 3: the subsampled grid doesn't wrap around consistently.

**why:** the circular convolutions and the ReLUs commute with every shift (stage 3), and the global max pool is invariant to every shift of its input. the subsampling is the only step that isn't: a stride-$s$ layer is equivariant only to shifts by multiples of $s$ (a shift by $s$ becomes a shift by 1 on the coarse grid), so two of them give equivariance to multiples of $s^2 = 4$, and the final pool turns that into invariance to multiples of 4, nothing else. for a 1-pixel shift, each stride-2 layer samples the *other* half of the positions, so the network computes a genuinely different function of the image. the size of the error is not small because nothing forces the even and odd sub-grids to agree; there is no "approximately".

## experiment 2: does it matter once the network is trained?

**what people usually predict:** after experiment 1, most people expect the trained network to be *more* robust than the random one ("it learns that position doesn't matter"), with flip rates under 1%; and that blurring fixes the problem.

**what happens:**

| model | test accuracy | flips at 1 px | 2 px | 3 px | 4 px |
|---|---|---|---|---|---|
| stride 2 | 0.892 [0.892–0.919] | 8.0% [6.7–8.0] | 11.2% [8.0–11.2] | 9.3% [6.2–9.3] | 0.0% |
| stride 2 + blur | 0.933 [0.914–0.947] | 3.6% [3.0–3.8] | 4.1% [4.1–5.1] | 3.5% [3.0–3.5] | 0.0% |
| stride 2 + shift augmentation | 0.888 [0.887–0.911] | 8.1% [6.5–8.1] | 9.5% [7.5–9.5] | 7.9% [6.4–7.9] | 0.0% |

one test digit in twelve changes its predicted class when moved by one pixel; one in nine when moved by two. a 4-pixel shift changes nothing, exactly, for every model. blurring before subsampling more than halves the flip rate (and here also improves accuracy), but leaves 3–4%. training with random shifts of up to ±3 pixels doesn't reduce the flip rate at all.

**why:** the trained network inherits the symmetry structure of its architecture: exact invariance to the subgroup of shifts by multiples of 4, nothing for the rest. training doesn't add invariance to the missing shifts because nothing in the loss asks for it, and a digit that sits near a decision boundary for one sub-grid can sit on the other side for the neighbouring one. the flip rate tracks how many test digits are near a boundary, which is why it's roughly the error rate's size. blurring removes the high spatial frequencies that alias when you subsample, so the even and odd sub-grids see more similar signals: much less sensitive, but still not invariant, since a 3×3 blur doesn't remove those frequencies completely. augmentation teaches the network to *classify* shifted digits (that's what it's for), not to give the *same output* for a digit and its shift; the two are different requirements, and only the architecture can deliver the second exactly. a stride-1 network would have 0.0% everywhere, at several times the compute.

## experiment 3: invariance by averaging, and what it costs

**what people usually predict:** that the averaged network is about as accurate as the plain one on upright digits (it's the same network, just made symmetric), and, from some, that being rotation-invariant should help at 45° too.

**what happens:**

| model | upright | rotated 90° | rotated 45° | max output change at 90° |
|---|---|---|---|---|
| plain (stride 2) | 0.892 | 0.092 | 0.258 | 19 |
| C4-averaged | 0.672 [0.672–0.771] | 0.672 | 0.303 | $1.9 \times 10^{-6}$ |

the averaged network is exactly invariant to quarter turns (its output changes by $2 \times 10^{-6}$, float32 rounding, against 19 for the plain network), so it classifies rotated digits exactly as well as upright ones. but it pays for that on upright digits: 0.67–0.77 against 0.89–0.92 for the plain network. the confusion matrices show 9s predicted as 6 (about 25% of the 9s with seed 0; 6→9 dominates with seed 2 instead), the pair a quarter-turn-invariant model genuinely can't separate. at 45°, outside $C_4$ and only defined by interpolation, both networks collapse (0.26 and 0.30).

**why:** a $C_4$-invariant classifier must give a 6 and an upside-down 6 the same label, and an upside-down 6 *is* a 9: the symmetry is not a symmetry of the task, and no amount of training can fix that. it's not the whole cost, though: the largest single confusion with seed 0 is 5 → 3, not a rotation pair. the averaged network has to recognize every digit in four orientations with the same capacity and the same two epochs, a harder learning problem, and more pairs of digits become hard to separate once orientation is no longer a cue. 45° is not in $C_4$, so the average gives no guarantee there, and the bilinear rotation also blurs the digit. the lesson for the rest of the chapter: build in exactly the symmetries the task has. a molecule's energy really is invariant to every rotation and relabelling of its atoms, which is why topics 02–05 can build those in with no cost at all.
