# linear state-space models (S4) · lab notes

numbers are from the reference solution on a laptop CPU (2 torch threads). the whole lab took 100–110 s on a lightly loaded machine and up to about 220 s under heavy load, nearly all of it in experiment 3; training times in the tables are machine-dependent, accuracies are exactly reproducible. experiments 1 and 2 are deterministic apart from timings; experiment 3 uses seeds 0, 1, 2 for initialization and training batches and a fixed test set (seed 99). the data are synthetic throughout: a hand-made signal in experiment 1, random inputs in experiment 2, and a synthetic token task in experiment 3.

## experiment 1: a fixed-size memory of the whole past

**what people usually predict:** the recent past is remembered well and the beginning is blurred or forgotten (as with an exponentially decaying memory), and the error falls steadily with $N$.

**what happens:**

| N | RMS error, all | first 100 steps | last 100 steps |
|---|---|---|---|
| 4 | 0.310 | 0.113 | 0.208 |
| 8 | 0.255 | 0.071 | 0.370 |
| 16 | 0.030 | 0.013 | 0.008 |
| 32 | 0.022 | 0.024 | 0.002 |
| 64 | 0.018 | 0.030 | 0.001 |

with 16 numbers, the 1,000-sample history is reconstructed with RMS error 0.03, and the first 100 steps are as good as the last (0.013 vs 0.008). at $N \le 8$ the reconstruction is poor everywhere, worst near the jump at step 500 and at the end. at large $N$ the end keeps improving but the beginning gets slightly worse (0.013 → 0.030).

**why:** LegS projects the history onto Legendre polynomials under the *uniform* measure on $[0, t]$: every part of the past has equal weight in the approximation, so nothing is preferentially forgotten. the price is resolution: $N$ polynomials can't represent detail finer than about $t/N$, which is why the jump at step 500 is smoothed (and causes Gibbs-style ringing at small $N$) whatever $N$ is. the beginning degrades slightly at large $N$ because of the discretization: the first steps use the step $1/k$ with $k$ small, where the bilinear rule is least accurate, and higher-order coefficients amplify that early error. S4 drops the $1/t$ scaling, trading this "remember everything equally" memory for a time-invariant one with a learned time scale.

## experiment 2: what the kernel remembers, and what the two views cost

**what people usually predict:** a smooth exponential decay, faster for larger $\Delta$; the convolution 100× or more faster than the recurrence at long lengths.

**what happens:**

| dt | lag of the first revival | its height (lag 0 = 1) | exp(−dt·lag/2) | median height between lag 0 and the revival |
|---|---|---|---|---|
| 0.001 | 1999 | 0.368 | 0.368 | 0.0048 |
| 0.01 | 199 | 0.370 | 0.370 | 0.0058 |
| 0.1 | 19 | 0.370 | 0.387 | 0.0536 |

the kernels don't decay smoothly: each falls by about two orders of magnitude soon after lag 0, stays low (the median height between lag 0 and the revival is under 1% for the two smaller steps), and then **comes back** at lag $\approx 2/\Delta$ with height $e^{-1}$, again at $4/\Delta$ with $e^{-2}$, and so on. the convolution is only about 4–7× faster than the step-by-step recurrence (earlier runs at $L$ = 256, 1024, 4096 gave 4×, 5×, 7× and 5×, 4×, 4×; the current lab, at $L$ = 256, 1024, 2048 with best-of-2–3 timings, gave 5×, 5×, 5×). timings are machine- and load-dependent; the ratio is the robust part.

**why:** S4D-Lin's frequencies are $\pi n$, so after ZOH mode $n$ rotates by $\pi n\Delta$ per step. at lag $l = 2/\Delta$ every mode has turned by a multiple of $2\pi$ and they all add in phase again, as at lag 0; in between they interfere and mostly cancel. the common decay $e^{-\Delta l/2}$ (every mode has real part $-1/2$) evaluates to $e^{-1}$ at the revival, whatever $\Delta$ is. so S4D-Lin is a damped Fourier basis on a window of $2/\Delta$ lags: by choosing $C$, a channel can shape its kernel freely within that window, and $N$ sets the resolution of the shapes (how many Fourier components), while $\Delta$ sets the window. that's why the layer spreads $\Delta$ over two decades: channels with windows from 20 to 2,000 steps. (with $C = 1$, as here, you see the raw basis; a trained $C$ would suppress the revivals if they hurt.)

the convolution wins by less than its $O(L\log L)$ vs $O(L)$-sequential story suggests because on a CPU the recurrence's per-step work ($H \times N$ complex multiply-adds, batched) is cheap, so its cost is mostly per-step overhead, while the convolutional pass still has to build the kernel, itself $O(HNL)$ work. (we didn't profile which part dominates.) on a GPU, where the parallel computation is nearly free and the sequential loop is not, the gap is orders of magnitude, which is why S4 trains in convolutional mode. you'd still use the recurrent mode for generation, where tokens arrive one at a time and each step costs $O(HN)$ with constant memory.

## experiment 3: long-range memory against an LSTM

**what people usually predict:** both models solve short sequences; the LSTM degrades gradually, the SSM a little more slowly.

**what happens:** (300 Adam steps, batch 32, lr 3e-3; test accuracy on 512 fresh sequences; chance 0.125)

| model | L | test accuracy per seed (0, 1, 2) | train seconds per run |
|---|---|---|---|
| S4D (2 blocks) | 16 | 1.00, 1.00, 1.00 | 1.4 |
| S4D (2 blocks) | 64 | 1.00, 1.00, 1.00 | 4.3 |
| S4D (2 blocks) | 256 | 0.85, 0.30, 0.95 | 10.5 |
| LSTM | 16 | 1.00, 1.00, 1.00 | 0.8 |
| LSTM | 64 | 0.12, 0.13, 0.12 | 2.4 |
| LSTM | 256 | 0.13, 0.12, 0.12 | 9.5 |

parameters: S4D 9,352, LSTM 9,224. the LSTM breaks abruptly between 16 and 64: at $L = 64$ it is at chance on all three seeds. the S4D model is perfect at 64 and solves 256 on two of three seeds (0.85, 0.95), with one failure (0.30). in preliminary runs with the same architecture and optimizer (seed 0), the LSTM *without* the forget-gate bias fix solved $L = 10$ but stayed at chance at $L = 20$ after 600 steps, and at $L = 64$ after 2,000 steps; with the fix, $L = 20$ is solved in 600 steps.

**why:** for the LSTM, the gradient from the last output to the first token's embedding is a product of $L - 1$ step Jacobians; with forget gates below 1, it shrinks geometrically, and early in training (when the gates aren't yet trained to stay open) the signal never arrives, so the model never learns to keep the token. initializing the forget gate high delays the decay, which buys a factor of a few in length, not an order of magnitude. the S4D model's last output depends on the first token through a single kernel entry $K_{L-1}$ per channel and layer; channels whose $\Delta$ gives a window longer than $L$ have a usable $K_{L-1}$ from the start, and the gradient adjusts $C$ to amplify it relative to the distractors. at $L = 256$ that has to happen against 255 distractors summed by the same kernel, so it's harder and seed-dependent in 300 steps. the honest summary is "S4D solves $L = 64$ reliably and $L = 256$ on 2 of 3 seeds with this budget; the LSTM fails beyond $L = 16$ with this budget", not "S4D solves long-range memory". more steps would help both models; this lab doesn't measure where each one's limit is with unlimited training.
