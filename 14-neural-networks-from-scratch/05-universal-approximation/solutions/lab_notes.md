# universal approximation · lab notes

## experiment 1: depth wins on the sawtooth

**what people usually predict:** that the shallow network's error shrinks smoothly as $k$ grows, and that "a few dozen" units are plenty for 32 pieces.

**what happens:** for the depth-5 sawtooth (32 pieces, computed exactly by the deep network with **10** units), the shallow interpolant's max error hovers between about 0.5 and 1 for every $k$ up to 32, jumps to **0 at $k = 33$**, rises again to about 0.5 at $k = 34$ and decays slowly, reaching 0 again only at $k = 65$. at $k = 17$ the interpolant is identically 0 (max error 1). the table of exact sizes: the deep network needs $2d$ units, the shallow one (with evenly spaced knots) $2^d + 1$: 10 against 33 at $d = 5$, 20 against 1,025 at $d = 10$.

**why:** a one-hidden-layer network with $k$ units has at most $k + 1$ pieces (on paper 2), and the sawtooth has $2^d$, so no shallow network with fewer than $2^d - 1$ units can be exact, and with far fewer pieces than teeth it can't follow the teeth at all: it misses by about half a tooth height or more. the evenly spaced interpolant hits zero error exactly when its knots include every kink ($k - 1$ a multiple of $2^d$); in between, its knots fall at the wrong phases of the teeth. at $k = 17$ the knots $j/16$ are exactly the troughs, so every sample is 0 and the interpolant is flat: aliasing, the same thing that makes wagon wheels spin backwards on film. the deep network gets its pieces from composition: each tent doubles the number of pieces, for the price of two units (on paper 4). this is Telgarsky's depth separation in miniature.

## experiment 2: do random deep networks have many pieces?

**what people usually predict:** that, since depth *can* multiply pieces, deeper random networks have many more pieces than shallow ones with the same units.

**what happens:** the opposite. with 48 units in total, the mean number of pieces on $[-3, 3]$ (10 networks each) falls steadily with depth: **40.9** (1 × 48), 39.5 (2 × 24), 34.7 (3 × 16), 28.9 (4 × 12), 20.2 (6 × 8), **9.7** (12 × 4). none comes close to the 49 a single layer of 48 units could have, let alone anything exponential.

**why:** the exponential count of the sawtooth needs weights that are precisely aligned: every piece produced by the earlier layers must be folded back across the whole range of the next layer's kinks. random weights don't do that. a later unit's kink lands somewhere in the previous layer's output range, splitting maybe one or two pieces, so each unit adds roughly one piece, and the count grows with the number of units, not exponentially with depth (Hanin & Rolnick, 2019). deep, narrow random ReLU networks lose more: with width 4, a sizable fraction of units are off for every input in the range (or on for every input, adding no kink), and once all units of a layer are off, the network's output is constant beyond that point. "depth gives exponentially many pieces" is a statement about what's *possible*, not about what's typical.

## experiment 3: can gradient descent find the deep solution?

**what people usually predict:** that the deep network, which represents sawtooths so efficiently, learns them more easily.

**what happens:**

| target | network | successes (max error < 0.1) | median final MSE | best max error |
|---|---|---|---|---|
| depth 2 (4 pieces) | shallow 1 × 24 | 3/5 | 0.0001 | 0.006 |
| depth 2 (4 pieces) | deep 4 × 4 | 0/5 | 0.0840 | 0.502 |
| depth 3 (8 pieces) | shallow 1 × 24 | 0/5 | 0.0655 | 0.551 |
| depth 3 (8 pieces) | deep 4 × 4 | 0/5 | 0.0840 | 0.502 |

the shallow network learns the 2-tooth sawtooth in 3 of 5 runs. the deep network never learns either target: in every run it ends at MSE 0.084, a flat line at about 1/2.

**why:** an MSE of $1/12 = 0.0833$ and a max error of 0.5 are exactly what predicting the mean everywhere gives, and that's where every deep run ends. dead units explain it. with the default initialization, about 40% of the units of these width-4 layers are already off for every input in $[0, 1]$ at the start (counted over the 5 runs: 0–3 dead units per layer), and training kills more: at the end, every run has a layer with at least 3 of its 4 units dead, and two runs have a layer with all 4 dead. a dead unit gets zero gradient forever; a layer with one live unit squeezes the whole signal through a single ReLU, which can't carry a function that goes up and down; and with a whole layer dead the output is constant, so the only thing left to learn is the output bias: the mean. the shallow network has 24 independent units in parallel, so it can afford to lose some, but it too fails on 8 pieces in 1,500 steps, because high-frequency targets are learned slowly (spectral bias) and its units have to line up with each tooth. the deep architecture's exact solution (experiment 1) is a needle in a haystack for gradient descent from a random start. the gap between representing and learning is the theme of chapter 15 (initialization, normalization and residual connections are what make deep networks trainable) and of chapter 19.
