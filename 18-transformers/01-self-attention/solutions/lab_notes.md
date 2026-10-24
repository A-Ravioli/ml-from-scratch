# self-attention · lab notes

numbers below are from the reference solution on a laptop CPU (seed 0 unless stated). your numbers will differ slightly with your own random draws; the patterns shouldn't.

## experiment 1: what the √d does at initialization

**what people predict.** most people expect the unscaled softmax to get "somewhat sharper" with width, and the gradient to shrink by a modest factor. few predict that the *scaled* numbers don't move at all.

**what happens.**

| $d$ | score std (scaled) | max weight | mean \|dL/dS\| | score std (unscaled) | max weight | mean \|dL/dS\| |
|---|---|---|---|---|---|---|
| 4 | 1.01 | 0.106 | 1.1e-02 | 2.0 | 0.279 | 1.0e-02 |
| 64 | 1.00 | 0.107 | 1.2e-02 | 8.0 | 0.792 | 4.6e-03 |
| 1024 | 1.00 | 0.108 | 1.2e-02 | 32.1 | 0.950 | 1.2e-03 |

(the lab also prints $d$ = 16 and 256.) scaled, every column is flat: the score standard deviation is 1.00, the top weight about 0.11 (rows of 64 keys, well spread but not uniform), the gradient 0.012, whatever $d$. unscaled, the scores' spread is exactly $\sqrt d$ (2, 8, 32), at $d = 1024$ the top weight averages 0.95 (attention is nearly one-hot from the very first step), and the gradient reaching the scores is ten times smaller than with the scaling.

**why.** on paper 1: a dot product of $d$ independent unit-variance products has standard deviation $\sqrt d$. a softmax over scores with spread 32 gives the largest score essentially all the weight. the softmax Jacobian $\operatorname{diag}(a) - aa^\top$ vanishes at a one-hot $a$, so the gradient through the scores, $a \odot (g - a^\top g)$, shrinks: in a row with $a_m \approx 1$, every entry is either multiplied by a tiny $a_j$ or by $1 - a_m \approx 0$. with the scaling, the score distribution doesn't depend on $d$, so neither does anything downstream of it. the upstream gradient was divided by $\sqrt d$ here so that the comparison isolates the softmax; that's also what happens in a real network, where layer norms and initializations keep activations at unit scale.

## experiment 2: does it matter once you train?

**what people predict.** "Adam rescales gradients, so the unscaled version will catch up", or the opposite, "the unscaled model can't learn at all". the truth depends on width, which most people don't anticipate.

**what happens.** accuracy after 600 steps (chance 0.016), seed 0, and in brackets seeds 1–3:

| $d$ | scaled | unscaled |
|---|---|---|
| 32 | 0.978 (0.955, 0.964, 0.955) | 0.910 (0.902, 0.908, 0.886) |
| 128 | 1.000 (1.0, 1.0, 1.0) | 0.986 (0.978, 0.999, 0.996) |
| 512 | 0.999 (1.0, 1.0, 0.999) | 0.697 (0.672, 0.649, 0.668) |

at $d = 32$ and 128 the unscaled model learns the task, a little slower. at $d = 512$ it's stuck around 0.67 and a training loss near 2, while the scaled model solves the task in its first 25 steps (its loss curve then has a few spikes: $10^{-3}$ is a large learning rate for that width, but it recovers). the result holds across all four seeds. in a longer side run (800 steps, $d = 1024$) the unscaled loss never left 3.5–4.3, close to chance.

**why.** PyTorch's default `nn.Linear` initialization draws weights from $U(-1/\sqrt{d}, 1/\sqrt{d})$, which makes the projected queries and keys have variance about 1/3 per entry, so the raw scores have standard deviation about $\sqrt d / 3$: 1.9 at $d = 32$ (mild, so the unscaled model is fine), 7.5 at $d = 512$ (saturated). once attention is saturated, the gradient that reaches $W_Q$ and $W_K$ is not just small but *uninformative*: it comes from the one or two keys that already have the weight, and says nothing about the 14 others. Adam rescales each parameter's step by its own running gradient size, which fixes the magnitude but can't create a signal that isn't there, and with near-one-hot attention the model can't explore which key to match. the scaled model starts with soft attention over all 16 pairs, so the gradient tells every query–key pair whether matching it would help. the small model barely cares because at $d = 32$ the unscaled scores aren't saturated to begin with: the $\sqrt d$ matters precisely because it lets you change the width without retuning anything else.

## experiment 3: what attention knows about order

**what people predict.** that shuffled tokens give "different" outputs, and that the causal mask makes no difference to whether position is visible without positional encodings.

**what happens.** without a mask, un-shuffling the outputs for the shuffled input reproduces the original outputs to $3 \times 10^{-8}$ (float32 rounding): attention computed exactly the same thing, just in a different order. with `is_causal=True` the difference is 0.49: order now matters. six copies of one token give identical outputs at all six positions, with or without the mask. and the causal output at position 0 is `out_proj(v_proj(x_0))` to $6 \times 10^{-8}$, for any input.

**why.** on paper 3: every step of self-attention either acts token by token or treats the keys as an unordered set, so permuting the input permutes the output. without positional information, a bidirectional transformer sees a bag of tokens: "dog bites man" and "man bites dog" are the same input. the causal mask is tied to positions, so it breaks the symmetry: position $t$ averages over exactly $t + 1$ values, and position 0 can only copy itself. that is a positional signal (Haviv et al., 2022 showed that causal language models without positional encodings learn to use it and do almost as well), but it isn't available when every token sees every other one, and with identical tokens even the causal model sees identical averages at every position. that's why topic 02 adds explicit positions.
