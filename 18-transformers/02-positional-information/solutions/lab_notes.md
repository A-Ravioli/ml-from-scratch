# positional information · lab notes

numbers below are from the reference solution on a laptop CPU, on the real tiny Shakespeare (the offline stand-in gives different numbers; the extrapolation pattern is the same in kind). experiment 2 was repeated with seeds 1 and 2; the ranges are given.

## experiment 1: what a position "looks like" to attention

**what people predict.** for RoPE, most people say "constant" (it's a relative scheme, and nothing about the vectors changes) or "oscillates forever" (it's made of rotations).

**what happens.** the sinusoidal dot-product matrix is constant along every diagonal: bright on the main diagonal (a position is most similar to itself), with bands that fade and return. for RoPE, the score of two identical all-ones vectors is 64 at distance 0, then drops fast and keeps shrinking on average while it wiggles: mean |score| 33.6 over distances 1–100, 12.9 over 500–1000, 9.1 over 1000–2000.

**why.** on paper 1: $\text{PE}(m)\cdot\text{PE}(n) = \sum_i \cos((m - n)\omega_i)$, a function of $m - n$ only, so the matrix is Toeplitz. for RoPE with $q = k = \mathbf{1}$, each channel pair contributes $2\cos((m - n)\theta_i)$, so the score is $2\sum_i \cos(r\theta_i)$ with $r$ the distance. at $r = 0$ all 32 cosines are 1. as $r$ grows, the fast pairs spin through many turns and their cosines average out to about zero; only the slow pairs, whose angles are still small, keep contributing. the longer the distance, the fewer pairs are still "in phase", so the score decays. Su et al. (2021) proved an upper bound with this shape and called it RoPE's long-term decay: built-in, mild preference for nearby tokens, without any parameter. the decay isn't monotone because the cosines of different frequencies come back into partial alignment at some distances.

## experiment 2: length extrapolation

**what people predict.** "sinusoidal and RoPE extrapolate, because they're defined at every position; learned can't; no positions at all is worst." almost nobody predicts that RoPE ends up the *worst* of the four that run, or that no positional information beats two of the formulas.

**what happens.** bits per character (seed 0; seeds 1–2 in brackets), 600 steps on 64-character windows:

| scheme | positions 16–63 (trained) | 64–127 | 128–255 | 256–511 |
|---|---|---|---|---|
| none | 3.40 (3.42–3.43) | 3.60 | 3.59 | 3.64 (3.66–3.70) |
| learned | 3.21 (3.21) | — | — | — |
| sinusoidal | 3.06 (3.07–3.09) | 4.35 | 4.57 | 4.51 (4.37–4.48) |
| rope | **2.78** (2.75–2.76) | 3.02 | 4.68 | **5.18** (5.56–5.75) |
| alibi | 2.99 (2.96–3.01) | 3.04 | 2.95 | 2.98 (2.94–2.96) |

inside the training length RoPE is the best scheme and ALiBi second, and both beat learned and sinusoidal positions; no positions is clearly worst. past position 64 the ranking flips. the sinusoidal model jumps by more than a bit per character immediately. RoPE holds up for another 40–50 positions, then degrades steadily to 5.2–5.8 bits per character, far worse than the model with no positional information at all (3.6–3.7). ALiBi's curve is flat all the way to 512. the model with no positional information loses only 0.2–0.3 bits. (these are small models trained for 600 steps, far from converged; with more training every in-distribution number would drop, but the extrapolation pattern is the one Press et al. report at scale.)

**why.** *sinusoidal*: the formula is defined everywhere, but the network isn't. it has only ever seen the 64 vectors $x + \text{PE}(p)$ with $p < 64$, and the slow hands of the clock (on paper 4 of topic 02's lesson) are at angles at position 300 that training never produced. layers that were never trained on those inputs produce garbage, and since the positions are added into the residual stream, every layer is affected. *RoPE*: positions never enter the residual stream, only the angle between queries and keys, and the fast pairs have made full turns during training, so short distances still look familiar: that's why it survives a bit past 64. but distances beyond 64 put the slow pairs at angles never seen in training, and attention scores to far-away tokens become arbitrary; past 128 the model attends to the wrong places. *ALiBi*: the bias at distance 300 is just a larger penalty of the same kind, so far-away tokens are attended to even less; the model effectively keeps using a local window and never sees anything new, so its loss stays exactly where it was. the price is that it can never make good use of really long contexts (hence the flat line rather than a falling one). *none*: the only positional signal is the causal mask's averaging (lab 01), which has no fixed scale, so nothing breaks at position 64; it simply wasn't very informative to begin with. this matches Kazemnejad et al. (2023), who found no positional encoding extrapolates better than RoPE or learned positions on their tasks. the fix that long-context LLMs use for RoPE is to rescale positions (position interpolation, Chen et al., 2023) or frequencies so long inputs stay inside the range of angles seen in training, followed by a short fine-tune.
