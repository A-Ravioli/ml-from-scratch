# the transformer block · lab notes

numbers below are from the reference solution on a laptop CPU, on the real tiny Shakespeare, seed 0. the stalled post-LN run was checked at three learning rates in a side run (below).

## experiment 1: where the gradient goes at initialization

**what people predict.** usually that the gradient *shrinks* towards the input in both models (the vanishing-gradient story from chapter 17), or that the two models look the same at initialization.

**what happens.** pre-LN: the gradient norm is nearly flat across the 24 blocks, 0.357 at the first block and 0.313 at the last (ratio 0.88), if anything slightly larger near the input. post-LN: it grows towards the output, from 0.469 at block 1 to 0.808 at block 24 (ratio 1.72). neither vanishes.

**why.** in pre-LN, every block sits on an untouched identity path to the output (on paper 1), so each block receives roughly the gradient that arrives at the top of the stream, plus whatever flows through the blocks above it; the profile is flat. in post-LN, the gradient reaching block $l$ has passed through the layer norms of every block above it, and at initialization each of those norms rescales it by roughly $\sqrt d / \|x\|$ with a stream whose norm, right after each residual addition, is larger than the normalized one; the blocks nearest the output get the largest gradients. Xiong et al. (2020) proved that the post-LN gradient near the output is independent of depth while pre-LN's scales like $1/\sqrt L$; combined with Adam, which turns any gradient into steps of about the learning rate, the last layers of a post-LN model take relatively large, poorly conditioned steps at the start of training. the static picture is modest; its consequence is not.

## experiment 2: training without a safety net

**what people predict.** "post-LN will be a bit slower" or "it will diverge to NaN". few predict a model that trains *stably* and learns almost nothing.

**what happens.** bits per character on the training batches, average of the last 25 of 200 steps:

| run | last 25 steps |
|---|---|
| pre-LN, constant $10^{-3}$ | **3.54** |
| post-LN, constant $10^{-3}$ | 4.78 |
| post-LN, 100 warmup steps to $10^{-3}$ | 3.67 |

the post-LN model drops to about 4.8 bits in 30 steps and then stays flat for the remaining 170. pre-LN keeps improving. with warmup, post-LN starts slower (it's taking tiny steps at first) but overtakes its constant-rate twin by step 60 and finishes 0.13 bits behind pre-LN. a side run (same setup, no warmup) found the same plateau at constant learning rates of $10^{-3}$, $3 \times 10^{-3}$ and $10^{-2}$ (4.74–4.75 bits after 200 steps), while pre-LN reached 3.52–3.59 at all three; with warmup, post-LN at $10^{-4}$ and $3 \times 10^{-4}$ was still improving after 200 steps (4.46 and 3.96).

**why.** 4.77 bits per character is the entropy of the character frequencies of the training text: the stalled model has learned exactly the unigram distribution, which only needs the output head's bias and the last layer, and nothing that uses context. at the start of training, Adam's first updates have a fixed size (about the learning rate per parameter), and in a deep post-LN stack those updates to the last blocks change the output a lot. the model quickly falls into the solution that ignores the input (predicting frequencies is the easiest way to reduce the loss) with the attention and MLP branches in a regime where their gradients are uninformative. Liu et al. (2020) describe post-LN's heavy dependence on its residual branches, which amplifies small parameter changes: the larger the early steps, the worse. warmup keeps the first few hundred updates tiny while Adam's second-moment estimates settle and the model moves into a better-conditioned region; then the full learning rate is safe. changing the constant learning rate doesn't help because the problem isn't the step size in the long run but the size of the very first steps relative to the curvature at initialization. pre-LN avoids it by construction, which is why GPT-2 and almost every large model since use it, and it's what you'd choose for a 100-layer model (with a final norm and scaled output-layer initialization). warmup is still used with pre-LN in practice (topic 04's `train_lm` has it), but as an optimization nicety rather than a requirement.
