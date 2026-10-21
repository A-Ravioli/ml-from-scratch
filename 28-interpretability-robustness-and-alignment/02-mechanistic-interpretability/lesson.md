# mechanistic interpretability

## why this matters

attribution (topic 01) tells you which *inputs* mattered. it can't tell you *how* the model turned them into an answer. mechanistic interpretability tries to read the algorithm itself off the weights and activations: which components compute what, and how they connect. the goal is ambitious, closer to reverse-engineering a compiled binary than to fitting a summary statistic, but it has had real successes. Olsson et al. (2022) found **induction heads**, pairs of attention heads that implement "if the current token appeared before, predict whatever followed it", and showed they form in a sudden phase change that coincides with the arrival of in-context learning. Meng et al. (2022) used **causal tracing** to locate where GPT-style models store factual associations ("the Eiffel Tower is in ___"), and then edited single facts by changing one MLP's weights. Wang et al. (2022) traced the full circuit, 26 heads in GPT-2 small, that completes "When Mary and John went to the store, John gave a drink to ___".

all of these rest on two operations you'll build: **reading** activations (caching every intermediate output of a forward pass) and **writing** them (running the model with one activation replaced). reading gives you correlations; writing gives you causes. the lab trains a small transformer that learns to copy in-context and uses your tools to find, test and trace its induction circuit.

## warm-up: predict first

1. in a pre-LN transformer, what does block 3 *read*, and where does it *write* its output?
2. you run a model on a "clean" prompt and a "corrupted" one that differs in one token, then rerun the corrupted prompt with layer 5's output at every position replaced by its clean value. could the output be fully clean again?
3. a two-layer model learns to continue repeated random sequences. why would it need *two* layers of attention to do it?
4. you zero out an attention head and the loss barely changes. does that mean the head is unimportant?

## prerequisites check

- [ ] you know the pre-LN block $x + \text{attn}(\text{LN}(x))$, $x + \text{MLP}(\text{LN}(x))$ and the causal mask (→ ch 18 · 03–04)
- [ ] you know what `model.named_modules()` returns and how module names are dotted paths (→ ch 15 · 07)
- [ ] your attention modules keep `last_weights` when `save_weights = True` (→ ch 18 · 01, ch 27 · 02)
- [ ] you know logits, log-probabilities and the cross-entropy loss of a language model (→ ch 18 · 04)

## core ideas

### idea 1: the residual stream, and reading it with hooks

a decoder-only transformer is a stack of blocks around one running vector per position, the **residual stream**. the embedding writes the first value; every attention layer and MLP *reads* the stream (through a layer norm), computes something, and *adds* its output back. the unembedding reads the final value. Elhage et al. (2021) made this the central picture: the stream is a shared communication channel, and each component's contribution to the logits is (almost) additive, so you can ask what each one wrote.

to look at these values you need them without rewriting the model's `forward`. PyTorch's **forward hooks** do that: `module.register_forward_hook(fn)` calls `fn(module, inputs, output)` every time the module runs, and returns a handle whose `remove()` takes the hook off again. a cache is a dictionary filled by one hook per module, keyed by the module's dotted name (`"blocks.0.attn"`, `"ln_f"`). two hygiene rules matter more than they look. **detach** what you store, or every cached tensor keeps the whole autograd graph alive. and **always remove the hooks**, even if the forward pass raises: a forgotten hook keeps running on every later call, filling a dictionary nobody reads or, worse, still patching (idea 2) a model you think is clean.

> **spec.** `run_with_cache(model, idx, *, names=None)`: registers a forward hook on each submodule whose name (from `model.named_modules()`) is in `names`, or on every submodule except the root (name `""`) if `names` is None; `KeyError` for a name that isn't a submodule. runs `output = model(idx)` and returns `(output, cache)` with `cache[name]` = that module's output, detached (tuple outputs become tuples of detached tensors; if a module runs twice, the last call wins). every hook is removed afterwards, even if the forward pass raises. the model's mode isn't changed.

> **▶ build: stage 1.** implement `run_with_cache`.

### idea 2: writing activations

reading tells you what's there; to learn what an activation *does*, change it and watch the output. a forward hook that **returns** a tensor replaces the module's output with it, so everything downstream computes from your value. three kinds of replacement come up again and again:

- **zero ablation**: replace the output with zeros. simple, but it can push the model far outside anything it saw in training, so a big effect may mean "this broke the model" rather than "this component carried the information".
- **mean ablation**: replace it with its average over a reference dataset. it removes what's specific to this input and keeps the model on-distribution.
- **patching**: replace it with the value the same module produced on a *different* input. this is the precise one: it asks "what if this component had seen the other input?"

often you want to replace only some positions of a `(B, T, d)` activation: patch the stream at the position of one token and leave the others alone. a causal model has a useful property here: patching position $t$ can't change any output at positions before $t$.

> **spec.** `patch_activation(model, idx, name, value, *, positions=None)`: runs `model(idx)` with the output of submodule `name` replaced by `value` (broadcastable to that output's shape); with `positions` (an int, a list of ints or a slice), only `output[:, positions]` is replaced by `value[:, positions]` (after broadcasting), along dim 1, the sequence dimension of `(B, T, ...)` activations. returns the model's output. `value` isn't modified, the hook is removed afterwards (even on errors), and `KeyError` for an unknown name.

> **▶ build: stage 2.** implement `patch_activation`.

### idea 3: activation patching

**activation patching** (also called causal tracing or interchange intervention) turns patching into an experiment. take a **clean** input on which the model does the thing you care about, and a **corrupted** input that differs minimally and makes the model do something else: in the lab, a repeated sequence where one earlier token is swapped, so the model should now copy a different token. pick a **metric** that measures the behaviour, usually a **logit difference** like $\text{logit}(\text{clean answer}) - \text{logit}(\text{corrupt answer})$ at the answer position: unlike a probability it's roughly linear in the residual stream, and unlike a loss it's specific to the two answers. then, for each component (and each position), run the corrupted input with that component's *clean* activation patched in, and record the metric. components where the metric jumps back to its clean value carry the information that distinguishes the two inputs; components where nothing happens don't.

this is a **denoising** experiment (clean into corrupt, "is this enough to restore the behaviour?"). its mirror, **noising** (corrupt into clean, "is this necessary?"), answers a different question, and the two can disagree when the model has redundant paths. sweeping positions turns the result into a map of where information *travels*: in a causal model it starts at the corrupted token's position and has to be moved, by attention, to the answer position at some layer. the grid of (layer, position) shows you at which layer the move happens. run the clean cache and every patched run in eval mode, without gradients: dropout would make the comparison meaningless.

> **spec.** `activation_patching(model, clean_idx, corrupt_idx, metric, names)`: caches the clean run's activations at `names`, then for each name runs the corrupted input with that clean activation patched in (all positions) and returns `{name: float(metric(output))}`. `position_patching(model, clean_idx, corrupt_idx, metric, names)`: the same, one position at a time: an ndarray `(len(names), T)` whose `[i, t]` entry patches `names[i]` at position `t` only. both run in eval mode under `torch.no_grad()` and restore the model's previous mode.

> **🔮 predict:** clean and corrupted inputs differ only at position 3. which entries of the `position_patching` grid for the embedding layer can differ from the corrupted metric?

> **▶ build: stage 3.** implement `activation_patching` and `position_patching`.

### idea 4: attention heads, and the induction circuit

each attention head does two separable things. its **QK circuit** decides *where* to look: the attention weight from query position $t$ to key position $s$ depends on the stream at $t$ (through $W_Q$) and at $s$ (through $W_K$). its **OV circuit** decides *what* to move: the value read from $s$ (through $W_V$) and written into the stream at $t$ (through $W_O$). you can read the QK part directly: it's the attention pattern your attention modules store with `save_weights = True`.

heads often have recognisable patterns. a **previous-token head** attends from each position to the one before it, so its output at position $s$ carries "the token at $s - 1$ was $u$". an **induction head** uses that: at position $t$, holding token $u$, its query looks for keys that say "the previous token was $u$", which is exactly position $s + 1$ for every earlier occurrence $s$ of $u$; its OV circuit copies the token found there, the one that came after $u$ last time, into the prediction. on a sequence that repeats with period $P$, an induction head at position $t \ge P$ attends to $t - P + 1$. the key at $s + 1$ only contains "previous token was $u$" because a head in an *earlier* layer wrote it there: the induction head's keys are composed from another head's output (**K-composition**), which is why induction needs at least two layers (warm-up 3).

scoring patterns turns a picture into a number per head: average the weight a head puts on the position its hypothesised role predicts. a high score is a hypothesis, not a proof; idea 5 tests it.

> **spec.** `attention_patterns(model, idx)`: for a model with `model.blocks[i].attn` exposing `save_weights` / `last_weights` (your GPT and your Llama), sets every flag, runs the model in eval mode without gradients, and returns the list of `(B, H, T, T)` weights, one per block; flags and mode are restored afterwards. `prev_token_score(weights)`: `(H,)`, the mean of `weights[b, h, t, t - 1]` over the batch and $t = 1, \dots, T-1$. `induction_score(weights, period)`: `(H,)`, the mean of `weights[b, h, t, t - period + 1]` over the batch and $t = \text{period}, \dots, T - 1$; `ValueError` unless $1 \le$ `period` $< T$.

> **▶ build: stage 4.** implement `attention_patterns`, `prev_token_score` and `induction_score`.

### idea 5: ablation, and why it under-reports

to test that a head *causes* the behaviour, remove it: zero its contribution and measure the loss. a head's contribution is its slice of the concatenated head outputs, the input to the output projection `out_proj` (`o_proj` in your Llama): head $h$ owns features $h d_h$ to $(h+1) d_h - 1$. zeroing that slice with a forward **pre**-hook (`register_forward_pre_hook`, which can replace the module's *input*) removes exactly what the head writes into the stream.

ablation answers "is this component *necessary*?", and models are often built with spare parts. Wang et al. (2022) found **backup heads** in GPT-2 that take over when the main heads are ablated, and McGrath et al. (2023) found whole layers that compensate ("self-repair"). if four heads do the same job, removing one barely shows. a small ablation effect is weak evidence that a component is unimportant; a large patching effect is strong evidence that it matters. combine both, and ablate groups as well as individuals.

> **spec.** `ablate_heads(model, idx, heads)`: `heads` is a list of `(layer, head)` pairs; runs `model(idx)` with each listed head's slice `[head * d_h, (head + 1) * d_h)` of the input to `model.blocks[layer].attn.out_proj` (or `.o_proj` when there is no `out_proj`) set to zero, `d_h` = `attn.head_dim`, and returns the output. weights are unchanged and hooks are removed afterwards.

> **🔮 predict:** a layer has four heads that all score above 0.65 as induction heads. you ablate the best one. how much does the loss on repeated text rise?

> **▶ build: stage 5.** implement `ablate_heads`.

## worked example

one head, the sequence `a b c a b c` (period 3), with this attention pattern (row = query position, column = key position):

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **0** | 1 | | | | | |
| **1** | 0.5 | 0.5 | | | | |
| **2** | 0 | 0.8 | 0.2 | | | |
| **3** | 0 | 0.9 | 0 | 0.1 | | |
| **4** | 0 | 0 | 0.6 | 0.2 | 0.2 | |
| **5** | 0 | 0 | 0 | 0.7 | 0.1 | 0.2 |

**previous-token score:** the entries $(t, t-1)$ for $t = 1..5$ are $0.5, 0.8, 0, 0.2, 0.1$, mean **0.32**.

**induction score** (period 3): for $t = 3, 4, 5$ the targets are $t - 2 = 1, 2, 3$, holding $0.9, 0.6, 0.7$: mean **0.733**. at $t = 3$ the current token is `a`; its previous occurrence is position 0, and the token after it, `b`, sits at position 1: attending there and copying `b` predicts the next token correctly. a head attending to position 0 itself (the *duplicate* token, $t - P$) would score 0 here.

**a patching grid.** in the lab, clean and corrupted inputs differ only at position 4. patching the embedding or block 0 at position 4 restores the clean prediction at position 13, and so does block 1 at position 13; every other cell keeps the corrupted value. the information sits at position 4 until a layer-1 head moves it to position 13.

your `prev_token_score` and `induction_score` must reproduce 0.32 and 0.733.

## on paper

1. **the logits are a sum.** in a one-layer, attention-only transformer with no layer norm, the final stream at position $t$ is $x_t = e_t + \sum_h o^h_t$ (embedding plus each head's output). show that the logits $W_U x_t$ split into a *direct path* and one term per head, and say what zero-ablating head $h$ does to them.

<details><summary>solution</summary>

$W_U x_t = W_U e_t + \sum_h W_U o^h_t$, by linearity. the first term is the direct path (embedding straight to unembedding, a bigram table $W_U W_E$); each head adds $W_U o^h_t$ with $o^h_t = W_O^h \sum_s A^h_{ts} W_V^h e_s$. zero-ablating head $h$ removes exactly its term and leaves the others unchanged. with layer norm and MLPs the decomposition is only approximate (the norm rescales the whole stream, the MLP is nonlinear), which is why patching is used to measure effects rather than reading them off.

</details>

2. **what patching can and can't restore.** clean and corrupt differ at position $j$ only, and the metric reads the logits at the last position $T - 1$. which single-position patches are guaranteed to restore the clean metric exactly?

<details><summary>solution</summary>

the embedding at position $j$ (the input to the first block): after that patch the corrupted input's activations at every layer and position equal the clean ones. a later layer's stream at position $j$ is *not* enough: by then, attention at positions after $j$ has already read the corrupted value, so those positions stay corrupted (in the tests' tiny GPT, patching block 0's output at $j$ gives −0.0582 against a clean −0.0581: close, but not exact). and the final block's output (or the final norm) at position $T-1$: the logits there depend on nothing else. in between, a single-position patch of block $\ell$'s output restores the metric only if all the information has been moved to that position by layer $\ell$, which is what the grid measures. positions before $j$ never matter: their clean and corrupt activations are equal.

</details>

3. **why induction needs composition.** a single attention layer has one QK circuit: the score between positions $t$ and $s$ is a bilinear function of the embeddings (plus positions) at $t$ and $s$. explain why it can't attend to "the position after an earlier copy of my token" when the period varies.

<details><summary>solution</summary>

position $s+1$ is the right target because of the token at $s$, but the key at $s + 1$ contains only the embedding of the token *at* $s + 1$ (and its position). in one layer the score can compare "my token" with "your token" (finding duplicates, at $t - P$) or with "your position", but not with "your predecessor's token". a previous-token head in an earlier layer writes the predecessor's token into the stream at $s+1$, after which the second layer's QK circuit can match it.

</details>

4. **necessity versus sufficiency.** four heads each implement induction perfectly and the model averages their outputs. what does ablating one head do to the loss? what do patching (denoising) one head's clean output into a corrupted run and ablating all four do?

<details><summary>solution</summary>

ablating one leaves three that still copy the right token; the averaged output still predicts it, a little less confidently, so the loss rises a little (in the lab, 0.30 → 0.41 nats). patching one head's clean output restores a quarter of the copying signal; it may or may not flip the metric. ablating all four removes copying entirely (the lab's loss goes to 5.2 nats, worse than guessing, because the rest of the model still acts as if copying were happening). single-component ablations under-report importance when components are redundant.

</details>

## experiment

the lab trains a 2-layer GPT on sequences that repeat with a random period, so the only way to predict them is to copy in-context. you'll watch the loss on repeated text and your head scores during training (does copying arrive gradually or all at once?), read the attention patterns of the heads that do it, ablate them singly and in groups, and trace the copied token through the network with a position-by-layer patching grid.

## check your understanding

1. why does `run_with_cache` detach what it stores?

<details><summary>answer</summary>

a stored tensor that requires grad keeps its whole computation graph alive: caching every module of a model would hold every intermediate of the forward pass in memory indefinitely. the cache is for reading, so detached copies are enough.

</details>

2. why is a logit difference a better patching metric than the probability of the clean answer?

<details><summary>answer</summary>

probabilities saturate: once the clean answer has 99% probability, large internal changes barely move it, and small ones near 50% move it a lot. the logit difference is close to linear in the residual stream and specific to the two competing answers, so partial restorations show up proportionally.

</details>

3. a patch at (block 2, position 7) restores the metric fully. what can you conclude, and what can't you?

<details><summary>answer</summary>

the stream at block 2's output, position 7, carries everything about the clean/corrupt difference that the metric depends on: it's *sufficient* for the behaviour (given the rest of the model). it doesn't tell you which component inside blocks 0–2 put it there, or that this is the only route.

</details>

4. an induction head's score is 0.77, not 1.0. name two reasons a real induction head doesn't score 1.

<details><summary>answer</summary>

it spends some attention elsewhere (on the current token, on the first position as a "resting" place) when it's confident or unsure; and the score averages over positions where the job is ambiguous (the token after the very first position, or repeated tokens inside the random half that create several candidate matches).

</details>

5. why does the corrupted run in the lab score −6.9, almost the negative of the clean run's +6.9?

<details><summary>answer</summary>

the model copies whatever followed the earlier occurrence: in the corrupted run that's the replacement $y$, so it predicts $y$ with the same confidence it predicted $x_4$ in the clean run, and the logit difference flips sign. the corruption changes the answer, not the model's certainty.

</details>

6. zero ablation and mean ablation of the same head give different loss changes. which do you trust more, and why?

<details><summary>answer</summary>

usually mean ablation: zeros may be a value the head never outputs, so the downstream layers see something off-distribution and the measured damage includes that confusion. mean ablation removes the input-specific information while keeping typical statistics.

</details>

## common bugs

- **hooks left behind.** an exception between `register_forward_hook` and `remove()` leaves a hook running forever; later "clean" runs are silently patched. use `try`/`finally`.
- **caching the root.** `named_modules()` yields `("", model)` first; caching it stores the model's output tuple, and patching it replaces the output wholesale.
- **patching the wrong axis.** activations are `(B, T, d)`; `out[positions] = ...` patches *batch rows*. positions go on dim 1.
- **patching with grads and dropout on.** a train-mode model gives a different clean cache every run; compare in eval mode under `no_grad`.
- **the induction off-by-one.** an induction head at $t$ attends to $t - P + 1$, the token *after* the earlier copy; $t - P$ is a duplicate-token head.
- **ablating the output instead of the input of `out_proj`.** `out_proj`'s output mixes all heads; a head's own slice exists only at its input.

## going deeper

- Elhage et al. (2021), "A mathematical framework for transformer circuits", Anthropic. [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)
- Olsson et al. (2022), "In-context learning and induction heads", Anthropic. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Meng et al. (2022), "Locating and editing factual associations in GPT", *NeurIPS* (causal tracing, ROME). [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- Wang et al. (2022), "Interpretability in the wild: a circuit for indirect object identification in GPT-2 small". [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Geiger et al. (2021), "Causal abstractions of neural networks", *NeurIPS* (interchange interventions). [arXiv:2106.02997](https://arxiv.org/abs/2106.02997)
- Heimersheim & Nanda (2024), "How to use and interpret activation patching". [arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
- McGrath et al. (2023), "The hydra effect: emergent self-repair in language model computations". [arXiv:2307.15771](https://arxiv.org/abs/2307.15771)

## connections

- ← builds on: your GPT and its attention modules with `save_weights` (ch 18), your Llama (ch 27 · 02), hooks and module names (ch 15 · 07), attribution and probes (topic 01).
- → used in: topic 03 asks what the directions in the cached activations mean (superposition) and learns a sparse dictionary for them; topic 04's adversarial perturbations are another way of writing to a model's inputs; the capstone uses `run_with_cache` to read the neurons of a transformer trained on modular addition and explains its algorithm.
