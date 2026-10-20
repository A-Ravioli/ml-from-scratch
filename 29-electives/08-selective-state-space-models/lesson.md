# selective state-space models

## why this matters

topic 07's S4D layer is a convolution: every input is filtered with the same kernel, whatever it is. that's exactly what makes it fast to train, and exactly what it can't get past. read "the password is *swordfish*. … [two thousand words] … what is the password?". to answer, a model has to decide, *based on the content*, that "swordfish" is worth keeping and the next two thousand words aren't. a time-invariant system can't make that decision: its kernel weighs a position by how long ago it was, never by what was there. attention can (it compares contents directly), but it pays $O(L^2)$ for the privilege, and at generation time it has to keep every past token around in its KV cache.

**Mamba** (Gu & Dao, 2023) keeps the state-space recurrence but makes it **selective**: the step size $\Delta$ and the matrices $B$ and $C$ become functions of the current input. a token that matters can open the state wide (a large $\Delta$ writes it in and forgets what was there); a token that doesn't can leave the state untouched ($\Delta \approx 0$). the price is the convolution trick: with parameters that change at every step there is no single kernel. the fix is an old idea from parallel computing, the **scan** (Blelloch, 1990): a linear recurrence is a composition of affine maps, composition is associative, and associative operations can be computed in $O(\log L)$ parallel steps. this topic builds both scans, the selective scan, Mamba's layer and block, and ends with the block's project: your Mamba against a transformer on a long-sequence task, including lengths neither model saw in training.

## warm-up: predict first

1. an S4D layer and a selective SSM each see a sequence of 1,000 tokens in which 8 are marked as "remember me", at random positions. which can, in principle, output those 8 tokens at the end, and why can't the other?
2. you want $h_t = a_th_{t-1} + b_t$ for $t = 1, \dots, 10^6$ on a machine with a million processors. the obvious loop takes $10^6$ steps. how many steps do you need?
3. in a selective SSM, what should the step size $\Delta_t$ be on a token you want to ignore? on a token that should overwrite the memory?
4. a transformer's memory at generation time grows with the sequence length; a Mamba layer's state is fixed. name one task where that fixed size must eventually hurt.
5. a transformer and a Mamba model both learn a task on sequences of length 64. you test them on length 1,024. which do you expect to hold up, and why?

## prerequisites check

- [ ] you've built the S4D layer: discretization, recurrent view, convolutional view (→ topic 07)
- [ ] you know why the convolutional view needs time invariance (→ topic 07, on paper 4)
- [ ] you've built multi-head causal attention and transformer blocks, and know their $O(L^2)$ cost and the KV cache (→ ch 18 · topics 01, 03, 06)
- [ ] you know gated recurrent units and what a gate does (→ ch 17 · topic 05)
- [ ] you've used RMSNorm and SiLU (→ ch 15 · topics 02, 07)

## core ideas

### idea 1: selection, and the recurrence it leaves you with

a time-invariant SSM computes $y = K * u$: the contribution of $u_s$ to $y_t$ is $K_{t-s}$, a function of the distance alone. to filter by content, let the parameters depend on the input. Mamba's selective SSM (it calls it **S6**) computes, at every position $t$ and for every channel $d$,

$$\Delta_t = \operatorname{softplus}(\text{linear}(x_t)), \qquad B_t = \text{linear}(x_t), \qquad C_t = \text{linear}(x_t),$$

and runs the diagonal recurrence with those time-varying parameters (idea 3 has the details). the step size is the key. in the discretized system, $\bar A_t = e^{\Delta_tA}$ with $A < 0$:

- $\Delta_t \to 0$: $\bar A_t \to 1$ and $\bar B_t \to 0$. the state passes through unchanged and the token is **ignored**.
- $\Delta_t \to \infty$: $\bar A_t \to 0$. the state is **reset** and filled with the current input.

in topic 07, $\Delta$ was a per-channel time scale fixed after training. here it is a per-token decision: how much to forget, and how much of this token to write in. with one state dimension, $A = -1$ and an exact discretization, the update is $h_t = e^{-\Delta_t}h_{t-1} + (1 - e^{-\Delta_t})x_t$: a gated recurrence with gate $g_t = 1 - e^{-\Delta_t}$, the same form as a GRU's update (on paper 2). selective SSMs are what you get when you take the gated RNNs of chapter 17, remove the nonlinearity from the recurrence, and give the state many dimensions.

removing the nonlinearity is what makes it fast. once the parameters are known (they depend only on the inputs, not on the state), what's left at each step is an elementwise affine map,

$$h_t = a_t \odot h_{t-1} + b_t,$$

with $a_t = \bar A_t$ and $b_t = \bar B_tu_t$. every selective SSM layer reduces to evaluating this **linear recurrence**, the **scan**, for all $t$.

> **spec.** `sequential_scan(a, b, h0=None)`: `a` and `b` are tensors of the same shape `(batch, L, ...)`; it returns `h` of that shape with $h_t = a_t\odot h_{t-1} + b_t$ along dimension 1, starting from `h0` (shape `(batch, ...)`) or zeros. a plain loop over $t$ is the point here: it's the reference the parallel version must match.

> **▶ build: stage 1.** implement `sequential_scan`.

### idea 2: the parallel scan

the loop takes $L$ sequential steps, no matter how many processors you have. but each step is an affine map, $f_t(h) = a_th + b_t$, and $h_t = (f_t \circ \dots \circ f_1)(h_0)$: a prefix of compositions. composing two affine maps gives another affine map. writing a map as the pair $(a, b)$, "first $(a_1, b_1)$, then $(a_2, b_2)$" is

$$(a_1, b_1) \bullet (a_2, b_2) = (a_1a_2,\ a_2b_1 + b_2),$$

and this operation is **associative** (on paper 1), though not commutative. associativity is all a parallel prefix algorithm needs: you can group the compositions any way you like.

the simplest such algorithm is **Hillis–Steele** (Hillis & Steele, 1986). keep an array of pairs, initially $(a_t, b_t)$. in round $r = 0, 1, 2, \dots$, with offset $s = 2^r$, replace every element $t \ge s$ by (element $t - s$) $\bullet$ (element $t$), all at once; elements $t < s$ stay. after round $r$, element $t$ holds the composition of the maps $t - 2^{r+1} + 1, \dots, t$ (or from 1 if that's further back). after $\lceil\log_2L\rceil$ rounds every element holds its full prefix, and its $b$ is $h_t$ (with $h_0 = 0$; fold a nonzero $h_0$ into $b_1$ first, $b_1 \leftarrow a_1h_0 + b_1$). each round is two whole-tensor multiplies and an add, with shifted copies padded by the identity map $(1, 0)$. it does $O(L\log L)$ work in $O(\log L)$ depth. **Blelloch's** work-efficient scan (Blelloch, 1990) does $O(L)$ work in $2\log_2L$ depth with an up-sweep and a down-sweep; Mamba's GPU kernel uses a variant of it, fused with the discretization so that the $(L, D, N)$ tensors never leave fast on-chip memory. either is fine here.

on a CPU with PyTorch, the parallel scan's advantage is that each round is one vectorized operation over the whole sequence instead of $L$ tiny ones; the lab measures what that buys.

> **spec.** `combine(first, second)` returns the pair for "apply `first`, then `second`", elementwise on tensors (or numbers). `parallel_scan(a, b, h0=None)` returns exactly what `sequential_scan` returns, using $O(\log L)$ rounds of whole-tensor operations (no Python loop over positions), for any $L \ge 1$, differentiably, without modifying its inputs.

> **🔮 predict:** after the first Hillis–Steele round on $L = 4$, element 3 holds which composition? and after the second?

> **▶ build: stage 2.** implement `combine` and `parallel_scan`.

### idea 3: the selective scan (S6)

with $D$ channels (the layer's width) and $N$ state dimensions per channel, S6 takes

- the input $u$: `(batch, L, D)`;
- step sizes $\Delta$: `(batch, L, D)`, positive, one per channel and position;
- $A$: `(D, N)`, real and negative (a fixed parameter, not input-dependent);
- $B$ and $C$: `(batch, L, N)`, input-dependent and shared by all channels at a position;

and computes, per channel $d$ and state dimension $n$,

$$h_t = \exp(\Delta_tA)\odot h_{t-1} + (\Delta_tu_t)\,B_t, \qquad y_t = \sum_nC_{t,n}h_{t,n} + Du_t .$$

$A$ is discretized with the zero-order hold, as in topic 07, but $B$ with the simpler Euler rule $\bar B_t = \Delta_tB_t$ (Gu & Dao, 2023: the difference is small when $\Delta A$ is small, and the simpler form is cheaper). the tensors $a = \exp(\Delta A)$ and $b = \Delta uB$ have shape `(batch, L, D, N)`, which is the memory cost of a naive selective scan: $N$ times the activations. that's why Mamba's kernel never materializes them in main memory.

when $\Delta$, $B$ and $C$ don't depend on $t$, S6 is a time-invariant SSM and therefore a convolution with $K_l = \sum_nC_n\Delta B_ne^{\Delta A_nl}$, the real-$A$ version of topic 07's kernel. selection is a strict generalization.

> **spec.** `selective_scan(u, delta, A, B, C, D=None, *, mode="parallel")` with the shapes above returns `y` of shape `(batch, L, D)` by the formulas above (no skip term if `D` is `None`). `mode="parallel"` uses your parallel scan, `mode="sequential"` your sequential scan, and any other mode raises `ValueError`; both modes must agree.

> **▶ build: stage 3.** implement `selective_scan`.

### idea 4: the selective layer

the `SelectiveSSM` layer produces $\Delta$, $B$, $C$ from its input $x$ `(batch, L, D)`:

- `x_proj`: `Linear(D, R + 2N, bias=False)`, whose output splits, in this order, into $\Delta_{\text{raw}}$ (size $R$), $B$ (size $N$) and $C$ (size $N$). $R$, the **rank** of $\Delta$, defaults to $\lceil D/16\rceil$: $\Delta$ is computed through a bottleneck, which is cheap and works as well as a full projection. (Mamba's reference code sets $R = \lceil d_{\text{model}}/16\rceil$ from the block's width $d_{\text{model}} = D/E$; this layer doesn't know $d_{\text{model}}$, so it uses its own width $D$, which gives an $E$ times larger rank. pass `dt_rank` explicitly to match Mamba.)
- `dt_proj`: `Linear(R, D)` with bias, then $\Delta = \operatorname{softplus}(\texttt{dt\_proj}(\Delta_{\text{raw}}))$. the bias is initialized so that $\operatorname{softplus}(\text{bias}) = \Delta_0$ with $\Delta_0$ log-uniform in $[\Delta_{\min}, \Delta_{\max}]$ (default $[10^{-3}, 10^{-1}]$, as in topic 07): that's the inverse softplus, $\text{bias} = \Delta_0 + \log(1 - e^{-\Delta_0})$. its weight starts uniform in $\pm R^{-1/2}$.
- `A_log` `(D, N)`: $A = -\exp(\texttt{A\_log})$, initialized to $A_{d,n} = -(n + 1)$ (Mamba's "S4D-Real" initialization). real $A$ suffices once $B$ and $C$ are selective; Gu & Dao (2023) report that complex $A$ helps on continuous signals like audio but not on text.
- `D` `(D,)`: the skip, initialized to ones.

for generation, the layer runs as a recurrence: one step costs $O(DN)$ and the state is a fixed `(batch, D, N)` tensor, however long the sequence has been. compare a transformer, whose KV cache grows by $2 \times$ layers $\times$ width numbers per token. that fixed state is both the strength (constant memory, constant time per token) and the limit (on paper 4).

> **spec.** `SelectiveSSM(d_inner, d_state=16, *, dt_rank=None, dt_min=1e-3, dt_max=1e-1)` with exactly the parameters above (`A_log`, `D`, `x_proj`, `dt_proj`). `params(x)` returns `(delta, B, C)` for `x` of shape `(batch, L, D)`. `forward(x)` is `selective_scan(x, delta, -exp(A_log), B, C, D)`. `step(x_t, h=None)` takes `x_t` `(batch, D)` and the state `h` `(batch, D, N)` (zeros if `None`) and returns `(y_t, h)`, so that stepping through a sequence reproduces `forward`.

> **▶ build: stage 4.** implement `SelectiveSSM`.

### idea 5: the Mamba block

a Mamba model is a stack of identical blocks with residual connections, like a transformer, but each block replaces both attention and the MLP. for width $d$ and expansion $E$ (default 2):

1. normalize: $x' = \text{RMSNorm}(x)$ (pre-norm, as in modern transformers);
2. project up and split: $(x_{\text{in}}, z) = \texttt{in\_proj}(x')$, with `in_proj` a `Linear(d, 2Ed, bias=False)`, each half of width $Ed$;
3. a short **causal depthwise convolution** over time on $x_{\text{in}}$ (kernel `d_conv` = 4, one filter per channel), then SiLU: it gives the SSM's input a little local context, so $\Delta$, $B$, $C$ can depend on the last few tokens, not just the current one;
4. the selective SSM on that, width $Ed$;
5. **gate**: multiply elementwise by $\operatorname{SiLU}(z)$;
6. project down with `out_proj`, a `Linear(Ed, d, bias=False)`, and add the residual: $x + \texttt{out\_proj}(\cdot)$.

"causal" in step 3 matters: the convolution must only look backwards (pad on the left), or the model can peek at future tokens during training and fails at generation. the gate (step 5) plays the role of the GLU in transformer MLPs and in topic 07's block.

what this block doesn't include is Mamba's real engineering contribution: the **hardware-aware** implementation that fuses the discretization, the scan and the output into one GPU kernel, keeps the `(L, D, N)` intermediates in on-chip SRAM, and recomputes them in the backward pass instead of storing them. on a GPU that makes the selective scan as fast as a convolution; on a CPU, as you'll see, the naive version's memory traffic dominates.

> **spec.** `MambaBlock(d_model, *, d_state=16, expand=2, d_conv=4, dt_rank=None)` with submodules `norm` (your chapter 15 `RMSNorm(d_model)`), `in_proj`, `conv1d` (an `nn.Conv1d(Ed, Ed, d_conv, groups=Ed)` holding the depthwise filters and their bias; compute the causal convolution with its weights however you like), `ssm` (`SelectiveSSM(Ed, d_state, dt_rank=dt_rank)`) and `out_proj`, wired as above. `forward(x)` maps `(batch, L, d_model)` to the same shape.

> **🔮 predict:** you zero `out_proj`'s weight. what does the block compute? and if you replace $x$ by $2x$, how does the output change?

> **▶ build: stage 5.** implement `MambaBlock`.

## worked example

**a scan by hand.** $a = (0.5, 0.5, 2)$, $b = (1, 1, 1)$, $h_0 = 0$: $h = (1,\ 0.5 + 1,\ 2\cdot1.5 + 1) = (1, 1.5, 4)$. with $h_0 = 2$: $h = (2, 2, 5)$.

**combine.** $(2, 3) \bullet (5, 7)$: first $h \mapsto 2h + 3$, then $h \mapsto 5h + 7$, so $h \mapsto 5(2h + 3) + 7 = 10h + 22$: the pair $(10, 22)$.

**Hillis–Steele on the same scan.** start: $(0.5, 1), (0.5, 1), (2, 1)$. round 1 (offset 1): element 1 becomes $(0.5, 1)\bullet(0.5, 1) = (0.25, 1.5)$, element 2 becomes $(0.5, 1)\bullet(2, 1) = (1, 3)$. round 2 (offset 2): element 2 becomes (element 0) $\bullet$ (element 2) $= (0.5, 1)\bullet(1, 3) = (0.5, 4)$. the $b$'s are now $(1, 1.5, 4)$. ✓

**selection.** with $\Delta_t = 0$, $\exp(0\cdot A) = 1$ and $\Delta_tu_tB_t = 0$: $h_t = h_{t-1}$, so token $t$ is skipped and changing it changes nothing. with $\Delta_t = 200$ and every $A \le -0.6$, $\exp(200A) < 10^{-50}$: the state is replaced by $200u_tB_t$, and nothing before $t$ affects any later output.

your code must reproduce all four.

## on paper

1. **associativity.** show that $\bullet$ is associative and that $(1, 0)$ is its identity. is it commutative?

<details><summary>solution</summary>

$((a_1, b_1)\bullet(a_2, b_2))\bullet(a_3, b_3) = (a_1a_2, a_2b_1 + b_2)\bullet(a_3, b_3) = (a_1a_2a_3,\ a_3a_2b_1 + a_3b_2 + b_3)$. $(a_1, b_1)\bullet((a_2, b_2)\bullet(a_3, b_3)) = (a_1, b_1)\bullet(a_2a_3, a_3b_2 + b_3) = (a_1a_2a_3,\ a_2a_3b_1 + a_3b_2 + b_3)$. equal. $(1, 0)\bullet(a, b) = (a, a\cdot0 + b)$ and $(a, b)\bullet(1, 0) = (a, b)$. not commutative: $(2, 3)\bullet(5, 7) = (10, 22)$ but $(5, 7)\bullet(2, 3) = (10, 17)$; order of application matters, as for any function composition. (it's associative because function composition is.)

</details>

2. **$\Delta$ is a gate.** take $N = D = 1$, $A = -1$, $B_t = C_t = 1$, and the exact (ZOH) discretization of $B$. show that the recurrence is $h_t = (1 - g_t)h_{t-1} + g_tu_t$ and find $g_t$. how does Mamba's Euler $\bar B = \Delta$ compare?

<details><summary>solution</summary>

ZOH: $\bar A = e^{-\Delta}$, $\bar B = \frac{e^{-\Delta} - 1}{-1} = 1 - e^{-\Delta}$. so $h_t = e^{-\Delta_t}h_{t-1} + (1 - e^{-\Delta_t})u_t$, with $g_t = 1 - e^{-\Delta_t} \in (0, 1)$: an interpolation between the old state and the input, like a GRU's update gate, with $\Delta_t = 0$ meaning "keep" and $\Delta_t \to \infty$ meaning "replace". Euler gives $\bar B = \Delta_t \approx g_t$ for small $\Delta_t$ (they differ at second order), and for large $\Delta_t$ it scales the input up instead of saturating; the following layers absorb the scale.

</details>

3. **work and depth.** count the combine operations of Hillis–Steele on $L = 2^k$, and compare with the sequential loop's $L - 1$. why can the parallel version still be faster?

<details><summary>solution</summary>

round $r$ updates the $L - 2^r$ elements with $t \ge 2^r$, so the total is $\sum_{r=0}^{k-1}(L - 2^r) = kL - (L - 1) = L(\log_2L - 1) + 1$: about $\log_2L$ times more work than the loop. but the $L - 2^r$ combines of a round are independent and run as one vectorized operation, so the *depth* (sequential steps) is $k = \log_2L$ instead of $L - 1$. with enough parallel hardware (a GPU, or SIMD lanes), time follows depth, not work. Blelloch's scan reduces the work to $O(L)$ at the cost of doubling the depth.

</details>

4. **what a fixed state can't do.** a model must reproduce a random string of $m$ tokens from a vocabulary of size $V$ after reading it once. show that a recurrent model with a state of $S$ numbers, each stored in $b$ bits, fails for $m > Sb/\log_2V$. why doesn't the same argument limit attention?

<details><summary>solution</summary>

after reading the string, everything the model knows about it is in the state: at most $Sb$ bits. a uniformly random string carries $m\log_2V$ bits, so if $m\log_2V > Sb$, two different strings must map to the same state, and the model outputs the same thing for both: it errs on at least one. attention keeps every past token's key and value (the KV cache grows with $m$), so its memory isn't bounded in advance. this is the trade-off behind "transformers are better than SSMs at copying" (Jelassi et al., 2024): selection decides what to keep, but it can't keep more than the state holds.

</details>

## experiment

the lab times your sequential and parallel scans as the sequence grows; compares your selective model with your topic 07 S4D model on selective copying, where the tokens to remember sit at random positions among noise; and runs the block's project: your Mamba against small transformers on a long-sequence task, trained at one length and tested at lengths up to 16 times longer, with seeds, training time and inference cost reported.

## check your understanding

1. why does making $\Delta$, $B$, $C$ input-dependent destroy the convolutional view but not the scan?

<details><summary>answer</summary>

the convolution needs one kernel for all positions, which requires time-invariant parameters (topic 07, on paper 4). the scan only needs each step to be an affine map of the previous state, $h_t = a_th_{t-1} + b_t$; it doesn't care whether $a_t$ and $b_t$ change with $t$, as long as they don't depend on $h$. so input-dependent parameters are fine; state-dependent ones (like an LSTM's gates, which depend on $h_{t-1}$) are not.

</details>

2. why does Mamba initialize $\Delta$ in $[10^{-3}, 10^{-1}]$ rather than around 1?

<details><summary>answer</summary>

with $A_n = -(n + 1)$ and $\Delta = 1$, $e^{\Delta A}$ ranges from $e^{-1}$ to $e^{-N}$: the state forgets almost everything in a step or two, so the layer starts with essentially no memory, and gradients to earlier tokens vanish. small $\Delta$ starts the layer with long memory (slowly decaying modes) and lets training learn where to forget, by raising $\Delta$ on the tokens that should reset the state.

</details>

3. a Mamba layer at generation: what is its memory cost per layer, and how does it compare with a transformer layer's KV cache after 10,000 tokens, for $d = 1024$, $E = 2$, $N = 16$?

<details><summary>answer</summary>

the SSM state is $Ed\times N = 2048\times16 = 32{,}768$ numbers, plus the convolution's last `d_conv - 1` inputs ($3\times2048$), regardless of length. the transformer's KV cache is $2\times d\times L = 2\times1024\times10{,}000 \approx 2\times10^7$ numbers per layer, 600 times more, and growing.

</details>

4. why does the Mamba block need the short convolution before the SSM?

<details><summary>answer</summary>

$\Delta_t$, $B_t$, $C_t$ are computed from the SSM's input at position $t$ only. the convolution mixes in the previous few tokens, so the selection decision can depend on a short local context (a bigram, the previous token) rather than on one token in isolation. ablations in the Mamba paper show it helps; it's the same role as the "shift" in RWKV or the short convolutions in H3.

</details>

5. the selective scan materializes `(batch, L, D, N)` tensors. what does that cost for batch 8, $L = 2048$, $D = 2048$, $N = 16$ in float32, and what does Mamba's kernel do instead?

<details><summary>answer</summary>

$8\times2048\times2048\times16\times4$ bytes $\approx 2.1$ GB per tensor, and a naive implementation stores several (for $a$, $b$, $h$, and their gradients). Mamba's kernel loads $\Delta$, $A$, $B$, $C$, $u$ into on-chip SRAM, discretizes and scans there, writes only $y$ `(batch, L, D)` back, and recomputes the intermediates in the backward pass.

</details>

6. you trained a selective SSM and a transformer on length-64 sequences and test on length 1,024. list the reasons each might fail.

<details><summary>answer</summary>

the transformer: position encodings it never saw (learned embeddings don't exist beyond the training length; sinusoids at new positions produce new combinations; ALiBi's linear bias favours recent tokens and may swamp a distant match), and attention spread over 16 times more keys, diluting a sharp match. the SSM: modes that decay too fast for the longer gap, and selection that isn't perfectly zero on distractors, so 16 times more of them leak into the state. which effect wins is an empirical question; the project measures it.

</details>

## common bugs

- **shifting with wraparound in the parallel scan.** symptom: early positions pick up contributions from the end of the sequence. fix: pad the shifted copies with the identity map, $a = 1$ and $b = 0$, not by `torch.roll`.
- **combining in the wrong order.** symptom: the parallel scan matches the sequential one only when all $a_t$ are equal. fix: (element $t - s$) $\bullet$ (element $t$), earlier map first: new $b = a_t\,b_{t-s} + b_t$.
- **updating $a$ before using it for $b$.** symptom: wrong results from round 2 on. fix: compute both new values from the old ones (a tuple assignment does this).
- **ZOH for $B$.** symptom: the selective scan doesn't match the reference; with small $\Delta$ the error is tiny, with large $\Delta$ it isn't. fix: Mamba's $\bar B = \Delta B$.
- **a symmetric (non-causal) convolution.** symptom: training loss far lower than generation quality suggests; the causality check fails. fix: pad `d_conv - 1` positions on the left only.
- **the bias of `dt_proj` set to $\Delta_0$ instead of its inverse softplus.** symptom: initial step sizes around 0.7, far above $\Delta_{\max}$. fix: bias $= \Delta_0 + \log(1 - e^{-\Delta_0})$.

## going deeper

- Gu & Dao (2023), "Mamba: linear-time sequence modeling with selective state spaces". selection (§3.1–3.2), the hardware-aware scan (§3.3), the block (§3.4), selective copying and induction heads (§4.1). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Blelloch (1990), "Prefix sums and their applications", Technical Report CMU-CS-90-190, Carnegie Mellon University. the work-efficient scan and the general recipe for linear recurrences. [cs.cmu.edu/~guyb/papers/Ble93.pdf](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- Hillis & Steele (1986), "Data parallel algorithms", *Communications of the ACM* 29(12). the log-depth scan you built. [doi:10.1145/7902.7903](https://doi.org/10.1145/7902.7903)
- Martin & Cundy (2018), "Parallelizing linear recurrent neural nets over sequence length", *ICLR*. the parallel scan for linear RNNs, before Mamba. [arXiv:1709.04057](https://arxiv.org/abs/1709.04057)
- Dao & Gu (2024), "Transformers are SSMs: generalized models and efficient algorithms through structured state space duality" (Mamba-2), *ICML*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- Arora et al. (2023), "Zoology: measuring and improving recall in efficient language models". multi-query associative recall, where fixed-state models fall behind attention. [arXiv:2312.04927](https://arxiv.org/abs/2312.04927)
- Jelassi, Brandfonbrener, Kakade & Malach (2024), "Repeat after me: transformers are better than state space models at copying", *ICML*. [arXiv:2402.01032](https://arxiv.org/abs/2402.01032)
- Ali, Zimerman & Wolf (2024), "The hidden attention of Mamba models". the implicit attention rows of lab experiment 3. [arXiv:2403.01590](https://arxiv.org/abs/2403.01590)
- Olsson et al. (2022), "In-context learning and induction heads", *Transformer Circuits Thread*. [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Press, Smith & Lewis (2022), "Train short, test long: attention with linear biases enables input length extrapolation" (ALiBi), *ICLR*. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)

## connections

- ← builds on: the S4D layer, discretization and the two views (topic 07); linear recurrences (topics 05–06); gated RNNs (ch 17); attention, transformer blocks, ALiBi and the KV cache (ch 18); RMSNorm and SiLU (ch 15).
- → used in: the chapter capstone, where this block's project becomes a portfolio entry (per-seed scores, a paired comparison with the transformer baseline, a budget table); the research chapter (ch 30), where "does the efficient architecture match attention at equal compute?" is a model research question.
