# selective state-space models · lab notes

numbers are from the reference solutions (this topic's, topic 07's S4D layer, and chapter 18's transformer blocks) on a laptop CPU with one torch thread. **timings in this lab are machine- and load-dependent**: the lab took 85–175 s on a lightly loaded machine and up to 300–460 s while many other jobs were running, and training and inference times varied by 2–4× between runs; compare ratios and orderings, not absolute numbers. the accuracies are reproducible exactly (fixed seeds for initialization, training batches and test sets). **all data are synthetic.** the project is one toy task at a tiny scale (6k–27k parameters, 1,000 training steps): read its numbers as evidence about this task and budget, not about language modelling.

## experiment 1: what the parallel scan buys on a CPU

**what people usually predict:** the parallel scan wins big at long lengths ("log depth beats linear depth"), or it loses because it does more work.

**what happens:** (best of 3, forward only, a `(4, L, 32, 8)` tensor; ranges over four runs, two of them under heavy load)

| L | sequential (ms) | parallel (ms) | speed-up | rounds |
|---|---|---|---|---|
| 64 | 0.3–0.4 | 0.4–1.2 | 0.4–1.0× | 6 |
| 256 | 1.1–2.2 | 1.3–4.5 | 0.4–0.8× | 8 |
| 1024 | 5.4–27 | 8.3–63 | 0.4–0.7× | 10 |
| 2048 | 19–28 | 80–108 | 0.2–0.3× | 11 |
| 4096 (earlier version of the lab) | 26–38 | 50–83 | 0.5× | 12 |

on this CPU the parallel scan is never faster than the loop and is up to 5× slower. the results agree to about $3\times10^{-6}$ (float32).

**why:** each loop iteration touches one `(4, 32, 8)` slice, 1,024 numbers, which stays in cache; the Python overhead per iteration is a few microseconds. each Hillis–Steele round instead reads and writes the *whole* `(4, L, 32, 8)` tensor several times (two shifted copies, two multiplies, an add), and there are $\log_2L$ rounds, so the parallel version moves about $\log_2L$ times more data through memory, and on a CPU with a few cores there's no spare parallelism to hide it. the speed-up the lesson promises needs hardware where the per-round cost is nearly independent of $L$ (a GPU with thousands of lanes) and an implementation that doesn't round-trip through main memory, which is exactly what Mamba's fused kernel provides. the residual differences come from float32 rounding: the two scans add the same terms in a different order.

## experiment 2 (the project): selective SSM vs transformer on long sequences

**what people usually predict:** the transformer wins at the training length (attention is made for retrieval) and the SSM generalizes better to long sequences; few people predict the S4D model's failure, and many expect both transformers to behave alike.

**what happens:** (single-query induction, trained at $L = 32$ for 1,000 steps, batch 32, AdamW lr 3e-3; 128 test sequences per length; chance 0.071; train seconds from the most recent run, under heavy load)

| model | seed | L = 32 | L = 64 | L = 128 | L = 256 | L = 512 | train seconds |
|---|---|---|---|---|---|---|---|
| Mamba (yours) | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40 |
| Mamba (yours) | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 41 |
| Mamba (yours) | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 39 |
| S4D (yours, topic 07) | 0 | 1.00 | 0.66 | 0.22 | 0.09 | 0.06 | 11 |
| S4D (yours, topic 07) | 1 | 0.15 | 0.09 | 0.07 | 0.05 | 0.05 | 10 |
| S4D (yours, topic 07) | 2 | 0.20 | 0.10 | 0.09 | 0.08 | 0.08 | 11 |
| transformer, ALiBi | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 15 |
| transformer, ALiBi | 1 | 0.98 | 1.00 | 0.98 | 0.98 | 0.91 | 15 |
| transformer, ALiBi | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16 |
| transformer, sinusoidal | 0 | 1.00 | 0.50 | 0.25 | 0.17 | 0.12 | 16 |
| transformer, sinusoidal | 1 | 1.00 | 0.49 | 0.30 | 0.16 | 0.16 | 16 |
| transformer, sinusoidal | 2 | 1.00 | 0.54 | 0.24 | 0.16 | 0.13 | 17 |

parameters: Mamba 5,888 · S4D 9,616 · each transformer 26,512. training time per run across our runs (same code, different machine load): Mamba 12–89 s, S4D 4–17 s, transformers 7–26 s. within a run, Mamba took between about 1× and 3.5× the transformers' training time (typically about 2×).

inference cost of one forward pass on one sequence (seed-0 models, best of 3, most recent run): Mamba 3 ms at $L = 256$ and 24 ms at 1024 (8.4×), S4D 10 → 33 ms (3.4×), ALiBi transformer 8 → 83 ms (10.3×), sinusoidal transformer 9 → 95 ms (11.1×). single-shot timings in earlier runs at $L$ = 512 → 2048 gave 4.8–11.4× for Mamba and 15–20× for the transformers, and an independent rerun measured 2.4× for Mamba and 8.1× for the sinusoidal transformer. at these tiny sizes and batch 1, fixed overheads and machine load move these ratios a lot. the robust statement is only that the transformers' cost grew faster with length than Mamba's in every run we made, as their $O(L^2)$ attention predicts; this lab is not a careful benchmark.

the headline, with its uncertainty: at the training length, Mamba, the ALiBi transformer and the sinusoidal transformer all solve the task (1.00 on 3 of 3 seeds for Mamba and sinusoidal; 1.00, 0.98, 1.00 for ALiBi). at 16× the training length, Mamba keeps 1.00, 0.98, 1.00 (mean 0.99) and the ALiBi transformer 0.99, 0.91, 1.00 (mean 0.97): with three seeds, Mamba is at least as good, and the difference at 512 rests on one ALiBi seed. the sinusoidal transformer collapses as soon as the sequence is longer than it trained on (0.49–0.54 at 2×, 0.12–0.16 at 16×). topic 07's time-invariant S4D fails: one seed fits the training length (1.00) without generalizing beyond it, the other two never get off chance.

the sinusoidal transformer embeds tokens with standard deviation $d^{-1/2}$ and multiplies by $\sqrt d$ (Vaswani et al.), so tokens and sinusoids both have unit scale. an earlier version of this lab used PyTorch's default $\mathcal N(0, 1)$ embeddings times $\sqrt{32}$, a token signal about 8× the position signal (RMS 5.7 against 0.71): it scored 0.91–0.98 at $L = 32$, 0.52–0.65 at 64 and 0.10–0.15 at 512, and an independent rerun of an unscaled variant (token std about 1) gave 0.98–1.00, 0.46–0.52 and 0.13–0.16. the collapse beyond the training length doesn't depend on the scaling.

a preliminary run with the same model definitions but a longer training length ($L = 64$; transformers at lr 1e-3 for 3,000 steps, Mamba at 3e-3 for 600 steps; seed 0 only; the sinusoidal model then still had $\mathcal N(0, 1)$ embeddings) agreed in shape: Mamba 0.99, 0.98, 0.99, 1.00, 0.98 at $L$ = 64 to 1,024; ALiBi 1.00, 1.00, 1.00, 0.97, 0.84; sinusoidal 0.62 at the training length, 0.08 at 1,024. the ALiBi transformer needed about 800 steps to learn the task, Mamba about 450.

**why:** *S4D:* its output at the last position is $\sum_sK_{L-s}u_s$ (plus nonlinearities between layers), with $K$ fixed after training. finding "the token after the earlier trigger" requires weighting position $s$ by what's *at* $s - 1$, which a fixed kernel can't do for arbitrary positions. at $L = 32$ one seed found a solution that works for that one length (we didn't investigate how; with only 30 possible trigger positions, a fixed kernel plus nonlinearities has room for a position-specific lookup), and it breaks as soon as the length changes. *Mamba:* $B_s$, $C_L$ and $\Delta$ depend on the tokens, so the effective weights $\alpha_{L,s}$ (experiment 3) can single out the right position wherever it is; distractors contribute almost nothing, so 16 times more of them barely matter. *ALiBi:* attention retrieves the right position easily, and ALiBi has no position embedding to fall off the end of; but its bias penalizes distance linearly, $-m_h(L - s)$, so at 16× the length the trigger is 16× further away and is heavily penalized in the heads with large slopes. a plausible reading (we didn't inspect the heads) is that the model relies on its flattest heads, and one seed's version of that is less robust. *sinusoidal:* at test lengths beyond 32 the model sees position vectors it never trained on, and the attention patterns it learned (which use position to find "the token right after") misfire. (separately: the chapter-18 reference `GPT` at this point added unit-amplitude sinusoids to embeddings of standard deviation 0.02, so position swamped content and it couldn't learn even a previous-token task in 300 steps; that has since been fixed in chapter 18, but this lab keeps its own small transformer built from your blocks.)

**what you can't conclude:** anything about large models or natural language (these models have thousands of parameters and the task has one bit of structure); that transformers can't extrapolate (ALiBi mostly does, and better position schemes exist); or that Mamba is better at recall in general: lesson on paper 4 shows a fixed state must fail when there's more to remember than it can hold, and the Zoology and "repeat after me" papers show attention winning on multi-query recall and copying at equal size. a fair verdict also needs equal compute: here Mamba had a quarter of the transformers' parameters but took about 1–3.5× their training time within a run (typically about 2×), because the naive selective scan is memory-bound (experiment 1).

## experiment 3: what did selection learn?

**what people usually predict:** both layers look at the first trigger, or at the target.

**what happens:** on one length-256 test sequence (first trigger at 107, target at 108, prediction correct), the last position's row of layer 1 puts its weight on the last few positions (top: 255, 252, 254; largest weight 0.020 against a median of 0.00045), and layer 2's row has a single spike at position **110**, with weight 24.6 against a median of 0.0026, four orders of magnitude.

**why:** the work splits like a transformer's induction circuit (Olsson et al., 2022), with the convolution playing the previous-token head. the width-4 causal convolution in front of each SSM means the SSM input at position $s$ mixes tokens $s - 3, \dots, s$. at positions 108–111 that window contains both the trigger and the target, so layer 2 can compute a $B_s$ that writes the target's identity into the state keyed by "a trigger was just here"; at the end, the final trigger produces a $C_L$ that reads with the same key, and the product $C_L\cdot B_s$ (decayed by $\exp(A\sum\Delta)$, which stays near 1 because $\Delta$ is small almost everywhere) is large only there. position 110 rather than 108 is where the learned write happened to be strongest within that window. layer 1's row is local because the model only needs it for short-range context. at longer test lengths the same write happens at the same relative offset from the trigger, and every distractor adds a weight of order $10^{-3}$, which is why accuracy holds at 16× the training length.
