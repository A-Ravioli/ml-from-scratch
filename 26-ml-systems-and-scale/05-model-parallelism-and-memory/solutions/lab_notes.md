# model parallelism & memory · lab notes

numbers below are from the reference solution on an Apple M2 laptop CPU (4 threads; experiment 3 one thread per process), one run, other jobs sharing the machine. the memory numbers are exact and machine-independent; the times move by 10–20% between runs.

## experiment 1: memory, predicted and measured

**what people predict.** that the activations are comparable to the model's state, maybe 2–5× larger at a big batch.

**what happens.** your `training_memory(1_816_896, bytes_per_param=4)` predicts 7.27 MB of parameters, 7.27 MB of gradients and 14.54 MB of AdamW state, and PyTorch holds exactly that (the optimizer's `step` counters are 0-d tensors, 4 bytes each, invisible at this precision). the saved activations at batch 64 and context 128 are **620 MB, 21×** the weights, gradients and optimizer state together. per token, they take 57.2 KB at context 32, 63.4 KB at 64 and 75.7 KB at 128: linear in the batch, superlinear in the context.

**why.** the state is a fixed 16 bytes per parameter; the activations are per token. per token and layer, the block saves its inputs to four layernorm-normalized projections, the $q$, $k$, $v$ heads, the MLP's 768-wide hidden layer before and after GELU, the residual streams and dropout-free intermediates: dozens of $d$-wide vectors, about 14 KB per token per layer at $d = 192$ in float32. the attention adds per-head $T \times T$ tensors that your chapter 18 attention materializes and autograd saves: per token that's $H \cdot T$ scores per layer, which grows with $T$: from 32 to 128 the per-token bytes grow by 18.5 KB, exactly $4 \text{ layers} \times 6 \text{ heads} \times 96 \text{ extra keys} \times 8$ bytes, i.e. two float32 $T \times T$ tensors per head and layer (the softmax output, and the scores feeding it). a fused attention kernel (FlashAttention, or PyTorch's `scaled_dot_product_attention`) saves only $q$, $k$, $v$, the output and one statistic per row, removing the $T^2$ term entirely; that's the capstone's first optimization.

## experiment 2: activation checkpointing

**what people predict.** a memory saving equal to the number of layers (4×), and the textbook 33% slowdown.

**what happens.** at batch 16, context 128: saved activations fall from **155 MB to 10 MB (15.5×)**, the forward + backward step slows from 160 ms to 229 ms (1.43×) in this run, and the gradients are identical. the slowdown is the least stable number in this lab: reruns on this machine and a reviewer's measured anywhere from **1.09× to 1.50×**, depending on load.

**why.** what's left is almost exactly the four block inputs, $4 \times 16 \times 128 \times 192 \times 4 = 6.3$ MB, plus what lives outside the blocks: the embedding output, the final layernorm and the logits over 65 characters for the loss (about 3.7 MB). the saving is far more than 4× because inside each block the activations are about 25 times larger than the block's input, and with checkpointing only one block's internals exist at a time, transiently, during its recomputation (which `saved_tensor_bytes` doesn't see: the peak memory of a real run includes one block's worth). on paper 3's 4/3 assumes the backward pass costs exactly twice the forward and that nothing else happens in a step. on a small model on a CPU neither holds well: the ratio of backward to forward time depends on which ops dominate (masking and softmax recompute cheaply; matmuls don't), and under load the timing noise is as large as the effect, which is why the measured range straddles 4/3. the optimizer step isn't part of this timing, so a full training step's relative overhead is a little smaller.

## experiment 3: ZeRO-1 on real processes, and the pipeline bubble

**what people predict.** a quarter of the optimizer state per rank with 4 ranks (correct), and final weights that are "close" to the 1-rank run's.

**what happens.**

| ranks | AdamW state per rank | your `training_memory`, ZeRO-1 | max difference between ranks | max difference from 1 rank |
|---|---|---|---|---|
| 1 | 3.19 MB | 3.19 MB | 0 | 0 |
| 2 | 1.60 MB | 1.60 MB | 0 | 3.6 × 10⁻⁶ |
| 4 | 0.80 MB | 0.80 MB | 0 | 5.4 × 10⁻⁶ |

the ranks agree bit for bit with each other, and differ from the single-process run by a few millionths after 20 steps. the GPipe bubble with 4 stages: 75% idle with 1 micro-batch, 43% with 4, 27% with 8, 16% with 16, 4% with 64.

**why.** every rank receives identical bits from the all-reduce and from the all-gather (each element of the final parameters was computed once, by one rank, and copied), so they can't disagree. the difference from one rank is floating-point association: with 2 ranks the gradient is $(\bar g_{1:32} + \bar g_{33:64})/2$, two means of 32 combined by the all-reduce, while one rank computes one mean over 64; the sums happen in a different order and round differently in the last bits, and AdamW's division by $\sqrt{v}$ amplifies tiny gradient differences for parameters whose gradients are near zero. mathematically the runs are identical; numerically they're as close as two orders of addition can be. in the schedule plot the bubble is three grey triangles: at the start, stage $s$ waits $s$ ticks for the first micro-batch to reach it; in the middle, stage $s$ waits $2(p - 1 - s)$ ticks between its last forward and its first backward (the backward wave starts at the last stage and has to travel back); at the end, stage $s$ idles $s$ ticks while the earlier stages finish. that's $2(p - 1)$ idle ticks per stage, whatever $m$ is. more micro-batches make the busy middle longer, so the fixed bubble becomes a smaller fraction.
