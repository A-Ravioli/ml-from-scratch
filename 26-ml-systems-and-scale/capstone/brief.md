# 26 · capstone: make your GPT 5× faster to train

## goal

chapter 18's capstone trained GPTs on tiny Shakespeare with your `train_lm`. here you take its middle model and make it reach the same validation loss **five times faster in wall-clock time**, on your own machine, with every optimization profiled, ablated and reported with uncertainty. if 5× isn't reachable on your hardware, the deliverable is the largest speedup you can reach and a measured account of what stands in the way.

every topic of the chapter is in it. you start with a profile (topic 01), replace your attention's score–softmax–weights chain with one fused kernel and let `torch.compile` fuse the rest (02), try float16 with your `GradScaler` and PyTorch's (03), keep the data on the device and use a fused optimizer, and trade steps against learning rate the way topic 04's batch-size arguments suggest. the honest-evaluation habits are older: one change at a time, the baseline rerun on the same machine, seeds, and bootstrap intervals (chapter 05).

## data and model

tiny Shakespeare from `mlfs.datasets.tiny_shakespeare()` (cached under `~/.cache/mlfs/`), characters through your `CharTokenizer`, the first 80% for training and the next 10% for validation, exactly as in chapter 18. the model is chapter 18's **M**: your `GPT` with width 192, 4 layers, 6 heads, context 128, dropout 0.1, learned positions, tied embeddings: 1,816,896 parameters. offline (`MLFS_OFFLINE=1`) the notebook runs the small config on a synthetic stand-in; the tests use tiny GPTs on random token streams.

## the rules

- **the baseline** is your chapter 18 `train_lm`, unchanged, with chapter 18's capstone settings for M: batch 64, 2,000 steps, AdamW at $10^{-3}$ ($\beta = 0.9, 0.95$, weight decay 0.1 on matrices), 100 warmup steps then cosine to $10^{-4}$, clipping at 1, evaluation every 250 steps on 20 batches. its wall-clock time includes its evaluations: that's what training with it costs.
- **the target** is the baseline's final validation loss, measured with `estimate_loss` on 50 batches of the second half of the validation block (the reporting half). the recipe search picks its runs using the first half (the selection half), so the reported loss isn't the one the search optimized.
- **the race**: any run that reaches the target (same `estimate_loss`, 50 batches of the reporting half) counts, with any change to *how* the model is trained (kernels, precision, compilation, data handling, optimizer implementation, learning rate, schedule length, dropout). the architecture, the data and the evaluation stay fixed.
- **timing**: same machine, back to back; the baseline is run again at the end, and every speedup is reported as a range over both baseline runs (and, for the winner, its seeds). the headline is the **same-recipe** speedup (the baseline's exact training, faster); the recipe search's further gain is reported separately, and its own training cost isn't charged.

## configurations

| config | what runs | runtime |
|---|---|---|
| **small** (tests, `verify.py`, offline, any machine without a GPU, or `MLFS_CAPSTONE=small`) | a 1-layer GPT of width 32 on the CPU, 60 baseline steps, no compilation or float16 (on a CPU, float16 autocast is many times slower than float32) | about a minute |
| **full** (your write-up; needs a GPU: `cuda` or `mps`) | model M on `get_device()`: the profile, the baseline (twice), the ablation (11 configurations, 5 windows of 20 steps each after warmup and compilation), the race (2,000 steps with the baseline's recipe; 600, 800 and 1,000 steps at $3 \times 10^{-3}$ without dropout), and 3 seeds of the winner | **52 minutes** measured on an Apple M2 (`mps`) shared with other jobs (load average 5–70 during the run; the two baseline runs alone took 677 s and 881 s). on a quiet machine it's shorter; to fit a tighter budget, drop the 600-step run and the extra seeds |

## deliverables

1. `capstone.py` with the five functions implemented and their checks green:
   - `fused_attention(model)`: patches every block's attention, in place, to call `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)` (attention dropout in training mode only), falling back to the original forward when a cache, a mask, a context, `is_causal=False` or `save_weights` is involved; same outputs and gradients as your chapter 18 attention.
   - `time_steps(step_fn, n_steps, *, warmup=3, repeats=3, sync=None)`: `warmup` untimed calls, then `repeats` windows of `n_steps` consecutive calls, each window bracketed by `sync()` calls; returns the mean seconds per step of each window.
   - `make_train_step(model, optimizer, *, amp_dtype=None, scaler=None, grad_clip=1.0, compile_backend=None)`: returns `step(x, y)`: forward (through `torch.compile(model, backend=compile_backend)` if given, under `torch.autocast` if `amp_dtype`), zero the gradients, backward (scaled, then `scaler.unscale_`, if a scaler is given), clip to `grad_clip` (None: no clipping), `scaler.step` + `scaler.update` or `optimizer.step()`; returns the loss detached, without `.item()`; `step.skipped` counts steps the scaler skipped.
   - `train_fast(model, train_data, block_size, *, val_data=None, batch_size=64, max_steps=1000, lr=1e-3, warmup_steps=100, weight_decay=0.1, grad_clip=1.0, eval_batches=20, device='cpu', seed=0, fused_attn=False, amp_dtype=None, compile_backend=None, fused_optimizer=False, scaler=None)`: `train_lm`'s recipe exactly (the same batches for the same seed, AdamW groups and betas, schedule, clipping), with the data copied to the device once, batches indexed there, no evaluation during training, and one switch per optimization; for float16 it uses `scaler` or a new `mlfs.systems.GradScaler`. returns `{"loss": per-step training losses, "lr": per-step learning rates, "wall": seconds (synchronized), "val_loss": estimate_loss on val_data with eval_batches (NaN without), "skipped": int}` and leaves the model in eval mode. with every switch off it must reproduce `train_lm`'s weights.
   - `speedup_ci(base_times, new_times, *, n_boot=2000, alpha=0.05, rng=None)`: `median(base) / median(new)` with a percentile bootstrap interval, resampling the two samples independently.
2. the full-config run, and the write-up at the end of the notebook answering its five prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass |
| profile | a per-op profile of the baseline step on the CPU and a per-part timing on your device, with the memory-bound ops identified |
| ablation | every optimization alone, all together, and all but one, each with a bootstrap interval |
| race | the fastest run that reaches the target, with 3 seeds, against two baseline runs |
| honesty | if the speedup is below 5×, a measured breakdown of where the remaining time goes and what limits it on your hardware |

## what the reference gets

measured on an Apple M2 laptop (`mps`) shared with other jobs, in the final full run (load average 5–70 during it); earlier complete runs on other days are quoted where they differ. treat any single timing as ±30%: the two baseline runs in the same notebook took 677 s and 881 s.

**the profile.** on the CPU (batch 8), matrix products are only 26% of the step's self time; the rest is dropout's random masks (`bernoulli_`, 15–21% across runs), copies from reshapes and transposes (`copy_`, 7–12%), your attention's causal mask (`masked_fill_`, 5–6%), GELU and its backward, and AdamW's many small per-parameter kernels. on `mps` (batch 64), timed piece by piece: 80–127 ms forward, a further 120–170 ms backward, 6–23 ms of optimizer, under 1 ms to sample and copy a batch. those pieces, measured separately and at different moments, don't add up exactly to the 328–356 ms full-step medians of the ablation: a full step also zeroes gradients and clips them, and the machine's load drifts between measurements (the GPU slowed by a third between this run's two baselines). topic 01's analysis puts the softmax, layernorms, GELU, residual adds and the attention-score matmul left of the ridge: memory-bound.

**the ablation** (ms per step, median of 5 windows of 20 steps; speedup over the baseline step, 95% bootstrap interval over windows):

| configuration | ms per step | speedup |
|---|---|---|
| baseline step | 328 | 1.00× |
| + on-device batches only | 340 | 0.97× [0.94, 1.02] |
| + fused attention only | 260 | 1.26× [1.21, 1.33] |
| + fused AdamW only | 340 | 0.96× [0.94, 1.02] |
| + fp16 + your `GradScaler` only | 317 | 1.04× [1.01, 1.09] |
| + fp16 + `torch.amp.GradScaler` only | 338 | 0.97× [0.95, 1.03] |
| + `torch.compile` only | 201 | **1.63×** [1.58, 1.72] |
| **all (fp16 with your `GradScaler`)** | **174** | **1.89×** [1.85, 1.99] |
| all − fp16 | 200 | 1.64× [1.59, 1.73] |
| all − `torch.compile` | 272 | 1.21× [1.18, 1.27] |
| all, fp16 with `torch.amp.GradScaler` | 181 | 1.81× [1.76, 2.37] |

`torch.compile` is the one optimization that matters on its own; fused attention helps alone (1.26×) and is mostly subsumed once the compiler fuses the attention's elementwise ops; on-device batching and fused AdamW are within noise for a model this small. float16 on its own is roughly neutral (a GPU without tensor cores gains only fewer bytes, and autocast adds casts), and adds about 15% on top of compilation. the scaler matters less than it did in earlier runs: in those, your `GradScaler` checked every gradient for inf/NaN with a separate `.all()` per parameter, each one a CPU–GPU sync, and fp16 with it was 0.85× alone. after changing `unscale_` to collect the checks on the GPU and sync once per step, it's on a par with `torch.amp.GradScaler`, which (with a fused optimizer) avoids even that one sync. one sync per step remains, because the contract has `step()` return a Python boolean. earlier full runs ranked the configurations the same way, with "all" at 1.36–1.79× depending on the scaler version and the load.

**the race** (target 1.5860 nats/char on the reporting half; baselines 677 s and 881 s; speedups are ranges over the two):

| run | steps | lr | dropout | wall | reaches target | speedup |
|---|---|---|---|---|---|---|
| **same recipe (systems only)** | 2,000 | $10^{-3}$ | 0.1 | 241 s | yes (1.5751) | **2.81–3.66×** |
| lr 3×, no dropout | 600 / 800 / 1,000 | $3 \times 10^{-3}$ | 0 | 109 / 159 / 194 s | only 1,000 (1.5799) | 3.48–4.53× (1,000 steps) |

- **headline: the systems changes make each step 1.89× faster [1.85, 1.99]** (back to back, same machine state). in the race the same training finished 2.8–3.7× sooner; that's more than the per-step figure because the baseline's wall clock also includes `train_lm`'s evaluation every 250 steps (40 extra forward batches each time), its per-step CPU-side batch sampling, and, in this run, a GPU that was slower during the baseline runs than during the race. in earlier runs on quieter days the same-recipe speedup was 1.8–1.9×. load drift isn't a systems speedup, so treat 1.89× per step as the systems factor and the race range as context: it depends mostly on how loaded the machine is when the baseline runs.
- **recipe factor, on top: 1.04–1.24×.** the search picked 1,000 steps at $3 \times 10^{-3}$ without dropout, the shortest run reaching the target on the selection half. over seeds 0, 1, 2: wall 194–231 s, reporting-half loss 1.5634–1.5810, all below the target. the search itself cost 462 s of training, not charged above.
- **total: 2.9–4.5×** against the two baselines. an earlier full run, with dropout kept in the search, measured 3.0–3.9×; one with a dropout-free search, 3.0–4.2×.

**5× was not reached reliably on this hardware**: the upper end touches 4.5× only against the slower baseline run. what stands in the way:

- **no tensor cores**: the 2–8× that 16-bit matmuls give on NVIDIA GPUs isn't available; float16 is worth about 15% here once compiled, and bfloat16 is emulated (topic 03).
- **a small, overhead-bound model**: the projections are $(8192, 192) @ (192, 192)$ matmuls, too small to fill the GPU, and a step is still dozens of kernel launches after compilation. the final configuration reaches about $6 \times 1.8\text{M} \times 8192 / 0.174\,\text{s} \approx 0.51$ TFLOP/s, about 19% of the 2.7 TFLOP/s float32 peak topic 01 measured on a quiet day (more if you count the attention FLOPs $6N$ leaves out).
- **no flash-attention kernel on `mps`**: `scaled_dot_product_attention` there removes some overhead but still materializes the scores; on CUDA it would dispatch to FlashAttention.
- **the recipe side** has more room than this search explored (warmup, schedule shape, batch size), but each probe costs minutes, and the 800-step run missed the target by 0.02 nats, so the true minimum lies between grid points.

on an NVIDIA GPU with tensor cores, the same switches (bf16 autocast, FlashAttention through SDPA, `torch.compile`, fused AdamW) are the ones nanoGPT and modded-nanoGPT rely on, and the two this laptop can't use (tensor-core 16-bit matmuls and FlashAttention) are where the large step-time gains are on that hardware, so 5× in time to target is plausible there; this course hasn't measured it. on this laptop, the honest answer is about 3–4.5×.

## stretch goals

- **PyTorch's flash attention path on CUDA.** on an NVIDIA GPU, `F.scaled_dot_product_attention` dispatches to FlashAttention; rerun the ablation there and compare its contribution with the one you measured.
- **a CUDA-graph or `mode="reduce-overhead"` compile.** remove the per-step launch overhead altogether, where your backend supports it.
- **a larger batch with a re-tuned learning rate.** is there a batch size at which tokens per second rise faster than the loss per token falls? (chapter 19's critical batch size.)
- **more of the step on the device.** write the sampling of batch offsets so nothing at all crosses to the CPU during training, and measure what that buys.
- **a second baseline seed.** how much of the speedup range is the baseline's own seed-to-seed variation?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Karpathy (2022), nanoGPT, whose `train.py` combines fused attention, `torch.compile`, mixed precision and fused AdamW. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Dao, Fu, Ermon, Rudra & Ré (2022), "FlashAttention: fast and memory-efficient exact attention with IO-awareness". [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Ansel et al. (2024), "PyTorch 2: faster machine learning through dynamic Python bytecode transformation and graph compilation", *ASPLOS*. [doi:10.1145/3620665.3640366](https://doi.org/10.1145/3620665.3640366)
- Micikevicius et al. (2018), "Mixed precision training", *ICLR*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Jordan et al. (2024), modded-nanoGPT, a public record of speedups on a fixed GPT-2 training target, each optimization measured. [github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- He (2022), "Making deep learning go brrrr from first principles". [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)
