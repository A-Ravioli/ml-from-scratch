# 26 · ML systems & scale

> make models fast and fit them onto hardware.

chapter 18's GPT trained in a few minutes on a laptop, and a GPU vendor's spec sheet says it should have taken one. this chapter is about the gap between the arithmetic a model needs and the time and memory it actually takes, and about the techniques that close it, from one laptop up to the thousands of GPUs behind every frontier model. you'll learn to predict an operation's speed from two numbers (its FLOPs and its bytes), measure your own machine's limits, write GPU kernels that fuse and tile, train in 16-bit without losing accuracy, spread training over many processes and shard what doesn't fit, and shrink a model for serving. the capstone puts all of it to work on your own GPT: make it reach the same loss five times faster, and prove every step of the way with a measurement.

**you will build:** a roofline calculator and a GPT-block FLOP/byte table, a benchmark and profiler wrapper (`mlfs.systems`); Triton-style fused softmax and tiled matmul kernels on a traffic-counting interpreter, and a real Metal GPU kernel; a float-format simulator and a dynamic loss scaler (`GradScaler`); a ring all-reduce simulation and data-parallel training across real processes; memory accounting, activation checkpointing, a tensor-parallel MLP, a pipeline schedule and a ZeRO-1 sharded AdamW; int8 quantization, an int8 linear layer, magnitude pruning and the distillation loss. all of `mlfs.systems`.
**prerequisites:** chapter 18 (your GPT, `train_lm`, KV cache and online softmax), chapter 15 (training loops, `get_device`, schedulers), chapter 05 (bootstrap intervals), chapter 00 · 03 (floating point).
**time · scaffolding:** L4 · ~80 h. the lessons' **spec** boxes and the checks are the contract; internal structure is yours.
**newly unlocked library components:** `torch.distributed` collectives and `DistributedDataParallel` (after 04), `torch.autocast` and `torch.amp.GradScaler` (after 03), `torch.utils.checkpoint` and FSDP (after 05), `torch.compile`, `F.scaled_dot_product_attention` as a fused kernel and fused optimizers (after 02), `torch.ao.quantization` (after 06). (build-before-you-use, see AUTHORING.md §9)

**hardware.** everything runs on a laptop CPU, and the chapter was written and measured on an Apple-silicon Mac (M2, 8 cores, unified memory) whose GPU PyTorch reaches through `mps`. there's no CUDA there, so two adaptations: topic 02 teaches kernels on a NumPy interpreter that mirrors Triton line for line (Triton needs an NVIDIA or AMD GPU; its code is shown alongside) plus one real kernel in Metal, Apple's GPU language, via `torch.mps.compile_shader`; and topics 04–05 run data-parallel and ZeRO training across CPU processes with the gloo backend. every timing in the lessons, lab notes and capstone is measured on that machine, shared with other jobs, and reported with its spread: yours will differ, and the ratios are what to compare. one quirk of that machine colours several labs: PyTorch runs CPU matrix products on Apple's AMX matrix coprocessor (through the Accelerate library), a unit shared by all cores, so one thread already gets about 1 TFLOP/s and adding threads or processes adds no matrix throughput. on Linux or x86, CPU matmuls scale with cores in the ordinary way, and a CUDA GPU is supported by every lab and the capstone.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build |
|---|---|---|
| 01 | [the performance model](01-the-performance-model/) | FLOP/byte counts, the roofline, honest benchmarking, a GPT-block intensity table, MFU, a profiler wrapper |
| 02 | [writing kernels](02-writing-kernels/) | fused softmax and tiled matmul as Triton-style block programs (counted traffic), and a fused softmax in Metal |
| 03 | [mixed precision](03-mixed-precision/) | a float-format simulator, master weights, `GradScaler`, a mixed-precision step with autocast |
| 04 | [data parallelism](04-data-parallelism/) | ring all-reduce simulation, communication cost model, gradient accumulation, data sharding, DDP over gloo |
| 05 | [model parallelism & memory](05-model-parallelism-and-memory/) | memory accounting, activation checkpointing, a tensor-parallel MLP, GPipe schedule, ZeRO-1 `ZeroAdamW` |
| 06 | [efficient inference](06-efficient-inference/) | int8/k-bit quantization, `Int8Linear`, magnitude pruning, distillation loss, the decoding roofline |

## capstone

*make your GPT 5× faster to train*, with every optimization profiled and documented. you profile chapter 18's trainer, ablate fused attention, fused AdamW, on-device batching, float16 with your `GradScaler`, and `torch.compile` one at a time and all together (with bootstrap intervals on every speedup), then race your `train_fast` against `train_lm` to the same validation loss. see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it.

## notation

| symbol | meaning |
|---|---|
| $P$ | peak arithmetic rate (FLOP/s) |
| $W$ | memory bandwidth (bytes/s) |
| $I$, $I^* = P/W$ | arithmetic intensity (FLOP/byte), ridge point |
| $N$, $D$ | parameters, training tokens ($C \approx 6ND$) |
| $B$, $T$, $d$, $H$, $d_h$, $d_\text{ff}$ | batch, context length, width, heads, head width, MLP width |
| $E$, $M$ | exponent and mantissa bits of a float format; $\varepsilon = 2^{-M}$ |
| $S$ | loss scale |
| $p$ | number of ranks (world size); $r$ a rank |
| $\alpha$, $\beta$ | message latency (s) and link bandwidth (bytes/s) |
| $\Psi$ | parameter count in memory formulas (ZeRO's notation) |
| $m$ | micro-batches per batch (pipelines) |
| $s$, $q$, $q_\text{max}$ | quantization scale, integer code, largest code $2^{b-1} - 1$ |
| $T$ (in distillation) | softmax temperature |

## how this connects

- builds on: matmul cost (ch 02), floating point (ch 00 · 03), the bootstrap (ch 05), training loops and schedules (ch 15), your GPT, KV cache and online softmax (ch 18).
- used later in: chapter 27's large language models (sized with `training_memory`, trained in bf16, quantized and distilled, served with batching and speculative decoding), chapter 19's scaling laws (written in $6ND$), and every capstone that needs to run faster than it does.

## core references

- Williams, Waterman & Patterson (2009), "Roofline: an insightful visual performance model for multicore architectures", *CACM*. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
- Micikevicius et al. (2018), "Mixed precision training", *ICLR*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Rajbhandari, Rasley, Ruwase & He (2020), "ZeRO: memory optimizations toward training trillion parameter models", *SC20*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Shoeybi et al. (2019), "Megatron-LM: training multi-billion parameter language models using model parallelism". [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Dao, Fu, Ermon, Rudra & Ré (2022), "FlashAttention: fast and memory-efficient exact attention with IO-awareness". [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Hinton, Vinyals & Dean (2015), "Distilling the knowledge in a neural network". [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
