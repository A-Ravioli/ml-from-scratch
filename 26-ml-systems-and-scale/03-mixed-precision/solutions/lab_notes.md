# mixed precision · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU with 4 threads for experiments 1–2; `mps` GPU for experiment 3), one run, seed fixed, other jobs sharing the machine. experiments 1 and 2 are deterministic up to thread scheduling; experiment 3's timings move by 10–20% between runs.

## experiment 1: where gradients live

**what people predict.** that a few percent of gradients would underflow, mostly the tiny ones in a long tail.

**what happens.** after 30 training steps, one float32 backward pass of the 2-layer GPT (batch 32 × 64 positions):

| gradients | loss scale | flushed to 0 | subnormal (imprecise) | overflow |
|---|---|---|---|---|
| parameter gradients | 1 | 0.15% | 33.6% | 0 |
| parameter gradients | 65,536 | 0.01% | 0.11% | 0 |
| activation gradients | 1 | 0.23% | **99.7%** | 0 |
| activation gradients | 65,536 | 0.00% | 0.01% | 0 |

few values are flushed to zero outright, but essentially *every* activation gradient lands in float16's subnormal range, where precision degrades (a value near $2^{-20}$ keeps only 4 of float16's 10 mantissa bits). scaling by $2^{16}$ moves the whole distribution into the normal range, and the largest scaled gradients (parameter gradients near $2^{-5}$, so about $2^{11}$) still sit about 5 binades below float16's max.

**why.** the loss is a mean over $B \times T = 2{,}048$ positions, so $\partial L / \partial h$ for any single position carries a factor $1/2048 \approx 2^{-11}$ on top of the per-token gradient (itself $\sim 2^{-4}$–$2^{-8}$). parameter gradients sum those contributions over all positions, which brings them back up by roughly the same factor. that's why the activation-gradient histogram sits 4–6 binades to the left of the parameter one, centred near $2^{-17}$, just below float16's smallest normal $2^{-14}$. bigger batches and longer contexts push it further left. a scale of 65,536 still leaves room: the largest scaled gradients are around $2^{11}$, against a max of $2^{16}$. (in experiment 2 the scaler settles lower, near $2^{14}$–$2^{17}$, because gradients early in training are larger than these, measured after 30 steps.)

## experiment 2: float16 overflow, seen live

**what people predict.** that a scale of $2^{32}$ causes a handful of skipped steps; many expect training to stall or diverge.

**what happens.** the first **15 steps are all skipped** (the loss curve is flat there), the scale halving each time from $2^{32}$ to $2^{17}$; the first real update is at step 15. a few steps later, three more overflows push the scale down to $2^{14}$, and from then on it grows every 40 clean steps ($2^{15}$, $2^{16}$, $2^{17}$) without another overflow in 160 steps: 18 skipped steps in all. final 20-step mean training losses: float32 2.456, bfloat16 2.453, float16 + `GradScaler` 2.500. the float16 run is behind only because it started 15 steps late; its curve has the same shape, shifted right.

**why.** at $2^{32}$ the scaled gradients (about $2^{-17} \cdot 2^{32} = 2^{15}$ for typical activation gradients, much more for the large ones) overflow float16's 65,504 immediately. each skip halves the scale, so it takes $\log_2(2^{32}/2^{17}) = 15$ skips to reach a scale at which nothing overflows: the cost of a bad initial scale is logarithmic, which is why the default can start high. the early extra overflows come from the gradients being largest at the start of training. the float32 and bfloat16 curves are indistinguishable at this scale (differences of 0.003 are within one seed's noise); a single seed can't separate them, and doesn't need to: the point is that bfloat16 trains without any scaler, and float16 trains with one.

## experiment 3: is 16-bit faster here?

**what people predict.** a 2× or larger speedup from 16-bit, since that's what tensor cores give, or at least something from halving the bytes.

**what happens.**

| device | matmul | fp32 | fp16 | bf16 | GPT step fp32 | fp16 + your scaler | bf16 |
|---|---|---|---|---|---|---|---|
| cpu | 1024³ | 1,380 GFLOP/s | **3** GFLOP/s | **3** GFLOP/s | – | – | – |
| mps | 2048³ | 2,859 GFLOP/s | 3,134 GFLOP/s | 1,541 GFLOP/s | 73 ms | 81 ms | 85 ms |

on this machine's CPU, 16-bit matmuls are 460× *slower* than float32. on its GPU, float16 matmuls are 10% faster and bfloat16 ones about half the speed. a full training step of model M (batch 32) is slower in both 16-bit modes than in float32.

**why.** the M2's CPU cores have no 16-bit matrix arithmetic that PyTorch uses, so 16-bit matmuls fall back to a slow scalar path. its GPU has no tensor cores: float16 runs through the same ALUs at about the same rate as float32 (the 10% is fewer bytes moved), and bfloat16 isn't native on M1/M2 GPUs at all, so it's emulated through conversions. in the training step, the small savings in memory-bound ops are outweighed by autocast's casts (every weight is cast to 16-bit at every use) and by your `GradScaler`'s per-step work: checking every gradient for inf/NaN and returning a Python boolean forces the CPU to wait for the GPU every step. so on this hardware, 16-bit training is a memory saving, not a speedup, unless something else removes the casts and the sync (the capstone measures whether `torch.compile` and PyTorch's own sync-free `torch.amp.GradScaler` change that). on an NVIDIA GPU with tensor cores the same step would be several times faster in bf16. measure before deciding: the right answer depends entirely on the hardware.
