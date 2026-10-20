# data parallelism · lab notes

numbers below are from the reference solution on an Apple M2 laptop (8 cores: 4 performance, 4 efficiency), run several times on different days while other jobs shared the machine: once at a load average around 10, once around 40. every rank is a process on that one machine, talking through gloo over shared memory. the shapes held in both runs; the individual timings didn't, so both are given where they differ.

## experiment 1: what an all-reduce costs

**what people predict.** that 100 small all-reduces of 4 KB cost about the same as one of 400 KB, since the bytes are the same, or at most 2–3× more.

**what happens.** the fit of your `allreduce_time` (least squares on relative error) gives a latency of **$\alpha \approx 110$–$160\,\mu$s** per message and a bandwidth of **$\beta \approx 1.3$–$1.5$ GB/s**, and follows the measurements from 1 KB to 16 MB within about 30% (an unweighted fit, tried first, matched the large messages and missed the small ones by 2–3×). below roughly 100 KB an all-reduce costs the same whatever its size: pure latency. the same 400 KB:

| ranks | 100 × 4 KB | 1 × 400 KB | ratio |
|---|---|---|---|
| 2 | 24–28 ms | 0.27–0.29 ms | **84–104×** |
| 4 | 91–153 ms | 1.0–1.6 ms | **87–94×** |

**why.** with the ring model, each small all-reduce costs $2(p-1)\alpha$ plus almost nothing for its bytes: at $p = 2$, about $2 \times 120\,\mu\text{s} = 0.24$ ms, and 100 of them about 24 ms; the one big all-reduce pays the latency once and moves its 400 KB at $\beta$. that's the case for buckets: a GPT has dozens of parameter tensors, many of them tiny (biases, layer-norm weights), and reducing each separately would pay the latency dozens of times per step. for large messages the ring model says the time should hardly depend on $p$; here 4 ranks are about twice as slow as 2 at 16 MB, because on one machine the "links" share the same memory system and cores rather than each rank having its own network card. on a real cluster the per-message latency is network latency (microseconds on InfiniBand) and the bandwidth is the slowest link's, but the latency-versus-bandwidth crossover is the same idea.

## experiment 2: does your DDP scale?

**what people predict.** close to 2× for 2 ranks and 3–4× for 4 ranks with one thread each, perhaps a bit less for communication; less with the cores split.

**what happens.** essentially **no speedup at all**.

| setting | ranks | time (load ~10) | time (load ~40) | time in all_reduce |
|---|---|---|---|---|
| 1 thread per rank | 1 | 3.46 s | 5.12 s | 1% |
| 1 thread per rank | 2 | 3.50 s | 6.01 s | 12–29% |
| 1 thread per rank | 4 | 3.77 s | 4.43 s | 13–30% |
| cores split between ranks | 1 | 2.88 s | 3.28 s | 0% |
| cores split between ranks | 2 | 3.42 s | 3.60 s | 13–14% |
| cores split between ranks | 4 | 3.95 s | 4.13 s | 13–14% |

two controls run the same work with no communication at all. control 1 switches every collective of `ddp_train` off: at loads of 35–50, 2 ranks took 3.0–4.4 s against an ideal 2.1 s, and 4 ranks 3.0–3.9 s against an ideal 1.0 s, nearly as slow as with communication. control 2 runs a fixed single-threaded workload as 1 process and as 4 processes started together: 512³ matmuls reached a total throughput of only **1.2–2.3×** across runs (1.6× on a quieter machine), while a cache-resident elementwise job (`exp` and a multiply on 64K floats) reached 2.9× under heavy load and nearly 4× when the machine was quiet.

**why.** communication is 12–30% of each rank's time (more under load), roughly what the fitted model predicts: a 5 MB gradient between 2 ranks costs about $2(\alpha + 2.6\,\text{MB}/\beta) \approx 4$ ms per step against 15–20 ms of compute. so communication isn't what kills the speedup; the controls show what does. the MLP's time is matrix products, and on an Apple-silicon Mac PyTorch hands every CPU matmul to Apple's Accelerate library, which runs it on the **AMX matrix coprocessor**: one matrix unit per cluster of cores, not one per core. a single thread already drives it at close to a TFLOP/s for a 512³ product, about ten times what one core's vector units could do (topic 01's lab measured the CPU's "peak" at about 1.1 TFLOP/s). four processes doing matmuls are four clients of the same unit, so they mostly take turns: adding ranks adds no matrix throughput. elementwise work runs on each core's own NEON units and does scale with processes, until the machine's other jobs take the cores. so strong scaling on this laptop measures a shared matrix unit, not data parallelism. the "cores split" setting gains nothing either, because its 1-rank run already had the whole AMX unit. on a Linux or x86 machine (and on a cluster, where each rank has its own GPU), per-rank compute doesn't shrink as ranks are added, and you'd see real scaling, with the efficiency set by the communication-to-compute ratio your `ddp_efficiency` models: 12–16% exposed communication would mean about 85% efficiency, and overlapping the all-reduce with the backward pass (DDP's buckets) would hide most of it.
