# modern LLM architecture · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, 4 threads), tiny Shakespeare characters from the cache, 400 steps of 32 × 64 per model, one seed (seed 0) in the notebook. seeds 1 and 2, run separately at lr 3e-3, are quoted for the spread. offline (`MLFS_OFFLINE=1`) the text is synthetic and the losses mean nothing in particular.

## experiment 1: GPT-2 recipe against Llama recipe

**what people predict.** a small win for the Llama (0.01–0.05 nats), or no difference at this size.

**what happens.**

| model | parameters | lr 1e-3 | lr 3e-3 | lr 1e-2 |
|---|---|---|---|---|
| GPT (ch 18) | 108,352 | 2.422 | 2.266 | **2.060** |
| Llama, n_kv_heads=4 | 104,832 | 2.193 | **2.009** | 2.033 |

at lr 3e-3 the Llama is **0.26 nats** better, a huge gap, and it's stable across seeds (GPT 2.266, 2.276, 2.315; Llama 2.009, 2.011, 2.004). but the GPT keeps improving as the learning rate rises (in separate runs, 2.064 at 2e-2 and 2.41 at 4e-2), while the Llama's best is around 3e-3 (2.033 at 1e-2, 2.055 at 2e-2). **compared at their own best learning rates, the gap is 0.05 nats.**

**why.** most of the 0.26 nats at a shared learning rate is that the GPT wants a larger one: its learned position embeddings and LayerNorm biases start at zero-ish values and need big steps to become useful in 400 steps, whereas RoPE gives the Llama positional information from step 1, and RMSNorm without biases leaves fewer parameters to tune. what remains, about 0.05 nats at equal size and compute, is the real recipe gain in this setting, of the order the papers report for each component separately (Shazeer, 2020, measured SwiGLU's gain in log-perplexity at the hundredths level). the lesson is a methodological one that recurs in every architecture paper: a comparison at one shared hyperparameter setting measures how well each model fits *that setting*. to rank architectures, tune each (at least its learning rate), use several seeds, and ablate one change at a time.

## experiment 2: how many key/value heads do you need?

**what people predict.** either "nothing lost" or a large loss for MQA.

**what happens.**

| key/value heads (of 4) | parameters | validation loss | toy cache per sequence | 7B-class cache per 4,096-token sequence |
|---|---|---|---|---|
| 4 (MHA) | 104,832 | 2.009 | 64 KiB | 2.00 GiB |
| 2 (GQA) | 96,640 | 2.023 | 32 KiB | 1.00 GiB |
| 1 (MQA) | 92,544 | 2.038 | 16 KiB | 0.50 GiB |

each halving of the key/value heads costs about 0.015 nats here; the ordering is the same in seeds 1 and 2 (2.011, 2.032, 2.047 and 2.004, 2.024, 2.043), with differences of 0.014–0.021 nats per halving, larger than the seed spread. the cache halves exactly each time.

**why.** keys and values are a summary of each token for the queries to look up; different query heads ask different questions of a largely shared summary, so a few key/value heads carry most of what's needed. part of the small loss here is also fewer parameters (the GQA model has 8% fewer). the cache scales linearly in the number of key/value heads, and at serving scale it's the cache that limits batch size: at batch 32 and 32k tokens the 7B-class MHA cache would be $32 \times 32{,}768 \times 0.5$ MiB = 512 GiB, far beyond any single GPU, and 64 GiB with 8 key/value heads. that's why GQA with 8 key/value heads became the default for large models (Llama 2 70B, Llama 3, Mistral): a loss of quality of this size buys a 4–8× larger batch.

## experiment 3: load balancing

**what people predict.** that without balancing one or two experts take everything, and that balancing costs loss.

**what happens.** effective experts at the end (layer 0, layer 1) and validation cross-entropy:

| aux coefficient | validation loss | effective experts |
|---|---|---|
| 0 | 2.012 | 6.64, 5.33 |
| 0.01 | 2.037 | 7.83, 7.95 |
| 0.1 | 2.010 | 7.99, 7.99 |

without the auxiliary loss the second layer's router starts badly skewed (about 2.4 effective experts after 20 steps), recovers only partly, and ends with three experts taking 18–32% of the slots each and three with 1–4%. with coefficient 0.01 or 0.1 the routers are balanced within the first 100 steps. seeds 1 and 2 give the same picture (layer 1 at 4.96 and 4.76 effective experts without balancing, 7.92–7.96 with 0.01). the validation losses don't separate: 2.012, 2.037, 2.021 (seeds 0–2) without balancing against 2.037, 2.024, 2.052 with 0.01. the MoE models (226,688 parameters, about 25k active MLP parameters per layer) land at the same loss as experiment 2's dense GQA model (2.023, 34k MLP parameters per layer).

**why.** routing is self-reinforcing: experts learn only from the tokens they receive, so an expert that wins a few tokens early improves on them and wins more. the second layer is hit harder because its inputs are already more specialized. a partial collapse doesn't show in the loss at this scale (the busy experts absorb the work), but it wastes the starved experts' parameters and, in a real deployment where experts live on different devices, makes the busiest device the bottleneck for every batch. the auxiliary loss uses $P_e$ because $f_e$ comes from a top-k selection and has no gradient. as for MoE not winning: sparsity pays off when the model is large enough that more parameters help at fixed compute, and when the hardware can spread experts across devices. with 64-wide experts trained for 400 steps the model is limited by data and steps, not by capacity, so extra (inactive) parameters buy nothing.
