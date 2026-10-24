# fine-tuning · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, 4 threads), tiny Shakespeare from the cache, a byte-level `Llama` (3 layers, width 96, 329,472 parameters) pretrained for 800 steps (held-out loss 1.901 nats per byte), then 300 fine-tuning steps of 32 examples per run. the instruction data is **synthetic** (`make_tasks`). the notebook runs one seed; the spreads quoted below come from three seeds per configuration (different LoRA initializations and batch orders), run separately. offline (`MLFS_OFFLINE=1`) the lab runs 200 + 80 steps on synthetic text, and its numbers mean little.

## experiment 1: rank against quality

**what people predict.** either that LoRA is always nearly as good as full fine-tuning (the headline of the LoRA paper), or that more rank is always better.

**what happens.**

| task | run | trainable parameters | response loss | exact match | Shakespeare loss |
|---|---|---|---|---|---|
| both | base model | 0 | 1.737 (continue), 6.458 (letters) | 0.00 | 1.901 |
| continue | full | 329,472 | 1.628 | – | 2.374 |
| continue | LoRA r=1 | 5,184 | **1.556** | – | **2.085** |
| continue | LoRA r=4 | 20,736 | 1.553 | – | 2.112 |
| continue | LoRA r=16 | 82,944 | 1.566 | – | 2.146 |
| letters | full | 329,472 | **0.046** | **0.97** | 4.956 |
| letters | LoRA r=1 | 5,184 | 1.439 | 0.35 | 5.327 |
| letters | LoRA r=4 | 20,736 | 0.830 | 0.32 | 5.926 |
| letters | LoRA r=16 | 82,944 | 1.048 | 0.35 | 5.509 |

on **continue**, every LoRA rank beats full fine-tuning on held-out responses (1.55–1.57 against 1.63) and damages Shakespeare less (2.09–2.15 against 2.37). three seeds agree closely: full 1.627–1.628 and 2.37–2.42; rank 1 1.545–1.563 and 2.05–2.06; rank 16 1.549–1.564 and 2.12–2.15. rank makes no difference.

on **letters**, full fine-tuning answers 95–98% of held-out prompts exactly (three seeds) and LoRA doesn't come close in 300 steps. LoRA is also erratic: over the notebook's run and three more seeds, rank 1 reaches 17–35% exact (loss 1.43–1.57) and rank 16 23–83% (loss 0.46–1.05); the notebook's rank-16 run landed at the low end. lowering LoRA's learning rate to $10^{-3}$ or $3 \times 10^{-4}$ makes it worse (rank 16: 17–28% and 10–17%). every run that learns the letters task wrecks the Shakespeare loss (4.5–6.3 nats from 1.9), LoRA included; only the runs that barely learn it (low learning rate) keep it near 3.5–4.6.

**why.** the two tasks need different things from fine-tuning. *continue* asks for something the base model already does (continue Shakespeare); fine-tuning only has to teach it the template and when to stop, a small, low-dimensional change, so rank 1 suffices, and full fine-tuning, with 64× more freedom, overfits its 3,000 examples a little and drifts further from the pretrained model. this is the regime the LoRA paper and LIMA's "superficial alignment" describe. *letters* asks for a skill the base model doesn't have (map each lowercase byte to its uppercase byte, count characters); that needs real changes to how the network computes, which a rank-limited update, starting from zero, learns slowly and with high variance. this mirrors Biderman et al. (2024), who found LoRA well behind full fine-tuning on code and math (new skills) while forgetting less. the forgetting half reproduces here only on *continue*: on *letters*, LoRA runs that learned anything forgot as much as full fine-tuning, because the task's outputs (capitals, short answers, an end-of-text token after a few bytes) pull hard against Shakespeare-style prose, and in a 3-layer model there's no spare capacity to keep both. before giving up on LoRA for the letters task you'd try more steps, a learning-rate sweep with warmup, adapting the embeddings and head too (PEFT's `modules_to_save`), and averaging over seeds.

## experiment 2: QLoRA and merging

**what people predict.** that 4-bit weights cost a lot of quality, and NF4 visibly less than a uniform grid.

**what happens.**

| base weights | relative squared weight error | Shakespeare loss |
|---|---|---|
| float32 | 0 | 1.901 |
| NF4 (blocks of 64) | 0.85% | 1.917 |
| uniform 4-bit (blocks of 64) | 1.17% | 1.915 |

NF4's weight error is 27% lower than the uniform grid's (on random normal weights the ratio is the same: 0.85% against 1.16%), but both cost the model only 0.015 nats, and their losses are indistinguishable. LoRA rank 16 on the letters task reached response loss 0.520 and 65% exact on the float32 base, and 0.763 and 55% on the NF4 base: inside the seed spread of experiment 1 (23–83%), so this single comparison can't separate them. merging changed the logits by at most $3.6 \times 10^{-5}$ (float32 round-off), removed 82,944 parameters (412,416 → 329,472), and cut the time for 60 greedy tokens from 42–43 ms to 27–30 ms (two runs).

**why.** this model's trained weights are close to normal (excess kurtosis 0.49), so quantiles of a normal distribution fit them better than an evenly spaced grid: more levels near zero where most weights are. but at 4 bits with blocks of 64, both grids are already fine enough that the rounding error is a small perturbation for this model; NF4's advantage matters more for larger models, smaller budgets (3 bits) or larger blocks, and QLoRA's paper measured it on 7–65B models. LoRA learns on a quantized base because the gradient flows *through* the frozen (dequantized) weights into $A$ and $B$, which stay in full precision: the 4-bit base only shifts the function the adapters correct. merging a QLoRA model gives a full-precision matrix $W_{\text{NF4}} + sBA$: quantizing it again would round away much of the small update, so QLoRA models are either served with the adapter kept separate or merged into a 16-bit copy of the base. the 13–15 ms saved by merging is the cost of 21 extra pairs of thin matmuls per token (7 adapted layers in each of 3 blocks), which at this tiny size is mostly per-call overhead.
