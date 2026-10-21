# preference learning · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, 4 threads), tiny Shakespeare from the cache, a byte-level `Llama` (3 layers, width 96) pretrained for 800 steps and fine-tuned for 300 steps to finish half-lines. the preference labels are **synthetic**: `gold_reward` (real words × distinct words − 0.15 × length mismatch) plays the human, through a Bradley–Terry coin with temperature 0.1. every curve in the notebook is one run; where the notes quote other seeds, they come from a separate script with the same recipe (same SFT model, different reward-model initializations, pair labels and sampling seeds). the whole notebook takes about three minutes; offline it runs a much smaller configuration on synthetic text, whose numbers mean nothing.

## experiment 1: a reward model, and best-of-n against it

**what people predict.** that a reward model which agrees with the labels most of the time makes best-of-n steadily better on the true objective, perhaps with diminishing returns.

**what happens.** 2,106 pairs; the noisy labels agree with the gold rule on 87% of them, and the trained reward model ranks only **59%** of held-out pairs the way their labels do (59–61% over three reward-model seeds).

| n | KL (nats) | proxy | gold, proxy picks | gold, oracle picks |
|---|---|---|---|---|
| 1 | 0.00 | −0.04 | 0.609 | 0.609 |
| 2 | 0.19 | 0.93 | 0.653 | 0.710 |
| 4 | 0.64 | 1.48 | 0.677 | 0.824 |
| 8 | 1.20 | 2.08 | **0.703** | 0.880 |
| 16 | 1.84 | 2.62 | 0.681 | 0.947 |
| 32 | 2.50 | 3.03 | 0.700 | 0.988 |
| 64 | 3.17 | 3.54 | 0.683 | 0.995 |

the proxy score of the chosen answer rises steadily with $n$; its gold score rises until about $n = 4$–$16$ and then stalls or falls, while picking by the gold rule itself keeps climbing to 0.995. in three runs (different reward-model seeds and sample sets) the gold curve peaked at $n$ = 8, 4 and 16 (0.70, 0.73, 0.73) and ended at $n = 64$ at 0.68, 0.63 and 0.68: below its peak every time, while the proxy rose monotonically.

**why.** best-of-n with a perfect reward would only improve; with a proxy it selects the answers with the largest *proxy* score, which at large $n$ increasingly means the answers with the largest *proxy error*: the more candidates you look through, the more likely the winner is one the reward model overrates. a 59%-accurate proxy is a weak signal, so the gains saturate early; Gao et al. (2022) found the same shape, with larger reward models (and more data) pushing the peak to higher KL. the gap to the oracle curve is how much a better reward model could buy at the same KL.

## experiment 2: RLHF with PPO

**what people predict.** that PPO without a KL penalty learns faster and ends better, or that it gets worse "a bit".

**what happens.**

| beta | iteration | proxy | gold | KL (nats) | a held-out answer |
|---|---|---|---|---|---|
| 0 | 0 | 0.05 | 0.63 | 0.0 | `rough stard whine domen` |
| 0 | 20 | 0.85 | 0.66 | 7.8 | `root to him then; I` |
| 0 | 30 | 1.69 | 0.65 | 13.8 | `to this thou thee thee!` |
| 0 | 40 | 2.23 | 0.40 | 24.4 | `thee thy thee thee thee` |
| 0 | 50 | **2.84** | **0.30** | 26.8 | `thee thy thee thee thee` |
| 0.1 | 30 | 1.21 | 0.71 | 4.9 | |
| 0.1 | 50 | 1.21 | **0.71** | 5.5 | |

without a leash the proxy reward climbs all the way, while the gold reward is flat for about 15 nats of KL and then collapses as the policy settles on repeating *thee*. with $\beta = 0.1$ (on standardized rewards) the KL stops near 5 nats and the gold reward ends higher (0.71–0.75 over the last iterations). in two runs of the separate script with the same learning rate, the unleashed policy collapsed the same way, to "the the the" and "it it it it" (gold 0.17 at KL 25–26). with half the learning rate, two of four runs had collapsed within 50 iterations (at KL 19 and 39) and two hadn't yet (KL 13–14): when, and whether within a given budget, the collapse comes depends on the run; in every run the KL kept growing and the proxy kept rising, which is the precondition for it.

**why.** the reward model was trained on answers from the SFT policy, none of which repeated one word over and over. its score for such strings is an extrapolation, and here it extrapolates upwards: short, common, valid-looking words score well. PPO finds the direction in which the proxy keeps rising and follows it far outside the reward model's training distribution; the gold rule, which penalizes repetition, shows what the proxy can't. the KL penalty stops the policy where the proxy is still trustworthy, at the cost of leaving some proxy reward on the table. if you only watched the proxy curve (right plot), the collapsed policy would look like the best checkpoint: that's why RLHF runs are monitored with independent evaluations (held-out reward models, human ratings) and stopped early.

## experiment 3: DPO, no reward model at all

**what people predict.** that DPO, having no reward model to exploit, can't over-optimize.

**what happens.**

| beta | best gold (KL) | at the end of 150 steps (KL) |
|---|---|---|
| 0.1 | 0.73 (4.0–5.9) | 0.65 (7.1) |
| 0.01 | 0.72 (3.5) | 0.47 (24.8) |

both start by improving the gold reward about as much as best-of-n and PPO did (to 0.72–0.73 at 3–6 nats). with $\beta = 0.01$ the KL keeps growing and the gold reward falls to 0.47; in a second run with other seeds, $\beta = 0.01$ went much further (KL 86 nats by step 150, gold 0.25, outputs full of invalid bytes), while $\beta = 0.1$ stayed at 5.4 nats with gold 0.72.

**why.** DPO's loss only asks that the chosen answer gain log-probability *relative* to the rejected one, and its gradient (on paper 2) has no term that keeps either probability up. on a fixed dataset the easiest way to keep widening the margin is to push the rejected answers' probability down without limit, which moves probability mass to sequences in neither set; a small $\beta$ means the same margin requires a larger log-ratio, so the policy travels further from the reference. the "reward model" being exploited is DPO's own implicit one, fitted to 1,790 noisy pairs. at their best, all three methods land at a similar gold reward (0.70–0.75) at a few nats of KL: in this setting the choice of method matters less than knowing when to stop, which needs a measurement the optimizer doesn't see (Rafailov et al., 2024, measured this for direct alignment methods at scale).
