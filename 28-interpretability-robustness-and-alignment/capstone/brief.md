# 28 · capstone: reverse-engineer a circuit

## goal

train a one-layer transformer to compute $(a + b) \bmod p$, watch it **grok** (memorise its training pairs within a few hundred steps, then generalise to the unseen pairs thousands of steps later), and then explain *how* it adds. Nanda et al. (2023) reverse-engineered exactly this kind of network and found an algorithm nobody designed: it represents each number as a set of rotations at a few **key frequencies**, multiplies rotations with trigonometric identities, and reads the answer off with a cosine that peaks at the right $c$. your job is to measure whether your model, built from your own chapter 18 parts, does the same, how cleanly, and when the algorithm appears during training. this is interpretability at its most demanding: not "which inputs mattered" but "here is the algorithm, with equations, and here are the measurements that support each step".

## background: the Fourier multiplication algorithm

write $w_k = 2\pi k / p$. the algorithm Nanda et al. describe has three steps.

1. **embed as rotations.** the embedding of a number $a$ is (approximately) a linear function of $\cos(w_k a)$ and $\sin(w_k a)$ for a handful of key frequencies $k$; every other frequency is absent. the embedding matrix, read in the Fourier basis over $a$, is sparse.
2. **multiply rotations.** attention moves the embeddings of $a$ and $b$ to the final position; the MLP neurons compute products such as $\cos(w_k a)\cos(w_k b)$, and the identities
   $$\cos(w(a+b)) = \cos(wa)\cos(wb) - \sin(wa)\sin(wb), \qquad \sin(w(a+b)) = \sin(wa)\cos(wb) + \cos(wa)\sin(wb)$$
   turn them into $\cos(w_k(a+b))$ and $\sin(w_k(a+b))$. so each neuron's activation, as a function of $(a, b)$, lives in the span of $\cos$ and $\sin$ of $w_k a$, $w_k b$, $w_k(a+b)$ and $w_k(a-b)$ for a single $k$.
3. **read out with a cosine.** the unembedding computes, for every candidate answer $c$,
   $$\cos(w_k(a + b - c)) = \cos(w_k(a+b))\cos(w_k c) + \sin(w_k(a+b))\sin(w_k c),$$
   and the logits are $\sum_k \alpha_k \cos(w_k(a+b-c))$. every term equals $\alpha_k$ at $c = a + b \bmod p$ and is smaller elsewhere; summed over several frequencies, the wrong answers interfere destructively and the right one wins.

two **progress measures** make the training dynamics visible. the **restricted loss** keeps only the part of the logits that this algorithm uses (each output's average plus the projection onto $\cos$ and $\sin$ of $w_k(a+b)$ at the key frequencies) and scores it on all pairs; it falls when the algorithm works, even while the full model is still dominated by memorisation. the **excluded loss** removes that part and scores what's left on the training pairs; it rises as the model stops relying on memorisation. Nanda et al. found the circuit forming gradually, long before the test loss moved, and the "sudden" generalisation to be the moment the memorising components were cleaned away by weight decay.

## data

synthetic: all $p^2$ pairs $(a, b)$ with $0 \le a, b < p$, as three tokens `[a, b, =]` (`=` is token $p$) with label $(a + b) \bmod p$, split by a seeded permutation into a training fraction and the rest. nothing is downloaded; offline runs are identical.

## deliverables

1. `capstone.py` with the functions below implemented and their checks green:
   - `modular_addition_data(p, *, train_frac=0.3, seed=0)`: a dict with `all_x` `(p², 3)` int64 in grid order (row $a p + b$), `all_y`, `train_idx` and `test_idx` (sorted indices from `np.random.default_rng(seed).permutation(p * p)`, the first `round(train_frac * p²)` for training), `train_x`, `train_y`, `test_x`, `test_y`, and `p`.
   - `make_model(p, *, d_model=128, n_heads=4, d_mlp=512, seed=0)`: your `mlfs.transformer.GPT` over $p + 1$ tokens with block size 3, one layer, `d_ff = d_mlp`, untied embedding and head, initialised after `torch.manual_seed(seed)`. `answer_logits(model, x)`: the `(n, p)` logits at the last position for the $p$ number tokens.
   - `train_grokking(model, data, *, steps, lr=1e-3, weight_decay=1.0, log_every=100, checkpoint_every=None, device='cpu')`: full-batch AdamW (betas 0.9, 0.98) on the training pairs; a history dict with `step`, `train_loss`, `test_loss`, `train_acc`, `test_acc` and `weight_norm` (the 2-norm of all parameters) logged in eval mode after $t$ updates for $t = 0$, every `log_every`, and `steps`; with `checkpoint_every`, also `checkpoints`, a list of `(t, state_dict copy on the CPU)` at the same kind of schedule. leaves the model in eval mode.
   - `fourier_basis(p)`: the `(p, p)` orthonormal rows $1/\sqrt p$, then $\sqrt{2/p}\cos(w_k x)$, $\sqrt{2/p}\sin(w_k x)$ for $k = 1, \dots, (p-1)/2$; `ValueError` for even $p$. `embedding_frequency_norms(W_E, p)`: the fraction of the embedding's non-constant energy (squared norm in the Fourier basis, cos and sin together) at each frequency, using only the rows of the $p$ number tokens. `key_frequencies(fractions, *, frac=0.9)`: the fewest frequencies (sorted, 1-based) whose fractions reach `frac`.
   - `logit_cosine_fit(full_logits, p, freqs)`: removes each input's mean over $c$, fits $\alpha_k$ by least squares against $\cos(w_k(a + b - c))$, and returns `(alphas, FVE)`, the fraction of the centred logits' energy explained. `progress_measures(full_logits, labels, train_idx, p, freqs)`: `{"restricted": ..., "excluded": ...}` as defined above (cross-entropy, restricted on all pairs, excluded on the training pairs).
   - `neuron_activations(model, x)`: the `(n, d_mlp)` ndarray of `blocks.0.mlp.act` outputs at the last position, read with your `mlfs.safety.run_with_cache` in eval mode (mode restored). `neuron_frequency_fit(acts, p)`: for each neuron, the frequency whose 8 functions ($\cos$, $\sin$ of $w_k a$, $w_k b$, $w_k(a+b)$, $w_k(a-b)$) explain the largest fraction of its variance over the grid, and that fraction: `(best_k, FVE)` (FVE 0 for a constant neuron).
2. the full-config run and the write-up at the end of the notebook, answering its five prompts.

## configurations

| config | what runs | runtime |
|---|---|---|
| **small** (tests, `verify.py`, offline, any machine without a GPU, or `MLFS_CAPSTONE=small`) | $p = 31$, 60% of pairs for training, width 64, 4 heads, 256 MLP neurons, 4,000 steps, one seed, on the CPU. it groks too (at step 1,350–1,400 on seed 0, depending on the machine) | about half a minute |
| **full** (your write-up; needs `cuda` or `mps`) | $p = 59$, 30% of the 3,481 pairs for training, width 128, 4 heads, 512 MLP neurons, 20,000 full-batch steps (lr $10^{-3}$, weight decay 1), seeds 0–4 | **12–13 minutes** on an Apple M2 (`mps`) shared with other jobs: about 2.5 minutes per seed |

**why $p = 59$.** grokking needs enough training pairs, and the training fraction is fixed at Nanda et al.'s 30%. with this model and recipe, measured on `mps` with 20,000 steps: $p = 31$ and $37$ never generalised (0 of 3 seeds); $p = 43$ grokked in 1 of 3 seeds (at step 11,500) and $p = 47$ in 2 of 3 (9,000 and 17,750); $p = 53$ in 5 of 5 in one sweep (6,000–10,000 steps) but only 4 of 5 within 15,000 steps in a rerun with the notebook's split, two of them with test accuracy slipping back after grokking; $p = 59$ grokked in 5 of 5 seeds in two separate sweeps. so 59 is the smallest prime that grokked reliably here.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass |
| grokking | the full config on at least 3 seeds, with the steps at which training and test accuracy reach 99% |
| the algorithm | key frequencies, the logits' FVE by their cosines, and the neuron statistics, per seed |
| progress | restricted and excluded loss over training for at least one seed |
| honesty | what the measurements support and what they don't, including how your model differs from Nanda et al.'s |

## what the reference gets

measured on an Apple M2 (`mps`), full config. GPU training isn't bit-reproducible, so the grokking steps below come from three complete runs of the notebook (15 (run, seed) pairs); the per-seed table is the latest run. one run took 12.2 minutes of training and 12.3 minutes in total.

**grokking.** every model reached 100% training accuracy by step 100 and then sat at chance-level-ish test accuracy (0.2–0.3, from the pairs memorisation happens to get right) for thousands of steps. test accuracy crossed 99% at steps **3,000–7,400** across all 15 (run, seed) pairs (latest run: 3,900, 3,100, 4,800, 6,500, 5,400). the total weight norm fell from about 21 to 19 while the training set was memorised, rose to 28–33 while the memorising and generalising solutions coexisted, and dropped to about 22 once the model generalised. training is punctuated by brief loss spikes (AdamW with weight decay 1 "slingshots"): in one run, seed 3's last logged step landed inside one (test accuracy 0.37 at step 20,000, back near 1.0 a few hundred steps later in other spikes), which is why the notebook analyses the latest checkpoint that still generalises if a run ends mid-spike (none did in the run below).

| seed | memorised (train acc ≥ 0.99) | grokked (test acc ≥ 0.99) | key frequencies (90% of embedding energy) | top-5 energy | logit FVE by their cosines | neurons with FVE > 0.8 | median neuron FVE | variance-weighted neuron FVE |
|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 3,900 | 2, 6, 7, 8, 12, 14, 16, 17, 23, 26 | 0.63 | **0.919** | 0.15 | 0.34 | **0.96** |
| 1 | 100 | 3,100 | 5, 8, 10, 11, 12, 14, 18, 21, 23, 24, 25, 28 | 0.56 | 0.961 | 0.20 | 0.30 | 0.98 |
| 2 | 100 | 4,800 | 1, 2, 5, 9, 12, 15, 17, 18, 21, 25, 28 | 0.50 | 0.953 | 0.18 | 0.33 | 0.97 |
| 3 | 100 | 6,500 | 1, 3, 5, 6, 11, 12, 20, 22, 24 | 0.56 | 0.929 | 0.15 | 0.29 | 0.95 |
| 4 | 100 | 5,400 | 1, 2, 4, 6, 8, 10, 12, 15, 17, 22, 24, 26, 28 | 0.48 | 0.935 | 0.19 | 0.31 | 0.97 |

what this says, and doesn't:

- **the logits are a sum of cosines.** across all the runs (mine and a reviewer's), $\sum_k \alpha_k \cos(w_k(a+b-c))$ over the key frequencies explained 74–97% of the centred logits of a generalising model, and 91–96% in the latest run's table above; one seed analysed in the middle of a loss spike scored 0.09, which is what a spike is: the circuit briefly knocked out. the logit map of a single output $c$ over the $(a, b)$ grid is a set of diagonal stripes, constant along $a + b$, exactly as the algorithm predicts.
- **the embedding is sparse in the Fourier basis, but less sparse than in Nanda et al.** 9–13 of the 29 frequencies hold 90% of the energy, and the top 5 hold 48–63% (Nanda et al. report about 5 key frequencies carrying almost everything, for $p = 113$ with no LayerNorm and much longer training). different seeds pick different frequencies (frequency 12 appears in all five, by chance or not), as expected: any set of frequencies works.
- **the active neurons compute single-frequency products.** the neurons' fit is bimodal: 15–20% of them are explained to more than 80% (most of those to 95–100%) by the 8 terms of one frequency, and the rest have low FVE (median 0.29–0.34); but the low-FVE neurons are almost inactive. weighted by their variance, a single frequency explains **95–98%** of the MLP's activity, and every neuron's best frequency is a key frequency.
- **the circuit forms before the test loss moves.** seed 0 (latest run): at step 3,000 the test loss is still 2.46, but the restricted loss (key frequencies only, scored on *all* pairs, including the 70% never trained on) is already 0.32 and the excluded loss (key frequencies removed, on the training pairs) is still 0.06: the generalising circuit exists and works, while memorisation still carries the training set. by step 3,500 the excluded loss has jumped to 2.16 (removing the key frequencies now breaks the training set too: memorisation has been cleaned away), and at 3,900 test accuracy crosses 99%. the "sudden" generalisation is the end of a gradual process.
- **what it doesn't show.** every measurement here is a fit or a projection: correlational evidence that the network's activations have the algorithm's form. the causal test (ablate the key frequencies and watch accuracy collapse; keep only them and watch it survive) is a stretch goal. the model also differs from Nanda et al.'s (LayerNorm, GELU, learned positions, smaller $p$, shorter training), which is the most likely reason the embedding is less sparse.

## stretch goals

- **a causal test.** zero every non-key frequency of the embedding (in the Fourier basis) and re-evaluate; then zero only the key frequencies. predict both test accuracies first.
- **Nanda's architecture.** build the model from your `MultiHeadAttention` and `FeedForward(activation='relu')` with no LayerNorm and untied embeddings, as in the paper. does it grok on the same schedule, and are its neurons cleaner?
- **no weight decay.** rerun with `weight_decay=0`. does it still grok within 20,000 steps? what happens to the weight norm and the excluded loss?
- **other operations.** train on $(a - b) \bmod p$ or $a \cdot b \bmod p$ and look for the same structure (Chughtai et al., 2023, generalise the algorithm to group operations).
- **clock or pizza?** Zhong et al. (2023) found a second algorithm in networks with attention that's closer to uniform; inspect your attention patterns and test which one your model uses.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Power et al. (2022), "Grokking: generalization beyond overfitting on small algorithmic datasets". [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Nanda et al. (2023), "Progress measures for grokking via mechanistic interpretability", *ICLR*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Liu, Michaud & Tegmark (2023), "Omnigrok: grokking beyond algorithmic data", *ICLR* (the role of weight norm). [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)
- Chughtai, Chan & Nanda (2023), "A toy model of universality: reverse engineering how networks learn group operations", *ICML*. [arXiv:2302.03025](https://arxiv.org/abs/2302.03025)
- Zhong et al. (2023), "The clock and the pizza: two stories in mechanistic explanation of neural networks", *NeurIPS*. [arXiv:2306.17844](https://arxiv.org/abs/2306.17844)
