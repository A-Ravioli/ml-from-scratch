# 28 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 12 out of 15, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass | all pass, plus your own checks: the Fourier basis reconstructs the embedding exactly, the restricted and excluded logits add up to the original plus the constant term, and a model you planted (hand-set cosine logits) gets FVE 1 |
| **results** | no numbers | one seed, or no step counts for memorisation and generalisation | the full config on at least 3 seeds: memorisation and grokking steps, key frequencies, logit FVE and neuron statistics per seed, with the spread reported | as 2, plus the progress measures over training for at least one seed and the steps at which the restricted loss and the test loss each cross a threshold you fix in advance |
| **analysis** | none | describes the plots | explains the algorithm (embedding → products of rotations in the MLP → $\cos(w(a+b-c))$ in the logits) with equations, and says which parts your measurements support and which they don't (for example, how clean the neurons are) | as 2, plus one causal test you designed (ablate or keep only the key frequencies in the embedding, patch neurons, or remove weight decay) with a prediction written before running it |
| **honesty about limits** | missing | generic caveats | names what differs from Nanda et al.'s setup (LayerNorm, GELU, $p$, training length) and how it shows up in the numbers, and separates correlational evidence (Fourier fits) from causal evidence (interventions) | as 2, plus a seed that behaved differently (grokked late, never, or unstably) analysed rather than dropped |
| **clarity** | hard to follow | readable | loss curves on a log scale, the embedding spectrum, the progress measures and a per-seed table, side by side with the narrative | a peer could rerun every number from the notebook alone and would know which differences between seeds are noise |
