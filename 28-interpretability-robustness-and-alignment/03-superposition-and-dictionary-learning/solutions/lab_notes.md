# superposition & dictionary learning · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, one thread). all data is synthetic by design (the true features are known), so the notebook gives the same numbers offline; it takes about 16 seconds. the notes quote two more seeds (1 and 2: model initialisation and data), run with the same code outside the notebook.

## experiment 1: when does a model use superposition?

**what people predict.** 5 features at every sparsity ("5 dimensions hold 5 things"), or, having read the warm-up, "more with sparsity", without a sense of how many or how the change happens.

**what happens.**

| sparsity | features represented ($\|W_i\| > 0.5$) | loss on a fresh batch |
|---|---|---|
| 0 | 5 | 0.0464 |
| 0.7 | 7 | 0.0333 |
| 0.9 | 9 | 0.0078 |
| 0.97 | 11 | 0.0016 |
| 0.99 | 12 | 0.0005 |

seeds 1 and 2 gave 5, 8, 9, 11, 13 and 5, 6, 9, 11, 12. with dense features $W^\top W$ is the identity on the 5 most important features and zero elsewhere. at 0.7 two antipodal pairs appear (the red off-diagonal entries at $-1$); at 0.9 four pairs plus one dedicated dimension; at 0.97 and 0.99 the off-diagonal blocks fill with smaller negative and positive entries, and every represented feature shares its direction.

**why.** with dense features, every feature is active in every input, so any overlap between two features adds interference to every reconstruction, and the best use of 5 dimensions is the 5 most important features (PCA, on paper 1). the interference between two superposed features only costs something when both are active, which happens with probability $(1-S)^2$; as $S$ grows, the expected cost of sharing falls below the benefit of representing one more feature, and features are added in importance order. the off-diagonal entries are negative because negative interference is free (the ReLU removes it) while positive interference leaks through (on paper 3).

## experiment 2: the geometry of superposition

**what people predict.** a continuum: features at every angle, $D$ spread out between 0 and 1.

**what happens.** the dimensionalities stick to a few values:

- sparsity 0: $D = 1$ (×5)
- 0.7: $1$ (×3) and $0.49$–$0.51$ (×4: two antipodal pairs)
- 0.9: $1$ and $0.46$–$0.54$ (×8: four pairs)
- 0.97: $0.48$–$0.51$ (×5) and $0.41$–$0.44$ (×5, near $\tfrac25$), one at 0.35
- 0.99: $0.42$–$0.47$ and $0.32$–$0.38$ (near $\tfrac25$ and $\tfrac13$)
- 40 equally important features in 10 dimensions at sparsity 0.97: **every** $D$ between 0.243 and 0.255

in every model the $D_i$ of the represented features add up to about the number of dimensions (5.00, 5.00, 4.96, ...; $40 \times \tfrac14 = 10$).

**why.** the sum $\sum_i D_i$ is at most $m$, with equality when the features form a *tight frame* in the space they use: the model always uses its whole space. within that budget, gradient descent finds symmetric arrangements where every feature in a group sees the same interference, because a lopsided arrangement has a feature with more interference than necessary and the gradient pushes it apart: pairs ($\tfrac12$), triangles ($\tfrac23$), pentagons ($\tfrac25$), tetrahedra ($\tfrac34$) and so on. Elhage et al. found these "sticky" fractions in their phase diagrams. the uniform model has no reason to treat any feature differently, so all 40 share the 10 dimensions equally, $D = 10/40$.

## experiment 3: a sparse autoencoder recovers the features

**what people predict.** that a smaller `l1_coeff` is always better (better reconstruction), or that MMCS falls steadily as `l1_coeff` grows.

**what happens.** the true number of active represented features per input is 0.91.

| l1_coeff | L0 | FVU | dead | MMCS |
|---|---|---|---|---|
| 0.0001 | 9.87 | 0.0010 | 0.03 | 0.775 |
| 0.001 | 8.84 | 0.0010 | 0.03 | 0.795 |
| 0.01 | 4.34 | 0.0011 | 0.05 | 0.959 |
| 0.03 | 1.87 | 0.0012 | 0.05 | **0.997** |
| 0.1 | 1.42 | 0.0082 | 0.05 | **1.000** |
| 0.3 | 1.20 | 0.0660 | 0.12 | 1.000 |
| 1 | 0.79 | 0.5162 | 0.10 | 0.997 |

seeds 1 and 2: MMCS 0.82 and 0.81 at $10^{-4}$, 0.98 and 0.99 at 0.03, 1.000 at 0.1, with L0 1.2 at 0.1 in both.

**why.** at small `l1_coeff` the SAE uses about 10 of its 40 features per input to reconstruct a 5-dimensional vector: with 40 directions in 5 dimensions there are infinitely many exact reconstructions, and nothing selects the sparse one, so the dictionary is a near-arbitrary overcomplete frame (MMCS 0.78: 40 *random* unit directions in 5 dimensions score 0.84 ± 0.03 against 9 true directions, so this dictionary is slightly worse than random). the penalty selects the sparsest explanation, which for data that really is a sparse combination of 9 directions is those directions. between 0.03 and 0.3 the dictionary is right and the reconstruction is still good; at 1 the penalty is so strong that the SAE shrinks its activations and leaves half the variance unexplained, even though its directions are still right (shrinkage, as in the lasso). without the true features you'd pick the knee of the L0–FVU curve, or the model's downstream loss with the reconstruction spliced in. the dropdown under the table trains one SAE at a chosen `l1_coeff` and shows its cosines with the true features: at 0.03 (L0 1.87, FVU 0.0012, MMCS 0.997) every true feature has one bright partner and a few dictionary features are dead; at $10^{-4}$ every true feature is smeared across several dictionary features.

## experiment 4: feature splitting

**what people predict.** that the 4-feature SAE learns 4 of the 16 children and the wider SAEs learn the rest, each dictionary containing the smaller one.

**what happens.**

| width | MMCS with the 4 parents | MMCS with the 16 children | FVU | dead |
|---|---|---|---|---|
| 4 | **0.989** | 0.862 | 0.257 | 0 |
| 8 | 0.950 | 0.922 | 0.151 | 0 |
| 16 | 0.859 | **0.997** | 0.005 | 0 |
| 32 | 0.883 | 1.000 | 0.005 | 0.03 |

the 4-feature SAE learns the four *parents*, directions that never occur on their own in the data (every input is made of children). the cosine grid between the width-4 and width-16 dictionaries is block-structured: each width-4 feature is close to exactly four width-16 features, its children. at width 32 the extra features duplicate children (and one is dead).

**why.** with 4 features and 16 children, the best a sparse code can do is one feature per family: the parent direction is the one that's closest, on average, to all four children (0.86 each), so it reconstructs any child with one active feature and a 26% error. with 16 features the SAE can afford one per child and gets almost perfect reconstruction; the parent features "split". neither dictionary is wrong: they describe the same data at two resolutions. Bricken et al. (2023) saw the same in a language model, where a single "base64" feature in a small SAE split into several more specific ones in larger SAEs. so "is this SAE feature the model's real feature?" is often the wrong question; the dictionary size is a choice of granularity.
