# 11 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and your full-config numbers are within the brief's ranges | all pass, plus your own sanity checks: e.g. your `purity` and ARI agree with a hand count on a small contingency table, and your t-SNE map's 10-NN agreement is reproducible across two seeds |
| **results** | no numbers | ARI or NMI for one pipeline | every pipeline in experiment A against the random baseline, with mean ± sd over seeds, plus the embedding and cluster-then-label numbers | the same, plus at least one ablation of your own (another $k$, another number of PCA components, standardized pixels, a full-covariance GMM on fewer components), with uncertainty |
| **analysis** | none | describes the tables | explains the results with the chapter's ideas: what PCA removes, k-means's preference for round equal-sized clusters, why a GMM's flexibility can hurt, why t-SNE maps flatter clusters, why purity rises with $k$ | also explains specific failures with evidence: which digits merge or split and why (look at the cluster means), and an experiment that tests one of your explanations |
| **clarity** | hard to follow | readable | labelled figures; all five prompts answered with numbers | a reader could reproduce every number (sizes, seeds, components, perplexity) and knows which numbers use labels (evaluation only) and which don't |

## what graders look for

- **no label leakage.** the clustering and embedding pipelines never see `y`; labels appear only in evaluation (ARI, NMI, purity, the 10-NN agreement, naming clusters in experiment C). say so explicitly.
- **baselines.** a random labeling (ARI 0) and k-means on raw pixels. without them, "ARI 0.36" means nothing.
- **uncertainty that means something.** the spread over seeds of the clustering, and an honest comment on what it does *not* cover (a different 10,000-image subset would move the numbers more).
- **honest negatives.** a method that did worse than expected (the diagonal GMM, PCA that didn't help k-means much) is a result. explain it rather than hiding it.
- **careful reading of t-SNE.** cluster sizes and gaps in the map aren't evidence; neighbourhoods are. clustering the t-SNE map is reported, if at all, with that caveat.
