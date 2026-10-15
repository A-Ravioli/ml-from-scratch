# anomaly detection · lab notes

the numbers below come from the reference solution with the lab's seeds; yours should be very close (the isolation forest's numbers move by about one standard deviation between implementations that draw their random numbers in a different order).

## experiment 1: the contamination rate

**what people usually predict:** a wrong contamination rate makes the detector worse across the board; or that a higher rate catches more anomalies "for free".

**what happens (mean ± sd over 10 seeds):** the AUC is **0.998** at every contamination setting. precision and recall trade off exactly as the number of flagged points dictates:

| contamination | precision | recall | F1 |
|---|---|---|---|
| 1% | 1.000 | 0.200 | 0.333 |
| 2% | 1.000 | 0.400 | 0.571 |
| 5% (the truth) | 0.893 ± 0.011 | 0.893 ± 0.011 | 0.893 ± 0.011 |
| 10% | 0.500 | 1.000 | 0.667 |
| 20% | 0.250 | 1.000 | 0.400 |

**why:** the trees, and therefore the scores, never see `contamination`; it only sets `threshold_` (the lab applies the five thresholds to the same scores, which is exactly what `fit` with each value would produce, since the trees depend only on the seed). the ranking is fixed, so the AUC is fixed. the threshold decides how many points are flagged: $0.01 \times 2{,}000 = 20$ at 1%, so at most 20 of the 100 anomalies can be caught (recall ≤ 0.2), and the top 20 are all real (precision 1). at 20%, 400 points are flagged, all 100 anomalies are among them (recall 1), and 300 are false alarms (precision 0.25). only at the true rate can both be high, and then precision equals recall by construction, because the number flagged equals the number of anomalies. the 0.893 is the detector's real limitation: about 11 planted points land just outside the bulk of the Gaussian and look ordinary, while a few genuine tail points of the Gaussian look as unusual as them. when the true rate is unknown, report ranking metrics (AUC, average precision, precision at the number of alerts you can investigate), and treat `contamination` as an operating decision, not a model parameter.

## experiment 2: masking, and the subsample size

**what people usually predict:** more data per tree is better, so all 1,100 points should win; the 5-NN score, being local, should handle the clump well.

**what happens:** the AUC falls steadily as `max_samples` grows: **1.000** at 16, 0.997 at 32, 0.980 at 64, 0.937 at 128, 0.910 at 256 (the default), 0.897 at 512, and **0.864** with all 1,100 points (mean of 3 seeds; standard deviations up to 0.02). the score maps show why: with 16 points per tree the clump is the darkest (most anomalous) region; with 1,100 it's only as anomalous as the Gaussian's outer ring, and a few isolated tail points outrank it. the 5-NN distance score is **worse than random (AUC 0.099)**; with $k = 150$ it's **0.999**.

**why:** this is **masking**: anomalies that are numerous and close together protect each other. with all 1,100 points in a tree, the 100 clumped points form a dense region of their own; random cuts separate the clump from the bulk quickly, but then need many more cuts to isolate each point inside the clump, as many as for a point in the middle of the normal data. their average path is barely shorter than typical. with 16 random points per tree, a tree contains on average $16 \times 100/1100 \approx 1.5$ clump points, often 0 or 1. a lone clump point in a sample of 16 is isolated by the first cut that falls between $(3, 3)$ and the bulk, so its path is very short; and when the tree has no clump point at all, new clump points are evaluated on the tree and still land in a small, shallow region far from where the tree's points were. subsampling turns a dense group into a few isolated points, which is exactly what the algorithm detects. the 5-NN score fails for the same reason in its own terms: each clump point's 5 nearest neighbours are other clump points 0.1 away, much closer than neighbours in the normal bulk, so the clump looks like the *densest* part of the data. once $k$ exceeds the clump size (100), the $k$-th neighbour must be in the bulk, far away, and the score recovers. the general lesson: every detector has a scale (the subsample size, $k$, a bandwidth), and anomalies that come in groups larger than that scale look normal.

## experiment 3: two modes, anomalies in the middle

**what people usually predict:** all four find the obvious clump between the blobs; maybe the isolation forest best, because it makes no assumptions.

**what happens:**

| detector | AUC | median rank of the 15 anomalies (of 1,015) |
|---|---|---|
| one Gaussian (Mahalanobis) | **0.012** | 1006 |
| GMM, 2 components | **0.995** | 11 |
| 10-NN distance | 0.902 | 86 |
| isolation forest | 0.822 | 197 |

the single Gaussian ranks the anomalies as the *most normal* points in the data set (median rank 1006 of 1,015).

**why:** a single Gaussian fitted to two blobs has its mean at the origin, between them, and its covariance stretched along the axis joining them. the Mahalanobis distance is smallest at the mean, exactly where the anomalies sit: the model's assumption (one mode) puts the highest density in an empty region. the two-component mixture has its means on the blobs, so the origin is 4 standard deviations from both and gets a very low density: nearly perfect. the 10-NN and isolation-forest scores make no shape assumption, but the 15 anomalies form a clump, and experiment 2's masking applies at their scale: with $k = 10 < 15$, each anomaly's 10 nearest neighbours are other anomalies, so their distances are only moderately large (rank 86); the isolation forest's default subsample of 256 contains about 4 anomalies, which partly shield each other, and points in the blobs' outer tails are about as easy to isolate (rank 197). the moral: a density model is only as good as its shape assumptions, and a "model-free" detector still has a scale. know your data's modes, and check a detector against planted anomalies like these before trusting it.
