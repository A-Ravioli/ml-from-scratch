# graph & density clustering · lab notes

the numbers below come from the reference solution with the lab's seeds (DBSCAN `eps = 0.2` in experiment 1); your implementation should give the same values, except for k-means, GMM and spectral clustering if your random draws come in a different order.

## experiment 1: every method on every shape

**what people usually predict:** spectral clustering or DBSCAN wins everywhere; k-means is worst everywhere; single linkage is a weaker version of the others.

**what happens (ARI):**

| data | k-means | GMM | spectral | DBSCAN | single | Ward |
|---|---|---|---|---|---|---|
| two moons | 0.48 | 0.35 | **1.00** | **1.00** | **1.00** | 0.26 |
| two rings | 0.00 | 0.00 | **1.00** | 0.55 | **1.00** | 0.02 |
| varied density | 0.90 | **0.99** | 0.97 | 0.75 | 0.00 | 0.98 |
| stretched | 0.27 | **0.99** | 0.30 | 0.80 | 0.00 | 0.28 |

no method is good everywhere. the best on the curved shapes (single linkage, spectral) are the worst or near-worst on the blobs, and vice versa.

**why:** each method encodes a definition of "cluster", and wins exactly when the data match it.

- **k-means and Ward** look for compact, round groups of similar size (Ward is greedy k-means, merging to minimize the increase in squared error). they're good on the varied-density blobs (0.90, 0.98: the blobs are compact even if their densities differ) and cut every other shape with straight lines.
- **the GMM** allows ellipses with different sizes and orientations, so it's best on both blob data sets (0.99), including the stretched ones that defeat every other method; curved shapes aren't ellipses.
- **spectral clustering** sees only the 15-nearest-neighbour graph, so a cluster is anything connected within itself and weakly connected to the rest: moons and rings are perfect. on the stretched blobs the strips nearly touch along their length, the graph has many edges between them, and the smallest normalized cut runs *across* the strips (0.30).
- **DBSCAN** with one `eps` finds dense connected regions. at `eps = 0.2` the moons are perfect, but the outer ring (sparser than the inner one) breaks into ten arcs plus noise (0.55), and the sparse blob of the varied-density data dissolves into small groups and noise (0.75). changing `eps` repairs one data set and breaks another: `eps = 0.3` fixes the rings (1.00) and improves the varied-density blobs (0.87), but merges the three stretched strips into one cluster (0.00, from 0.80); at 0.1 or 0.15 nothing is good.
- **single linkage** merges along the closest pair of points: a cluster is a chain. it follows the moons and rings perfectly, and with blobs it does the worst possible thing: its last merge separates one or two outlying points from everything else, so the 3-cluster cut is "everything" plus a few stragglers (ARI 0.00).

the practical lesson: before choosing a method, decide what a cluster *means* for your problem, and look at the data (in 2-D, or through an embedding) before trusting any score.

## experiment 2: DBSCAN's density threshold

**what people usually predict:** with the right `eps` DBSCAN finds the three blobs; or that it can't get above about 0.5.

**what happens:** the ARI climbs as `eps` grows, as the noise fraction falls from 88% (eps 0.03) to a few percent: 0.72 at eps ≈ 0.19, 0.85 at 0.26, and a best of **0.934 at eps = 0.405** (3 clusters, **4%** noise). it's still 0.933 at the next value, 0.42, and then, one step later at **0.435**, it falls to **0.002**: everything is one cluster. the number of clusters found jumps around between 2 and 9 for small `eps`, settles at 3 between 0.26 and 0.42, then drops to 1. the k-distance plot rises smoothly (10th percentile 0.042, median 0.144, 90th percentile 0.377) with a sharp upturn only for the last few outliers.

**why:** the median 5th-neighbour distance is 0.05 in the dense blob, 0.14 in the middle one and 0.26 in the sparse one (on the standardized scale), with the sparse blob's outskirts reaching 0.4 and more. for small `eps`, the sparse blob doesn't have enough core points and fragments into many small clusters and noise (the erratic cluster count). to keep it together `eps` must reach the scale of the sparse blob, but the gap between the sparse blob and its neighbours is only slightly larger than that, so the best value sits right next to the point where a few sparse points bridge the gap and the clusters merge. the k-distance plot has no knee because there is no single density: it's a smooth mixture of three. "choose `eps` from the knee" works when clusters share a density; with varied densities, use HDBSCAN (which considers every density level) or a method with per-cluster scales (the GMM gets 0.99 here).

## experiment 3: a thin bridge

**what people usually predict:** single linkage survives a few bridge points; perhaps 10–15 points to break it. Ward a few more.

**what happens:** single linkage is perfect (ARI 1.00) with 0, 2 or 4 bridge points and fails completely (**0.00**) from **6 points on**. Ward stays at **1.00** for every bridge size up to 20.

**why:** without a bridge, the last single-linkage merge joins the two blobs at the height of the smallest gap between them (several units). a bridge of $m$ evenly spaced points across the 5-unit gap has steps of $5/(m - 1)$; at $m = 6$ that's 1.0, already shorter than the distances to the most outlying points of each blob. single linkage then connects both blobs through the bridge before it connects those outliers, so its final merge (the 2-cluster cut) separates an outlier from everything else. a single chain of close points is all it takes, which is called **chaining**. Ward's criterion is the increase in squared error, which for merging the two big blobs is enormous whatever the bridge looks like; the bridge points just join whichever blob is nearer. real data with bridges: noise points between clusters, transitional states (cells in the middle of differentiating, customers drifting between segments), and measurement artefacts. single linkage is still the right tool when chains *are* the structure you want (like the moons), and it's closely related to the minimum spanning tree.
