# k-means · lab notes

the numbers below come from the reference solution with the lab's seeds; your implementation should give the same or very similar values (restarts and random draws can differ in order).

## experiment 1: shapes k-means can't see

**what people usually predict:** k-means does badly on both, because it got stuck in a poor local minimum, so the true labeling must have a lower inertia.

**what happens:** k-means cuts both data sets with a straight line: ARI **0.28** on the moons and **0.00** on the rings. and the true labeling is *worse* by k-means's own measure: inertia **251.9** for the true moons against **164.9** for k-means's answer, and **225.8** against **143.4** for the rings.

**why:** this isn't an optimization failure. k-means found a partition with a lower objective than the truth, so even a perfect optimizer would return something like it. the objective itself asks for compact, round groups around a centre: a crescent's points are far from the crescent's mean (which lies off the crescent), and the two rings have the *same* mean, so splitting them by radius buys nothing in squared distance to a centre. every k-means cluster is a convex Voronoi cell (on paper 4), and neither a crescent nor a ring is convex. to cluster them you have to change what "close" means: a graph of nearest neighbours and spectral clustering (topic 05), density-based clustering (DBSCAN, topic 05), single linkage, or a feature map in which the groups become round, like the radius $\|x\|$ for the rings, or kernel PCA (topic 03).

## experiment 2: how lucky is one run?

**what people usually predict:** with blobs this well separated, k-means++ succeeds nearly always and random starts most of the time.

**what happens:** of 200 single runs from uniformly random starts, **0** find all 25 blobs; from k-means++ starts, **81** (41%). median ARI: 0.817 (random) and 0.949 (k-means++). the inertia histogram has separate spikes: the optimum near 900 and bad local minima around 2,700, 4,600 and higher, each one more misplaced centre. with `n_init=10`, k-means++ finds the blobs in **100%** of 50 fits; random starts still in **0%**.

**why:** a bad start puts two centres in one blob and none in another. Lloyd's moves are local: the two centres split their blob between them, and the orphaned blob is absorbed by a neighbouring centre, which settles between the two blobs. no single assignment or update step can move a centre across the empty space to the orphan, because it would first have to give up the points it owns. the median random run has several such defects (three centres sitting between blobs in the figure). k-means++ makes the first defect unlikely, but with 25 blobs it gets 24 sequential choices to make: even if each extra centre lands in an empty blob with probability 0.96, all 24 do with probability $0.96^{24} \approx 0.38$. the measured single-run success $p \approx 0.41$ predicts $1 - 0.59^{10} \approx 0.995$ for ten restarts, matching the 50/50 observed. uniform starts succeed with probability $25!/25^{25} \approx 10^{-10}$ before Lloyd, and Lloyd repairs only a little, so ten restarts don't help. the moral: k-means++ plus restarts is not a nicety; it's what makes k-means work.

## experiment 3: choosing k when clusters come in pairs

**what people usually predict:** the silhouette picks 4, the number of blobs you can see.

**what happens:** at $d = 6$ the silhouette scores are 0.809 ($k = 2$), 0.752 ($k = 3$), 0.691 ($k = 4$), then falling: it picks **$k = 2$**, one cluster per pair. the elbow plot, by contrast, bends sharply at $k = 4$ (inertia 2,490 at $k = 2$, 480 at $k = 4$). sliding $d$, the silhouette switches to 4 at $d = 8$ (where 2, 3 and 4 are within 0.005 of each other) and prefers 4 clearly from $d = 10$ on.

**why:** at $k = 2$, a point's own cluster is its pair: $a(i)$ is roughly the average distance to points of its own blob (about 1.8) and of its sibling blob (about 6), so $a \approx 4$, while the other pair is about 20 away: $b \approx 21$ and $s \approx 0.8$. at $k = 4$, $a(i) \approx 1.8$ but the nearest *other cluster* is now the sibling blob, only 6 away: $b \approx 6$ and $s \approx 0.7$. the silhouette rewards the ratio of separation to compactness, and the coarse level has the better ratio until the siblings are about as far from each other as the pairs are. the inertia doesn't care about separation at all, only about compactness, so it shows the four blobs clearly. neither is "right": the data really has two levels of structure. if you need groups for a downstream decision, choose $k$ for that decision; if you want the structure, look at more than one number (or at a hierarchy, topic 05).
