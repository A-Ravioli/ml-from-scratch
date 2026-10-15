# GNN expressivity · lab notes

numbers measured with the reference solution on a laptop CPU (about half a minute for the whole lab). every graph is generated in the notebook, so `verify.py`'s offline run gives the same numbers.

## experiment 1: pairs that look the same

**what people usually predict:** that a trained-for-structure network like GIN surely separates a ring of six from two triangles ("they have different numbers of triangles!"), and that GCN, being the classic, does at least as well as WL.

**what happens:**

| pair | 1-WL | triangles | GIN embedding difference | GCN embedding difference |
|---|---|---|---|---|
| C6 vs 2 × C3 | indistinguishable | 0 vs 2 | 0.0 | 0.0 |
| C12 vs 3 × C4 | indistinguishable | 0 vs 0 | 0.0 | 0.0 |
| decalin vs bicyclopentyl | indistinguishable | 0 vs 0 | 0.0 | 0.0 |
| 3-regular vs 4-regular (12 nodes) | distinguished | 0 vs 5 | 2.6 | $3 \times 10^{-17}$ |
| path P6 vs C4 + edge | distinguished | 0 vs 0 | $4.6 \times 10^{-3}$ | $3.1 \times 10^{-4}$ |

on the three pairs 1-WL can't separate, both networks give embeddings that agree *exactly* (0.0 in float64, not merely small). on the regular pair, which 1-WL separates in one round, GIN's embeddings differ by 2.6 and GCN's by $3 \times 10^{-17}$, i.e. not at all. on $P_6$ versus $C_4 + K_2$ both networks see a difference.

**why:** on paper 1: equal WL colours force equal embeddings in any message-passing network, at every layer, for any weights, and the three WL-hard pairs have equal colour histograms in every round, so the readouts are sums of the same multiset of vectors. exactness is the tell: "0.0" isn't the network failing to learn, it's a theorem. the triangle counts show what this means in practice: $C_6$ and $2 \times C_3$ differ in the most basic cyclic substructure, and no message-passing network can see it. the GCN failure on regular graphs is the worked example: on a $d$-regular graph with identical features, every row of $\hat A$ sums to 1, so $\hat A x = x$ for both $d = 3$ and $d = 4$. GCN is *weaker* than 1-WL; GIN's unnormalized sum computes $(1 + \epsilon + d)x$ and sees the degree. ($P_6$ versus $C_4 + K_2$ isn't regular, so GCN's degree normalization produces different numbers there, though 15 times smaller than GIN's difference with these random weights.)

## experiment 2: learning to tell regular graphs apart

**what people usually predict:** close to 100% for every model, since the classes differ in every node's degree; maybe a few points less for the GCN.

**what happens:**

| model | test accuracy |
|---|---|
| GCN, mean readout | 0.500 |
| GCN, sum readout | 0.500 |
| GIN (sum aggregation, sum readouts) | 1.000 |

both GCNs are exactly at chance; GIN is perfect.

**why:** with identical node features, every GCN layer maps every node of every regular graph to the same vector (the worked example: $\hat A x = x$, then the same weights and ReLU everywhere), whether the graph is 3- or 4-regular. after three layers, every node of every graph in the dataset has the same embedding. the sum readout then gives $12 \times$ that vector for every graph, since all graphs have 12 nodes, so it doesn't help either: the information was destroyed in the first layer, before any readout. no training can recover it, which is why the accuracy is exactly 0.500 and not "a bit worse". GIN's first layer already computes $\operatorname{MLP}((1 + d)x)$, different for $d = 3$ and $d = 4$. the general lesson: an architecture's expressivity is decided by its aggregation, not by its size or training, and degree normalization, harmless-looking, throws away exactly the multiplicity information that WL uses.

## experiment 3: breaking the symmetry with random features

**what people usually predict:** that random noise can't help a classifier, or, from those who've read about it, that random features "solve" expressivity.

**what happens** (GIN, 3 layers, 600 epochs, $C_{2k}$ vs $C_k + C_k$ for $k = 3, 4, 5$):

| random features | test accuracy (seeds 0, 1, 2) | std of p(label 1) for the same graph over 20 draws |
|---|---|---|
| 0 | 0.495 (0.50, 0.50, 0.50) | 0.000 |
| 8 | 0.767 (0.76, 0.79, 0.75) | 0.215 |

without random features GIN gives every graph the same output and scores the majority-class rate. with 8 random features per node it reaches 77%, but the prediction for one fixed graph changes from draw to draw with a standard deviation of 0.2 in probability.

**why:** all the graphs here are 2-regular, so they're indistinguishable to 1-WL and every node looks like every other node. random features give each node an (almost surely) unique identity, so message passing can now detect a cycle closing: in a short cycle, information sent out from a node comes back to it (around the cycle, or from both directions at once) within the network's 3 hops, carrying the node's own random vector, which a 3-layer network can in principle notice; in a longer cycle it doesn't come back in time. it's only 77% because this detection is approximate (the network must learn to compare noisy vectors) and a node's 3-hop neighbourhood contains a whole cycle only if the cycle has at most 7 nodes, which covers $C_3, C_4, C_5$ and $C_6$ but not $C_8$ or $C_{10}$. (with $k$ up to 8, the same setup reaches only about 60%.) the price is in the last column: the network is no longer a function of the graph. permutation invariance, the property this whole chapter is about, now holds only on average over the random draws; a single prediction for a molecule is a sample. in practice you'd average many draws at test time, and generalization to larger graphs suffers. that's why most practical models prefer deterministic extra information: substructure counts, or, when nodes have positions, geometry (topic 05).
