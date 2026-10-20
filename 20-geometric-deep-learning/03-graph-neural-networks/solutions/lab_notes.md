# graph neural networks · lab notes

numbers measured with the reference solution on a laptop CPU (about two minutes for the whole lab). the graphs are synthetic stochastic block models (300 nodes, 3 communities of 100, average degree about 8), so `verify.py`'s offline run gives the same numbers. accuracies are means over seeds 0, 1, 2, where the seed changes the graph, the features, the split and the initialization.

## experiment 1: what does repeated averaging converge to?

**what people usually predict:** that all nodes converge to the same vector (the global average), "like heat diffusion".

**what happens:**

| k | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.028 | 0.189 | 0.456 | 0.816 | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

by 8 steps the feature vectors point in nearly the same direction, and by 16 they're indistinguishable at three decimals. but they don't become *equal*: after 256 steps, feature 0 of every node lies exactly on a line through the origin when plotted against $\sqrt{\tilde d_i}$ (correlation 1.0000). nodes differ only by their degree.

**why:** on paper 2. $\hat A$'s top eigenvector is $\tilde D^{1/2}\mathbf 1$ with eigenvalue 1, and every other eigenvalue is smaller in magnitude, so $\hat A^k X \to u_1 u_1^\top X$: each feature column becomes a multiple of $\sqrt{\tilde d}$. (the random-walk normalization $\tilde D^{-1}(A + I)$ would converge to a constant instead; the symmetric one converges to $\sqrt{\text{degree}}$.) on these graphs the next eigenvalues are 0.78 and 0.76, and they belong to the community structure, so the community information decays like $0.78^k$ relative to the degree component: 2% left at $k = 16$, $10^{-7}$ (float32's resolution) at $k = 64$. the cosine reaching 1.000 at $k = 16$ doesn't mean the information is gone yet, as experiment 2 shows.

## experiment 2: how deep should a GCN be?

**what people usually predict:** a smooth decline with depth "because of oversmoothing", a residual connection that helps somewhat, and no strong expectation for the MLP controls.

**what happens.** left, logistic regression on standardized $\hat A^k X$:

| k | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| test accuracy | 0.533 | 0.747 | 0.803 | 0.864 | 0.906 | **0.925** | 0.447 | 0.303 | 0.303 | 0.303 |

right, trained networks (test accuracy, training accuracy in brackets, mean of 3 seeds):

| depth | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| GCN | 0.808 (0.97) | 0.900 (0.98) | 0.872 (0.93) | 0.289 (0.39) | 0.311 (0.43) | 0.292 (0.38) |
| GCN, kaiming init | 0.792 (0.93) | 0.889 (0.97) | 0.892 (0.96) | **0.817** (0.82) | 0.328 (0.39) | 0.308 (0.41) |
| GCN + residual | 0.772 (1.00) | 0.858 (1.00) | 0.869 (0.99) | 0.886 (1.00) | 0.914 (1.00) | 0.886 (0.99) |
| MLP | 0.525 (0.99) | 0.506 (1.00) | 0.536 (1.00) | 0.314 (0.33) | 0.314 (0.33) | 0.314 (0.33) |
| MLP, kaiming init | 0.511 (1.00) | 0.492 (0.96) | 0.519 (0.92) | 0.461 (0.96) | 0.386 (0.79) | 0.419 (0.68) |

the linear model on propagated features keeps *improving* up to 16 propagation steps (0.925, better than any trained network), then falls off a cliff: 0.45 at 32 steps and chance from 64 on. with PyTorch's default initialization, the plain GCN collapses at 8 layers, and so does the plain MLP of the same depth, which never sees the graph; both stop fitting even the training set. with kaiming initialization and zero biases the 8-layer GCN trains again (0.82). at 16 and 32 layers the kaiming GCN collapses anyway, fitting only 39–41% of its 60 training nodes, while the kaiming MLP of the same depth still fits 68–79% of them. with residual connections the GCN is at 0.89 with 32 layers.

**why:** three things are going on, and the controls separate them.

1. **oversmoothing, in its pure form** (left panel, no optimization involved). each propagation step averages away noise (the features carry a weak signal in a lot of noise, and neighbours mostly share the community), which is why accuracy *rises* for 16 steps. the community signal shrinks like $0.78^k$ relative to the degree component, the noise like $0.70^k$ or faster, so the signal-to-noise ratio improves while both are still representable. once the community component is $10^{-3}$–$10^{-4}$ of the dominant one ($k = 32$), a regularized classifier trained on 60 labels can't use it; at $10^{-7}$ ($k = 64$) it's below float32 resolution and gone. in exact arithmetic the information never vanishes on a connected graph; in practice it does, fast. one caveat: the logistic regression gets each propagated feature *standardized* (rescaled to unit variance), which rescues small components that a trained GCN, with no such step, would have to recover through its weights. so 0.906 at $k = 8$ is an upper bound on what averaging leaves, not what a GCN gets.
2. **trainability, at depth 8 with the default initialization.** the default-init MLP collapses at exactly the same depth without any graph, and fixing the initialization (chapter 15) brings the 8-layer GCN back to 0.82. so the depth-8 collapse in the first row is mostly the familiar deep-ReLU-network problem, not something about graphs.
3. **something graph-specific, at depth 16–32.** with a trainable initialization the MLP control still fits most of its training set at 16 and 32 layers, while the GCN of the same depth can't fit even its 60 training nodes. that is not oversmoothing of a fixed representation (a trained GCN has weights between the propagations that could, in principle, undo it); it's an optimization failure that only the graph version has. Cong et al. (2021) argue exactly this: deep GCNs remain expressive, and their trouble is in training and generalization rather than oversmoothing; Zhang et al. (2022) separate the depth of propagation from the depth of transformations and blame the latter ("model degradation").

the residual connection fixes both kinds of trainability failure, and the residual GCN keeps working at depth 32. it isn't really doing 32 rounds of averaging: the identity path lets each layer contribute a small correction, so the network can behave like a much shallower one, which is how it avoids the left panel's cliff. the lesson: when a deep GNN fails, check a same-depth MLP *with a trainable initialization* before blaming, or exonerating, the graph.

## experiment 3: when neighbours are not like you

**what people usually predict:** that more information can't hurt, so every GNN beats the MLP on both graphs, perhaps by less on the heterophilous one; and that GAT, which can learn whom to listen to, handles heterophily best.

**what happens:**

| graph | edge homophily | MLP | GCN | GraphSAGE | GAT |
|---|---|---|---|---|---|
| homophilous | 0.73 | 0.506 | **0.900** | 0.781 | 0.839 |
| heterophilous | 0.09 | 0.506 | 0.450 | **0.650** | 0.447 |

(edge homophily: the fraction of edges that join two nodes of the same community, seed 0.) on the homophilous graph every GNN beats the MLP by a lot, and GCN is best. on the heterophilous graph GCN and GAT are no better than ignoring the graph (a little worse here; a 10-seed run gives MLP 0.516 ± 0.016, GCN 0.478 ± 0.022, GAT 0.492 ± 0.027), while GraphSAGE gains 11–14 points over the MLP (14 in the 3-seed table; 0.624 ± 0.019 vs 0.516 over 10 seeds).

**why:** GCN puts a node's own features into the same weighted average as its neighbours'. with 91% of neighbours from other communities, that average is dominated by the *other two* classes' signals, and the node's own weak signal is diluted to about $1/\tilde d$ of the mix: the layer destroys the information the MLP was using. GAT's output is also a weighted mean over the same neighbourhood (self-loop included); attention can reweight the neighbours, but on these features (one noisy hint of the class) it has little basis for telling an informative neighbour from a misleading one, and a mean can't separate "me" from "them". GraphSAGE keeps the node's own features on a separate weight $W_1$, so nothing is diluted, and the neighbour mean is informative too: if most of my neighbours look like classes B and C, I'm probably class A. the model can learn exactly that "opposite" rule with a negative weight on the neighbour term. separating the ego embedding from the neighbourhood embedding is the first of the designs Zhu et al. (2020) recommend for heterophilous graphs.
