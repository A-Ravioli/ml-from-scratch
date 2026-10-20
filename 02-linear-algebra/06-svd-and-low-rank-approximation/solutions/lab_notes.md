# SVD & low-rank approximation · lab notes

## experiment 1: image compression, and what "low rank" means for a picture

**what people usually predict:** (a) a face needs a large fraction of its 64 ranks to be recognizable. (b) the two squares are equally simple, so they need the same rank.

**what happens:** (a) on the real Olivetti faces, most people recognize the face somewhere around rank 10–20. storage is $k(64 + 64 + 1)$ numbers against $64 \cdot 64 = 4096$, so about 3% per rank: rank 10 costs 31%, rank 20 costs 63%, and compression stops paying at all around $k \approx 32$. a single small image is a poor case for SVD compression. it pays off for big matrices with fast-decaying spectra (and for *collections* of images, which is the capstone's PCA). the singular values of a real face fall off steeply at first, then flatten into a long tail of fine detail and noise. (the offline synthetic stand-in is a smooth blob plus pixel noise. it needs more rank for 99.9% energy because noise is incompressible, but the smooth part appears within the first few ranks.) (b) the upright square is **exactly rank 1**: one nonzero singular value, perfect at k = 1. the 45° square needs about **21 ranks for 99.9% of its energy**, and its numerical rank is 22.

**why:** an upright square is $\mathbf 1_{[a,b]}(i)\cdot\mathbf 1_{[a,b]}(j)$, an outer product $xy^\top$ of two indicator vectors, so it's rank 1. rotation turns the edges into staircases. every row is a different-length run of ones, centered at the same place, and runs of different lengths aren't multiples of each other. the diamond spans 44 rows, but it is mirror-symmetric top to bottom, so its rows come in identical pairs: only 22 are distinct, hence rank 22, not 44. the SVD measures simplicity *with respect to the coordinate axes*. that's the right notion when the axes mean something (users × items, samples × features). for images it's a weakness, which is why image codecs use localized bases (DCT blocks, wavelets) and why convolutional networks (chapter 16) assume translation structure instead of low rank.

## experiment 2: why not just use the eigenvalues of $A^\top A$?

**what people usually predict:** both methods get all or nearly all of the 16 values. maybe the Gram approach loses one or two at the very bottom.

**what happens:** your `svd` gets **14 of 16** right to 3 digits. it only misses $10^{-14}$ and $10^{-15}$, which are within a few rounding errors of zero ($\varepsilon\sigma_1 \approx 2 \times 10^{-16}$). the $\sqrt{\operatorname{eig}(A^\top A)}$ recipe gets only **7 of 16**. everything below about $10^{-7}$ comes out as a flat floor of noise near $10^{-8}$, and `eigh` returns several *negative* eigenvalues for a matrix that is mathematically positive semidefinite.

**why:** forming $A^\top A$ squares the singular values, so the spread goes from $10^{0}$–$10^{-15}$ to $10^{0}$–$10^{-30}$. but any eigensolver, however good, can only resolve eigenvalues to an absolute accuracy of about $\varepsilon\|A^\top A\| \approx 10^{-16}$. every $\sigma_i^2$ below $10^{-16}$ (that's $\sigma_i < 10^{-8}$) drowns in rounding noise, and its square root, about $10^{-8}$, is meaningless. the damage is done by the multiplication $A^\top A$, before any solver runs. one-sided Jacobi works on $A$'s columns directly and gets *absolute* error $\approx \varepsilon\sigma_1$ for every singular value. (it's actually better than that for many matrices: Demmel & Veselić showed one-sided Jacobi often gets high *relative* accuracy.) the same lesson as normal equations versus QR in topic 04: **never square a matrix you could factor instead.**

## experiment 3: PCA is an SVD (of centered data)

**what people usually predict:** both arrows point along the long axis $(1, -1)/\sqrt2$, perhaps one a little off.

**what happens:** the centered SVD's first right singular vector is about $(0.69, -0.72)$, along the cloud. the uncentered one points about $(0.85, 0.52)$: **toward the mean** $(5, 3)$ from the origin, nearly perpendicular to the cloud's long axis.

**why:** $\|Xv\|^2 = \sum_i (x_i^\top v)^2 = n(\bar x^\top v)^2 + \|X_cv\|^2$ (the cross term vanishes because the centered rows sum to zero). for uncentered data, the first term is $300 \cdot \|(5,3)\|^2 \approx 10{,}000$ along the mean direction, against $\|X_cv\|^2 \approx 300 \cdot 4 = 1{,}200$ along the long axis. the mean wins. centering removes the first term, so $\|X_cv\|^2/(n-1)$ is exactly the sample variance along $v$, and the top singular vector is the direction of greatest variance. every PCA implementation (including the capstone's and chapter 11's) centers first. a model that "discovers" that the main direction of variation points toward the mean has almost always skipped this step.
