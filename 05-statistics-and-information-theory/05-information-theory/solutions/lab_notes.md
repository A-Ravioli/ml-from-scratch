# information theory · lab notes

## experiment 1: how close can a compressor get to the entropy?

**what people usually predict:** an optimal code should get close to the entropy; perhaps 0.5–0.6 bits per symbol for single symbols, a little better with blocks.

**what happens ($p = 0.1$, entropy 0.469 bits per symbol):**

| block size k | expected bits/symbol | achieved on 25,200 symbols |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.645 | 0.647 |
| 3 | 0.533 | 0.536 |
| 4 | 0.493 | 0.495 |
| 5 | 0.480 | 0.484 |
| 6 | 0.470 | 0.474 |
| 8 | 0.476 | 0.480 |
| 10 | 0.477 | 0.482 |

coding one bit at a time gives **no compression at all**: exactly 1 bit per symbol, more than twice the entropy. blocks fix it fast: within 5% of the entropy by $k = 5$, and within 0.3% at $k = 6$. the improvement isn't monotone (k = 6 beats k = 8 and 10), but every rate stays below the bound $H + 1/k$. the achieved rate on the actual message tracks the expected length to within about 1%.

**why:** a prefix code must spend at least one whole bit on every symbol it codes, so with a two-symbol alphabet nothing can be saved, however skewed the source. the source coding theorem bounds the waste at under one bit *per coded symbol*; coding blocks of $k$ shares that bit among $k$ source symbols, so the per-symbol overhead is below $1/k$. within that bound the overhead depends on how close the block probabilities are to powers of 2, which wobbles with $k$: that's the non-monotonicity. the measured rate differs from the expected one only through the LLN fluctuation of a 25,200-symbol message. arithmetic coding removes the integer-length problem altogether and gets within a couple of bits of $nH$ for the whole message.

**text (offline stand-in, 76,688 characters, 37 distinct):**

| method | bits per character |
|---|---|
| uniform code, $\log_2 37$ | 5.209 |
| entropy of the character frequencies | 3.924 |
| Huffman, one character at a time | 3.956 |
| entropy of character-pair frequencies, per character | 2.942 |
| Huffman on character pairs | 2.955 |
| zlib, level 9 | 2.250 |

Huffman on characters lands within 1% of the character-frequency entropy, and zlib beats that "entropy" by 43%. there's no contradiction: 3.92 bits is the entropy of a model that treats characters as **independent** draws from their frequencies. text isn't like that: after "th", "e" is far more likely than its overall frequency suggests. the pair frequencies already capture some of that (2.94 bits per character), and zlib, which replaces repeated strings with back-references, captures more, especially here, where the text is built from a small vocabulary of repeated words. the real limit is the **entropy rate** of the source, $\lim_n H(X_1, \dots, X_n)/n$, which accounts for all the dependence, and every compressor is implicitly a model of it. online, on the first 200,000 characters of real tiny Shakespeare (62 distinct characters), the same experiment gives:

| method | bits per character |
|---|---|
| uniform code, $\log_2 62$ | 5.954 |
| entropy of the character frequencies | 4.761 |
| Huffman, one character at a time | 4.799 |
| entropy of character-pair frequencies, per character | 4.113 |
| Huffman on character pairs | 4.128 |
| zlib, level 9 | 3.156 |

the ordering is the same, and zlib's margin over the character-frequency entropy is again large (34%), now from repeated words, names and phrases rather than from a tiny fixed vocabulary. for comparison, on the *full* text (different numbers, measured in the ch 04 capstone): a unigram model's test perplexity was 28.4, about 4.8 bits per character, and the best n-gram model reached 2.83 bits per character, below zlib. the neural language models of chapters 17–18 go lower still, which is why good language models make good compressors.

## experiment 2: dependence out of thin air

**what people usually predict:** close to 0, perhaps slightly positive by noise, and noise should average out over 200 repetitions.

**what happens (mean plug-in estimate in nats, true value 0; in brackets $(k - 1)^2/2n$):**

| categories k | n = 50 | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|---|
| 2 | 0.0100 (0.0100) | 0.0049 (0.0050) | 0.0005 (0.0005) | 0.0001 (0.0001) |
| 5 | 0.1896 (0.1600) | 0.0854 (0.0800) | 0.0080 (0.0080) | 0.0008 (0.0008) |
| 10 | 0.8148 (0.8100) | 0.4743 (0.4050) | 0.0416 (0.0405) | 0.0040 (0.0040) |

with 10 categories each and 100 samples, two variables that are independent look like they share 0.47 nats, about 20% of the maximum possible $\ln 10 = 2.3$. it's a bias, not noise: averaging repetitions doesn't remove it.

**why:** the plug-in estimate is $\mathrm{KL}(\hat p(x, y) \,\|\, \hat p(x)\hat p(y))$, and a KL is never negative. the empirical table always deviates a bit from an exact product of its marginals, and every deviation counts as dependence; nothing can cancel it. for large $n$, $2n\hat I$ is approximately chi-squared with $(k_x - 1)(k_y - 1)$ degrees of freedom (it's the likelihood-ratio statistic of the independence test), so $\mathbb{E}\hat I \approx (k_x - 1)(k_y - 1)/(2n)$, the bracketed numbers. the approximation underestimates when cells are nearly empty (k = 10, n = 50 has 0.5 samples per cell). the bias grows with the number of cells and shrinks only like $1/n$. so if you rank features by estimated mutual information with a label (feature selection, or decision-tree splits by information gain, ch 10), features with **many categories** win even when they're pure noise: an ID-like column with a different value in every row has the maximum possible plug-in MI with any label. corrections include subtracting the expected bias, a permutation test (topic 04), or the "gain ratio" used by C4.5.

## experiment 3: which way round is KL?

**what people usually predict:** both fits sit between the peaks with a large width; or both pick a peak.

**what happens:**

| fit | mu | sigma | KL(p ‖ q) | KL(q ‖ p) |
|---|---|---|---|---|
| forward, min KL(p ‖ q) | 0.00 | 2.10 | 0.554 | 1.255 |
| reverse, min KL(q ‖ p) | −2.00 | 0.60 | 10.419 | 0.692 |

the forward fit is a single broad bump centered at 0, *between* the peaks, where $p$ has almost no mass. the reverse fit sits on one peak (the left one here; the right one is equally good) with exactly that peak's width, and ignores the other entirely. each is terrible by the other's criterion.

**why:** in $\mathrm{KL}(p \,\|\, q) = \sum p\log(p/q)$ the expectation is under $p$. wherever $p$ has mass and $q$ is tiny, $\log(p/q)$ is huge, so $q$ must cover *all* of $p$'s mass: **mode covering** (or "mean seeking"). the cheapest single Gaussian that covers both peaks is the one matching $p$'s mean and variance, $\sigma = \sqrt{2^2 + 0.6^2} = 2.09$, and maximum likelihood finds exactly that, mass in the valley included. in $\mathrm{KL}(q \,\|\, p)$ the expectation is under $q$. wherever $q$ puts mass and $p$ is tiny, the cost is huge, so $q$ must stay *inside* $p$'s mass, and it can safely ignore regions of $p$ it doesn't visit: **mode seeking**. hugging one peak exactly costs $\log 2 = 0.693$ (the fitted value is 0.692): $q$ matches the peak's shape but has twice the density, since $p$ splits its mass between two peaks. for generative models, forward KL (maximum likelihood) gives blurry samples that cover all the data, including implausible in-between images; reverse-KL-like objectives (some GAN variants) give sharp samples that may miss modes ("mode collapse"). variational inference minimizes reverse KL, so its approximate posteriors tend to be overconfident and to lock onto one mode (ch 12).
