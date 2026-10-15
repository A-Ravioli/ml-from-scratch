# embeddings

## why this matters

the n-gram model of topic 01 treats every word as an unrelated symbol. it learns that "the king" is often followed by "said", and has no idea that "the queen" probably is too: "queen" and "king" are as different to it as "queen" and "spoon". every fact has to be learned separately for every word, which is why its counts thin out so fast. the fix, the idea under every neural language model since Bengio et al. (2003), is to represent each token as a **dense vector**, an embedding, so that similar tokens get similar vectors and whatever the model learns about one transfers to its neighbours.

where do good vectors come from? from the text itself. words that appear in similar contexts tend to mean similar things ("you shall know a word by the company it keeps", Firth, 1957). **word2vec** (Mikolov et al., 2013) turned that observation into a fast, simple training objective: predict which words appear near each word. the vectors it learns were the surprise of their year: nearest neighbours that make sense, and directions that seem to encode relations, so that king − man + woman lands near queen. in this topic you'll build it in PyTorch, your first model of the chapter on PyTorch, and then find out how much of the famous arithmetic survives on a small corpus, and what the vectors are really measuring.

## warm-up: predict first

1. a vocabulary has 50,000 words. computing a softmax over it for every training pair costs about how many multiply-adds per pair, with 100-dimensional vectors?
2. "love" and "hate": in a model that learns from neighbouring words, will they end up close together or far apart?
3. you sample negatives from word frequencies raised to the power 3/4 instead of the raw frequencies. which words gain probability: the frequent ones or the rare ones?
4. at initialization the output vectors are all zero. what is the loss of a pair with $K = 5$ negatives?
5. king − man + woman: is the vector nearest to that result usually "queen"?

## prerequisites check

- [ ] you know logistic regression, the sigmoid and the log-loss (→ ch 08 · topic 04)
- [ ] you can write a PyTorch module, a loss and a training loop with `torch.optim` (→ ch 15 · topic 07)
- [ ] you know cosine similarity and the SVD (→ ch 02 · topics 01 and 06)
- [ ] you know pointwise mutual information and KL divergence (→ ch 05 · topic 05)
- [ ] you've sampled from a categorical distribution (→ ch 04 · topic 02)

## core ideas

### idea 1: words are known by the company they keep

a one-hot vector for word $i$ is a row of the identity matrix: every pair of words is equally far apart, and nothing learned about one transfers to another. an **embedding** replaces it with a learned row $v_i \in \mathbb{R}^D$ of a matrix $V \in \mathbb{R}^{|V| \times D}$ (with $D$ in the tens to hundreds). in PyTorch that matrix is `nn.Embedding(vocab_size, embed_dim)`: indexing it with a tensor of ids is the same as multiplying one-hot vectors by the matrix, without ever building them.

**skip-gram** learns $V$ by turning raw text into a supervised problem: for each position $i$ (the **center** word), predict each word within a window of $\pm w$ positions (its **context**). the training set is every (center, context) pair:

$$\{(x_i, x_j) : 0 < |i - j| \le w\}.$$

for `[5, 6, 7, 8]` with $w = 1$: (5, 6), (6, 5), (6, 7), (7, 6), (7, 8), (8, 7). positions near the ends have fewer neighbours, so a text of $n$ words gives $\sum_i \big(\min(i, w) + \min(n - 1 - i, w)\big)$ pairs, about $2wn$. a word that appears in many of the same pairs as another word will be pushed towards the same place.

your `skipgram_pairs` returns them as a $(P, 2)$ `int64` array, ordered by center position, then by context position. with a megabyte of text that's over a million pairs, so build them with NumPy rather than a Python loop: an $(n, 2w)$ array of context positions `np.arange(n)[:, None] + offsets`, a mask of the ones inside the text, and two fancy-indexing lookups.

> **▶ build: stage 1.** implement `skipgram_pairs(ids, window=2)`.

### idea 2: negative sampling instead of a softmax

the obvious model predicts the context word with a softmax over the vocabulary, $P(o \mid c) = \exp(u_o^\top v_c)/\sum_{w}\exp(u_w^\top v_c)$, where $v_c$ is the center word's input vector and $u_o$ the context word's **output** vector (a second table). but the denominator touches every word: 50,000 dot products of length 100 per pair, five million multiply-adds (warm-up 1), for each of millions of pairs.

**negative sampling** asks an easier question: is this (center, context) pair real, or noise? for every real pair, draw $K$ "negative" words $n_1, \dots, n_K$ from a noise distribution and train a logistic classifier to tell them apart, with the dot product as its logit:

$$\ell(c, o) = -\log\sigma(u_o^\top v_c) - \sum_{k=1}^{K}\log\sigma(-u_{n_k}^\top v_c).$$

each pair now costs $K + 1$ dot products instead of $|V|$. it isn't the softmax's likelihood any more, but it rewards exactly what you want: real context words get large dot products with the center, random words get small ones.

**which noise?** Mikolov et al. found empirically that the best noise distribution is the unigram distribution raised to the power $3/4$:

$$P_n(w) = \frac{c(w)^{3/4}}{\sum_{w'}c(w')^{3/4}}.$$

the power flattens the distribution: frequent words ("the", "and") are sampled less often than their frequency, rare words more often (warm-up 3; on paper 3). with raw frequencies, most negatives would be the same handful of function words and rare words would almost never be pushed away from anything. your `sample_negatives(counts, n, *, power=0.75, generator=None)` draws $n$ ids with replacement using `torch.multinomial` and the given `generator`, so the course's tests and your runs are reproducible. one trap: `0 ** 0 == 1`, so with `power=0` a word with count 0 would be sampled; keep it out explicitly.

> **🔮 predict:** with counts $(1000, 10)$, what share of the negatives goes to the rare word with power 1? with power 3/4?

> **▶ build: stage 2.** implement `sample_negatives`.

### idea 3: the model is two lookup tables

`SkipGram(vocab_size, embed_dim)` holds two tables, `in_embed` (the center vectors $v$, the embeddings you keep) and `out_embed` (the context vectors $u$). its `forward(center, context, negatives)` takes a batch of $B$ centers, $B$ contexts and a $(B, K)$ tensor of negatives and returns the batch mean of $\ell$. three details make it correct and stable:

- **use `F.logsigmoid`**, not `torch.log(torch.sigmoid(x))`. for a score of $-100$, $\sigma(-100)$ underflows to 0 in float32 and its log is $-\infty$; `logsigmoid` computes $-\log(1 + e^{-x})$ stably.
- **batch the negatives.** `out_embed(negatives)` is $(B, K, D)$ and `v` is $(B, D)$; `torch.bmm(u_neg, v.unsqueeze(-1))` gives the $(B, K, 1)$ scores in one call.
- **initialize like word2vec.** the input vectors uniform in $(-0.5/D, 0.5/D)$, the output vectors all zero. then every score starts at 0, every sigmoid at $1/2$, and the loss of every pair is exactly $(K + 1)\log 2$ (warm-up 4: $6\log 2 = 4.159$ for $K = 5$), a free sanity check for the first batch. zero output vectors don't cause a symmetry problem: the input vectors are random, so the gradients the output vectors receive differ from the first step.

the property `embeddings` returns `in_embed.weight`. why two tables, and why keep the input one? a word rarely appears in its own context, so with a single table the model would have to make $v_w^\top v_w$ small for every word, fighting the pressure to make similar words similar. two tables avoid that; after training, the input vectors are the conventional choice (averaging the two also works).

> **▶ build: stage 3.** implement `SkipGram.__init__` and `SkipGram.forward`.

### idea 4: similarity is a cosine, analogies are offsets

after training, the length of a vector mostly tracks how frequent and how varied its word's contexts are, while the *direction* carries what it co-occurs with. so similarity is measured by the cosine, $\cos(a, b) = a^\top b/(\|a\|\|b\|)$: normalize every row once, and a matrix-vector product gives the similarity of one word to all others.

- **`most_similar(embeddings, idx, k=10)`** returns the $k$ rows with the highest cosine to row `idx`, excluding `idx` itself (which always has cosine 1), as `(indices, sims)` NumPy arrays, best first. accept a tensor or an array.
- **`analogy(embeddings, a, b, c, k=1)`** answers "$a$ is to $b$ as $c$ is to ?" with the 3CosAdd rule: normalize the rows, form $t = \hat v_b - \hat v_a + \hat v_c$, and return the $k$ words with the highest cosine to $t$, **excluding $a$, $b$ and $c$**.

the exclusion is not a detail. $\hat v_b - \hat v_a$ is the difference of two nearby unit vectors, often short compared with $\hat v_c$, so $t$ stays close to $c$ (and to $b$). without the exclusion the "answer" is very often one of the query words (Linzen, 2016). the lab measures how often.

> **🔮 predict:** on a corpus of 200,000 words, what fraction of simple analogies (he : him :: she : her, king : queen :: lord : lady, come : came :: go : went) will word2vec answer exactly right?

> **▶ build: stage 4.** implement `most_similar` and `analogy`.

### idea 5: training, and what it converges to

**`train_skipgram(ids, vocab_size, *, embed_dim=50, window=2, n_negatives=5, epochs=1, batch_size=512, lr=0.01, seed=0, device="cpu")`** puts the pieces together: build all pairs once, count the words with `np.bincount` (for the noise distribution), create the model, and for each epoch go through the pairs in a fresh random order in minibatches, drawing new negatives for every batch, with Adam. it returns the model and the list of batch losses.

**reproducibility without global state.** the course convention (CONTRACTS.md) is that torch code never relies on the global random generator. use one `torch.Generator().manual_seed(seed)` for the shuffling (`torch.randperm(..., generator=g)`) and the negatives. the initialization inside `SkipGram` uses the global generator, so seed it inside `with torch.random.fork_rng():`, which restores the global state afterwards.

**what does it learn?** Levy & Goldberg (2014) showed that if the vectors are free enough, the loss is minimized when every center-context score equals a shifted pointwise mutual information,

$$u_c^\top v_w = \log\frac{P(w, c)}{P(w)P_n(c)} - \log K,$$

so skip-gram with negative sampling implicitly factorizes a (shifted) PMI matrix (on paper 2). that's why the lab can compare it with a plain SVD of a co-occurrence matrix. it also explains the "love"/"hate" puzzle (warm-up 2): PMI is high for words that share contexts ("I ___ thee", "to ___ him"), and opposites share almost all of their contexts. these vectors measure **relatedness of usage**, not synonymy.

> **▶ build: stage 5.** implement `train_skipgram`.

## worked example

**pairs.** `skipgram_pairs([5, 6, 7, 8], window=1)` is `[[5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7]]`. with 37 words and window 2 there are $2 + 3 + 4 \cdot 33 + 3 + 2 = 142$ pairs.

**the loss at initialization.** every output vector is 0, so every score is 0 and $\ell = -\log\tfrac12 - 5\log\tfrac12 = 6\log 2 = 4.159$ for every pair, whatever the input vectors.

**the loss by hand.** $D = 2$, $v_c = (1, 0)$, $u_o = (1, 1)$, one negative with $u_n = (-1, 0)$. the positive score is $u_o^\top v_c = 1$, the negative score $u_n^\top v_c = -1$, and

$$\ell = -\log\sigma(1) - \log\sigma(-(-1)) = 2\log(1 + e^{-1}) = 2 \times 0.3133 = 0.6265.$$

**cosines.** rows $(1, 0)$, $(10, 1)$, $(0, 1)$, $(-1, 0)$, $(1, 1)$. the neighbours of row 0 are row 1 (cosine $10/\sqrt{101} = 0.995$, despite its length), row 4 ($1/\sqrt2 = 0.707$), row 2 (0); row 3 has cosine $-1$.

**an analogy.** man $= (0, 0, 1)$, woman $= (0, 1, 1)$, king $= (1, 0, 1)$, queen $= (1, 1, 1)$, plus a distractor $(1, 0.05, 1.05)$ very close to king. normalized, $t$ = woman − man + king has cosine 0.975 with queen, against 0.750 for the distractor and 0.733 for king itself (excluded anyway). a second case shows why the exclusion exists: with rows $a = (1, 0)$, $b = (1, 0.1)$, $c = (0, 1)$ and $d = (-0.3, 1)$, the offset $b - a$ is tiny, $t$ lands almost on $c$, and only the exclusion makes $d$ the answer. the tests check all of these.

## on paper

1. **the gradient.** differentiate $\ell$ with respect to $v_c$, $u_o$ and $u_{n_k}$. what does each update do geometrically?

<details><summary>solution</summary>

with $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$: $\partial\ell/\partial v_c = -(1 - \sigma(u_o^\top v_c))u_o + \sum_k\sigma(u_{n_k}^\top v_c)u_{n_k}$, $\partial\ell/\partial u_o = -(1 - \sigma(u_o^\top v_c))v_c$, $\partial\ell/\partial u_{n_k} = \sigma(u_{n_k}^\top v_c)v_c$. gradient descent moves $v_c$ towards the context's output vector and away from the negatives', each weighted by how wrong the classifier currently is: a pair already scored confidently ($\sigma \approx 1$ for the real one, $\approx 0$ for noise) contributes almost nothing.

</details>

2. **the optimum is shifted PMI.** treat each score $x = u_c^\top v_w$ as a free variable. the expected loss over the data and the noise contributes $-\#(w, c)\log\sigma(x) - K\,\#(w)P_n(c)\log\sigma(-x)$ for that pair, where $\#$ are counts. show that the minimum is at $x = \log\frac{\#(w, c)}{\#(w)P_n(c)} - \log K$.

<details><summary>solution</summary>

the derivative is $-\#(w, c)(1 - \sigma(x)) + K\#(w)P_n(c)\sigma(x)$. setting it to 0: $\frac{\sigma(x)}{1 - \sigma(x)} = e^x = \frac{\#(w, c)}{K\#(w)P_n(c)}$, so $x = \log\frac{\#(w, c)}{\#(w)P_n(c)} - \log K$. with $P(w, c) = \#(w, c)/N$ and $P(w) = \#(w)/N$, that's $\log\frac{P(w, c)}{P(w)P_n(c)} - \log K$: the PMI with the smoothed context distribution, shifted by $\log K$ (Levy & Goldberg, 2014). with only $D$ dimensions per word the model can't hit every target exactly, so it finds a low-rank approximation, like an SVD does.

</details>

3. **the 3/4 power.** two words have counts 1,000 and 10. what share of the negatives does the rare one get with power 1, 3/4 and 0?

<details><summary>solution</summary>

power 1: $10/1010 = 1.0\%$. power 3/4: $10^{0.75} = 5.62$, $1000^{0.75} = 177.8$, share $5.62/183.4 = 3.1\%$, three times more. power 0: 50%. the 3/4 power is a compromise: rare words get pushed away from things often enough to get useful vectors, without negatives being dominated by the rare tail.

</details>

## experiment

`lab.py` trains your skip-gram on the words of tiny Shakespeare (3,330 words that occur at least 5 times):

1. nearest neighbours, and the cosines of synonym, antonym and related pairs.
2. a hand-made analogy set, with and without excluding the query words, for two seeds.
3. a count-based alternative with no training loop: positive PMI plus an SVD, against your word2vec.

## check your understanding

1. why does negative sampling need a separate output table, rather than scoring $v_o^\top v_c$ with one table?

<details><summary>answer</summary>

with one table, a word that occurs in its own context rarely would need $v_w^\top v_w = \|v_w\|^2$ small, and words that co-occur would need large dot products, so similar words with similar vectors would also predict each other as context, whether or not they co-occur. two tables decouple "what I am" (input) from "what I predict around me" (output). with two tables, synonyms that never appear next to each other can still get similar input vectors because they predict the same contexts.

</details>

2. your loss starts at 4.159 with $K = 5$ and never moves. what's the likeliest bug?

<details><summary>answer</summary>

no gradient reaches the parameters: the optimizer was built on the wrong parameters (e.g. before moving the model, or on a different model), `zero_grad`/`step` are missing, or the scores were computed from detached tensors. 4.159 is exactly $6\log 2$, the loss with every score at 0, i.e. with the output table still all zeros.

</details>

3. the vectors of "love" and "hate" have cosine above 0.5. is the embedding broken?

<details><summary>answer</summary>

no. skip-gram measures shared contexts, and antonyms share nearly all of theirs: both are verbs and nouns of feeling with the same subjects and objects. the vectors capture relatedness of usage; distinguishing synonyms from antonyms needs other signals (lexical resources, or the much richer contexts of large language models).

</details>

4. why are analogy benchmarks reported with the query words excluded, and why should you be suspicious of that?

<details><summary>answer</summary>

because otherwise the nearest vector to $b - a + c$ is usually $b$ or $c$, and the benchmark would measure almost nothing. but the exclusion means "the closest word that isn't one of the inputs", which can look like reasoning when it's mostly a neighbourhood search around $c$. in the lab, without the exclusion, 92% of the answers are one of the query words.

</details>

5. why is the dot product of two embeddings a reasonable logit, rather than, say, their distance?

<details><summary>answer</summary>

it's bilinear and cheap: a batch of scores is one matrix product, and the gradients are simple (on paper 1). it also matches the optimum: the score approximates a log-ratio of probabilities (PMI), and a low-rank matrix of log-ratios is exactly a matrix of dot products. the same reasoning makes the output layer of every language model a dot product between a hidden state and token embeddings.

</details>

6. you train twice with different seeds and the nearest neighbours of a rare word change completely. what does that tell you?

<details><summary>answer</summary>

the rare word's vector is poorly determined: it appears in few pairs, so its position depends on initialization and sampling noise. frequent words' neighbourhoods are much more stable. that's why word2vec drops words below a minimum count, and why conclusions drawn from one rare word's neighbours are unreliable.

</details>

## common bugs

- **`torch.log(torch.sigmoid(x))`.** symptom: the loss becomes `inf` or `nan` once scores get large. use `F.logsigmoid`.
- **summing over the batch instead of averaging.** symptom: the loss scales with the batch size, and the learning rate that works depends on it.
- **sampling negatives from raw counts, or forgetting zero counts with power 0.** symptom: the distribution test fails; negatives dominated by "the", "and", "i".
- **`analogy` without normalizing the rows first.** symptom: frequent words with long vectors win every analogy.
- **forgetting to exclude $a$, $b$, $c$.** symptom: the "answer" to king : queen :: lord : ? is "lord".
- **seeding with `torch.manual_seed` and leaving it there.** symptom: other code's randomness changes after you train; or two runs with the same seed differ because the shuffling used an unseeded generator.

## going deeper

- Mikolov, Sutskever, Chen, Corrado & Dean (2013), "Distributed representations of words and phrases and their compositionality", *NeurIPS*. negative sampling, the 3/4 power, subsampling of frequent words. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
- Mikolov, Chen, Corrado & Dean (2013), "Efficient estimation of word representations in vector space". skip-gram, CBOW and the analogy benchmark. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
- Levy & Goldberg (2014), "Neural word embedding as implicit matrix factorization", *NeurIPS*. skip-gram factorizes shifted PMI. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html)
- Levy, Goldberg & Dagan (2015), "Improving distributional similarity with lessons learned from word embeddings", *TACL* 3: 211–225. count-based methods with word2vec's tricks match it. [doi:10.1162/tacl_a_00134](https://doi.org/10.1162/tacl_a_00134)
- Linzen (2016), "Issues in evaluating semantic spaces using word analogies", *RepEval workshop*. what the exclusion rule hides. [arXiv:1606.07736](https://arxiv.org/abs/1606.07736)
- Bengio, Ducharme, Vincent & Jauvin (2003), "A neural probabilistic language model", *JMLR* 3: 1137–1155. the first neural language model with learned embeddings. [jmlr.org/papers/v3/bengio03a](https://www.jmlr.org/papers/v3/bengio03a.html)
- Pennington, Socher & Manning (2014), "GloVe: global vectors for word representation", *EMNLP*. [aclanthology.org/D14-1162](https://aclanthology.org/D14-1162/)

## connections

- ← builds on: logistic regression and the log-loss (ch 08), PyTorch modules and training loops (ch 15 · topic 07), cosines and the SVD (ch 02), PMI (ch 05), sampling (ch 04), the vocabulary and tokens of topics 01–02.
- → used in: every model from here on starts with `nn.Embedding`, now unlocked: the RNN and LSTM language models (topics 04–05), seq2seq (topic 06), the capstone's character LSTM, GPT's token embeddings and their tied output layer (ch 18), contrastive objectives like InfoNCE, which generalize negative sampling (ch 25), and retrieval with cosine similarity (ch 27).
