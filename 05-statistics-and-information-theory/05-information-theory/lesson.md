# information theory

## why this matters

every classifier you'll train in this course minimizes a **cross-entropy**. language models are scored by **perplexity**, an exponentiated cross-entropy. variational autoencoders and variational inference minimize a **KL divergence**; so do knowledge distillation and the trust regions of RL algorithms like PPO. contrastive representation learning maximizes a bound on **mutual information**. decision trees pick splits by **information gain**. these words are everywhere in ML, and they all come from one 1948 paper about sending messages down telephone wires.

Shannon's question was: how many bits does it take to transmit the output of a random source? his answer, the **entropy**, turns out to measure uncertainty itself, and once you have it, a whole family of quantities follows: how many extra bits you waste by using the wrong distribution (KL divergence), how much one variable tells you about another (mutual information). this topic builds them all, for discrete distributions and for Gaussians, and makes them concrete with an actual compressor: a **Huffman coder**, which turns a probability distribution into a code whose average length you can measure and compare with the entropy. in the lab you'll watch the measured compression rate close in on the entropy, and see a real compressor appear to beat it.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. a source emits one of four symbols with probabilities $(1/2, 1/4, 1/8, 1/8)$. how many bits per symbol do you need on average, at best? less than 2?
2. is the KL divergence a distance? is $\mathrm{KL}(p \,\|\, q) = \mathrm{KL}(q \,\|\, p)$?
3. you train a classifier by minimizing cross-entropy. what is the smallest the loss can possibly get, and what is it then measuring?
4. can the entropy of a continuous distribution be negative?
5. a source emits 1 with probability 0.1 and 0 otherwise. its entropy is 0.47 bits. how many bits per symbol does an optimal code for *single* symbols use?

## prerequisites check

- [ ] you know expectation, and $\mathbb{E}[g(X)] = \sum_x p(x) g(x)$ (→ ch 04 · topic 03)
- [ ] you know logs, $\log_b x = \ln x / \ln b$, and $\log(ab) = \log a + \log b$ (→ ch 01 · topic 01)
- [ ] you know the multivariate Gaussian and its log-density (→ ch 04 · topic 04)
- [ ] you can use your `cholesky` and `cho_solve` (→ ch 02 · topic 03)
- [ ] you know maximum likelihood (→ topic 02)

## core ideas

### idea 1: surprise, entropy, and the cost of being wrong

how surprised should you be when an event of probability $p$ happens? a natural measure is $-\log p$: certain events ($p = 1$) carry no surprise, rare ones a lot, and the surprises of independent events add, because probabilities multiply. the **entropy** of a distribution is the expected surprise:

$$H(p) = -\sum_x p(x)\log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)].$$

with base-2 logs it's measured in **bits**, with natural logs in **nats** ($1 \text{ nat} = 1/\ln 2 \approx 1.44$ bits). terms with $p(x) = 0$ contribute 0 (the limit of $p\log p$ as $p \to 0$). entropy is 0 for a certain outcome, and maximal, $\log k$, for the uniform distribution on $k$ outcomes. a fair coin has 1 bit; the distribution $(1/2, 1/4, 1/4)$ has 1.5.

the coding interpretation makes this concrete (idea 4 proves it): $H(p)$ in bits is the minimum average number of bits per symbol needed to encode symbols drawn from $p$. the ideal code gives symbol $x$ a codeword of about $-\log_2 p(x)$ bits: short codes for common symbols. for warm-up 1, that's lengths $(1, 2, 3, 3)$, averaging $1.75 = H$ bits.

now suppose the symbols come from $p$, but you designed your code for a different distribution $q$. you pay $-\log q(x)$ per symbol, so on average the **cross-entropy**

$$H(p, q) = -\sum_x p(x)\log q(x).$$

the excess over the best possible is the **Kullback–Leibler divergence**:

$$\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ \ge\ 0.$$

it's zero exactly when $q = p$ (Gibbs' inequality, on paper 1). it's infinite if $q(x) = 0$ for some $x$ with $p(x) > 0$: a code that has no codeword for something that happens. and it is **not symmetric** (warm-up 2): $\mathrm{KL}(p \,\|\, q) \ne \mathrm{KL}(q \,\|\, p)$ in general, and it violates the triangle inequality, so it's a divergence, not a distance.

this is warm-up 3. a classifier's cross-entropy loss is $H(p_{\text{data}}, q_{\text{model}})$ estimated on the training set, and it equals $H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}} \,\|\, q_{\text{model}})$. the model can only shrink the KL term. the floor is the entropy of the labels given the inputs: the irreducible noise. minimizing cross-entropy is minimizing KL from data to model, which is maximum likelihood (topic 02).

> **▶ build: stage 1.** implement `entropy(p, *, base=None)`, `cross_entropy(p, q, *, base=None)` and `kl_divergence(p, q, *, base=None)` in `exercise.py` (`marimo edit exercise.py`). normalize inputs first (a helper, `normalize`, is provided).

### idea 2: mutual information

how much does knowing $Y$ tell you about $X$? measure it as the drop in uncertainty. the **conditional entropy** $H(X \mid Y) = \sum_y p(y) H(X \mid Y = y)$ is the average uncertainty left about $X$ once you know $Y$, and the **mutual information** is the reduction:

$$I(X; Y) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X, Y).$$

the second form shows it's symmetric: $Y$ tells you exactly as much about $X$ as $X$ tells you about $Y$. there's a third form, the most useful one:

$$I(X; Y) = \sum_{x,y} p(x, y)\log\frac{p(x, y)}{p(x)p(y)} = \mathrm{KL}\big(p(x, y) \,\|\, p(x)p(y)\big).$$

mutual information is the KL divergence from the joint distribution to the one you'd have if $X$ and $Y$ were independent. so it's $\ge 0$, and it's 0 exactly when they're independent. unlike correlation, which only sees linear relationships, MI detects any dependence at all: if $Y = X^2$ with $X$ symmetric, the correlation is 0 but the MI is large. at the other extreme, if $Y$ is a copy of $X$, then $I(X; Y) = H(X)$: everything there is to know.

you'll compute it from a joint probability table $p(x, y)$, or from a table of counts, normalized. in practice those counts come from data, and the "plug-in" estimate from a finite table has a bias you'll measure in the lab.

> **🔮 predict:** $X$ and $Y$ are independent and uniform over 10 values each. you draw 100 pairs, tabulate the counts, and plug the table into the formula. is the average estimate of $I(X; Y)$ close to 0? (the lab measures it.)

> **▶ build: stage 2.** implement `mutual_information(joint, *, base=None)` for a 2-D table.

### idea 3: entropy and KL for Gaussians

for a continuous density $f$, replace sums with integrals: the **differential entropy** is $h(f) = -\int f(x)\log f(x)\,dx$. it behaves like entropy in most ways, with one big difference (warm-up 4): it can be negative, and it changes when you rescale. a uniform distribution on $[0, a]$ has $h = \log a$, which is negative for $a < 1$. differential entropy measures uncertainty *relative to the units*; it's not a bit count. differences of differential entropies, like KL and MI, don't have this problem: they're invariant to rescaling and always $\ge 0$.

for a $d$-dimensional Gaussian $\mathcal{N}(\mu, \Sigma)$, the expected negative log-density works out to (on paper 2)

$$h = \frac d2\log(2\pi e) + \frac12\log\det\Sigma.$$

the mean doesn't matter, only the volume of the covariance ellipsoid, through $\det\Sigma$. among all distributions with a given covariance, the Gaussian has the largest entropy: it's the "least informative" distribution with those second moments, one reason it's the default noise model everywhere.

the KL divergence between two Gaussians also has a closed form:

$$\mathrm{KL}\big(\mathcal{N}(\mu_0, \Sigma_0) \,\|\, \mathcal{N}(\mu_1, \Sigma_1)\big) = \frac12\Big[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^\top\Sigma_1^{-1}(\mu_1 - \mu_0) - d + \log\frac{\det\Sigma_1}{\det\Sigma_0}\Big].$$

in 1-D it reads $\log\frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2} - \frac12$. this is the KL term in every variational autoencoder (ch 21). compute it the way ch 02 taught you: with a Cholesky factor $\Sigma = LL^\top$, $\log\det\Sigma = 2\sum_i\log L_{ii}$, and $\Sigma_1^{-1}v$ comes from `cho_solve`, never from an explicit inverse.

> **▶ build: stage 3.** implement `gaussian_entropy(cov)` and `gaussian_kl(mu0, cov0, mu1, cov1)` using your `cholesky` and `cho_solve`.

### idea 4: codes that achieve the entropy

a **prefix code** assigns each symbol a string of bits so that no codeword is the beginning of another. that makes a stream of codewords decodable on the fly, with no separators: read bits until they spell a codeword, emit it, repeat. which codeword lengths $\ell_1, \dots, \ell_k$ are possible? the **Kraft inequality** says a prefix code with these lengths exists if and only if

$$\sum_i 2^{-\ell_i} \le 1.$$

(think of the codewords as leaves of a binary tree: a leaf at depth $\ell$ uses up a $2^{-\ell}$ share of the tree.) minimizing the expected length $\sum_i p_i\ell_i$ subject to Kraft gives, if lengths could be fractional, $\ell_i = -\log_2 p_i$ and expected length exactly $H(p)$. lengths must be integers, so round up: $\ell_i = \lceil -\log_2 p_i\rceil$ still satisfies Kraft and costs less than one extra bit. that's **Shannon's source coding theorem** for symbol codes: the best expected length $L$ satisfies

$$H(p) \le L < H(p) + 1.$$

**Huffman's algorithm** (1952) finds an optimal code exactly, greedily. put every symbol in a min-heap keyed by probability. repeatedly pop the two least probable nodes, merge them into one node with the summed probability, and push it back. each merge adds one bit to the codewords of every symbol inside the two merged nodes: a '0' for one side, a '1' for the other. after $k - 1$ merges, one node is left and every symbol has its codeword. when probabilities tie, any choice gives an optimal code (different codes, same expected length).

the "+1" in the bound is the cost of integer lengths, and it can dominate. for warm-up 5: a source with $P(1) = 0.1$ has entropy 0.469 bits, but any code for single symbols needs at least one bit per symbol, so the best you can do is 1.0: no compression at all. the fix is to code **blocks** of $k$ symbols as one super-symbol. the block has entropy $kH$ (for independent symbols), so the bound becomes $kH \le L_k < kH + 1$, and per symbol, $H \le L_k/k < H + 1/k$. the overhead is shared among $k$ symbols.

> **🔮 predict:** coding blocks of $k$ bits from the $P(1) = 0.1$ source, how big must $k$ be before the rate is within 5% of the entropy? (the lab measures it.)

> **▶ build: stage 4.** implement `huffman_code(probs)`, returning one bit string per symbol, and `expected_length(probs, code)`.

### idea 5: compressing a message

with a code in hand, compression is concatenation, and decompression is the prefix property in action. to **encode** a sequence of symbols, join their codewords. to **decode**, scan the bits left to right, accumulating a buffer; whenever the buffer equals a codeword, output its symbol and clear the buffer. because the code is prefix-free, the first match is the only possible one. if the bits run out with a non-empty buffer, the input was corrupt.

that's a working compressor. its rate on a long message from the source approaches its expected length, by the law of large numbers. real compressors go further in two ways. **arithmetic coding** avoids the integer-length overhead entirely, coding a whole message in about $nH + 2$ bits. and **modeling** improves the $p$ you code with: English letters aren't independent, and a model that predicts the next character from the previous ones assigns better probabilities, hence shorter codes. that's where compression and machine learning meet: a better predictive model *is* a better compressor, and a language model's cross-entropy loss is literally the number of bits per token it would need.

> **▶ build: stage 5.** implement `encode(symbols, code)` and `decode(bits, code)`.

## worked example

**entropy and Huffman for a dyadic source.** for $p = (1/2, 1/4, 1/8, 1/8)$: $H = \frac12 \cdot 1 + \frac14 \cdot 2 + \frac18 \cdot 3 + \frac18 \cdot 3 = 1.75$ bits. Huffman merges the two $1/8$s into $1/4$, then the two $1/4$s into $1/2$, then the two $1/2$s. the codeword lengths are $(1, 2, 3, 3)$, for example $(0, 10, 110, 111)$, and the expected length is exactly 1.75: every probability is a power of 2, so $-\log_2 p_i$ are integers and the code matches the entropy. for $(1/2, 1/4, 1/4)$, $H = 1.5$ bits.

**KL is asymmetric.** with $p = (0.5, 0.5)$ and $q = (0.9, 0.1)$: $\mathrm{KL}(p \,\|\, q) = 0.5\ln\frac{0.5}{0.9} + 0.5\ln\frac{0.5}{0.1} = 0.511$ nats, while $\mathrm{KL}(q \,\|\, p) = 0.9\ln 1.8 + 0.1\ln 0.2 = 0.368$ nats.

**mutual information.** for the joint table $\begin{pmatrix}0.4 & 0.1\\0.1 & 0.4\end{pmatrix}$, both marginals are uniform (1 bit each), and $H(X, Y) = -2(0.4\log_2 0.4 + 0.1\log_2 0.1) = 1.722$ bits, so $I = 1 + 1 - 1.722 = 0.278$ bits: knowing $Y$ removes about 28% of the uncertainty about $X$.

**a Gaussian KL.** $\mathrm{KL}(\mathcal{N}(0, 1) \,\|\, \mathcal{N}(1, 4)) = \log 2 + \frac{1 + 1}{8} - \frac12 = 0.443$ nats.

**encoding.** with the code $(0, 10, 11)$, the symbols $(0, 2, 1, 0)$ encode to $0\,|\,11\,|\,10\,|\,0$ = `011100`, and decoding reads 0 → symbol 0, 1 (not a codeword), 11 → symbol 2, 1, 10 → symbol 1, 0 → symbol 0. the bits `01` decode to symbol 0 and a dangling 1: an error.

the tests check all of these.

## on paper

1. **Gibbs' inequality.** prove $\mathrm{KL}(p \,\|\, q) \ge 0$ using $\ln t \le t - 1$, and find when equality holds.

<details><summary>solution</summary>

$-\mathrm{KL}(p \,\|\, q) = \sum_{x: p(x) > 0} p(x)\ln\frac{q(x)}{p(x)} \le \sum_{x: p(x) > 0} p(x)\Big(\frac{q(x)}{p(x)} - 1\Big) = \sum_{x: p(x) > 0} q(x) - 1 \le 0.$ equality needs $q(x)/p(x) = 1$ wherever $p(x) > 0$ (since $\ln t = t - 1$ only at $t = 1$) and no $q$-mass outside $p$'s support, so $q = p$. (Jensen's inequality on the concave log gives the same result.)

</details>

2. **Gaussian entropy.** compute $h = \mathbb{E}[-\log f(X)]$ for $X \sim \mathcal{N}(\mu, \Sigma)$, using $\mathbb{E}[(X - \mu)^\top\Sigma^{-1}(X - \mu)] = d$.

<details><summary>solution</summary>

$-\log f(x) = \frac d2\log 2\pi + \frac12\log\det\Sigma + \frac12(x - \mu)^\top\Sigma^{-1}(x - \mu)$. taking expectations, the last term is $d/2$ (it's $\frac12\operatorname{tr}(\Sigma^{-1}\mathbb{E}[(X - \mu)(X - \mu)^\top]) = \frac12\operatorname{tr}(I_d)$), so $h = \frac d2(\log 2\pi + 1) + \frac12\log\det\Sigma = \frac d2\log(2\pi e) + \frac12\log\det\Sigma$.

</details>

3. **the 1-D Gaussian KL.** derive $\mathrm{KL}(\mathcal{N}(\mu_0, \sigma_0^2) \,\|\, \mathcal{N}(\mu_1, \sigma_1^2))$ from the definition.

<details><summary>solution</summary>

$\mathrm{KL} = \mathbb{E}_0[\log f_0(X) - \log f_1(X)]$ with $X \sim \mathcal{N}(\mu_0, \sigma_0^2)$. $\log f_0 - \log f_1 = \log\frac{\sigma_1}{\sigma_0} - \frac{(X - \mu_0)^2}{2\sigma_0^2} + \frac{(X - \mu_1)^2}{2\sigma_1^2}$. take expectations: $\mathbb{E}(X - \mu_0)^2 = \sigma_0^2$ and $\mathbb{E}(X - \mu_1)^2 = \sigma_0^2 + (\mu_0 - \mu_1)^2$. so $\mathrm{KL} = \log\frac{\sigma_1}{\sigma_0} - \frac12 + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2}$.

</details>

4. **Shannon code lengths.** show that $\ell_i = \lceil -\log_2 p_i\rceil$ satisfies Kraft, and that its expected length is less than $H + 1$.

<details><summary>solution</summary>

$\ell_i \ge -\log_2 p_i$ gives $2^{-\ell_i} \le p_i$, so $\sum_i 2^{-\ell_i} \le \sum_i p_i = 1$. and $\ell_i < -\log_2 p_i + 1$, so $\sum_i p_i\ell_i < \sum_i p_i(-\log_2 p_i + 1) = H + 1$. Huffman is at least as good as this code, so it satisfies the same bound.

</details>

## experiment

`lab.py` runs three experiments with your code:

1. Huffman coding of single bits and of blocks of bits, against the entropy; then text, against `zlib`.
2. estimating the mutual information of two independent variables from samples.
3. fitting one Gaussian to a two-peaked distribution by minimizing KL in each direction.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. what's the entropy, in bits, of a fair six-sided die? of a die that always shows 6?

<details><summary>answer</summary>

$\log_2 6 \approx 2.585$ bits, and 0 bits.

</details>

2. a classifier's validation cross-entropy is 0.30 nats per example and the labels' entropy given the inputs is 0.25 nats. what's the KL from the data to the model?

<details><summary>answer</summary>

$0.30 - 0.25 = 0.05$ nats. that's the part of the loss the model could still remove; the 0.25 is irreducible label noise.

</details>

3. why is $\mathrm{KL}(p \,\|\, q) = \infty$ when $q(x) = 0$ for some $x$ with $p(x) > 0$? what does this mean for a model's predicted probabilities?

<details><summary>answer</summary>

the term $p(x)\log\frac{p(x)}{0}$ is infinite: the model calls something impossible that happens. a model trained with cross-entropy is punished without limit for confident mistakes, which is why its probabilities should never be exactly 0 (and why `log(0)` shows up as `inf` or `nan` in buggy training loops).

</details>

4. two variables have correlation 0. can their mutual information be positive? can it be zero while the correlation is 0.9?

<details><summary>answer</summary>

yes to the first ($Y = X^2$ for symmetric $X$). no to the second: correlation $\ne 0$ means dependence, and MI is zero only for independent variables.

</details>

5. the differential entropy of a 1-D Gaussian with variance $10^{-4}$ is negative. is that a bug?

<details><summary>answer</summary>

no: $\frac12\log(2\pi e \cdot 10^{-4}) \approx -3.2$ nats. differential entropy isn't a count of bits; it's relative to the units. measure the same variable in different units and it shifts by the log of the scale factor.

</details>

6. why does Huffman coding of single bits never compress, however skewed the source?

<details><summary>answer</summary>

with two symbols, each codeword needs at least one bit, so the expected length is at least 1 bit per symbol whatever the probabilities. the gap to the entropy can approach 1 bit. block coding (or arithmetic coding) is needed to spread that overhead over many symbols.

</details>

7. a language model has a test cross-entropy of 1.0 bits per character. how big would a file of 1 million characters be if you compressed it with an arithmetic coder driven by this model?

<details><summary>answer</summary>

about $10^6$ bits $\approx 125$ kB (plus a few bytes of overhead). the cross-entropy is exactly the average code length the model would assign. better language models are better compressors, which is the basis of using compression benchmarks to evaluate them.

</details>

## common bugs

- **`0 * log 0 = nan`.** `p * np.log(p)` with a zero entry gives `nan`, and the whole sum becomes `nan`. mask out zero probabilities.
- **mixing bits and nats.** a formula with `np.log` gives nats; dividing by `np.log(2)` gives bits. symptom: answers off by a factor of 1.44.
- **KL with the arguments swapped.** $\mathrm{KL}(p \,\|\, q)$ weights by $p$, the first argument. symptom: a mismatch with `scipy.stats.entropy(p, q)`.
- **`np.linalg.inv` in the Gaussian KL.** works for small, well-conditioned matrices; loses accuracy otherwise. use the Cholesky factor.
- **a max-heap instead of a min-heap** in Huffman, merging the most probable nodes first. symptom: long codewords for common symbols, expected length far above the entropy.
- **forgetting a tie-breaker in the heap.** `heapq` compares tuples element by element; with equal probabilities it will try to compare the next elements, which may be lists or dicts. an insertion counter as the second element avoids it.

## going deeper

- Shannon (1948), "A mathematical theory of communication", *Bell System Technical Journal* 27(3): 379–423. [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- Huffman (1952), "A method for the construction of minimum-redundancy codes", *Proceedings of the IRE* 40(9): 1098–1101. [doi:10.1109/JRPROC.1952.273898](https://doi.org/10.1109/JRPROC.1952.273898)
- Cover & Thomas (2006), *Elements of Information Theory*, 2nd ed., Wiley, chapters 2, 5 and 8: entropy, data compression, and differential entropy.
- MacKay (2003), *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press, chapters 2, 4–6: entropy, the source coding theorem, symbol codes and arithmetic coding, written for ML. free at [inference.org.uk/itila](http://www.inference.org.uk/itila/).
- Delétang et al. (2024), "Language modeling is compression", ICLR 2024. a modern look at the prediction–compression equivalence. [arXiv:2309.10668](https://arxiv.org/abs/2309.10668)

## connections

- ← builds on: expectations (ch 04 · topic 03), the multivariate Gaussian (ch 04 · topic 04), Cholesky solves (ch 02 · topic 03), maximum likelihood (topic 02), the ch 04 capstone's perplexity.
- → used in: exponential families and Fisher information, where KL's curvature is the Fisher information (topic 06), cross-entropy loss for logistic and softmax regression (ch 08), information gain in decision trees (ch 10), variational inference and the ELBO (ch 12), language model perplexity (ch 17–18, 27), the VAE's KL term (ch 21), InfoNCE and mutual-information objectives (ch 25), and KL-regularized RL (ch 24).
