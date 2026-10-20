# speech recognition with CTC

## why this matters

a speech corpus gives you audio and a transcript: "the cat sat", or a phoneme string. it doesn't tell you *when* each sound happens. your acoustic model, though, produces one prediction per 10–20 ms frame, a hundred or so per second. to train it with cross-entropy you'd need a label for every frame, which means someone (or an older model) must first **align** the transcript to the audio. that's how speech recognition worked for decades: hidden Markov models, forced alignment, then frame-level training on the alignment, with every mistake in the alignment baked in.

**connectionist temporal classification** (CTC; Graves et al., 2006) removes that step. it defines the probability of a transcript as the sum over *every* way of aligning it to the frames, computes that sum exactly with dynamic programming, and lets gradient descent work out the alignment by itself. the price is one extra symbol, the **blank**, and the lab is about why it's needed. CTC trained the first end-to-end speech recognizers (Graves et al., 2013; Hannun et al., 2014), and it's still used today, often next to an attention decoder.

you'll build the alignment space, the forward algorithm in NumPy, a batched differentiable CTC loss in PyTorch that matches `F.ctc_loss` value for value and gradient for gradient, greedy and prefix-beam decoding, and a small bidirectional-GRU recognizer. **all audio here is synthetic**: the lab builds a toy "phoneme" corpus from topic 24's source–filter vowels and filtered noise bursts. it shows every mechanism of CTC, but its error rates say nothing about real speech.

## warm-up: predict first

1. 100 frames, a transcript of 10 distinct phonemes. roughly how many frame-level alignments does CTC sum over: thousands, billions, or more?
2. you label every frame with its phoneme and merge repeated frame labels to get the transcript. what goes wrong for a word with a double letter, like "too"?
3. after training, what fraction of frames do you expect the model to label *blank* (its most likely class)?
4. greedy decoding picks the most likely label at each frame. is the result the most likely transcript?
5. what's the fewest frames that can carry the label sequence a a b?

## prerequisites check

- [ ] you've built log mel spectrograms (→ topic 24)
- [ ] you know the HMM forward algorithm, a sum over hidden paths by dynamic programming (→ ch 12 · topic 04)
- [ ] you've trained GRUs in PyTorch and written a beam search (→ ch 17 · topics 05–06)
- [ ] you can compute $\log(e^a + e^b)$ without overflow (→ ch 00 · topic 03)
- [ ] softmax and cross-entropy gradients: $\partial(-\log\operatorname{softmax}(z)_k)/\partial z = y - e_k$ (→ ch 08)

## core ideas

### idea 1: alignments and the blank

the model outputs, for each of $T$ frames, a distribution $y_t$ over $V$ classes: the labels plus the **blank**, which this course writes as class 0. a **path** $\pi = (\pi_1, \dots, \pi_T)$ picks one class per frame. the **collapse** map $\mathcal B$ turns a path into a label sequence in two steps: *first merge runs of repeated classes, then delete the blanks*. so $\mathcal B(\text{a a – a b b –}) = \text{a a b}$ (– is blank): the blank between the two a-runs keeps them apart. CTC assumes the frames' classes are independent given the input, so a path's probability is $\prod_t y_t(\pi_t)$, and a label sequence $\ell$ gets the total probability of every path that collapses to it:

$$p(\ell \mid x) = \sum_{\pi:\,\mathcal B(\pi) = \ell}\ \prod_{t=1}^T y_t(\pi_t).$$

the blank does two jobs. it **separates repeats**: without it, merging repeats would turn every "a a" into "a", so doubled letters (and repeated phonemes) would be unrepresentable. and it **means "nothing new here"**: silence, transitions, or frames where the model isn't ready to commit. a phoneme lasting 8 frames can be one label frame and seven blanks.

a convenient way to see the valid paths is the **extended label sequence** $\ell' = (–, \ell_1, –, \ell_2, \dots, \ell_U, –)$, of length $2U + 1$. every valid path walks through $\ell'$ from left to right, one state per frame: it starts in the first blank or the first label, and at each frame it either **stays**, **advances by one**, or **skips** a blank to jump to the next label, a skip allowed only when the next label differs from the current one (otherwise the two would merge). it must end in the last label or the final blank. so a repeated pair needs a blank frame between them, and the fewest frames a label sequence can use is $U$ plus its number of adjacent equal pairs. the number of valid paths grows combinatorially: with no repeats it's $\binom{T + U}{2U}$ (on paper 1), about $4\times10^{21}$ for 10 labels in 100 frames.

> **spec.** blank is class 0 unless a keyword `blank=` says otherwise. `ctc_collapse(path, *, blank=0)` returns the list of ints $\mathcal B(\text{path})$. `ctc_expand(labels, *, blank=0)` returns $\ell'$ as a list. `min_frames(labels)` returns $U$ + the number of adjacent equal pairs. `ctc_alignments(labels, T, *, blank=0)` returns the sorted list of every length-$T$ path (tuples of ints) with $\mathcal B(\pi) = $ `labels`, which you can build by walking $\ell'$ as described (enumerating all $V^T$ paths would take exponential time); `[]` if there's none.

> **▶ build: stage 1.** implement `ctc_collapse`, `ctc_expand`, `min_frames` and `ctc_alignments`.

### idea 2: the forward algorithm

summing over $4\times10^{21}$ paths is out of the question, but the paths share structure: whether a path can continue from state $s$ at time $t$ depends only on $s$, not on how it got there. that's the same observation that gives the HMM forward algorithm (ch 12). define $\alpha_t(s)$ as the total probability of all path prefixes $\pi_1..\pi_t$ that are in state $s$ of $\ell'$ at time $t$. then

$$\alpha_1(1) = y_1(\ell'_1)\ (\text{the blank}),\qquad \alpha_1(2) = y_1(\ell'_2)\ (\text{the first label}),\qquad \alpha_1(s) = 0 \text{ otherwise},$$

$$\alpha_t(s) = \Big(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + [\text{skip ok}]\,\alpha_{t-1}(s-2)\Big)\,y_t(\ell'_s),$$

where "skip ok" means $\ell'_s$ is not blank and $\ell'_s \ne \ell'_{s-2}$. the answer is $p(\ell \mid x) = \alpha_T(2U + 1) + \alpha_T(2U)$: ending in the final blank or on the last label. (states here are numbered from 1; the code numbers them from 0.) that's $O(T\cdot U)$ work instead of exponential.

products of hundreds of probabilities underflow to 0 in float64 long before $T = 1000$, so you run the recursion on $\log\alpha$, replacing $+$ with $\operatorname{logaddexp}$ and $\times$ with $+$. unreachable states have $\log\alpha = -\infty$, and `np.logaddexp` handles that correctly. if the sequence is too short for the labels ($T <$ `min_frames`), no state at the end is reachable and $-\log p = +\infty$.

> **spec.** `ctc_forward(log_probs, labels, *, blank=0)`: `log_probs` is a `(T, V)` NumPy array of per-frame log-probabilities; returns the `(T, 2U + 1)` array of $\log\alpha_t(s)$ ($-\infty$ where unreachable). `ctc_nll(log_probs, labels, *, blank=0)` returns $-\log p(\ell \mid x)$ as a float: `inf` when no path fits, $-\sum_t\log y_t(\text{blank})$ for empty labels.

> **🔮 predict:** with uniform frame probabilities $1/V$, what is $p(\ell \mid x)$ in terms of the number of alignments?

> **▶ build: stage 2.** implement `ctc_forward` and `ctc_nll`.

### idea 3: a loss you can train with

to train, you need the loss for a padded batch, differentiable with respect to the network's outputs. two pieces of bookkeeping make batches work. **padding**: utterances have different lengths, so `log_probs` is `(B, T, V)` with item $b$ using only its first $T_b$ frames, and the targets are a `(B, S)` array whose row $b$ holds $U_b$ labels followed by junk. run the recursion for every item and every state at once (one loop over time, vectorized over batch and states), freeze item $b$'s $\alpha$ once $t \ge T_b$, and read each item's result at *its* last two states, $2U_b$ and $2U_b - 1$ (0-based). **gradients**: write the recursion in differentiable torch ops and autograd does the backward pass, which turns out to be the CTC backward (β) recursion of Graves et al. (on paper 3 gives the result in closed form). one trap: `torch.logsumexp` over entries that are *all* $-\infty$ returns $-\infty$ with a NaN gradient, and unreachable states are exactly that. use a large finite stand-in like $-10^{30}$ for "impossible", and set the loss of items with $T_b <$ `min_frames` to $+\infty$ separately.

the result should match `torch.nn.functional.ctc_loss`, which uses three conventions worth knowing. its `log_probs` are time-first, `(T, B, V)`, unlike the rest of this course. `reduction="mean"` divides each item's loss by its target length (at least 1) *before* averaging over the batch, so long and short transcripts weigh the same per label. and `zero_infinity=True` replaces impossible items' infinite losses (and their gradients) with zero, so one bad utterance can't poison a batch. (torch's backward computes the gradient for log-softmax outputs, so compare gradients with respect to the *logits*, not the log-probabilities.)

one assumption hides in all of this: CTC models the frames as **conditionally independent** given the input. the network sees the whole utterance (with a bidirectional encoder) but outputs no frame conditioned on the labels chosen at other frames. so CTC has no built-in language model; that's why decoding often adds one.

> **spec.** `ctc_loss(log_probs, targets, input_lengths, target_lengths, *, blank=0, reduction="mean", zero_infinity=False)`: `log_probs` `(B, T, V)` torch tensor (any float dtype); `targets` `(B, S)` integers, where entries past `target_lengths[b]` are arbitrary (even out of range) and must be ignored; frames past `input_lengths[b]` must be ignored too. per-item losses $-\log p(\ell_b \mid x_b)$ (0 for an empty transcript in zero frames), $+\infty$ when $T_b <$ `min_frames`$(\ell_b)$ (0 with `zero_infinity=True`, with zero gradient). reductions `"none"` `(B,)`, `"sum"`, `"mean"` as in torch; anything else raises `ValueError`. gradients must be finite and equal torch's.

> **▶ build: stage 3.** implement `ctc_loss`.

### idea 4: decoding

after training you want $\arg\max_\ell p(\ell \mid x)$. that's intractable in general, so you approximate. **greedy (best-path) decoding** takes the most likely class at every frame and collapses: fast and often good. but the most likely *path* need not collapse to the most likely *label sequence*, because a labeling's probability is spread over many paths (worked example).

**prefix beam search** (Hannun, Maas, Jurafsky & Ng, 2014; Maas et al., 2015) searches over label sequences instead. it keeps a beam of prefixes, and for each prefix two numbers: $p_b$, the probability of all paths so far that collapse to the prefix and *end in blank*, and $p_{nb}$, those that *end in the prefix's last label*. the split matters for repeats. at each frame, for every prefix $\rho$ in the beam, with $p = p_b + p_{nb}$:

- a **blank** keeps the prefix: $p_b(\rho) \mathrel{+}= p\cdot y_t(–)$;
- a label $c$ different from $\rho$'s last label extends it: $p_{nb}(\rho c) \mathrel{+}= p\cdot y_t(c)$;
- the **same label** $c$ as $\rho$'s last does two things: paths that ended in $c$ just continue the run, $p_{nb}(\rho) \mathrel{+}= p_{nb}(\rho)\,y_t(c)$, while paths that ended in blank start a new $c$, $p_{nb}(\rho c) \mathrel{+}= p_b(\rho)\,y_t(c)$.

then keep the `beam_size` prefixes with the largest $p_b + p_{nb}$. several paths that collapse to the same prefix *merge* into one entry, which is why this isn't the token-level `mlfs.seq.beam_search` from ch 17: that one scores token sequences, here many frame paths map to one label sequence and their probabilities must be added. with an unlimited beam the search is exact: it returns $p(\ell \mid x)$ for every $\ell$. real systems add a language model's score when a prefix is extended, which is the main reason to run a beam at all.

> **spec.** both take `(T, V)` log-probabilities (NumPy or torch). `greedy_decode(log_probs, *, blank=0)` returns `ctc_collapse` of the per-frame argmax. `prefix_beam_search(log_probs, *, beam_size=8, blank=0)` runs the recursion above in log space from the empty prefix ($p_b = 1$, $p_{nb} = 0$), prunes to the `beam_size` best totals after every frame, and returns a list of `(prefix tuple, log p)` sorted best first, dropping prefixes with probability 0.

> **🔮 predict:** two frames, $y_t = (0.6, 0.4)$ for (blank, a) at both. what does greedy return, and what does prefix beam search rank first?

> **▶ build: stage 4.** implement `greedy_decode` and `prefix_beam_search`.

### idea 5: measuring errors, and a model to train

recognizers are scored by **edit distance**: the fewest substitutions, insertions and deletions turning the hypothesis into the reference (Levenshtein, 1966), by the usual $O(|r|\,|h|)$ dynamic program. the **error rate** (word error rate for words, character or phoneme error rate for smaller units) divides the *total* edits over a test set by the *total* reference length. it is not the average of per-utterance rates, which would let short utterances dominate. it can exceed 1 when the hypotheses are full of insertions.

the acoustic model is a **bidirectional GRU**: one GRU reads the features forward, another backward, and their states are concatenated, so every frame's output can use the whole utterance. a linear layer maps each frame to $V$ logits, and a log-softmax turns them into the $y_t$ that CTC needs. the input is topic 24's log mel spectrogram, normalized per utterance. that's the shape of the first end-to-end recognizers; modern ones replace the GRU with convolutions and transformers (conformers) but keep the CTC head.

> **spec.** `edit_distance(ref, hyp)` takes any two sequences and returns the Levenshtein distance (all costs 1). `error_rate(refs, hyps)` returns $\sum_i d(r_i, h_i) / \sum_i |r_i|$, raising `ValueError` if the lists differ in length or all references are empty. `CTCRecognizer(n_features, n_classes, *, hidden_size=64, num_layers=2)` is an `nn.Module` holding a bidirectional, batch-first `nn.GRU(n_features, hidden_size, num_layers=num_layers)` and an `nn.Linear(2 * hidden_size, n_classes)` (no other parameters); `forward(x)` maps `(B, T, n_features)` to `(B, T, n_classes)` log-probabilities.

> **▶ build: stage 5.** implement `edit_distance`, `error_rate` and `CTCRecognizer`.

## worked example

$T = 2$ frames, classes (–, a), and $y_1 = y_2 = (0.6, 0.4)$.

| path | – – | – a | a – | a a |
|---|---|---|---|---|
| probability | 0.36 | 0.24 | 0.24 | 0.16 |
| collapses to | "" | a | a | a |

greedy picks – at both frames (0.6 > 0.4) and outputs "": the single best path has probability 0.36. but "a" collects three paths, $p(\text{a}) = 0.24 + 0.24 + 0.16 = 0.64$, so the best labeling is "a". prefix beam search finds it: after frame 1, $p_b(\text{""}) = 0.6$ and $p_{nb}(\text{a}) = 0.4$; after frame 2, "" has $p_b = 0.36$, while "a" has $p_{nb} = 0.6\cdot0.4 + 0.4\cdot0.4 = 0.40$ and $p_b = 0.4\cdot0.6 = 0.24$, total 0.64.

the forward algorithm agrees. with $\ell' = (–, \text{a}, –)$: $\alpha_1 = (0.6, 0.4, 0)$, then $\alpha_2(1) = 0.6\cdot0.6 = 0.36$, $\alpha_2(2) = (0.4 + 0.6)\cdot0.4 = 0.40$, $\alpha_2(3) = (0 + 0.4)\cdot0.6 = 0.24$, and $p(\text{a}) = \alpha_2(2) + \alpha_2(3) = 0.64$, so `ctc_nll` returns $-\log 0.64 = 0.446$. for counting: "a" in 3 frames has $\binom{4}{2} = 6$ alignments, and "a a" in 3 frames exactly one (a – a). your `ctc_nll`, `greedy_decode`, `prefix_beam_search` and `ctc_alignments` must reproduce these.

## on paper

1. **counting alignments.** show that a label sequence of $U$ labels with no adjacent repeats has $\binom{T + U}{2U}$ alignments in $T$ frames, and that each adjacent equal pair reduces $T$ by one in that formula.

<details><summary>solution</summary>

a valid path is $U$ nonempty runs of labels separated and surrounded by $U + 1$ runs of blanks, which may be empty. writing each label run as $1 + r_i$ with $r_i \ge 0$, the $2U + 1$ nonnegative run lengths sum to $T - U$: by stars and bars, $\binom{T - U + 2U}{2U} = \binom{T + U}{2U}$. an adjacent equal pair forces the blank run between them to be nonempty, so subtract 1 from that run too: $T$ becomes $T - 1$. for "a" in 3 frames, $\binom42 = 6$; for "a a" in 3 frames, $\binom{3 + 2 - 1}{4} = 1$.

</details>

2. **why the skip rule.** explain why the recursion may skip from label $\ell'_{s-2}$ over a blank to $\ell'_s$ only when $\ell'_s \ne \ell'_{s-2}$, and what goes wrong if you always allow it.

<details><summary>solution</summary>

skipping the blank means two label frames are adjacent in the path. if the labels differ (a then b), the collapse keeps both. if they're equal (a then a), the collapse merges them into one a, so the path really belongs to a *shorter* label sequence. allowing the skip would add those paths to $p(\text{a a})$: the "probability" would count paths that collapse to "a", and the sum over all labelings would exceed 1.

</details>

3. **the gradient.** with $y_t = \operatorname{softmax}(z_t)$, show $\partial(-\log p(\ell\mid x))/\partial z_t(k) = y_t(k) - \gamma_t(k)$, where $\gamma_t(k)$ is the posterior probability, given $\ell$, that a path uses class $k$ at frame $t$.

<details><summary>solution</summary>

$p = \sum_\pi\prod_{t'}y_{t'}(\pi_{t'})$, so $\partial\log p/\partial\log y_t(k) = \sum_{\pi:\,\pi_t = k}P(\pi)/p = \gamma_t(k)$. through the log-softmax, $\partial\log y_t(j)/\partial z_t(k) = [j = k] - y_t(k)$, so $\partial\log p/\partial z_t(k) = \gamma_t(k) - y_t(k)\sum_j\gamma_t(j) = \gamma_t(k) - y_t(k)$, since the $\gamma_t(j)$ sum to 1 over $j$. negate for the loss. it's cross-entropy's gradient with the one-hot target replaced by the posterior over alignments: CTC is cross-entropy against the model's own soft alignment. computing $\gamma$ explicitly takes a backward recursion β; autograd does it for you.

</details>

4. **the minimum length.** why does "a a b" need 4 frames but "a b a" only 3?

<details><summary>solution</summary>

the two adjacent a's must be separated by at least one blank frame or they merge (on paper 2), so "a a b" needs a – a b. in "a b a" no neighbours are equal, so a b a works with no blanks at all.

</details>

## experiment

the lab builds a synthetic corpus of 3–6 "phonemes" per utterance (five vowels from topic 24's source–filter model, three noise-burst fricatives, random pitch, durations and gaps, with repeated neighbours in about 30% of positions), featurizes it with your topic 24 log mel spectrogram, and trains your `CTCRecognizer` with your `ctc_loss`. experiment 1 trains the same model without a usable blank and compares error rates on utterances with and without repeats. experiment 2 looks at what the trained model actually outputs, frame by frame. experiment 3 compares greedy decoding with your prefix beam search.

## check your understanding

1. why does CTC need a blank, when an HMM recognizer doesn't?

<details><summary>answer</summary>

an HMM has several hidden states per phoneme (and explicit silence models), so the state sequence itself marks where one phoneme ends and the next, even an identical one, begins. CTC has exactly one output class per label, and merging repeats is how it lets a label span several frames, so it needs a separate symbol to mark a boundary between identical labels, and something to emit when no label applies.

</details>

2. why must the loss for item $b$ read $\alpha$ at frame $T_b$, not at the batch's last frame?

<details><summary>answer</summary>

frames after $T_b$ are padding. letting the recursion run through them multiplies in the padding frames' blank (or label) probabilities, so the loss would depend on garbage and the model would learn to predict something specific for padding. freezing $\alpha$ after $T_b$ (or reading it at $T_b$) makes the loss independent of padding, which the checks test by scrambling the padded frames.

</details>

3. why divide by the target length in `reduction="mean"`?

<details><summary>answer</summary>

$-\log p(\ell \mid x)$ grows roughly linearly with the transcript's length, so a plain mean would weight long utterances more and make the loss scale depend on the batch's composition. per-label normalization makes the numbers comparable across batches and learning rates transferable across datasets with different utterance lengths.

</details>

4. a model's greedy and beam-search outputs are almost always identical. what does that say about its frame posteriors?

<details><summary>answer</summary>

that they're peaky: at almost every frame one class (usually blank) has nearly all the probability, so one path dominates its labeling and best-path ≈ best-labeling. beam search helps when probability is spread across many paths, or when a language model adds information the acoustic model lacks.

</details>

5. what does CTC's conditional independence assumption prevent the model from learning, and how do real systems compensate?

<details><summary>answer</summary>

it can't make one frame's output depend on what was emitted at other frames, so it can't learn that "q" is usually followed by "u", or any spelling or grammar beyond what the encoder infers from audio. real systems add an external language model during beam search (shallow fusion), or replace CTC with models that condition on previous outputs (RNN-transducers, attention decoders), often keeping a CTC loss as an auxiliary term.

</details>

6. an utterance has 40 frames and a 25-label transcript with 20 adjacent repeats. what does the loss do, and what should you do about it?

<details><summary>answer</summary>

it needs at least $25 + 20 = 45$ frames, so no alignment exists and the loss is $+\infty$, which would wreck a batch's gradient. `zero_infinity=True` drops it, but the real fix is upstream: a frame rate too low for the label rate (common with heavily downsampling encoders and character targets) means the model can't express the transcript at all.

</details>

## common bugs

- **removing blanks before merging repeats.** then "a – a" collapses to "a" instead of "a a". merge first, then delete blanks.
- **allowing the skip between equal labels.** $p(\text{a a})$ then includes paths for "a", and the "probabilities" over all labelings sum to more than 1. the brute-force check catches it.
- **working in probability space.** fine for $T = 10$, zero for $T = 400$; then $-\log 0 = \infty$ and NaN gradients.
- **`-inf` in torch's logsumexp.** an all-`-inf` logsumexp has a NaN gradient. use a large negative finite number for unreachable states.
- **ignoring `input_lengths`.** the loss then depends on what the padding frames contain; the model can lower it by predicting blanks in the padding.
- **the per-utterance average as an error rate.** short utterances dominate. sum the edits, sum the reference lengths, divide.

## going deeper

- Graves, Fernández, Gomez & Schmidhuber (2006), *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*, ICML. the original: the blank, the forward–backward recursions, the gradient. [doi:10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
- Hannun (2017), *Sequence modeling with CTC*, Distill. the clearest visual explanation of the alignment lattice and of prefix beam search. [doi:10.23915/distill.00008](https://doi.org/10.23915/distill.00008)
- Graves (2012), *Supervised Sequence Labelling with Recurrent Neural Networks*, Springer, chapter 7. the full derivation, including the backward variables. [doi:10.1007/978-3-642-24797-2](https://doi.org/10.1007/978-3-642-24797-2)
- Graves, Mohamed & Hinton (2013), *Speech recognition with deep recurrent neural networks*, ICASSP. bidirectional LSTMs with CTC on TIMIT. [arXiv:1303.5778](https://arxiv.org/abs/1303.5778)
- Hannun et al. (2014), *Deep Speech: scaling up end-to-end speech recognition*. CTC at scale, with a language model in the prefix beam search. [arXiv:1412.5567](https://arxiv.org/abs/1412.5567)
- Hannun, Maas, Jurafsky & Ng (2014), *First-pass large vocabulary continuous speech recognition using bi-directional recurrent DNNs*. the CTC prefix beam search with a language model. [arXiv:1408.2873](https://arxiv.org/abs/1408.2873)
- Maas, Xie, Jurafsky & Ng (2015), *Lexicon-free conversational speech recognition with neural networks*, NAACL. prefix beam search with a character language model. [aclanthology.org/N15-1038](https://aclanthology.org/N15-1038/)
- Levenshtein (1966), *Binary codes capable of correcting deletions, insertions, and reversals*, Soviet Physics Doklady 10(8). the edit distance.
- Zeyer, Schlüter & Ney (2021), *Why does CTC result in peaky behavior?* the blank-dominated posteriors you'll see in the lab, explained. [arXiv:2105.14849](https://arxiv.org/abs/2105.14849)

## connections

- ← builds on: log mel spectrograms (topic 24); the HMM forward algorithm (ch 12 · topic 04); GRUs and beam search (ch 17); log-sum-exp (ch 00); cross-entropy gradients (ch 08).
- → used in: topic 26 swaps recognition for compression, turning audio into discrete tokens instead of labels; the attention-based alternative to CTC is the seq2seq model of ch 17 with a speech encoder. after this topic you may use `torch.nn.functional.ctc_loss` and `torch.nn.CTCLoss` (time-first `log_probs`) and library CTC decoders such as `torchaudio`'s, since you've built each of them.
