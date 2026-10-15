# building GPT

## why this matters

you now have every part: attention with a causal mask, positions, and a block that stacks. this topic assembles them into the model behind ChatGPT, Claude's ancestors and every model of chapter 27: a **decoder-only transformer language model**, GPT. it's surprisingly little code. the model is an embedding, a stack of causal blocks and a linear layer; the training objective is the one from chapter 17 (predict the next token, measure cross-entropy); the training loop is chapter 15's recipe (AdamW, warmup, cosine, clipping). what made GPT-2 and GPT-3 remarkable wasn't a new idea in any of these pieces but how well the combination scales: the same code, made bigger and fed more text, keeps getting better (the lab sees the first hint of this, and the capstone measures it).

two more things turn a model into a usable language model, and you'll build both. the first is **generation**: a loop that samples a token, appends it and runs the model again. the second is **decoding rules**. the same trained model writes fluent text or repetitive garbage depending on how you choose each next token, and temperature, top-k and nucleus sampling are the standard controls. chapter 17 compared a character LSTM with an n-gram; the capstone of this chapter puts your GPT on the same data and the same split.

## warm-up: predict first

1. a window of 256 tokens goes through a causal language model. how many next-token predictions does it provide training signal for?
2. a model's output layer maps a hidden vector to 50,000 logits; its input layer maps 50,000 token ids to vectors. could they share one matrix? what would you save?
3. at initialization, before any training, what loss should a language model over a vocabulary of 65 characters have?
4. you always pick the single most likely next character. what does the text look like after a few hundred characters?
5. temperature divides the logits by $T$ before the softmax. what distribution do you get as $T \to 0$, and as $T \to \infty$?

## prerequisites check

- [ ] you know what a language model, cross-entropy and perplexity are, and can convert nats per token to bits per character (→ ch 17 · topic 01)
- [ ] you can turn text into token ids with your `CharTokenizer` or `BPETokenizer` (→ ch 17 · topic 02)
- [ ] you know AdamW, warmup + cosine schedules and gradient clipping, and you have `CosineSchedule` (→ ch 15 · topics 03–04)
- [ ] your `TransformerBlock` works with `is_causal=True` and passes a `cache` through (→ topic 03)
- [ ] you have learned and sinusoidal positions (→ topic 02)

## core ideas

### idea 1: language modeling, one window at a time

a language model assigns probabilities to sequences by the chain rule: $P(w_1, \dots, w_N) = \prod_t P(w_t \mid w_{<t})$. a GPT models each factor with a transformer that reads the context and outputs a distribution over the vocabulary. training minimizes the average negative log-probability of the actual next token, exactly chapter 17's objective.

the training data is one long stream of token ids (the whole corpus, concatenated). a batch is $B$ windows of $T + 1$ consecutive tokens at random starting points: the first $T$ are the input $x$, the last $T$ the targets $y$, so $y_t = x_{t+1}$. because attention is causal, position $t$ of the output depends only on $x_1, \dots, x_t$, and one forward pass over the window gives $T$ predictions at once: given 1 token, given 2, ..., given $T$. that's the efficiency that made transformers win over RNNs in training: a whole window of predictions is a few matrix multiplications, with no loop over time.

the targets are **pre-shifted**: the batch already holds, at position $t$, the token the model should predict there. the model compares `logits[:, t]` with `targets[:, t]` and never shifts anything itself. this is the course-wide causal-LM protocol (`CONTRACTS.md`), and it leaves room for a convention the later chapters rely on: a target of $-100$ means "no loss here" (the prompt in fine-tuning, padding).

> **spec.** `get_batch(data, block_size, batch_size, *, generator=None, device='cpu')`: `data` is a 1-D int ndarray or int64 tensor of token ids. returns `(x, y)`, int64 tensors `(batch_size, block_size)` on `device`, with starting points $s$ drawn uniformly (via `torch.randint` with `generator`) from $0, \dots, \text{len} - T - 1$, `x[b] = data[s:s+T]` and `y[b] = data[s+1:s+T+1]`.

> **▶ build: stage 1.** implement `get_batch`.

### idea 2: the model

$$h_0 = E[x] + P[\text{positions}], \qquad h_l = \text{Block}_l(h_{l-1}), \qquad \text{logits} = \text{LN}_f(h_L)\, E^\top.$$

the token embedding $E$ is a $V \times d$ table; the position embedding is learned (`nn.Embedding(block_size, d)`, GPT-2's choice) or sinusoidal (no parameters). the blocks are your pre-LN `TransformerBlock`s called with `is_causal=True`, then a final layer norm (pre-LN needs it, topic 03) and a linear head to $V$ logits.

**weight tying.** the head's weight matrix has the same shape as the embedding, $V \times d$, and GPT shares them: the logit of token $w$ is the dot product of the final hidden state with $w$'s own embedding. this makes sense (tokens that should be predicted in similar contexts get similar embeddings and similar output vectors anyway), it halves the vocabulary-related parameters, and Press & Wolf (2017) found it usually improves perplexity. in PyTorch it's one line, `self.lm_head.weight = self.tok_emb.weight`, after which the two modules hold *the same tensor*, and `model.parameters()` lists it once. the head has no bias.

**initialization.** GPT-2 initializes every linear and embedding weight from $\mathcal{N}(0, 0.02^2)$ and biases at zero, then scales the weights of the two layers that write into the residual stream (each attention's `out_proj` and each MLP's `fc2`) by a further $1/\sqrt{2L}$, because there are $2L$ such writes and their contributions add up (topic 03, on paper 3). with small weights the logits start near zero, so the first prediction is nearly uniform and the initial loss is about $\ln V$: 4.17 nats for 65 characters. a model that starts far above $\ln V$ is confidently wrong before it has learned anything, which wastes the first hundreds of steps.

**the cache hook and the context limit.** a learned position table has `block_size` rows, so the model refuses longer inputs with a `ValueError`. for topic 06's KV cache, `forward` accepts `cache`: an object where `cache[i]` is block $i$'s store and `cache.seq_len` is the number of tokens already cached. the new tokens sit at positions `cache.seq_len`, ..., so the positions must be read *before* running the blocks (which append to the cache). you can build and test all of this now with a stand-in cache.

> **spec.** `GPT(vocab_size, *, block_size=256, n_layers=4, n_heads=4, d_model=128, d_ff=None, dropout=0.0, pos_encoding='learned', tie_weights=True)` has `tok_emb`, `pos_emb` (an `nn.Embedding(block_size, d)`, or `None` for `'sinusoidal'`), `blocks` (an `nn.ModuleList` of pre-LN `TransformerBlock`s), `ln_f`, `lm_head` (`nn.Linear(d, V, bias=False)`, the same tensor as `tok_emb.weight` if `tie_weights`), an attribute `n_layers`, dropout with rate `dropout` on the embeddings (token + position) as well as inside the blocks, and GPT-2's initialization above. `forward(idx, targets=None, *, cache=None)` returns `(logits (B, T, V), loss)`, `loss` the mean cross-entropy with `ignore_index=-100` or `None` without targets; with a cache, positions start at `cache.seq_len` and block $i$ gets `cache[i]`; more than `block_size` positions raise `ValueError`. `hidden_states(idx)` returns the `(B, T, d)` output of `ln_f`.

> **🔮 predict:** the default `GPT(65)` has 834,432 parameters. how many of them are in the embeddings, and how many would an untied head add?

> **▶ build: stage 2.** implement `GPT.__init__`, `forward` and `hidden_states`.

### idea 3: choosing the next token

the model gives a distribution $p = \operatorname{softmax}(z)$ over the next token. how you pick from it is a separate decision, and it changes the text more than you'd guess.

**greedy decoding** takes $\arg\max_w p_w$. it maximizes the probability of each step, not of the text, and it gets stuck: once a phrase is the most likely continuation of itself, it repeats forever (the lab shows it). **sampling** draws $w \sim p$. it's faithful to the model, but the long tail of a 50,000-word vocabulary is full of tokens that are each unlikely and together likely, so every few dozen tokens sampling picks something absurd, and the text derails.

three controls trade faithfulness for safety. **temperature**: sample from $\operatorname{softmax}(z/T)$. $T < 1$ sharpens the distribution towards the top tokens (as $T \to 0$, greedy), $T > 1$ flattens it (as $T \to \infty$, uniform). **top-k** (Fan et al., 2018): keep only the $k$ largest logits, set the rest to $-\infty$, renormalize. **top-p**, or nucleus sampling (Holtzman et al., 2020): sort the probabilities and keep the smallest set of top tokens whose total mass reaches $p$. top-p adapts to the distribution: after "the capital of France is" the nucleus is one token, after "I like" it's hundreds, while top-k keeps $k$ in both cases. in this course the order is: temperature, then top-k, then top-p on what remains.

> **spec.** `sample_next_token(logits, *, temperature=1.0, top_k=None, top_p=None, generator=None)`: `logits` `(B, V)` → `(B,)` int64. `temperature == 0` returns the argmax. otherwise divide by `temperature`; if `top_k` is set, keep the logits of each row that are at least its `top_k`-th largest (ties at the cut are all kept, so more than `top_k` tokens can survive); if `top_p` is set, keep, per row, the smallest set of most likely remaining tokens whose probability mass is at least `top_p` (mass exactly reaching `top_p` is enough; the top token always survives; `top_p <= 0` raises `ValueError`); then draw one token per row with `torch.multinomial(..., generator=generator)`.

> **▶ build: stage 3.** implement `sample_next_token`.

### idea 4: generation

generation is a loop: run the model on the sequence so far, take the logits at the last position, choose a token, append it, repeat. two details. the model can't see more than `block_size` positions, so the context fed to it is the last `block_size` tokens (the model "forgets" anything earlier). and the loop runs in eval mode (no dropout) under `torch.no_grad()`, restoring the mode afterwards.

notice how wasteful this is: to produce token 200, the loop recomputes the keys and values of all 199 previous tokens in every layer, though they haven't changed since the previous step. generating $n$ tokens costs $O(n^2)$ token-passes through the model. topic 06's KV cache stores those keys and values and makes each step cost one token-pass.

> **spec.** `GPT.generate(idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None)`: `idx` `(B, T)` → `(B, T + max_new_tokens)`, the prompt followed by the new tokens; each step runs the model on the last `block_size` tokens and picks the next token with `sample_next_token` (same keyword arguments) from the last position's logits; eval mode, no gradients, the original mode restored.

> **🔮 predict:** greedy decoding writes the text the model considers most probable. will it be the best text of the lab's six decoding rules?

> **▶ build: stage 4.** implement `GPT.generate`.

### idea 5: training and evaluation

the training loop is chapter 15's recipe, with transformer conventions. **AdamW** with $\beta = (0.9, 0.95)$, and weight decay applied only to parameters with two or more dimensions (the weight matrices and embeddings): biases and layer-norm gains are few, and decaying them toward zero just fights the normalization. a linear **warmup** and a **cosine** decay to a tenth of the peak (your `CosineSchedule`, stepped once per optimizer step), and **gradient clipping** at norm 1.

to evaluate, average the loss over a number of random batches in eval mode and without gradients. use a *fixed* generator seed so that every evaluation sees the same batches: then two evaluations differ only because the model changed, not because the batches did.

> **spec.** `estimate_loss(model, data, block_size, *, batch_size=32, n_batches=20, device='cpu')`: the mean of `model(x, y)[1]` over `n_batches` batches from `get_batch` with `torch.Generator().manual_seed(0)`, in eval mode under `torch.no_grad()`, restoring the mode; a Python float. `train_lm(model, train_data, block_size, *, val_data=None, batch_size=32, max_steps=1000, lr=3e-4, weight_decay=0.1, warmup_steps=100, grad_clip=1.0, eval_interval=100, eval_batches=20, device='cpu', seed=0)`: AdamW as above with $\beta = (0.9, 0.95)$, `CosineSchedule(opt, max_steps, warmup_steps=min(warmup_steps, max_steps - 1), min_lr=lr / 10)` stepped once per optimizer step (so the first step uses learning rate 0, step `warmup_steps + 1` the full `lr`, and the last steps approach `lr / 10`), batches from a generator seeded with `seed`, clipping with `nn.utils.clip_grad_norm_` at `grad_clip` unless it is `None`. at every multiple of `eval_interval` and at the last step, append the step, `estimate_loss` on the training and validation data (`nan` without `val_data`) and the learning rate that step used to the history `{'step', 'train_loss', 'val_loss', 'lr'}`. returns the history, leaving the model in eval mode.

> **▶ build: stage 5.** implement `estimate_loss` and `train_lm`.

## worked example

**parameters of `GPT(65)`** ($d = 128$, 4 layers, `block_size` 256, tied): token embedding $65 \times 128 = 8{,}320$; position embedding $256 \times 128 = 32{,}768$; each block $4(128^2 + 128) + (8 \cdot 128^2 + 5 \cdot 128) + 4 \cdot 128 = 198{,}272$ (topic 03's formula), four of them $793{,}088$; final norm 256. total **834,432**; an untied head would add another 8,320. at initialization the loss should be close to $\ln 65 = 4.17$.

**decoding**, with next-token probabilities $p = (0.5, 0.3, 0.15, 0.05)$:

- `top_k=2` keeps tokens 0 and 1 and renormalizes: $(0.625, 0.375, 0, 0)$.
- `top_p=0.75`: the top token's mass 0.5 isn't enough, the top two reach 0.8, so the nucleus is $\{0, 1\}$: again $(0.625, 0.375, 0, 0)$. `top_p=0.9` keeps three tokens (mass 0.95); `top_p=0.3` keeps only token 0.
- `temperature=0.5` divides the log-probabilities by 0.5, i.e. squares the probabilities before renormalizing: $(0.25, 0.09, 0.0225, 0.0025)/0.365 = (0.685, 0.247, 0.062, 0.007)$.

your code must reproduce these.

## on paper

1. **one window, $T$ examples.** write the loss of one window of length $T$ as a sum over positions. into the computations at which positions does the loss at position $t$ send gradient, and into which does it send none?

<details><summary>solution</summary>

$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p_\theta(y_t \mid x_{\le t})$. the prediction at position $t$ depends on $x_1, \dots, x_t$ through attention, so its loss sends gradient into the keys and values computed at every position $s \le t$, in every layer, and into the shared weights. what's true is the converse: the loss at position $s$ never depends on, and so sends no gradient through, the representations at positions $t > s$. that's why one pass trains all $T$ predictions without leaking the answers.

</details>

2. **temperature and entropy.** for logits $z$, let $p_T = \operatorname{softmax}(z/T)$. show that $p_T \to$ uniform as $T \to \infty$ and $p_T \to$ one-hot on $\arg\max z$ as $T \to 0^+$ (assuming a unique maximum). how does sampling at $T = 0.5$ relate to the probabilities $p_1$?

<details><summary>solution</summary>

$p_{T,i} = e^{z_i/T}/\sum_j e^{z_j/T}$. as $T \to \infty$ all $z_j/T \to 0$, so every term tends to 1 and $p_{T,i} \to 1/V$. as $T \to 0^+$, divide numerator and denominator by $e^{z_m/T}$ with $m$ the argmax: $p_{T,i} = e^{(z_i - z_m)/T}/\sum_j e^{(z_j - z_m)/T}$; every exponent with $z_j < z_m$ goes to $-\infty$, so only $j = m$ survives. at $T = 0.5$, $p_{0.5,i} \propto e^{2z_i} = (e^{z_i})^2 \propto p_{1,i}^2$: sampling with probabilities proportional to the squared probabilities.

</details>

3. **why tie?** with tied weights the logit of token $w$ is $e_w \cdot h$. show that the gradient of the loss with respect to the embedding $e_w$ has two parts, and explain why a token that is rare in the *input* still gets a useful embedding.

<details><summary>solution</summary>

$e_w$ is used as an input (when $w$ appears in the context) and as an output vector (in every softmax, for every position). the output part contributes $(p_w - \mathbb{1}[y = w])\,h$ at every position of every batch, whether or not $w$ occurs in the input. so even a token that rarely appears as input is trained on every step through the softmax, pushed towards contexts in which it's predicted and away from others. without tying, its input embedding would only be trained on the rare occasions it appears.

</details>

4. **top-k vs top-p.** a model is confident: $p = (0.97, 0.01, 0.01, 0.01, \dots)$; then uncertain: 200 tokens of probability 0.004 each and a long tail. compare what `top_k=40` and `top_p=0.9` keep in each case.

<details><summary>solution</summary>

confident: top-k keeps 40 tokens, including 39 that share only 3% of the mass but will still be sampled 3% of the time; top-p keeps just the first token (0.97 ≥ 0.9). uncertain: top-k keeps 40 of the 200 plausible tokens (cutting real alternatives, mass 0.16), top-p keeps the 200 plausible ones (mass 0.8) plus enough of the tail to reach 0.9. top-p's set size adapts to the model's confidence; top-k's doesn't.

</details>

## experiment

the lab trains four GPTs of different sizes with your `train_lm` for the same number of steps on the same batches of Shakespeare and asks which learns most from the same data. then it lets the largest one write under six decoding rules, and scores each text by how probable the model itself finds it, how repetitive it is and how much is copied from the training set.

## check your understanding

1. why must a GPT's targets be pre-shifted *outside* the model rather than shifted inside `forward`?

<details><summary>answer</summary>

so that the caller controls what's predicted: masked prompts ($-100$), packed sequences, or targets that aren't simply the next token of the input (chapter 27's fine-tuning only scores the response). a model that shifted internally would make all of those impossible and would silently double-shift data that was already shifted.

</details>

2. your model's initial loss is 7.9 on a 65-character vocabulary. what's wrong?

<details><summary>answer</summary>

$\ln 65 = 4.17$, so it starts confidently wrong: the logits are too large at initialization. likely causes: default PyTorch init (larger than 0.02) with an untied head, embeddings $\mathcal{N}(0, 1)$ multiplied by a tied head, or a missing final layer norm feeding a growing residual stream into the head.

</details>

3. why is weight decay not applied to layer-norm weights and biases?

<details><summary>answer</summary>

they're scale and shift parameters with no tendency to overfit (there are only $2d$ per norm), and the "right" value of a layer-norm gain is around 1, not 0. decaying them toward zero fights the normalization. weight decay is meant to keep large weight matrices small.

</details>

4. generation crops the context to the last `block_size` tokens. what does the model lose, and why can't it just read further back?

<details><summary>answer</summary>

anything earlier than `block_size` tokens is invisible: a character name introduced 300 characters ago is forgotten. with learned positions there's no embedding for positions beyond `block_size`, and the model was never trained on longer contexts; topic 02 showed what happens when you push positions past their training range.

</details>

5. with the same trained model, $T = 0.8$ text is readable and $T = 1.5$ text is garbled. the model hasn't changed. what has?

<details><summary>answer</summary>

only the distribution sampled from. $T = 1.5$ flattens it, moving probability mass from the plausible tokens into the tail of implausible ones; every so often a tail token is drawn, the model conditions on its own mistake, and the errors compound. $T = 0.8$ moves mass the other way.

</details>

## common bugs

- **shifting the targets twice.** `get_batch` already returns $y_t = x_{t+1}$; computing the loss on `logits[:, :-1]` vs `targets[:, 1:]` predicts two tokens ahead. symptom: a loss that plateaus far above the n-gram baseline.
- **tying weights by copying.** `self.lm_head.weight.data = self.tok_emb.weight.data` makes two parameters that start equal and then drift apart. assign the `Parameter` itself.
- **positions restarting at 0 with a cache.** whole-sequence outputs are fine; cached generation is gibberish. read `cache.seq_len` before running the blocks.
- **top-p that drops the top token.** removing tokens whose cumulative mass *including themselves* exceeds $p$ removes the first token when $p_1 > p$; compare the mass *before* each token.
- **generating in training mode.** dropout stays on, and the samples are noisier and not reproducible.
- **decaying everything.** `AdamW(model.parameters(), weight_decay=0.1)` also decays layer-norm gains and biases: training still works, slightly worse, and it's the wrong default to copy into chapter 27.

## going deeper

- Radford, Narasimhan, Salimans & Sutskever (2018), "Improving language understanding by generative pre-training" (GPT-1). [cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Radford, Wu, Child, Luan, Amodei & Sutskever (2019), "Language models are unsupervised multitask learners" (GPT-2), section 2.3: pre-LN, the final norm and the $1/\sqrt{N}$ residual initialization. [cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Brown et al. (2020), "Language models are few-shot learners" (GPT-3), section 2.1 and table 2.1 for the model sizes. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Press & Wolf (2017), "Using the output embedding to improve language models", *EACL*. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
- Holtzman, Buys, Du, Forbes & Choi (2020), "The curious case of neural text degeneration", *ICLR*. nucleus sampling, and why greedy and beam search repeat. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Fan, Lewis & Dauphin (2018), "Hierarchical neural story generation", *ACL*. top-k sampling. [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
- Karpathy (2022), nanoGPT, a minimal GPT training codebase close to what you've built. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## connections

- ← builds on: language modeling, perplexity and tokenizers (ch 17 · topics 01–02), the training recipe and `CosineSchedule` (ch 15), positions (topic 02) and the block (topic 03).
- → used in: the KV cache and `generate_cached` (topic 06), the capstone's nanoGPT and scaling curve; scaling laws (ch 19); conditional generation (ch 21–22); the systems view of training cost (ch 26); chapter 27's pretraining, fine-tuning, preference learning and decoding all start from this model and its causal-LM protocol; activation patching on a GPT (ch 28).
