# BERT-style encoders

## why this matters

GPT reads left to right, because it was built to write. but most uses of text aren't writing: deciding whether a message is spam, finding the names in a contract, judging whether two sentences mean the same thing. for these, a word's meaning depends on both sides of it ("the bank of the river" vs "the bank raised rates"), and a model that can only look left throws away half the evidence at every position.

the obvious fix, removing the causal mask, breaks training. a bidirectional model asked to predict the next token can simply look at it. Devlin et al. (2019) found a different game for a bidirectional model to play: hide a random 15% of the tokens and ask it to fill them in, using everything else. that's **masked language modeling**, and the model trained with it, BERT, changed how NLP was done. you pretrain one encoder on a large pile of unlabeled text, once, and then **fine-tune** it on each task with a small labeled dataset and a tiny classification head. for a few years nearly every leaderboard in NLP was topped by a BERT variant, and the pattern (self-supervised pretraining, then fine-tuning) is what chapter 25 generalizes to images and chapter 27 to LLMs.

this topic builds the masking, the encoder, the pretraining loop and fine-tuning, and the lab measures what pretraining on unlabeled text buys when you only have 40 labels.

## warm-up: predict first

1. a bidirectional transformer is trained to predict token $t + 1$ at position $t$. what does it learn?
2. BERT computes a loss on 15% of the tokens, GPT on 100%. which gets more training signal per pass over the data, and by how much?
3. at fine-tuning time the input never contains a [MASK] token. why might that be a problem for a model pretrained only on inputs full of them?
4. a classifier needs one vector per sentence, but the encoder outputs one per token. what are two ways to get one?
5. with 40 labeled spam messages, would you rather have a 100,000-parameter transformer trained from scratch, or a logistic regression on character counts?

## prerequisites check

- [ ] you know how a key-padding mask works and how to pass it as `(B, 1, 1, T)` (→ topic 01)
- [ ] your pre-LN `TransformerBlock` works without a causal mask (→ topic 03)
- [ ] you understand weight tying and the $-100$ ignore index (→ topic 04)
- [ ] you can compute balanced accuracy and know why it's used for imbalanced classes (→ ch 07)
- [ ] you've used a logistic regression as a baseline (→ ch 08)

## core ideas

### idea 1: masked language modeling

pick a random subset of positions, corrupt them, and train the model to recover the original tokens at those positions only. the context on *both* sides is visible, and nothing leaks, because the answer has been removed from the input.

BERT's recipe is specific, and each number has a reason. each position is selected with probability 15%: fewer and training is slow (only selected positions give a loss), more and too little context is left to reconstruct from. of the selected positions, 80% are replaced by a special [MASK] token, 10% by a random token, and 10% are left unchanged. the [MASK] replacements are the main task. but if *every* selected token were [MASK], the model would only need good representations at [MASK] positions, and [MASK] never appears in fine-tuning inputs. the 10% random replacements force it to doubt every token (any position might be a corrupted one, so every representation has to encode "what should be here"), and the 10% unchanged ones keep it from assuming that a visible token is always wrong. special tokens ([CLS], [PAD], separators) are never selected.

the labels are the original tokens at selected positions and $-100$ everywhere else, the same ignore convention as GPT's targets, so the cross-entropy only counts the positions the model had to reconstruct.

> **spec.** `mask_tokens(idx, *, mask_token_id, vocab_size, mask_prob=0.15, special_ids=(), generator=None)`: `idx` an int64 tensor of any shape (not modified). each position whose token isn't in `special_ids` is selected independently with probability `mask_prob`; of the selected, 80% become `mask_token_id`, 10% a token drawn uniformly from `0, ..., vocab_size - 1`, 10% stay as they are (decided independently per position). returns `(inputs, labels)`: `labels` holds the original token at selected positions and $-100$ elsewhere. all randomness comes from `generator`.

> **🔮 predict:** with 100 non-special tokens, how many differ from the original after `mask_tokens`, on average? (on paper 2.)

> **▶ build: stage 1.** implement `mask_tokens`.

### idea 2: the encoder

BERT is the encoder half of the original transformer: token embeddings plus learned position embeddings, a stack of blocks **without** a causal mask, and (since ours are pre-LN) a final layer norm. it outputs one contextual vector per token. batches of different-length texts are padded to a common length, and an **attention mask** `(B, T)` (True = real token) is turned into a key-padding mask `(B, 1, 1, T)` so no token attends to padding.

every input starts with a [CLS] token whose output is used as a summary of the whole sequence for classification (idea 4). the original BERT also had sentence-pair inputs with a separator and segment embeddings, for its next-sentence-prediction objective; later work (RoBERTa) found that objective unnecessary, and we skip it.

the **MLM head** turns hidden states into logits over the vocabulary: a dense layer, a GELU and a layer norm, then a decoder whose weight is tied to the token embedding (as in GPT), plus a bias per token. it's only used for pretraining and thrown away when fine-tuning.

> **spec.** `BERT(vocab_size, *, max_len=128, n_layers=4, n_heads=4, d_model=128, d_ff=None, dropout=0.1)` has `tok_emb`, `pos_emb` (`nn.Embedding(max_len, d)`), `blocks` (pre-LN `TransformerBlock`s, no causal mask), `ln_f`, attributes `vocab_size` and `d_model`, and weights initialized $\mathcal{N}(0, 0.02^2)$ (biases 0). `forward(idx, attention_mask=None)`: `(B, T)` int64 and an optional `(B, T)` bool mask → hidden `(B, T, d)` after `ln_f`; `T > max_len` raises `ValueError`. `mlm_logits(hidden)` → `(B, T, V)` = `mlm_norm(GELU(mlm_dense(hidden))) @ tok_emb.weight.T + mlm_bias`, with `mlm_dense` an `nn.Linear(d, d)`, `mlm_norm` an `nn.LayerNorm(d)` and `mlm_bias` a `(V,)` parameter; `tok_emb.weight` is the only `(V, d)` parameter.

> **▶ build: stage 2.** implement `BERT`.

### idea 3: pretraining

pretraining is a loop like GPT's: sample a batch of sequences, corrupt them, compute the MLM loss, step. one detail matters: draw **fresh masks every step**. the original BERT masked each sequence once during data preparation (with 10 copies); RoBERTa (Liu et al., 2019) showed that *dynamic* masking, a new random mask every time a sequence is seen, is at least as good and simpler. with dynamic masking, the same sentence teaches the model different fill-in-the-blank questions on every pass.

masked modeling gets less signal per token than next-token prediction: a loss on 15% of positions instead of 100%. that's the price of bidirectionality, and it's one reason BERT-style models were never used to generate text and why ELECTRA (Clark et al., 2020) replaced the objective with a cheaper-to-learn "was this token replaced?" task over every position.

> **spec.** `mlm_loss(model, inputs, labels, attention_mask=None)`: the cross-entropy of `model.mlm_logits(model(inputs, attention_mask))` against `labels`, ignoring $-100$. `pretrain_mlm(model, data, *, mask_token_id, special_ids=(), attention_mask=None, steps=500, batch_size=32, lr=1e-3, mask_prob=0.15, seed=0)`: `data` `(N, T)` int64; each step draws `batch_size` random rows (`torch.randint` with a generator seeded by `seed`, which also drives `mask_tokens`), masks them afresh, passes the rows' `attention_mask` to the model, and takes an AdamW step (weight decay 0.01, gradients clipped at norm 1 with `nn.utils.clip_grad_norm_`) on `mlm_loss`. returns the list of per-step losses, leaving the model in eval mode.

> **▶ build: stage 3.** implement `mlm_loss` and `pretrain_mlm`.

### idea 4: fine-tuning

to classify a sequence, you need one vector for it. **[CLS] pooling** takes the final hidden state of the first token: during fine-tuning, attention learns to gather whatever the task needs into that position. **mean pooling** averages the hidden states of the real tokens (never the padding). a linear head maps the pooled vector to class logits. then the *whole* network, pretrained encoder included, is trained on the labeled data with a small learning rate. the encoder starts from weights that already represent text well; the head starts from scratch.

the alternative, a **linear probe**, freezes the encoder and trains only the head. it's cheaper and tells you how much task-relevant information the frozen features already contain (chapter 25 uses probes to evaluate representations), but fine-tuning usually wins when the task differs from pretraining.

> **spec.** `BERTClassifier(encoder, num_classes, *, pool='cls', dropout=0.1)` keeps `encoder` (a `BERT`) and a `head` `nn.Linear(encoder.d_model, num_classes)`, with dropout before the head; `pool` is `'cls'` or `'mean'` (otherwise `ValueError`). `forward(idx, attention_mask=None)` → logits `(B, num_classes)` from the first position's hidden state (`'cls'`) or the mean over real tokens (`'mean'`). `finetune(model, X, y, *, attention_mask=None, epochs=5, batch_size=32, lr=3e-4, seed=0)`: AdamW (weight decay 0.01, clipping at 1) on the cross-entropy over minibatches of a fresh `torch.randperm` (generator seeded by `seed`) each epoch; returns the list of per-epoch mean training losses and leaves the model in eval mode.

> **🔮 predict:** with 40 labeled messages, how much better is the pretrained encoder than the same encoder trained from scratch? and with all 4,460? (lab, experiment 1.)

> **▶ build: stage 4.** implement `BERTClassifier` and `finetune`.

## worked example

**what the corruption does**, for a sequence of 100 non-special tokens over a vocabulary of $V = 50$: on average 15 positions are selected; of those, 12 become [MASK], 1.5 get a random token (which equals the original with probability $1/50$, so 1.47 actually change), and 1.5 are left alone. so the input differs from the original in $12 + 1.47 = 13.47$ positions, 13.5%, while the loss is computed on 15. the tests check these proportions on 40,000 tokens.

**the loss**, for one selected position where the model gives the original token probability 0.25: its contribution is $-\ln 0.25 = 1.386$ nats. unselected positions contribute nothing, however wrong the model is about them. at initialization, with small weights, every token gets probability about $1/V$, so the MLM loss starts near $\ln V$: 3.91 for $V = 50$.

**training signal.** for a 128-token sequence, GPT gets 128 predictions per forward pass and BERT about 19. to see as many predictions, BERT needs about $6.7\times$ more passes over the data.

## on paper

1. **the leak.** a transformer *without* a causal mask is trained on next-token prediction ($y_t = x_{t+1}$). describe a set of weights in a single attention layer that achieves zero loss without learning anything about language.

<details><summary>solution</summary>

an attention head that attends from position $t$ to position $t + 1$ (possible with positional information: a query that matches the key of the next position) and copies its token identity through the values to the output, which the head maps back to that token. every prediction is then exactly right, the loss is 0 on the training data, and the model has learned to copy, not to model language. at test time, when the next token isn't in the input, it's useless. masking removes the answer from the input, so no such shortcut exists.

</details>

2. **how much of the input changes.** with selection probability $m$ and the 80/10/10 split over a vocabulary of $V$ tokens, what is the expected fraction of non-special input tokens that differ from the original? evaluate for $m = 0.15$, $V = 50$ and $V = 30{,}000$.

<details><summary>solution</summary>

a position changes if it's selected and then either masked (0.8) or replaced by a random token different from the original ($0.1 \cdot (1 - 1/V)$): fraction $m\,(0.8 + 0.1(1 - 1/V))$. for $m = 0.15$: $V = 50$ gives $0.15 \times 0.898 = 13.47\%$; $V = 30{,}000$ gives $0.15 \times 0.89999 \approx 13.5\%$. the model sees an uncorrupted input at 86.5% of positions and has to reconstruct 15%.

</details>

3. **why pretraining helps with few labels.** a classifier with $p$ parameters is trained on $n$ labeled examples. give a bias–variance argument for why starting from pretrained weights (and training briefly with a small learning rate) beats random initialization when $n$ is small, and why the advantage shrinks as $n$ grows.

<details><summary>solution</summary>

with small $n$, a randomly initialized network can fit the labels in many ways, most of which generalize badly: high variance. fine-tuning from pretrained weights with a small learning rate keeps the solution close to the starting point, an implicit prior that says "features useful for modeling text are useful here"; the effective number of free parameters is much smaller (much of the solution is already fixed by pretraining). this trades a little bias (the prior may not fit the task perfectly) for a large reduction in variance. as $n$ grows, the data alone pins down a good solution, the variance term shrinks, and the prior's help becomes small (though usually not zero: the pretrained model has seen far more text than the labeled set contains).

</details>

## experiment

the lab pretrains a small BERT on the text of 4,460 SMS messages, without their labels, then fine-tunes it to detect spam with 40, 160, 640 or all 4,460 labels, against the same encoder trained from scratch and against your logistic regression from chapter 08 on character counts. then it asks the pretrained model to fill in hidden characters, to see what it has learned.

## check your understanding

1. why are the 10% "unchanged" positions still included in the loss?

<details><summary>answer</summary>

so the model must produce a good prediction for a token it can see, which pushes its representation of every token towards "what belongs here given the context", not just "what's written here". if unchanged positions weren't in the loss, there'd be no training signal tying visible tokens' outputs to anything, and the model could learn that only [MASK] and suspicious tokens matter.

</details>

2. BERT can't generate text the way GPT does. why not, and what would you have to do to make it try?

<details><summary>answer</summary>

it was never trained to model $P(x_t \mid x_{<t})$: it models a token given context on both sides. to generate, you could append [MASK] tokens and fill them in one at a time (or iteratively refine all at once), but the model's conditionals don't form a consistent left-to-right factorization, and quality is poor without special training (that's what models like XLNet and masked diffusion models address).

</details>

3. the fine-tuning learning rate is usually 10–100× smaller than the pretraining one. why?

<details><summary>answer</summary>

the point of fine-tuning is to adapt the pretrained weights, not overwrite them. a large learning rate on a few labeled examples moves the encoder far from its pretrained solution in a few steps ("catastrophic forgetting"), throwing away what pretraining learned, and with few examples it overfits fast.

</details>

4. you fine-tune with [CLS] pooling on padded batches but forget the attention mask. what goes wrong, and would mean pooling make it better or worse?

<details><summary>answer</summary>

every token, including [CLS], attends to the padding tokens, so the representation of the same message depends on how much padding its batch needed: predictions change with batch composition. mean pooling without the mask is worse: it averages the padding positions' outputs directly into the sentence vector.

</details>

5. the lab's logistic regression on character counts is a strong baseline for spam detection. what does that tell you about the task, and why should papers report such baselines?

<details><summary>answer</summary>

much of the signal is in surface statistics (digits, capitals, currency symbols, message length), which a bag of characters captures. a transformer that beats a weak baseline might still lose to a simple one; without the simple baseline you can't tell whether the model's complexity is buying anything. chapter 07's "honest evaluation" applies to deep learning too.

</details>

## common bugs

- **masking special tokens.** [CLS] or [PAD] get selected and replaced; the model learns to predict padding, and [CLS] loses its role. symptom: a low MLM loss that doesn't help fine-tuning.
- **corrupting the input in place.** `idx[sel] = mask_id` without a clone destroys the data tensor, so the next epoch masks an already-masked corpus.
- **labels everywhere.** computing the loss on all positions rewards copying the visible tokens; with 85% of positions unchanged, the loss looks great and nothing is learned.
- **the padding mask as `(B, T)`.** it must reach attention as `(B, 1, 1, T)`; otherwise broadcasting fails, or worse, succeeds along the wrong axis.
- **mean pooling over padding.** divide by the number of real tokens, not by $T$.
- **the same mask every epoch.** static masks work, but dynamic ones are free and at least as good.

## going deeper

- Devlin, Chang, Lee & Toutanova (2019), "BERT: pre-training of deep bidirectional transformers for language understanding", *NAACL*. section 3.1 and appendix C.2 for the 80/10/10 ablation. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Liu et al. (2019), "RoBERTa: a robustly optimized BERT pretraining approach". dynamic masking, no next-sentence prediction, more data. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- Clark, Luong, Le & Manning (2020), "ELECTRA: pre-training text encoders as discriminators rather than generators", *ICLR*. [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)
- Howard & Ruder (2018), "Universal language model fine-tuning for text classification", *ACL*. pretrain-then-fine-tune with an LSTM, just before BERT. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
- Raffel et al. (2020), "Exploring the limits of transfer learning with a unified text-to-text transformer" (T5), *JMLR*. a systematic comparison of pretraining objectives. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Almeida, Gómez Hidalgo & Yamakami (2011), "Contributions to the study of SMS spam filtering: new collection and results", *DocEng*. the lab's dataset. [doi:10.1145/2034691.2034742](https://doi.org/10.1145/2034691.2034742)

## connections

- ← builds on: attention with padding masks (topic 01), the pre-LN block (03), weight tying and the ignore index (04), balanced accuracy (ch 07), logistic regression (ch 08).
- → used in: the ViT's [CLS] token and pooling choices (topic 07); masked autoencoders and linear probes (ch 25); fine-tuning, LoRA and reward models on top of pretrained transformers (ch 27); probing representations (ch 28).
