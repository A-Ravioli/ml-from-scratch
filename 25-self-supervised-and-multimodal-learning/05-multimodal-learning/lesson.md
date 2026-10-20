# multimodal learning

## why this matters

every classifier you've trained so far has a fixed list of classes, baked into its last layer. to add "zebra" you collect labelled zebras and retrain. CLIP (Radford et al., 2021) removed that wall. it trains an image encoder and a text encoder together, on 400 million images paired with the text that appeared next to them on the web, so that an image and its caption land close together in one shared embedding space. afterwards a classifier for *any* set of classes can be written in words: embed "a photo of a zebra", "a photo of a horse", …, and label each image with the nearest caption. no zebra labels, no retraining. CLIP's zero-shot ImageNet accuracy matched the original supervised ResNet-50 without using a single one of ImageNet's 1.28 million labels.

the training objective is the contrastive loss you built in topic 02, applied *across* modalities: an image's positive is its own caption and its negatives are the other captions in the batch. in this topic you build a mini-CLIP: the symmetric loss, a small text transformer, the two projections and the learnable temperature, zero-shot classifiers with prompt ensembles, and the training loop. the lab trains it on an image–caption dataset we generate ourselves, and finds out how much the *wording* of a prompt changes what the model sees.

## warm-up: predict first

1. a batch of 256 image–caption pairs. how many negatives does each image get in CLIP's loss, and how does that differ from SimCLR's $2B - 2$?
2. two images in a batch have captions "a red digit" and "a red digit". what does that do to the loss?
3. CLIP learns its temperature instead of fixing it. why parameterise it as $\log(1/T)$ rather than $T$?
4. you ask a zero-shot classifier for "the digit 7", but the model only ever saw digits written as words ("seven"). what happens?
5. CLIP itself doesn't generate captions. how could you use it to caption an image anyway?

## prerequisites check

- [ ] you have InfoNCE and know why the temperature matters (→ topic 02)
- [ ] you have `TransformerBlock` with masks, and know key-padding masks (→ ch 18 · topics 03 and 05)
- [ ] you know embeddings and word-level tokenisation (→ ch 17 · topic 01)
- [ ] you have the evaluation ideas of topic 01 (zero-shot classification is a kind of probe with no training)

## core ideas

### idea 1: the symmetric contrastive loss

take a batch of $B$ image–caption pairs. encode the images to unit vectors $u_1, \dots, u_B$ and the captions to unit vectors $v_1, \dots, v_B$ in the same $d$-dimensional space. the $B \times B$ matrix of scaled cosine similarities $L_{ij} = s\,u_i^\top v_j$ (with $s = 1/T$ the **logit scale**) has the true pairs on its diagonal. CLIP's loss reads it both ways:

$$\mathcal L = \frac12\Big[\underbrace{\frac1B\sum_i -\log\frac{e^{L_{ii}}}{\sum_j e^{L_{ij}}}}_{\text{image} \to \text{text: each row}} + \underbrace{\frac1B\sum_j -\log\frac{e^{L_{jj}}}{\sum_i e^{L_{ij}}}}_{\text{text} \to \text{image: each column}}\Big].$$

each row is a $B$-way classification "which caption is mine?", each column "which image is mine?". compared with SimCLR's NT-Xent, the negatives are only the *other modality's* items: an image is never compared with another image, so each anchor has $B - 1$ negatives, not $2B - 2$. both directions matter: with only the rows, the loss is satisfied if every image finds its caption, even when one caption is the best match for several images.

the loss can't always reach zero. real captions are ambiguous ("a dog on grass" fits many photos), and duplicates in a batch create pairs where the "negative" is as correct as the positive (worked example). our generated captions are deliberately vague sometimes, so in the lab the loss settles well above zero.

> **spec.** `clip_loss(image_emb, text_emb, logit_scale)`: `(B, d)` tensors (normalise them yourself) and a float or scalar tensor. returns the scalar $\mathcal L$ above: the mean of the cross-entropy over rows and the cross-entropy over columns of `logit_scale * normalize(image_emb) @ normalize(text_emb).T`, with targets $0, \dots, B - 1$. differentiable in all three arguments.

> **🔮 predict:** our captions are generated from 6 templates, and some leave out the digit or the position. will the training loss approach 0, level off near $\log B$, or something in between? (the lab plots it, for $B = 256$.)

> **▶ build: stage 1.** implement `clip_loss`.

### idea 2: a text encoder

CLIP's text side is a transformer over byte-pair tokens. ours is a small one over words: an `nn.Embedding` for the tokens, fixed sinusoidal positions (so "red seven" and "seven red" differ), a stack of your `TransformerBlock`s, a final LayerNorm, and a pooling step that turns $T$ token vectors into one caption vector.

captions have different lengths, so a batch is padded with a `pad_id` token. padding must change nothing: no real token may **attend** to a pad (pass a key mask of shape `(B, 1, 1, T)`, True for real tokens, to every block), and pads must not be **pooled**. we pool by averaging the real tokens' outputs. CLIP instead uses a causal mask and takes the output at the end-of-text token, which then summarises the whole caption; mean pooling is simpler and works well for short captions.

> **spec.** `TextTransformer(vocab_size, *, max_len=16, d_model=64, n_layers=2, n_heads=4, pad_id=0)` is an `nn.Module` with `tok_emb = nn.Embedding(vocab_size, d_model)`, a buffer `pos_emb = sinusoidal_positions(max_len, d_model)`, `blocks` (an `nn.ModuleList` of `n_layers` `TransformerBlock(d_model, n_heads)`) and `ln_f = nn.LayerNorm(d_model)`. `forward(tokens)` takes int64 `(B, T)` with $T \le$ `max_len` (`ValueError` otherwise) and returns `(B, d_model)`: the mean over real (non-pad) positions of `ln_f(blocks(tok_emb(tokens) + pos_emb[:T]))`, each block called with `mask = (tokens != pad_id)[:, None, None, :]`.

> **▶ build: stage 2.** implement `TextTransformer`.

### idea 3: two encoders, one space

MiniCLIP glues an image encoder (any network producing `image_dim` features) and a text encoder (`text_dim` features) together. each gets a linear **projection** (no bias, as in CLIP) into a shared `embed_dim`-dimensional space, followed by L2 normalisation, so similarity is a cosine.

the temperature is learned. CLIP stores $\log s$ as a parameter, initialised to $\log(1/0.07)$, and uses $s = e^{\log s}$: the exponential keeps $s$ positive whatever the optimiser does, and makes steps multiplicative (a step of 0.1 in $\log s$ scales $s$ by 10% whether $s$ is 14 or 90; on paper 2). left alone, $s$ grows during training (sharper softmax, lower loss on easy batches), and CLIP clamps it at 100 to keep training stable. in the lab you'll watch it climb.

> **spec.** `MiniCLIP(image_encoder, text_encoder, image_dim, text_dim, *, embed_dim=128, init_temperature=0.07)` is an `nn.Module` with the two encoders as attributes `image_encoder` and `text_encoder`, `image_proj = nn.Linear(image_dim, embed_dim, bias=False)`, `text_proj = nn.Linear(text_dim, embed_dim, bias=False)` and `log_scale = nn.Parameter(torch.tensor(log(1 / init_temperature)))`. `encode_image(images)` and `encode_text(texts)` return unit vectors `(B, embed_dim)`. `forward(images, texts)` returns `(encode_image(images), encode_text(texts), log_scale.exp().clamp(max=100))`, with the logit scale still connected to the graph.

> **▶ build: stage 3.** implement `MiniCLIP`.

### idea 4: zero-shot classification, prompts and captions

once images and captions share a space, a classifier is a list of sentences. for each class $c$, embed a **prompt** such as "a photo of a {c}"; the class embeddings $w_1, \dots, w_K$ form the weights of a linear classifier on the image embeddings, and the prediction is $\arg\max_c \cos(u, w_c)$. nothing is trained: it's **zero-shot transfer**.

the wording matters, because the text encoder has learned a distribution of captions, and a prompt is only as good as its resemblance to that distribution. a bare label like "boxer" is ambiguous (a dog? an athlete?); "a photo of a boxer, a type of dog" is not. Radford et al. report that prompt templates gain several points over bare class names, and that **ensembling** helps too: embed the class with many templates ("a photo of a {c}", "a blurry photo of a {c}", "a drawing of a {c}", …), normalise each embedding, average, and renormalise. the ensemble cancels the quirks of individual phrasings, and costs nothing at test time because the averaged embedding is computed once per class.

CLIP can't write captions, but it can **choose** them: embed a set of candidate captions and pick the one closest to the image. that's retrieval, and it's what CLIP is used for in search engines (and, the other way round, text → image search, which the lab tries). *generating* a caption needs a decoder: a language model that cross-attends to the image encoder's features (the encoder–decoder transformer of ch 18, with image patches as the source sequence), trained with next-token prediction on captions. CoCa (Yu et al., 2022) trains both objectives at once, contrastive and captioning, on one image encoder.

> **spec.** `zero_shot_predict(image_emb, class_emb)`: `(B, d)` and `(n_classes, d)` tensors → `(B,)` int64, the index of the class with the highest cosine similarity (normalise both). `class_embeddings(model, class_names, templates, tokenize)`: for each name (in order), `tokenize([t.format(name) for t in templates])`, encode with `model.encode_text` in eval mode under `torch.no_grad()` (restore the model's mode afterwards), normalise each embedding, average over templates and renormalise. returns `(n_classes, d)`. `tokenize` maps a list of strings to whatever `encode_text` takes.

> **🔮 predict:** our model only ever saw digits written as words ("seven"). what accuracy do you expect from the prompts "the digit 0", …, "the digit 9"?

> **▶ build: stage 4.** implement `zero_shot_predict` and `class_embeddings`.

### idea 5: training

the loop is the familiar one: shuffle the pairs, take a batch of images and their captions *together* (shuffling them separately would teach the model that random captions match), compute both embeddings and the logit scale, step on `clip_loss`. the temperature is trained with everything else.

batch size matters even more than in SimCLR: CLIP used batches of 32,768 pairs, so each image chose its caption out of 32,768. at our scale (batches of 256, 20,000 pairs) the model sees each pair a few times, and its negatives often include captions that are just as true as the positive.

> **spec.** `train_clip(model, images, texts, *, epochs=1, batch_size=256, lr=1e-3, seed=0, device='cpu')`: `images` an array `(N, …)` (cast to float32), `texts` an int array `(N, T)` of token ids (int64), row $i$ of each a matching pair. `torch.manual_seed(seed)` first; move the model to `device`; Adam (`lr`) over all parameters. each epoch: `model.train()`, a fresh permutation from `np.random.default_rng(seed)` (one generator per run), `N // bs` full batches with `bs = min(batch_size, N)`, the same indices for images and texts; per batch, `img, txt, scale = model(...)` and a step on `clip_loss(img, txt, scale)`. returns `{'loss': [...], 'logit_scale': [...]}`, one float per step (the scale used in that step).

> **▶ build: stage 5.** implement `train_clip`.

## worked example

**the loss, with an ambiguous batch.** with orthogonal pairs, $u = v = (e_1, e_2)$ and $s = 1$: $L = I$, every row and column is $-\log\frac{e}{e + 1} = \log(1 + e^{-1}) = 0.3133$, so $\mathcal L = 0.3133$; at $s = 10$, $\log(1 + e^{-10}) \approx 4.5 \times 10^{-5}$.

now two *identical* images, $u_1 = u_2 = e_1$, with captions $v_1 = e_1$, $v_2 = e_2$ and $s = 1$: $L = \begin{pmatrix}1 & 0\\ 1 & 0\end{pmatrix}$. rows: image 1 picks caption 1 with probability $e/(e + 1)$, loss 0.3133; image 2 should pick caption 2 but prefers caption 1, loss $\log(1 + e) = 1.3133$. columns: caption 1 can't tell the two images apart, loss $\log 2 = 0.6931$, and caption 2 neither, $0.6931$. so $\mathcal L = \frac12\big(\frac{0.3133 + 1.3133}{2} + 0.6931\big) = 0.7532$. no encoder can get below $\log 2 = 0.6931$ on this batch: the two images are the same, so each column's two logits are always equal, and the two rows share one softmax, whose two probabilities can't both exceed $\frac12$. an encoder that makes the two captions equally similar to the image reaches that floor; this configuration pays 0.06 more.

**zero-shot with misleading norms.** image embedding $(1, 0.1)$, class embeddings $(10, 10)$, $(1, 0)$ and $(0, 1)$. cosines: $0.77$, $0.995$, $0.0995$, so class 1. a dot product would give $11$, $1$, $0.1$ and pick class 0, just because its vector is long.

**a prompt ensemble.** "a cat" embeds to $(2, 0)$ and "the cat" to $(0, 0.5)$. normalised: $(1, 0)$ and $(0, 1)$; averaged: $(0.5, 0.5)$; renormalised: $(0.707, 0.707)$. averaging the raw vectors would give $(1, 0.25)$, renormalised $(0.970, 0.243)$: the longer embedding would dominate the ensemble.

your code must reproduce these numbers.

## on paper

1. **two InfoNCEs.** show that $\mathcal L$ is the average of two InfoNCE losses with $B - 1$ negatives each. what is $\mathcal L$ for an untrained model whose similarities are all equal? if a batch contains $k$ images that all carry the identical caption text, show that the image → text loss of each of those images can't go below $\log k$.

<details><summary>solution</summary>

row $i$ is $-\log\frac{e^{L_{ii}}}{\sum_je^{L_{ij}}}$, an InfoNCE with the positive $v_i$ and $B - 1$ negatives $v_{j \ne i}$; the columns are the same with the roles swapped. with equal similarities every softmax is uniform over $B$ candidates, so $\mathcal L = \log B$: $5.55$ at $B = 256$, the chance level in the lab's loss plot. identical caption texts get identical embeddings $v$, so the row of one of those $k$ images has the same logit $s\,u_i^\top v$ in all $k$ of their columns. its softmax can put at most $1/k$ on its own caption, so its row loss is at least $\log k$, whatever the encoders do. (the same argument with identical *images* bounds the column terms.) generated captions like "a red digit in the top left" repeat often within a batch, which is one reason the lab's loss levels off well above zero.

</details>

2. **the learnable temperature.** with $s = e^{\theta}$, show that $\partial\mathcal L/\partial\theta = s\,\partial\mathcal L/\partial s$, and that a gradient step $\theta \leftarrow \theta - \eta g$ multiplies $s$ by $e^{-\eta g}$. why does $s$ tend to grow during training, and why clamp it?

<details><summary>solution</summary>

by the chain rule $\partial\mathcal L/\partial\theta = (\partial\mathcal L/\partial s)(ds/d\theta) = s\,\partial\mathcal L/\partial s$; the update gives $s' = e^{\theta - \eta g} = s\,e^{-\eta g}$, a relative change, and $s$ stays positive. once most positives already have the largest similarity in their row, scaling all logits up makes each softmax more confident in the right answer and lowers the loss, so $\partial\mathcal L/\partial s < 0$ and $s$ grows; on batches with true ambiguity, very large $s$ produces huge losses and gradients on the "hard" pairs, which destabilises training. CLIP clamps at 100 (a temperature of 0.01).

</details>

3. **a zero-shot classifier is a linear head.** show that prediction by $\arg\max_c\cos(u, w_c)$ with unit-norm image embedding $u$ is the same as a linear classifier with weight matrix $W = [\bar w_1, \dots, \bar w_K]$ (the normalised class embeddings) and no bias. what does that suggest about combining zero-shot classifiers with a few labelled examples?

<details><summary>solution</summary>

$\cos(u, w_c) = u^\top\bar w_c$ when $\|u\| = 1$, so the scores are $W^\top u$ and the argmax is a linear classifier's. a natural use of a few labels is therefore to *initialise* a linear probe (topic 01) with $W$ and fine-tune it, rather than starting from zero: the text-derived classifier is a strong prior. this is the idea behind methods such as WiSE-FT (Wortsman et al., 2022), which interpolate between the zero-shot and the fine-tuned weights.

</details>

## experiment

`lab.py` builds a paired dataset (MNIST digits painted in 7 colours and placed in 4 positions, with generated captions) and trains three mini-CLIPs on it, with your code.

1. does the wording of the prompt matter? zero-shot digit classification with six kinds of prompt.
2. combinations never seen in training: colour and digit of held-out colour–digit pairs, and text → image retrieval.

## check your understanding

1. why must images and captions be shuffled with the same permutation?

<details><summary>answer</summary>

the loss assumes row $i$ of the image batch matches row $i$ of the caption batch. shuffle them independently and every "positive" is a random caption: the model learns that nothing matches anything, and the loss stays near $\log B$.

</details>

2. why does CLIP compare images only with captions, never with other images?

<details><summary>answer</summary>

its goal is a shared space where the two modalities align; image–image negatives would add a SimCLR-like objective that needs augmentations (and, later work found, can help: SLIP, Mu et al., 2022, adds it). the cross-modal loss alone already forces images with different captions apart, because their captions differ.

</details>

3. in the lab, prompts with unknown words map to `<unk>`. what's the analogue for real CLIP, which has no unknown tokens?

<details><summary>answer</summary>

byte-pair encoding can spell any word, but a word (or phrasing) that rarely appeared in the training captions gets an embedding that the model has learned little about: the prompt lands in a poorly trained region of text space. "a photo of a {c}" works well for CLIP precisely because web captions often look like that.

</details>

4. a zero-shot classifier for "cat" vs "dog" works well; one for "a photo of a tabby" vs "a photo of a Maine Coon" doesn't. why?

<details><summary>answer</summary>

zero-shot accuracy is bounded by how often (and how consistently) the distinction appeared in the training captions. fine-grained breed names are rare and noisy on the web; the image encoder may not have learned the visual features that separate them, and no prompt can retrieve what was never learned.

</details>

5. why mean-pool the text tokens over real positions instead of over all $T$?

<details><summary>answer</summary>

pads carry no meaning; averaging over them dilutes the caption vector by an amount that depends on the caption's length (and the batch's padding), so the same caption would embed differently in different batches.

</details>

6. what does the logit scale climbing from 14 to 20 during training tell you?

<details><summary>answer</summary>

the model is becoming more confident: most positives are already ranked first in their rows and columns, so sharpening the softmax lowers the loss. if it shot to the clamp at 100 early, that could signal batches that are too easy (e.g. captions that trivially identify their images).

</details>

## common bugs

- **one direction only.** the loss trains images to find their captions but not captions to find their images; zero-shot classification (which goes text → image weights) suffers.
- **temperature stored as $T$ or $1/T$ directly:** it can go negative, and steps are additive; or fixed by accident (a float, not a parameter, or a `.detach()`).
- **padding attended to, or pooled:** the same caption embeds differently depending on how much padding its batch has.
- **dot products instead of cosines in zero-shot prediction:** the class with the longest embedding wins.
- **averaging unnormalised prompt embeddings:** one long embedding dominates the ensemble.
- **images and captions shuffled separately:** the loss never leaves $\log B$.

## going deeper

- Radford et al. (2021), *Learning transferable visual models from natural language supervision* (CLIP). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Jia et al. (2021), *Scaling up visual and vision-language representation learning with noisy text supervision* (ALIGN). [arXiv:2102.05918](https://arxiv.org/abs/2102.05918)
- Zhai, Mustafa, Kolesnikov & Beyer (2023), *Sigmoid loss for language image pre-training* (SigLIP): a pairwise sigmoid instead of the softmax. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- Yu et al. (2022), *CoCa: contrastive captioners are image-text foundation models*. [arXiv:2205.01917](https://arxiv.org/abs/2205.01917)
- Thrush et al. (2022), *Winoground: probing vision and language models for visio-linguistic compositionality*. [arXiv:2204.03162](https://arxiv.org/abs/2204.03162)
- Yuksekgonul, Bianchi, Kalluri, Jurafsky & Zou (2023), *When and why vision-language models behave like bags-of-words, and what to do about it?* [arXiv:2210.01936](https://arxiv.org/abs/2210.01936)
- Wortsman et al. (2022), *Robust fine-tuning of zero-shot models* (WiSE-FT). [arXiv:2109.01903](https://arxiv.org/abs/2109.01903)

## connections

- ← builds on: InfoNCE (topic 02), transformer blocks and masks (ch 18), word embeddings (ch 17), the evaluation harness (topic 01).
- → used in: embedding search for retrieval-augmented generation (ch 27 · topic 07) works exactly like this lab's text → image retrieval; conditional generation with guidance (ch 22 · topic 06) is how text embeddings like these steer image generators; linear probes of representations return in ch 28 · topic 01.
