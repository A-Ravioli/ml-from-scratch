# 18 · transformers

> build, train, and run efficient inference on the architecture behind modern AI.

chapter 17 ended with attention bolted onto a recurrent network. this chapter removes the recurrence and builds everything out of attention, the way Vaswani et al. did in 2017, and then follows the architecture to where it went: GPT, BERT, fast inference and vision. you'll write scaled dot-product and multi-head attention with exactly the masking and cache conventions the rest of the course relies on; three ways to tell attention where tokens are (sinusoids, rotations, a linear bias); the pre-LN and post-LN blocks and the full encoder–decoder; a GPT with weight tying, GPT-2 initialization, temperature/top-k/nucleus sampling and a training loop; a BERT with masked-language-model pretraining and fine-tuning; a KV cache, FlashAttention's online softmax, grouped-query sharing and linear attention; and a vision transformer. the capstone is nanoGPT: three GPTs on tiny Shakespeare, a scaling curve, the test set chapter 17's LSTM was measured on, and cached generation. on the way you'll meet an unscaled attention layer that can't learn a lookup at width 512, a rotary-embedding model that is the best of five at the length it was trained on and the worst of four at eight times that length, a 24-layer post-LN transformer that trains for 200 steps without learning anything but letter frequencies, a greedy decoder that writes "the the the the", a pretrained encoder that is barely better than guessing a space at fill-in-the-blank yet lifts spam detection from 40 labels by six points, a KV cache that does 500 times less work and runs 10 to 20 times faster, and a ViT that loses to a ResNet with 1,000 images and still loses with 20,000.

**you will build:** `causal_mask`, `scaled_dot_product_attention`, `split_heads`, `merge_heads`, `MultiHeadAttention` (topic 01); `sinusoidal_positions`, `RotaryEmbedding`, `alibi_slopes`, `alibi_bias`, `PositionalLM` (02); `FeedForward`, `TransformerBlock`, `EncoderDecoder`, `block_grad_norms` (03); `get_batch`, `GPT` (with `generate`), `sample_next_token`, `estimate_loss`, `train_lm` (04); `mask_tokens`, `BERT`, `mlm_loss`, `pretrain_mlm`, `BERTClassifier`, `finetune` (05); `LayerKVCache`, `KVCache`, `generate_cached`, `online_softmax_attention`, `repeat_kv`, `kv_cache_bytes`, `linear_attention`, `linear_attention_step` (06); `patchify`, `PatchEmbedding`, `ViT`, `attention_distance` (07). together they make up **`mlfs.transformer`** (PyTorch), which chapters 19, 21, 22, 25, 26, 27 and 28 build on.
**prerequisites:** chapter 17's tokenizers, language modeling, perplexity and Bahdanau attention (`mlfs.text`, `mlfs.seq`); chapter 15's PyTorch toolkit, layer norm, AdamW, warmup + cosine and clipping (`mlfs.torchnn`, `mlfs.train`); chapter 16's ResNet for topic 07 (`mlfs.cnn`); softmax and its gradient (ch 03, 08); variance of sums (ch 04); the bootstrap (ch 05); logistic regression (ch 08).
**time · scaffolding:** about 90 hours · L4 (spec-driven): each stage is specified in a **spec** box in the lesson (names, shapes, conventions) and checked live; the internal structure is yours, the notebooks give signatures and one-line docstrings, and hints have one level. the capstone gives the experiment cells and the brief; the five functions it needs are yours.
**newly unlocked library components:** from here on you may use `nn.MultiheadAttention` and `F.scaled_dot_product_attention` (topic 01; note `nn.MultiheadAttention`'s masks use True = *blocked*, and `F.scaled_dot_product_attention`'s `is_causal` aligns to the top left), rotary-embedding helpers from other libraries (02), `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer` and `nn.Transformer` (03), pretrained causal LMs and their generation utilities such as Hugging Face `transformers` (04), pretrained BERT-style encoders (05), fused attention kernels (PyTorch's flash/memory-efficient SDPA backends) and library KV caches (06), and `torchvision`/`timm` vision transformers (07). inside `mlfs`, `GPT`, `KVCache` and `generate_cached` are what chapter 27's LLM pipeline builds on.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [self-attention](01-self-attention/) | `causal_mask`, `scaled_dot_product_attention`, `split_heads`, `merge_heads`, `MultiHeadAttention` → `mlfs.transformer` | without the $\sqrt d$, scores spread to ±32 at $d = 1024$ and the gradient through the softmax is 10× smaller; trained on a lookup task, the unscaled layer is fine at $d = 32$ and stuck at 67% at $d = 512$ while the scaled one solves it in 25 steps; shuffled inputs give exactly shuffled outputs, unless the mask is causal |
| 02 | [positional information](02-positional-information/) | `sinusoidal_positions`, `RotaryEmbedding`, `alibi_bias`, `PositionalLM` → `mlfs.transformer` | trained on 64 characters and read at 512: ALiBi stays flat (2.98 bits/char), sinusoidal jumps by 1.3–1.5 bits, RoPE, the best at 64, degrades to 5.2–5.8, worse than no positional information at all (3.6) |
| 03 | [the transformer block](03-the-transformer-block/) | `FeedForward`, `TransformerBlock` (pre/post-LN, cross-attention), `EncoderDecoder`, `block_grad_norms` → `mlfs.transformer` | a 24-layer post-LN model without warmup sits at 4.78 bits/char, the unigram entropy, for 200 steps (at any learning rate tried), while pre-LN reaches 3.54; 100 warmup steps rescue it |
| 04 | [building GPT](04-building-gpt/) | `get_batch`, `GPT`, `sample_next_token`, `GPT.generate`, `estimate_loss`, `train_lm` → `mlfs.transformer` | after the same 300 steps, every larger model is better at every step (3.67 → 3.05 bits/char from 17k to 1.3M parameters); greedy decoding writes the text the model finds most probable (1.62 bits/char) and it's "the the the the" (7% distinct chunks) |
| 05 | [BERT-style encoders](05-bert-style-encoders/) | `mask_tokens`, `BERT`, `mlm_loss`, `pretrain_mlm`, `BERTClassifier`, `finetune` → `mlfs.transformer` | MLM pretraining on unlabeled SMS text lifts spam detection with 40 labels from 0.78 to 0.84 balanced accuracy (and beats every from-scratch budget), though the pretrained model recovers hidden characters barely better than guessing a space (18.6% vs 18.2%); a logistic regression on character counts ties a from-scratch transformer |
| 06 | [efficient attention & inference](06-efficient-attention-and-inference/) | `KVCache`, `generate_cached`, `online_softmax_attention`, `repeat_kv`, `kv_cache_bytes`, `linear_attention` → `mlfs.transformer` | the cache cuts token-passes 500-fold for 1,000 new tokens and runs only 10–20× faster on a CPU (machine-dependent); 16 conversations of 4,096 tokens on a 7B model need 32 GiB of cache against 12.5 GiB of weights; online softmax saves 94% of the memory and almost no CPU time |
| 07 | [vision transformers](07-vision-transformers/) | `patchify`, `PatchEmbedding`, `ViT`, `attention_distance` → `mlfs.transformer` | at equal size, recipe and (short) budget the ResNet wins by 9 points with 1,000 CIFAR-10 images and by 8–10 with 5,000 or 20,000; removing the ViT's positions costs under 2 points; after training, no head has become local |

## capstone

*nanoGPT*: three GPTs (0.35M, 1.8M and 7.5M parameters) trained with your `train_lm` on tiny Shakespeare at the character level, with the 80/10/10 split of chapters 04 and 17, a power-law fit through their validation losses, the best model measured on every test character with sliding windows and block-bootstrap intervals against chapter 17's LSTM (2.353 bits/char) and n-gram (2.631), and generation timed with and without your KV cache. the reference run (25 minutes on a shared laptop GPU) finds validation loss falling with size but bending (local slopes 0.097 from 0.35M to 1.8M parameters, 0.043 from 1.8M to 7.5M, at a fixed step budget), and its 7.5M-parameter GPT at **2.457 bits per character on test** [2.400, 2.514]: far ahead of the n-gram, tied with the LSTM on validation (2.033 vs 2.026), and 0.10 bits behind it on test, with a 128-character context against the LSTM's unlimited one. the cache makes greedy generation 4.1× faster on the CPU, and only 2.0× on the laptop GPU, which generates more slowly than the CPU at this size. see [`capstone/brief.md`](capstone/brief.md). small CPU config: under a minute on the offline synthetic stand-in.

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (an attention output with and without the causal mask, an online-softmax pass, and parameter and cache counts), and a 60-minute blind rebuild of masked scaled dot-product and multi-head attention with the cache hook, checked by topic 01's tests.

## research milestone

after this chapter: [**R1 · reproduce a deep learning paper**](../milestones/R1.md): the reproduction protocol, ten candidate papers (ResNet's degradation experiment, random-label memorization, dropout, batch norm, lottery tickets, pre-LN vs post-LN, ALiBi, the transformer on Multi30k, scaling laws, ViT's data scaling), a laptop compute budget, deliverables and a self-assessment.

## notation

| symbol | meaning |
|---|---|
| $B$, $T$ ($T_q$, $T_k$), $d$ ($d_{\text{model}}$) | batch size, sequence length (queries, keys), model width |
| $H$, $d_h = d/H$, $H_{kv}$ | attention heads, head width, key/value heads (grouped-query attention) |
| $q_i$, $k_j$, $v_j$; $Q$, $K$, $V$ | query, key, value vectors; their stacked matrices |
| $s_{ij} = q_i \cdot k_j / \sqrt{d_h}$, $a_{ij}$ | attention score and weight (softmax over $j$) |
| $W_Q, W_K, W_V, W_O$ | projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) |
| $\text{PE}(p)$, $\omega_i$ | sinusoidal position vector, its frequencies $10000^{-2i/d}$ |
| $\theta_i$, $R(\phi)$ | RoPE frequencies $\text{base}^{-2i/d_h}$, a 2-D rotation |
| $m_h$ | ALiBi slope of head $h$ |
| $L$, $d_{ff}$ | number of blocks, MLP hidden width (default $4d$) |
| $E$, $V$ | token embedding matrix, vocabulary size |
| $T$ (sampling), $k$, $p$ | temperature, top-k, top-p (nucleus mass) |
| $m$, $\ell$, $a$ | online softmax's running max, normalizer, numerator |
| $\phi$, $S$, $z$ | linear attention's feature map ($\operatorname{elu} + 1$), state matrix and normalizer |
| $p$ (images), $N$ | patch size, number of patches |
| $N$ (scaling), $L(N)$ | parameter count, loss as a function of it |

## how this connects

- builds on: Bahdanau attention, seq2seq, tokenizers and language modeling (ch 17), PyTorch training (ch 15), residual networks and the CIFAR recipe (ch 16), softmax and its gradient (ch 03, 08), the bootstrap (ch 05).
- used in: loss landscapes, the NTK and scaling laws of transformers (ch 19); attention as message passing on graphs (ch 20); autoregressive and conditional generation (ch 21); the diffusion transformer (ch 22); MAE, CLIP and multimodal models (ch 25); FLOPs, memory and kernels (ch 26); the Llama-style model, LoRA, preference learning, decoding and retrieval of chapter 27; attention patterns, activation patching and attribution (ch 28).

## core references

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin (2017), "Attention is all you need", *NeurIPS*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford, Wu, Child, Luan, Amodei & Sutskever (2019), "Language models are unsupervised multitask learners" (GPT-2). [cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Devlin, Chang, Lee & Toutanova (2019), "BERT: pre-training of deep bidirectional transformers for language understanding", *NAACL*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Dao, Fu, Ermon, Rudra & Ré (2022), "FlashAttention: fast and memory-efficient exact attention with IO-awareness", *NeurIPS*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dosovitskiy et al. (2021), "An image is worth 16x16 words: transformers for image recognition at scale", *ICLR*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Prince (2023), *Understanding Deep Learning*, MIT Press, chapter 12 (transformers). [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)
- Phuong & Hutter (2022), "Formal algorithms for transformers". [arXiv:2207.09238](https://arxiv.org/abs/2207.09238)
