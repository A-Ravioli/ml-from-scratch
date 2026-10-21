# 27 · large language models

> the full LLM lifecycle, from data to aligned assistant.

a chat assistant is the end of a long pipeline, and most of the decisions that make it good happen before anyone types a question: which pages of the web are kept, how many copies of each, how big a model the compute budget buys, what the architecture swaps in since GPT-2, how a pretrained text predictor learns to answer instead of rambling, how human preferences become a training signal, how its outputs are sampled, and how anyone can tell whether it got better. this chapter walks that pipeline end to end at a scale you can run on a laptop. you'll deduplicate a crawl, build a Llama with grouped-query attention and a mixture-of-experts layer, fine-tune it with LoRA, align it with DPO (and watch a policy hack its reward model with PPO), make it decode faster with speculative sampling, evaluate it honestly, and give it a retriever and tools. the capstone puts every step together: your own small assistant, trained from raw text, with a model card that says exactly what it is.

**you will build:** MinHash + LSH deduplication, quality filters, data mixtures and the value of repeated data; `SwiGLU`, `GroupedQueryAttention`, `MoE`, `LlamaBlock`, `Llama`; `LoRALinear`, `apply_lora`, `merge_lora`, SFT data and loss, NF4 quantization; sequence log-probabilities, a `RewardModel`, the Bradley–Terry and DPO losses, a DPO step, KL-shaped RLHF rewards; repetition penalty, min-p, speculative and constrained decoding; a log-likelihood eval harness with bootstrap and paired intervals, contamination checks, a debiased pairwise judge; chunking, a hashing TF-IDF embedder, exact and IVF vector search, a RAG prompt, an agent loop and majority voting. all of it is `mlfs.llm`.
**prerequisites:** chapter 18 (your GPT, `RotaryEmbedding`, `repeat_kv`, KV cache and `generate_cached`), chapter 17 (`BPETokenizer`, `CharTokenizer`), chapter 26 (the memory-bound decoding argument, quantization), chapter 24 (`ppo_clip_loss`), chapter 19 (scaling laws), chapter 11 (`KMeans`), chapter 05 (the bootstrap).
**time · scaffolding:** L4 · ~110 h. the lessons' **spec** boxes and the checks are the contract; internal structure is yours.
**newly unlocked library components:** `torch.nn.RMSNorm` and `F.scaled_dot_product_attention(..., enable_gqa=True)` (after 02), PEFT's LoRA and bitsandbytes 4-bit loading (after 03), TRL's reward and DPO trainers (after 04), Hugging Face `generate(assistant_model=...)` and grammar-constrained generation libraries (after 05), lm-evaluation-harness (after 06), FAISS and pretrained sentence embedders (after 07). (build-before-you-use, see AUTHORING.md §9)

**data.** there's no web crawl, instruction dataset, preference dataset or document collection on this machine, and nothing is downloaded. every topic says plainly what it generates: a templated crawl with mirrors, syndicated copies and spam (01), templated instructions such as "Write *crown* in capitals." and "Continue: *half a line*" (03, capstone), preferences labelled by a hidden "gold" rule standing in for human judgement (04, capstone), a cloze benchmark from held-out lines (06), and a knowledge base of invented countries (07). the language underneath is always tiny Shakespeare. **hardware.** the labs run on a laptop CPU in a few minutes each; the capstone's full configuration took 23 minutes on an Apple-silicon GPU through `mps` (a CUDA GPU works too), and its small configuration runs anywhere in about a minute.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build |
|---|---|---|
| 01 | [pretraining data & compute](01-pretraining-data-and-compute/) | shingles, MinHash, LSH dedup, Gopher-style quality filters, mixture weights, effective tokens of repeated data |
| 02 | [modern LLM architecture](02-modern-llm-architecture/) | SwiGLU, rotary grouped-query attention with a KV cache, a sparse MoE with load balancing, the Llama block and model |
| 03 | [fine-tuning](03-fine-tuning/) | LoRA (wrap, freeze, merge), instruction examples with a response-only loss, NF4 for QLoRA |
| 04 | [preference learning](04-preference-learning/) | sequence log-probs, a reward model, Bradley–Terry and DPO losses, a DPO step, KL-penalized RLHF rewards |
| 05 | [decoding & inference](05-decoding-and-inference/) | repetition penalty, min-p, the speculative acceptance step, speculative and constrained decoding |
| 06 | [evaluation](06-evaluation/) | log-likelihood scoring, multiple choice, bootstrap and paired intervals, n-gram contamination, a debiased judge |
| 07 | [retrieval, tools & reasoning](07-retrieval-tools-and-reasoning/) | chunking, a hashing TF-IDF embedder, top-k cosine search, an IVF index, RAG prompts, an agent loop, majority voting |

## capstone

*your own assistant pipeline*: pretrain a small Llama on filtered, deduplicated Shakespeare, fine-tune it on synthetic instructions, align it with DPO to a hidden preference rule, evaluate every stage with intervals (skills, gold reward, win rate, held-out loss) over several seeds, and write its model card. see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it.

## notation

| symbol | meaning |
|---|---|
| $J(A, B)$ | Jaccard similarity of two sets |
| $b$, $r$ | LSH bands and rows per band ($n_\text{perm} = br$) |
| $N$, $D$, $C$ | parameters, training tokens, compute ($C \approx 6ND$) |
| $U$, $R$, $R^*$ | unique tokens, repetitions (epochs − 1), the repetition half-life constant |
| $d$, $d_\text{ff}$, $H$, $H_{kv}$, $d_h$ | width, MLP width, query heads, key/value heads, head width |
| $E$, $k$ | experts and experts per token (MoE); also the top-$k$ of retrieval |
| $f_e$, $P_e$ | share of routing slots and mean router probability of expert $e$ |
| $W_0$, $A$, $B$, $r$, $\alpha$ | frozen weight, LoRA factors ($A$: $r \times d_\text{in}$, $B$: $d_\text{out} \times r$), rank, scale |
| $\pi$, $\pi_\text{ref}$, $\beta$ | policy, reference policy, KL / DPO coefficient |
| $r_\phi(x, y)$ | reward model's score of response $y$ to prompt $x$ |
| $y_w$, $y_l$ | preferred (chosen) and dispreferred (rejected) responses |
| $p$, $q$, $\alpha$ (decoding) | target and draft distributions, acceptance probability $\sum\min(p, q)$ |
| $\ell(c \mid x)$ | log-likelihood of continuation $c$ given context $x$ |
| $L$, $p$ (retrieval) | IVF lists and lists probed |

## how this connects

- builds on: your GPT, RoPE, KV cache and `generate_cached` (ch 18), tokenizers (ch 17), scaling laws (ch 19), PPO (ch 24), quantization and the decoding roofline (ch 26), k-means (ch 11), the bootstrap (ch 05).
- used later in: chapter 28 opens up your `Llama` to look inside (activation patching, probes, sparse autoencoders), revisits reward hacking as an alignment failure, and builds safety evals with your harness; chapter 30's research projects start from `Llama`, LoRA, `sequence_logprobs`, `dpo_loss`, the eval harness and the retriever.

## core references

- Touvron et al. (2023), "LLaMA: open and efficient foundation language models". [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Hoffmann et al. (2022), "Training compute-optimal large language models". [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Lee et al. (2022), "Deduplicating training data makes language models better", *ACL*. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- Hu et al. (2021), "LoRA: low-rank adaptation of large language models". [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback". [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov et al. (2023), "Direct preference optimization: your language model is secretly a reward model". [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Leviathan, Kalman & Matias (2023), "Fast inference from transformers via speculative decoding". [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Lewis et al. (2020), "Retrieval-augmented generation for knowledge-intensive NLP tasks". [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
