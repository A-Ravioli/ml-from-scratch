# fine-tuning

## why this matters

a pretrained language model is a remarkable text predictor and a terrible assistant. ask it a question and it may continue with three more questions, because that's what text on the web does after a question. **supervised fine-tuning** (SFT) fixes this with a modest dataset of instructions paired with the responses you want: InstructGPT (Ouyang et al., 2022) used about 13,000 demonstrations, and LIMA (Zhou et al., 2023) got a strong assistant from 1,000 carefully chosen ones. the model already has the knowledge; fine-tuning mostly teaches it the format, the voice, and when to stop.

the second problem is cost. full fine-tuning updates every weight, and with AdamW that means storing gradients and two moments for each: about 16 bytes per parameter in mixed precision (chapter 26 · 05), over 100 GB for a 7B model. and every fine-tuned variant is a full copy of the model. **LoRA** (Hu et al., 2021) freezes the pretrained weights and learns a low-rank correction to each weight matrix: for GPT-3 it cut the trainable parameters by 10,000× and the memory by 3× with no loss in quality on their benchmarks, and the correction can be folded into the weights afterwards, so inference costs nothing extra. **QLoRA** (Dettmers et al., 2023) goes further and stores the frozen weights in 4 bits, fine-tuning a 65B model on a single 48 GB GPU.

this topic builds LoRA, the code that wraps a model with it and merges it back, the data pipeline for instruction tuning (with the loss on the response only), and the 4-bit NormalFloat format behind QLoRA.

## warm-up: predict first

1. a 4096 × 4096 attention projection gets a rank-8 LoRA update. how many trainable numbers is that, as a fraction of the layer?
2. why initialize one of the two LoRA factors to zero? what would go wrong if both were random?
3. in instruction tuning, should the loss include the prompt tokens? what changes if it does?
4. LoRA trains 1% of the parameters. on what kind of task do you expect it to match full fine-tuning, and on what kind not?

## prerequisites check

- [ ] you have a `Llama` that follows the causal-LM protocol (→ topic 02)
- [ ] you know how cross-entropy with `ignore_index=-100` works and why targets are pre-shifted (→ ch 18 · 04)
- [ ] you know what a matrix's rank is and why $BA$ with $B \in \mathbb{R}^{d \times r}$ has rank at most $r$ (→ ch 02)
- [ ] you can quantize a block of weights with an absmax scale (→ ch 26 · 06)

## core ideas

### idea 1: low-rank adaptation

fine-tuning changes a pretrained weight $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ to $W_0 + \Delta W$. Aghajanyan et al. (2021) observed that fine-tuning works even when the update is confined to a random subspace of small dimension: the change a task needs has low **intrinsic dimension**. LoRA builds that assumption into the parameterization:

$$h = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times d_\text{in}},\; B \in \mathbb{R}^{d_\text{out} \times r},\; r \ll \min(d_\text{in}, d_\text{out}).$$

$W_0$ is frozen; only $A$ and $B$ train: $r(d_\text{in} + d_\text{out})$ numbers instead of $d_\text{in} d_\text{out}$. $B$ starts at **zero**, so at step 0 the wrapped layer computes exactly what the pretrained one did and fine-tuning starts from the pretrained model, not from a perturbed one. $A$ starts random (here, `nn.Linear`'s default Kaiming-uniform), so that the gradient with respect to $B$, which is $\propto (\partial L / \partial h)(Ax)^\top$, isn't zero. if both started at zero, neither would ever move.

the factor $\alpha / r$ fixes the scale of the update: Hu et al. set $\alpha$ once and kept it as they varied $r$, so they didn't have to retune the learning rate for each rank. dropout, when used, is applied to the input of the low-rank path only: the frozen path stays exactly the pretrained function.

> **spec.** `LoRALinear(base, r=8, *, alpha=16.0, dropout=0.0)`: `ValueError` for `r < 1`; keeps `base` (an `nn.Linear`) with its parameters frozen; trainable `lora_A` of shape `(r, in_features)` (Kaiming-uniform with `a=sqrt(5)`) and `lora_B` of shape `(out_features, r)` (zeros), on base's device and dtype; attributes `r`, `alpha`, `scaling = alpha / r`, `dropout` (an `nn.Dropout(dropout)`). `forward(x)` = `base(x) + scaling * dropout(x) @ A.T @ B.T`. `merge()` returns a new `nn.Linear` with weight $W_0 + \text{scaling} \cdot BA$, base's bias, and base's `requires_grad`.

> **▶ build: stage 1.** implement `LoRALinear`.

### idea 2: wrapping a model

to apply LoRA you replace chosen `nn.Linear` submodules in place with `LoRALinear`s that wrap them, then freeze everything else: embeddings, norms, biases, the head. which layers? the original paper adapted only the attention query and value projections; later work (e.g. QLoRA) found that adapting every linear layer works better at the same parameter budget. the head is usually left out: it's tied to the embedding in many models, and adapting it changes the output vocabulary distribution directly. selecting layers by **substring of the module's dotted name** (`"q_proj"` matches `blocks.3.attn.q_proj`) is the convention of the PEFT library. two practical details: collect the names first and modify afterwards (changing a module tree while iterating over it skips or revisits modules), and don't wrap a layer that is already the `.base` of a `LoRALinear`.

> **spec.** `apply_lora(model, *, r=8, alpha=16.0, dropout=0.0, targets=None)`: replaces, in place, every `nn.Linear` whose dotted name contains any string in `targets` (with `targets=None`: every `nn.Linear` whose last name component isn't `lm_head`) by `LoRALinear(layer, r, alpha=alpha, dropout=dropout)`, skipping layers inside an existing `LoRALinear`; freezes every parameter except the LoRA factors (which stay trainable); returns `model`.

> **🔮 predict:** with rank 1 on every linear layer of a 3-layer Llama (5,184 trainable numbers, 1.6% of the model), how close will you get to full fine-tuning?

> **▶ build: stage 2.** implement `apply_lora`.

### idea 3: merging

after training, $W_0 + \frac{\alpha}{r} BA$ is just a matrix. **merging** replaces each `LoRALinear` by a plain `nn.Linear` with that weight: same outputs, the original parameter count, and no extra matmuls at inference (the two thin matmuls of the low-rank path are cheap in FLOPs but cost kernel launches and memory traffic). the other option is to keep adapters separate: one frozen base in memory and a small adapter per customer or task, swapped per request. a rank-8 adapter for every linear layer of a 7B model is about 20M parameters, 40 MB in 16 bits. merging needs eval-mode semantics (dropout off), and merged models can't be un-merged exactly in low precision, so keep the adapter.

> **spec.** `merge_lora(model)`: replaces every `LoRALinear` in `model` by its `merge()`, in place; returns `model`.

> **▶ build: stage 3.** implement `merge_lora`.

### idea 4: instruction data

an SFT example is a prompt and a response laid out in a fixed **chat template**, here

```
User: {prompt}
Assistant: {response}<|endoftext|>
```

real templates (ChatML, Llama's `[INST]`) add special tokens for roles; the idea is the same. three details matter more than they look.

**the loss is on the response only.** the model should learn to produce answers, not to produce questions: set the targets of prompt positions to $-100$ so cross-entropy ignores them. the prompt still shapes every prediction through attention; it just isn't a target.

**the response ends with the end-of-text token**, and that token is a target. it's how the model learns to stop. a model trained without it rambles on into a new "User:" turn, and if the token never appeared in pretraining, its embedding starts untrained, which is why the lab inserts it between speeches during pretraining.

**tokenize the two halves separately.** encode `"User: {prompt}\nAssistant:"` and `" {response}<|endoftext|>"` and concatenate: then the boundary between masked and unmasked tokens is exact (with byte-level BPE the space attaches to the following word, which is why it starts the second half). with pre-shifted targets, input position $t$ predicts token $t + 1$, so for a sequence of $H$ prompt tokens and $R$ response tokens, `input_ids` is the first $H + R - 1$ tokens and `targets` is $H - 1$ times $-100$ followed by the $R$ response tokens. batches are padded on the right with a pad token in the inputs and $-100$ in the targets, so padding never contributes to the loss.

> **spec.** `build_sft_example(tokenizer, prompt, response, *, max_len=None)`: `head = tokenizer.encode(f"User: {prompt}\nAssistant:")`, `tail = tokenizer.encode(f" {response}<|endoftext|>")`; returns `(input_ids, targets)` as lists: `(head + tail)[:-1]` and `[-100] * (len(head) - 1) + tail`, both truncated to `max_len`. `collate_sft(examples, *, pad_id=0)`: list of `(input_ids, targets)` → `(x, y)` int64 tensors `(B, T_max)`, right-padded with `pad_id` and $-100$. `sft_loss(logits, targets)`: mean cross-entropy over positions whose target isn't $-100$.

> **🔮 predict:** you fine-tune on the spelling tasks (capitalize a word, count its letters). which run damages the model's Shakespeare more: full fine-tuning or rank-1 LoRA?

> **▶ build: stage 4.** implement `build_sft_example`, `collate_sft` and `sft_loss`.

### idea 5: QLoRA's 4-bit NormalFloat

LoRA removes the optimizer memory of the frozen weights, but the weights themselves still take 2 bytes each. QLoRA stores them in 4 bits. trained weights are roughly normally distributed around zero, so a uniform grid of 16 levels wastes levels in the sparse tails and has too few near zero, where most weights live. **NormalFloat-4** (NF4) places the 16 levels at quantiles of a standard normal, so each level is used about equally often, with an exact zero, normalized to $[-1, 1]$:

$$\text{levels} = \text{sort}\Big(\Phi^{-1}(p_1^+), \dots, \Phi^{-1}(p_8^+),\; 0,\; -\Phi^{-1}(p_1^-), \dots, -\Phi^{-1}(p_7^-)\Big) \big/ \max|\cdot|,$$

with $p^+$ the first 8 of 9 points evenly spaced from $\delta = 0.9677083$ down to $0.5$, and $p^-$ the first 7 of 8 such points (the asymmetry, 8 positive and 7 negative levels, is what leaves room for an exact zero; $\delta$ is the offset bitsandbytes uses to avoid $\Phi^{-1}(1) = \infty$). each block of 64 consecutive weights stores one absmax scale; every weight is divided by it and rounded to the nearest level. that's 4 bits plus 32/64 bits of scale per weight (QLoRA's "double quantization" compresses the scales further).

during QLoRA training the 4-bit weights are dequantized on the fly for each matmul, the LoRA factors stay in 16 or 32 bits, and gradients flow *through* the frozen dequantized weights into $A$ and $B$. you simulate it here by replacing each weight with its NF4 rounding (fake quantization, chapter 26 · 06): numerically the same model, without the memory savings.

> **spec.** `nf4_levels()`: a float32 tensor of the 16 levels above, ascending. `nf4_quantize(w, *, block_size=64)`: `ValueError` if `w.numel()` isn't a multiple of `block_size`; splits the flattened tensor into blocks of `block_size` consecutive elements; each block's absmax $s$ (1 for an all-zero block); returns $s \times$ (the level nearest to $w/s$) with `w`'s shape and dtype.

> **▶ build: stage 5.** implement `nf4_levels` and `nf4_quantize`.

## worked example

**counting.** a $4096 \times 4096$ projection with $r = 8$: $8 \cdot (4096 + 4096) = 65{,}536$ trainable numbers against 16,777,216: 0.39%.

**a LoRA layer by hand.** $W_0 = I_2$, $r = 1$, $\alpha = 2$ (scaling 2), $A = (1, 2)$, $B = (0.5, -1)^\top$, $x = (1, 1)$. base: $(1, 1)$. low-rank path: $Ax = 3$, $B \cdot 3 = (1.5, -3)$, times 2: $(3, -6)$. output $(4, -5)$. merged: $W_0 + 2BA = \begin{pmatrix} 2 & 2 \\ -2 & -3 \end{pmatrix}$, and $W x = (4, -5)$: the same.

**an SFT example.** with a byte-level tokenizer, prompt "hi" and response "hello": the head `User: hi\nAssistant:` is 19 tokens, the tail ` hello<|endoftext|>` is 7 (six bytes and the special token). `input_ids` has 25 tokens; `targets` starts with 18 $-100$s, then the 7 tail tokens.

**an NF4 block.** $w = (0.3, -1.2, 0.05, 0.9)$, one block: absmax 1.2; $w/s = (0.25, -1, 0.042, 0.75)$; nearest levels $(0.2461, -1, 0.0796, 0.7230)$; dequantized $(0.295, -1.2, 0.096, 0.868)$. your functions must reproduce these.

## on paper

1. **why zero-initialize $B$?** show that with $B = 0$, the gradient of the loss with respect to $A$ is zero at step 0 while the gradient with respect to $B$ is not. what happens after the first step?

<details><summary>solution</summary>

with $h = W_0 x + s BAx$ and upstream gradient $g = \partial L / \partial h$: $\partial L / \partial B = s\, g (Ax)^\top$, nonzero since $A$ is random; $\partial L / \partial A = s\, B^\top g x^\top = 0$ since $B = 0$. after one step $B \neq 0$ and both factors train. this asymmetric start keeps the initial function equal to the pretrained one while still breaking the symmetry that would freeze both factors at zero.

</details>

2. **memory.** for a model with $N$ parameters fine-tuned with AdamW in mixed precision, estimate the memory for full fine-tuning (16 bytes per parameter) and for LoRA with $N_\text{lora} = 0.01N$ trainable parameters and a 16-bit frozen base. evaluate for $N = 7 \times 10^9$, ignoring activations. what does a 4-bit base change?

<details><summary>solution</summary>

full: $16N = 112$ GB. LoRA: frozen base $2N = 14$ GB, plus $16 N_\text{lora} = 0.16N$ = 1.1 GB for the adapters: about 15 GB. with a 4-bit base (4.5 bits including scales): $0.56N = 3.9$ GB plus 1.1 GB: about 5 GB, which is why QLoRA fits a 65B model on one 48 GB GPU.

</details>

3. **rank and expressiveness.** a LoRA update of rank $r$ on a $d \times d$ layer can represent any $\Delta W$ of rank $\le r$. how many parameters does a full update need, and for what $r$ does LoRA have as many parameters as full fine-tuning of that layer? can a rank-$r$ LoRA match an arbitrary full update with that many parameters?

<details><summary>solution</summary>

full: $d^2$. LoRA: $2rd$, equal at $r = d/2$. no: at $r = d/2$ the update still has rank at most $d/2$, a strict subset of all $d \times d$ matrices (and the parameterization is redundant: $BA = (BM)(M^{-1}A)$ for any invertible $M$). LoRA's bet is that the needed update is close to low rank, not that low-rank updates are universal.

</details>

## experiment

the lab pretrains a small Llama on tiny Shakespeare and fine-tunes it on two **synthetic** instruction tasks (there's no instruction dataset on this machine): finishing a line of the play from its first half, a format the base model can already almost do, and spelling tasks (capitalize a word, count its letters, name its first letter), a skill it doesn't have. it compares full fine-tuning with LoRA at ranks 1, 4 and 16 on held-out response loss, exact answers, and how much Shakespeare each run forgets. then it rounds the base to 4 bits with your NF4 and with a uniform grid, trains LoRA on the 4-bit base (QLoRA), and merges a LoRA model to check outputs, size and speed.

## check your understanding

1. why does LoRA scale the update by $\alpha / r$ instead of just letting $B$ learn the right size?

<details><summary>answer</summary>

with Adam, the size of each step is roughly the learning rate regardless of the gradient's scale, so the effective size of the update depends on the parameterization. fixing $\alpha$ and dividing by $r$ keeps the update's scale roughly comparable across ranks, so one learning rate works for several ranks (approximately; rsLoRA argues $\alpha/\sqrt{r}$ is better at large $r$).

</details>

2. after `apply_lora`, the model's output is exactly the pretrained model's. why is that the right starting point, and which line of your code guarantees it?

<details><summary>answer</summary>

fine-tuning should start from the pretrained function and move from there; a random perturbation would destroy some of what pretraining learned before the first gradient step. `lora_B` initialized to zeros makes the low-rank term vanish.

</details>

3. a learner forgets to mask the prompt. what happens to the model?

<details><summary>answer</summary>

it also learns to generate prompts: part of the gradient goes to modeling user questions (often the larger part, since prompts can be long), and the model may continue its answer with a new "User:" line. with the mask, all the capacity spent goes to answers.

</details>

4. why does the response end with `<|endoftext|>`, and why is that token a target?

<details><summary>answer</summary>

generation stops when the model emits it. it's a target so the model learns *when* to stop: without it, the model keeps generating after its answer.

</details>

5. you serve 100 customers, each with their own fine-tuned 7B model. compare storing full fine-tunes with storing LoRA adapters.

<details><summary>answer</summary>

full: 100 × 14 GB = 1.4 TB, and each request needs its model in GPU memory. LoRA: one 14 GB base plus 100 adapters of ~40 MB each (4 GB total); requests for different customers can even share a batch, with the adapter applied per sequence (as in S-LoRA and Punica).

</details>

6. why is NF4 better than a uniform 4-bit grid for weights, and when would it not be?

<details><summary>answer</summary>

weights are roughly normal, so quantiles of a normal put more levels where most weights are and fewer in the tails, lowering the average rounding error. for a distribution far from normal (heavy outliers, or uniform weights) the advantage shrinks or reverses, which is why outlier handling (smaller blocks, keeping outlier features in high precision) matters too.

</details>

## common bugs

- **both LoRA factors random** (or both zero): the first changes the model before training; the second never trains.
- **forgetting to freeze** the base, the embeddings or the norms: you're doing full fine-tuning with extra steps.
- **modifying modules while iterating over `named_modules()`**, or wrapping the `.base` of an existing `LoRALinear` when applying LoRA twice.
- **off-by-one masks**: masking $H$ targets instead of $H - 1$ drops the first response token from the loss; masking $H - 2$ trains on predicting the colon.
- **padding targets with the pad id instead of $-100$**: the model learns to predict padding.
- **an end-of-text token that's never trained**: if pretraining never saw it and its embedding is frozen, a LoRA model struggles to learn to stop.

## going deeper

- Hu et al. (2021), "LoRA: low-rank adaptation of large language models", *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Aghajanyan, Zettlemoyer & Gupta (2021), "Intrinsic dimensionality explains the effectiveness of language model fine-tuning", *ACL*. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
- Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), "QLoRA: efficient finetuning of quantized LLMs", *NeurIPS*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Biderman et al. (2024), "LoRA learns less and forgets less", *TMLR*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback" (InstructGPT; the SFT stage). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Wei et al. (2022), "Finetuned language models are zero-shot learners" (FLAN), *ICLR*. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
- Zhou et al. (2023), "LIMA: less is more for alignment", *NeurIPS*. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)

## connections

- ← builds on: your `Llama` (topic 02), cross-entropy with ignored targets (ch 18 · 04), block quantization (ch 26 · 06), the optimizer-memory accounting of ch 26 · 05.
- → used in: topic 04 starts preference learning from an SFT model (and uses LoRA for the policy), topic 06 evaluates fine-tuned models, the capstone's assistant is pretrained, then fine-tuned with your `build_sft_example`, and chapter 28 studies what fine-tuning changes inside a model.
