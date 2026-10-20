# efficient inference

## why this matters

training happens once; inference happens every time someone uses the model. for a deployed language model, the cost of serving it quickly overtakes the cost of training it, and the economics are set by two numbers: how many bytes the model occupies, and how many tokens per second a device can produce.

generating text is memory-bound (topic 01): each new token needs every weight read from memory once, and does only two FLOPs per weight with it. a 7B model in 16-bit reads 14 GB per token; at 100 GB/s that's 7 tokens per second, no matter how fast the arithmetic is. so the levers of inference are levers on bytes. store weights in **8 or 4 bits** instead of 16 and you read 2–4× fewer bytes per token. **prune** weights to zero and (with the right kernels) skip them. **distil** the model into a smaller one that imitates it. and **batch** many requests together, so one read of the weights serves many tokens.

this topic builds each: symmetric int8 quantization with per-channel scales (and k-bit fake quantization to see how far it goes), an int8 linear layer, magnitude pruning, the distillation loss, and the roofline model that says how much batching buys.

## warm-up: predict first

1. a weight matrix has one entry of 8.0 and the rest between −0.1 and 0.1. you quantize it to 4 bits with one scale for the whole matrix. what happens to the small weights? what if each row has its own scale?
2. you zero out the 50% smallest-magnitude weights of a trained MLP. how much does test accuracy drop? does the model run faster?
3. a student network trained to match a teacher's softened predictions beats the same student trained on the labels alone. where does the extra information come from?
4. a GPU generates 7 tokens per second for one user. how many tokens per second can it generate for 32 users at once?

## prerequisites check

- [ ] you can round to a format and reason about rounding error relative to the step size (→ topic 03)
- [ ] you know the roofline and why matrix–vector products are memory-bound (→ topic 01)
- [ ] you can compute softmax, cross-entropy and KL divergence, and know which direction KL is (→ ch 05 · 05)
- [ ] you can replace submodules of an `nn.Module` and use `register_buffer` (→ ch 15 · 07)

## core ideas

### idea 1: symmetric quantization

represent a float tensor $w$ by integers $q$ and a scale $s$: $w \approx q \cdot s$. for **symmetric** $b$-bit quantization the integers run from $-q_\text{max}$ to $q_\text{max}$ with $q_\text{max} = 2^{b-1} - 1$ (127 for int8; the code $-128$ is left unused so the range is symmetric), and

$$s = \frac{\max |w|}{q_\text{max}}, \qquad q = \text{clamp}\big(\text{round}(w / s),\, -q_\text{max},\, q_\text{max}\big).$$

the largest weight maps to $\pm q_\text{max}$, zero maps exactly to zero, and every weight is off by at most $s/2$. that bound is the whole story: the error is set by the **largest** value sharing the scale. one outlier among small weights makes $s$ large, and the small weights all round to 0.

so the choice of what shares a scale matters more than anything else. **per-tensor** quantization uses one $s$ for the whole matrix: cheapest, most fragile. **per-channel** uses one $s$ per output channel (a row of an `nn.Linear` weight of shape $(\text{out}, \text{in})$, i.e. reducing over all axes but `axis=0`): an outlier spoils only its own row, at the cost of one float per row. **per-group** uses one $s$ per group of, say, 32 or 128 consecutive weights within a row: the standard for 4-bit LLM weights (as in GPTQ and llama.cpp's formats), at the cost of one 16-bit scale per group (0.5 extra bits per weight for groups of 32). LLMs make this choice urgent: Dettmers et al. (2022) found that from about 6.7B parameters, a few hidden dimensions carry activations 20× larger than the rest, which per-tensor schemes can't survive.

**fake quantization** (quantize then immediately dequantize, in float) measures the effect of any bit width on accuracy without needing integer kernels, which is how you choose a scheme before building one.

> **spec.** `quantize_int8(w, *, axis=0)`: `amax` = max of $|w|$ over every dimension except `axis` (keepdim); `scale = amax / 127` as float32, with scale 1 where `amax == 0`; `q = clamp(round(w / scale), -127, 127)` as int8 (`torch.round`, halves to even). returns `(q, scale)` with `scale` broadcastable against `w`. `dequantize_int8(q, scale)` = `q.float() * scale`. `fake_quantize(w, bits, *, axis=0, group_size=None)`: the same symmetric rounding with $q_\text{max} = 2^{b-1} - 1$, returned dequantized; scales per slice along `axis`, per tensor if `axis is None`, or, if `group_size` is given, per group of `group_size` consecutive elements of the flattened tensor (groups never straddle a row when the last dimension is a multiple of `group_size`).

> **▶ build: stage 1.** implement `quantize_int8`, `dequantize_int8` and `fake_quantize`.

### idea 2: an int8 layer

an `Int8Linear` stores what a quantized layer needs, as **buffers** (moved by `.to(device)` and saved in the `state_dict`, but not trained): the int8 weight $(\text{out}, \text{in})$, a float scale per output channel $(\text{out}, 1)$, and the bias. its forward pass dequantizes the weight and runs an ordinary linear layer. that saves memory (a quarter of float32's weight bytes) but not time here: real speedups need kernels that read the int8 weights and dequantize *inside* the matmul (topic 02's fusion), so the float weights never exist in DRAM; for memory-bound decoding that makes the speedup nearly the byte ratio. weight-only quantization like this is the simplest scheme; quantizing activations too (W8A8) enables integer matmul units but needs care with activation outliers (Dettmers et al., 2022; Xiao et al., 2023).

converting a model means walking its module tree and replacing every `nn.Linear` by `Int8Linear.from_linear(layer)`. some layers are often kept in float: the final classifier or LM head, whose errors go straight into the logits, and the first layer. so the converter takes a list of qualified module names to skip, as `named_modules()` names them (`"blocks.0.mlp.fc1"`, or `"2.0"` for the first child of the third child of an `nn.Sequential`).

> **spec.** `Int8Linear(weight_q, scale, bias=None)` registers buffers `weight_q` (int8), `scale` (float32) and `bias` (or `None`). `Int8Linear.from_linear(linear)` is a classmethod: `quantize_int8(linear.weight, axis=0)` and the bias. `forward(x)` = `F.linear(x, dequantize_int8(weight_q, scale).to(x.dtype), bias)`. `quantize_linears(model, *, skip=())` returns a deep copy of `model` with every `nn.Linear` whose qualified name isn't in `skip` replaced by an `Int8Linear`, leaving `model` untouched.

> **🔮 predict:** you quantize the three linear layers of a trained MNIST MLP to int8, per channel. how many accuracy points do you lose? and at 4 bits per tensor?

> **▶ build: stage 2.** implement `Int8Linear` and `quantize_linears`.

### idea 3: pruning

many trained weights are close to zero and contribute little. **magnitude pruning** sets the fraction $\rho$ of weights with the smallest $|w|$ to zero (LeCun et al., 1990, used second derivatives; Han et al., 2015, showed plain magnitude works well with retraining). two ways to choose the threshold. **per-layer**: each layer loses the same fraction. **global**: one threshold for all layers together, so layers with many small weights (typically the big ones) lose more and small, sensitive layers less. global pruning usually keeps more accuracy at the same overall sparsity, but can empty a layer entirely, which disconnects the network.

a pruned model is only smaller and faster if something exploits the zeros. a dense matmul multiplies by zero at full price; unstructured sparsity needs sparse formats and kernels that only pay off at very high sparsity on most hardware. **structured** pruning (removing whole neurons, heads or layers) and semi-structured patterns (NVIDIA's 2:4 sparsity: two zeros in every four weights, supported by sparse tensor cores) trade some accuracy for real speed. the lab shows the gap between "zeros" and "faster" directly. pruned networks are normally fine-tuned afterwards, re-applying the mask after every update so pruned weights stay zero, which is why the function returns the masks.

> **spec.** `magnitude_prune(model, sparsity, *, scope='global')`, in place and without gradients: over the weights of every `nn.Linear` (biases untouched), finds the $k = \text{round}(\rho n)$-th smallest $|w|$ (globally across layers, or within each layer for `scope='layer'`) and zeroes every weight with $|w|$ at or below it (nothing for $k = 0$). returns `{module name: bool keep-mask}`. `ValueError` for another scope. (with ties at the threshold, "at or below" removes every tied weight, so the sparsity can come out slightly above $\rho$; with trained float weights, exact ties are rare.)

> **▶ build: stage 3.** implement `magnitude_prune`.

### idea 4: distillation

a trained classifier's output says more than its top class. for a handwritten 7, a good teacher might give "1" a probability of 0.01 and "6" 0.0001: the relative sizes of the wrong answers encode similarities between classes that a one-hot label throws away. Hinton, Vinyals & Dean (2015) proposed training a small **student** to match the large **teacher**'s output distribution, softened with a **temperature** $T$: $p_i = \text{softmax}(z / T)_i$. at $T = 1$ the wrong classes' probabilities are tiny; at $T = 2$–$4$ they become visible and carry gradient.

the loss mixes the soft targets with the true labels:

$$\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\big(\text{softmax}(z_t/T) \,\big\|\, \text{softmax}(z_s/T)\big) + (1 - \alpha)\, \mathrm{CE}(z_s, y).$$

the KL is the teacher's distribution against the student's (the student is pulled towards covering everything the teacher believes), averaged over the batch. the $T^2$ keeps the two terms balanced: the gradient of the soft term with respect to the student's logits scales as $1/T^2$ (on paper 2), so without the factor, raising the temperature would quietly turn the soft term off. the teacher is fixed: its logits are constants, and no gradient flows into it. distillation is how many production models are made: DistilBERT keeps 97% of BERT's quality with 40% fewer parameters (Sanh et al., 2019), and small LLMs are routinely trained on larger models' outputs.

> **spec.** `distillation_loss(student_logits, teacher_logits, targets, *, temperature=2.0, alpha=0.5)`: $\alpha T^2\,\mathrm{KL}(\text{softmax}(t/T) \| \text{softmax}(s/T))$ (summed over classes, averaged over the batch: `F.kl_div(..., reduction='batchmean')`) $+ (1 - \alpha)\,\mathrm{CE}(s, \text{targets})$, with the teacher's logits detached.

> **▶ build: stage 4.** implement `distillation_loss`.

### idea 5: batching

back to the roofline. one decoding step of an $N$-parameter model at batch size $B$ reads all the weights once ($N \cdot$ bytes per parameter) and does $2NB$ FLOPs (ignoring attention over the KV cache). its time is at least the larger of the memory time and the compute time, so

$$\text{tokens/s} = \frac{B}{\max\left(\frac{N \cdot \text{bytes}}{W},\; \frac{2NB}{P}\right)}.$$

while the memory term dominates, the step time doesn't depend on $B$ at all: serving 32 users costs the same per step as serving one, and throughput grows linearly with the batch. only when $B$ exceeds about $P \cdot \text{bytes} / (2W)$, the ridge point in batch units, does arithmetic take over. that's why inference servers batch aggressively, why **continuous batching** (Yu et al., 2022; vLLM, Kwon et al., 2023) adds and removes requests from a running batch every step rather than waiting for a whole batch to finish, and why the KV cache (chapter 18), which grows with batch and context, is what limits the batch in practice. quantization helps both ways: fewer bytes per weight speeds up the memory-bound regime and leaves more memory for KV caches.

> **spec.** `decode_tokens_per_second(n_params, batch_size, *, bytes_per_param=2, bandwidth, peak_flops)` = $B / \max(N \cdot \text{bytes} / W,\; 2NB / P)$.

> **🔮 predict:** with your GPT from chapter 18 on your laptop, will tokens per second really grow linearly from batch 1 to batch 32?

> **▶ build: stage 5.** implement `decode_tokens_per_second`.

## worked example

**quantizing a row.** $w = (0.5, -1.27, 0.0)$: $s = 1.27/127 = 0.01$, $q = (50, -127, 0)$, exact. $w = (2.54, 1.0, -0.01)$: $s = 0.02$, $q = (127, 50, \text{round}(-0.5) = 0)$, so $-0.01$ is lost: its error, 0.01, is exactly half a step. quantized with one scale for both rows ($s = 0.02$), the first row's 0.5 would become $q = 25$, still exact, but anything below 0.01 in it would vanish.

**an outlier at 3 bits.** the row $(8.0, 0.5, 0.25, -4.0)$ at 3 bits ($q_\text{max} = 3$): $s = 8/3$, and $0.5$ and $0.25$ round to 0, $-4.0$ to $-2s = -5.33$ ($-1.5$ rounds to even). with groups of 2: $(8.0, 0.5)$ has $s = 8/3$ and loses 0.5; $(0.25, -4.0)$ has $s = 4/3$ and keeps $-4.0$ exactly.

**decoding a 7B model** at $W = 100$ GB/s, $P = 10$ TFLOP/s, 16-bit: batch 1 gives $1/0.14 = 7.1$ tokens/s; batch 32, 229 tokens/s (still memory-bound); compute-bound beyond $B = P \cdot 2 / (2W) = 100$, at $P/(2N) = 714$ tokens/s. int8 weights double the batch-1 speed to 14.3. your functions must reproduce these numbers.

## on paper

1. **the quantization error.** for $b$-bit symmetric quantization of weights uniform on $[-a, a]$ with per-tensor scale, the rounding error is approximately uniform on $[-s/2, s/2]$. show that its variance is $s^2/12$ and that each extra bit divides the error's standard deviation by 2 (about 6 dB of signal-to-noise per bit).

<details><summary>solution</summary>

a uniform variable on $[-s/2, s/2]$ has variance $\frac{1}{s}\int_{-s/2}^{s/2} e^2\, de = s^2/12$. with $s = a / (2^{b-1} - 1) \approx a/2^{b-1}$, one more bit halves $s$, so it halves the standard deviation $s/\sqrt{12}$ and quarters the variance: $10\log_{10} 4 \approx 6$ dB per bit.

</details>

2. **why $T^2$.** let $p = \text{softmax}(z_t/T)$, $q = \text{softmax}(z_s/T)$. show that $\partial\, \mathrm{KL}(p \| q) / \partial z_{s,i} = (q_i - p_i)/T$, and that for large $T$ (logits small relative to $T$) this is approximately $(z_{s,i} - z_{t,i})/(KT^2)$ for $K$ classes with zero-mean logits. conclude why the loss is multiplied by $T^2$.

<details><summary>solution</summary>

$\mathrm{KL}(p\|q) = \sum p_i \log p_i - \sum p_i \log q_i$; with $\log q_i = z_{s,i}/T - \log\sum_j e^{z_{s,j}/T}$, the derivative is $-p_i/T + q_i/T$. for $|z|/T$ small, $q_i \approx (1 + z_{s,i}/T)/K$ (zero-mean logits), the same for $p_i$, so $q_i - p_i \approx (z_{s,i} - z_{t,i})/(KT)$, and the gradient is $\approx (z_{s,i} - z_{t,i})/(KT^2)$: it shrinks like $1/T^2$. multiplying by $T^2$ keeps the soft term's gradients the same size as you change $T$ (Hinton et al., 2015). at high temperature distillation becomes matching logits (up to a constant), a regression on $z_t$.

</details>

3. **when batching stops being free.** for a model in 16-bit on a device with ridge point $I^*$ FLOP/byte, at what batch size does decoding become compute-bound? evaluate for $I^* = 30$ (a laptop GPU) and $I^* = 150$ (an A100).

<details><summary>solution</summary>

compute time equals memory time when $2NB/P = 2N/W$, i.e. $B = P/W = I^*$ with 2 bytes per parameter (in general $B = I^* \cdot \text{bytes}/2$). 30 sequences on the laptop, 150 on the A100: below that, extra sequences in the batch are free (until the KV cache fills the memory).

</details>

## experiment

the lab trains a small MLP on MNIST (a synthetic stand-in offline) and measures accuracy against bits for per-tensor, per-channel and per-group quantization; converts it with your `quantize_linears` and checks memory and speed. it prunes the MLP globally and per layer and measures accuracy and time against sparsity. it distils a large MLP into a small one trained on a few hundred labels, over several seeds. last, it measures your chapter 18 GPT's decoding throughput against batch size and compares with your roofline model.

## check your understanding

1. why does one large weight hurt per-tensor quantization of all the others?

<details><summary>answer</summary>

the scale is set by the largest $|w|$, and the step between levels is $s = \max|w| / q_\text{max}$. every other weight is rounded to a multiple of that step, so weights much smaller than $s/2$ become zero.

</details>

2. your int8 model uses a quarter of the memory but runs no faster than the float one. why, and what would fix it?

<details><summary>answer</summary>

the forward pass dequantizes to a full float weight in memory and then runs a float matmul: the bytes read per token are those of the float weights again, plus the conversion. a fused kernel that reads int8 and dequantizes in registers inside the matmul reads a quarter of the bytes, which for memory-bound decoding is nearly 4× faster than float32.

</details>

3. global pruning removed a whole layer. why can that happen, and why is it bad?

<details><summary>answer</summary>

if one layer's weights are all smaller than the global threshold (a layer with small-scale weights, say after normalization), all of them go. a layer of zeros outputs only its bias, cutting the information path through the network; accuracy collapses. guard against it with per-layer minimum density or per-layer pruning.

</details>

4. in the distillation loss, why KL(teacher ‖ student) rather than the reverse?

<details><summary>answer</summary>

the forward KL $\sum p_t \log(p_t/p_s)$ penalizes the student for putting low probability where the teacher puts mass, so the student learns to cover the teacher's whole distribution, including its informative small probabilities. it's also cross-entropy with soft targets up to a constant, the natural generalization of the hard-label loss.

</details>

5. why do inference servers batch requests, and what eventually limits the batch size?

<details><summary>answer</summary>

decoding is memory-bound: one read of the weights can serve many sequences at almost no extra cost, so throughput grows nearly linearly with the batch until the compute roof. in practice the KV caches (one per sequence, growing with context) fill the memory first.

</details>

6. what does fake quantization measure, and what doesn't it?

<details><summary>answer</summary>

it measures the accuracy effect of the rounding exactly (the forward pass sees the same values a real quantized model would). it says nothing about speed or memory, which depend on kernels that operate on the integers.

</details>

## common bugs

- **a zero scale for an all-zero channel.** $0/0$ gives NaN weights; give such channels scale 1.
- **reducing over the wrong axis.** an `nn.Linear` weight is $(\text{out}, \text{in})$; per-output-channel scales reduce over `in` (`axis=0` keeps the outputs).
- **storing quantized weights as parameters.** the optimizer or `.half()` then touches them; register buffers.
- **forgetting the $T^2$**, or computing KL in the wrong direction or with `reduction='mean'` (which divides by the number of classes too).
- **letting gradients flow into the teacher.** detach its logits; otherwise distillation trains the teacher towards the student.
- **pruning, then fine-tuning without re-applying the masks.** the optimizer revives pruned weights within a few steps.

## going deeper

- Jacob et al. (2018), "Quantization and training of neural networks for efficient integer-arithmetic-only inference", *CVPR*. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
- Dettmers, Lewis, Belkada & Zettlemoyer (2022), "LLM.int8(): 8-bit matrix multiplication for transformers at scale". [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Frantar, Ashkboos, Hoefler & Alistarh (2023), "GPTQ: accurate post-training quantization for generative pre-trained transformers", *ICLR*. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Han, Pool, Tran & Dally (2015), "Learning both weights and connections for efficient neural networks", *NeurIPS*. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Hinton, Vinyals & Dean (2015), "Distilling the knowledge in a neural network". [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Kwon et al. (2023), "Efficient memory management for large language model serving with PagedAttention", *SOSP* (vLLM, continuous batching). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Xiao et al. (2023), "SmoothQuant: accurate and efficient post-training quantization for large language models", *ICML*. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

## connections

- ← builds on: rounding and formats (03), the roofline (01), the KV cache and decoding (ch 18 · 06), cross-entropy and KL (ch 05).
- → used in: chapter 27 quantizes and distils its Llama, and its speculative decoding is another way around memory-bound generation; chapter 28's model compression connects to robustness.
