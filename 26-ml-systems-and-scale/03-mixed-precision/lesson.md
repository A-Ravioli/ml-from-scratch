# mixed precision

## why this matters

every number in your GPT so far has been a 32-bit float. halve that, and every memory-bound op from topic 01 moves half the bytes, every activation saved for the backward pass takes half the memory, and on GPUs with matrix units (NVIDIA's tensor cores) the matmuls run 8–16× faster than in float32. that's why essentially every large model since 2018 trains in 16-bit arithmetic.

the catch is that 16 bits can't hold what training needs. float16 overflows at 65,504 and rounds anything below about $6 \times 10^{-8}$ to zero, and gradients of a deep network routinely live down there. bfloat16 has the range but only 8 bits of precision, so adding a small update to a weight can do literally nothing. run training naively in 16-bit and it either diverges in a shower of NaNs or quietly stops learning.

**mixed precision** is the recipe that makes it work (Micikevicius et al., 2018): compute in 16-bit, keep a 32-bit master copy of the weights, accumulate sums in 32-bit, and scale the loss so gradients stay inside float16's range. this topic builds each piece: a simulator for any float format (to *see* the rounding), the master-weights argument, a dynamic loss scaler (the `GradScaler` that joins `mlfs.systems`), and a mixed-precision training step.

## warm-up: predict first

1. float16 has 5 exponent bits and 10 mantissa bits; bfloat16 has 8 and 7. which can represent $10^{-6}$? which can represent 1.001?
2. a weight equals 1.0 in bfloat16 and every step adds $10^{-3}$ to it. after 100 steps, what is the weight?
3. multiplying the loss by 65,536 multiplies every gradient by 65,536. why doesn't that change what the optimizer does?
4. on a laptop GPU without tensor cores, will float16 matmuls be faster than float32?

## prerequisites check

- [ ] you know floating point as sign × mantissa × 2^exponent, and what machine epsilon and overflow mean (→ ch 00 · 03)
- [ ] you can write a training step with `zero_grad`, `backward`, `clip_grad_norm_`, `step` (→ ch 15 · 07)
- [ ] you know which ops of a network are matmuls and which are reductions (→ topic 01)
- [ ] you know AdamW's update and its moment estimates (→ ch 06, ch 15 · 04)

## core ideas

### idea 1: what a float format can hold

a binary floating-point number is $(-1)^s \times 1.f \times 2^{e - \text{bias}}$: a sign bit, $E$ **exponent** bits (with bias $2^{E-1} - 1$) and $M$ **mantissa** bits $f$. the exponent sets the **range**, the mantissa the **precision**. in IEEE-style formats the all-ones exponent is reserved for $\pm\infty$ and NaN, so the largest finite value is

$$\text{max} = (2 - 2^{-M}) \cdot 2^{\text{bias}}.$$

the smallest *normal* value is $2^{1 - \text{bias}}$. below it, the exponent field is 0 and the leading 1 becomes a 0: **subnormal** numbers fill the gap to zero in even steps of $2^{1 - \text{bias} - M}$, the smallest positive value. between 1 and 2 the spacing is $\varepsilon = 2^{-M}$, so a number's relative rounding error is at most $\varepsilon / 2$. rounding is to the nearest representable value, with exact ties going to the one whose last mantissa bit is 0 (**round half to even**); values that round past the max become $\infty$.

| format | $E$, $M$ | max | smallest normal | smallest subnormal | $\varepsilon$ |
|---|---|---|---|---|---|
| float32 | 8, 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | $1.4 \times 10^{-45}$ | $1.2 \times 10^{-7}$ |
| float16 | 5, 10 | 65,504 | $6.1 \times 10^{-5}$ | $6.0 \times 10^{-8}$ | $9.8 \times 10^{-4}$ |
| bfloat16 | 8, 7 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | $9.2 \times 10^{-41}$ | $7.8 \times 10^{-3}$ |
| float8 E5M2 | 5, 2 | 57,344 | $6.1 \times 10^{-5}$ | $1.5 \times 10^{-5}$ | 0.25 |

float16 trades range for precision; **bfloat16** ("brain float", from Google's TPUs) is float32 with the mantissa cut to 7 bits: the same range, so no overflow or underflow problems, at a quarter of float16's precision. the 8-bit formats of recent GPUs come in two flavours: E5M2 (range, for gradients) and E4M3 (precision, for weights and activations; the common "E4M3FN" variant drops infinities to reach a max of 448). with only 2–3 mantissa bits, 8-bit training needs a scale factor per tensor, a generalization of the loss scaling you'll build below.

you can simulate any of these formats in float64 NumPy. for $|x|$ with leading bit $2^{e}$ (from `np.frexp`), the spacing of the format there is $q = 2^{\max(e,\, 1 - \text{bias}) - M}$ (the `max` makes the spacing stop shrinking in the subnormal range), and the rounded value is $q \cdot \text{round}(|x| / q)$. NumPy's `np.round` already rounds halves to even, and division by a power of two is exact in float64, so the result is exact.

> **spec.** `float_format(exp_bits, man_bits)` returns `{"max", "min_normal", "min_subnormal", "eps"}` for the IEEE-style format (top exponent reserved). `round_to_format(x, exp_bits, man_bits)`: float64 array of `x`'s shape, each element rounded to the nearest representable value (ties to even, subnormals included), $\pm\infty$ past the max, sign preserved (including $-0$), NaN and $\pm\infty$ passed through. for `(5, 10)`, `(8, 7)` and `(5, 2)` it must agree bit for bit with PyTorch's `float16`, `bfloat16` and `float8_e5m2` casts.

> **▶ build: stage 1.** implement `float_format` and `round_to_format`.

### idea 2: master weights

a training step changes each weight by a tiny fraction: with a learning rate of $10^{-3}$ and gradient-like quantities of order 1, AdamW moves a weight of size 1 by about $10^{-3}$. in bfloat16 the neighbours of 1.0 are $1 - 2^{-8}$ and $1 + 2^{-7}$, so $1 + 10^{-3}$ rounds straight back to 1.0. the update is lost, not slightly wrong; do it a hundred times and the weight still hasn't moved. float16 has the same problem a few times further down (its spacing at 1 is $2^{-10} \approx 10^{-3}$): updates smaller than half a spacing vanish, and slightly larger ones are rounded to multiples of it.

the fix is to keep a **master copy** of the weights in float32 and apply the updates there, where their effect is retained, and make a 16-bit copy of the result for the next forward and backward pass. the optimizer's moments live in float32 too. this is why mixed precision doesn't save memory on the weights and optimizer state: a 16-bit copy and 16-bit gradients plus float32 master weights and two float32 AdamW moments come to $2 + 2 + 4 + 4 + 4 = 16$ bytes per parameter, exactly as many as plain float32 training ($4 + 4 + 8$). it saves on activations, which dominate at large batch sizes, and on time (topic 05 counts this exactly).

in PyTorch you rarely make the 16-bit copy yourself: the parameters stay float32 (they *are* the master copy) and `torch.autocast` casts them to 16-bit on the fly as each matmul uses them.

> **spec.** `low_precision_sgd(w0, grads, lr, exp_bits, man_bits, *, master=False)`: applies $w \leftarrow w - \text{lr} \cdot g$ for each `g` in `grads`, every `g` first rounded to the format. with `master=False` the weights are stored in the format (rounded at the start and after every update); with `master=True` they're kept in float64 and only the returned value is rounded. returns the final weights, rounded to the format.

> **🔮 predict:** with `master=False` in bfloat16, how large must `lr * g` be for an update of a weight equal to 1.0 to have any effect at all? and in float16?

> **▶ build: stage 2.** implement `low_precision_sgd`.

### idea 3: loss scaling

float16's other problem is underflow. gradients of a deep network span many orders of magnitude, and a large fraction can be smaller than $6 \times 10^{-8}$ (Micikevicius et al. found most of a detector's activation gradients below float16's normal range). in a float16 backward pass these become zero, and whatever they would have taught the network is lost.

the trick: multiply the loss by a constant $S$ before `backward()`. by the chain rule every gradient is multiplied by $S$ too, which shifts the whole distribution up into float16's range. before the optimizer step, divide the gradients (now in float32, on the master weights) by $S$ again: the update is exactly the same as without scaling, except that the small gradients survived. choosing $S$ by hand is fragile, so **dynamic loss scaling** chooses it automatically:

1. start large ($S = 2^{16}$).
2. after the backward pass, unscale the gradients and check them. if any is $\infty$ or NaN, $S$ was too large and something overflowed: **skip this step** (don't touch the weights or the optimizer state) and multiply $S$ by a backoff factor (0.5).
3. if the gradients are finite, step, and count a clean step. after `growth_interval` (2,000) clean steps in a row, multiply $S$ by a growth factor (2) to probe whether a larger scale would fit.

skipped steps are rare in practice (one per few thousand), so they cost almost nothing, and the scale settles near the largest value that doesn't overflow. gradient clipping must happen *between* unscaling and stepping: clipping scaled gradients at a norm of 1 would clip true gradients at $1/S$. bfloat16 has float32's exponent range, so it needs no loss scaling at all; that's its main attraction.

> **spec.** `GradScaler(init_scale=65536.0, *, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)`. `get_scale()` returns the current scale (float). `scale(loss)` returns `loss.float() * scale`. `unscale_(optimizer)` divides every non-`None` `.grad` in `optimizer.param_groups` by the scale in place and records whether any is inf/NaN; calling it twice before `update()` raises `RuntimeError`. `step(optimizer)` calls `unscale_` if it hasn't been called since the last `update()`, then returns `False` without calling `optimizer.step()` if an inf/NaN was found, else calls `optimizer.step()` and returns `True`. `update()` multiplies the scale by `backoff_factor` and resets the clean-step count after an overflow; otherwise counts a clean step and, when the count reaches `growth_interval`, multiplies by `growth_factor` and resets it.

> **▶ build: stage 3.** implement `GradScaler`.

### idea 4: autocast and the mixed-precision step

which ops should run in 16-bit? matmuls, where the speed is: their many products are accumulated in float32 inside the hardware, so only the inputs and output are rounded. which shouldn't? reductions over many elements (sums, means, softmax normalizers, layer-norm statistics) and losses (log of a probability near 0), whose results lose too much in 16-bit. `torch.autocast(device_type, dtype)` implements exactly this policy per op: inside the context, `nn.Linear` and `matmul` cast their inputs to `dtype`, while `softmax`, `log_softmax`, `layer_norm`, losses and the like run in float32. the parameters themselves stay float32, and the backward pass runs each op in the dtype its forward used.

a mixed-precision training step is then:

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type=x.device.type, dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()        # scaled gradients, flowing through float16 activations
scaler.unscale_(optimizer)           # true gradients, on the float32 parameters
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)               # skipped if anything overflowed
scaler.update()
```

whether this is faster depends on the hardware. NVIDIA tensor cores run 16-bit matmuls many times faster than float32. Apple's M1 and M2 GPUs have no such units, and their float16 arithmetic peak is close to float32's, so the gain there comes only from moving fewer bytes in memory-bound ops, minus the cost of the casts; bfloat16 arithmetic isn't native on those chips at all. the lab measures what *your* machine does, and the capstone decides with measurements whether fp16 earns its place.

> **spec.** `amp_step(model, loss_fn, optimizer, x, y, *, dtype=torch.float16, scaler=None, grad_clip=None)`: zeroes the gradients; computes `loss_fn(model(x), y)` under `torch.autocast(x.device.type, dtype)`; with a `scaler`, backpropagates `scaler.scale(loss)`, calls `scaler.unscale_`, clips the gradient norm to `grad_clip` if given, `scaler.step` and `scaler.update`; without one, backpropagates the loss, clips and steps. returns `(loss as a Python float, whether the optimizer stepped)`.

> **🔮 predict:** in the check for stage 4, a loss is multiplied by $10^{-7}$ and trained one step in float16 without a scaler. will the first layer's weights change at all? with bfloat16?

> **▶ build: stage 4.** implement `amp_step`.

## worked example

**rounding 0.1.** it lies in $[2^{-4}, 2^{-3})$. in float16 ($M = 10$) the spacing there is $2^{-14}$: $0.1 / 2^{-14} = 1638.4$, which rounds to 1638, giving $1638 \cdot 2^{-14} = 0.0999755859375$ (relative error $2.4 \times 10^{-4}$, under $\varepsilon/2 = 4.9 \times 10^{-4}$). in bfloat16 ($M = 7$) the spacing is $2^{-11}$: $0.1 \cdot 2048 = 204.8 \to 205$, giving $0.10009765625$ (relative error $9.8 \times 10^{-4}$).

**a vanishing gradient.** a gradient of $10^{-8}$ in float16: the smallest subnormal is $2^{-24} = 5.96 \times 10^{-8}$, and $10^{-8}$ is less than half of it, so it rounds to 0. scaled by $2^{16}$, it's $6.55 \times 10^{-4}$, comfortably normal; unscaled again in float32, it's $10^{-8}$ to float32's precision.

**a vanishing update.** $1 + 10^{-3}$ in bfloat16: the spacing just above 1 is $2^{-7} = 0.0078$, and $10^{-3}$ is less than half of it, so the sum rounds to 1. a hundred such updates leave the weight at 1.0; a float64 master copy reaches 1.1, stored as 1.1015625 in bfloat16. your `round_to_format` and `low_precision_sgd` must reproduce all three.

## on paper

1. **the format constants.** derive float16's max, smallest normal and smallest subnormal from $E = 5$, $M = 10$, and check them against the table. why is float8 E5M2's range almost the same as float16's?

<details><summary>solution</summary>

bias $= 2^4 - 1 = 15$. max $= (2 - 2^{-10}) \cdot 2^{15} = 65{,}536 - 32 = 65{,}504$. smallest normal $2^{1-15} = 2^{-14} \approx 6.1 \times 10^{-5}$. smallest subnormal $2^{-14-10} = 2^{-24} \approx 6.0 \times 10^{-8}$. E5M2 has the same 5 exponent bits, so the same bias and nearly the same range (max $(2 - 2^{-2}) \cdot 2^{15} = 57{,}344$); it's float16 with the mantissa cut to 2 bits.

</details>

2. **why scaling doesn't change the update.** let $L$ be the loss and $S$ a power of two. show that in exact arithmetic, the update computed from $\nabla(SL)/S$ equals the one from $\nabla L$. why does it matter that $S$ is a power of two?

<details><summary>solution</summary>

$\nabla(SL) = S \nabla L$ by linearity, so $\nabla(SL)/S = \nabla L$. multiplying and dividing by a power of two only changes the exponent of a float, never its mantissa, so as long as nothing overflows or underflows the round trip is exact even in floating point. with an arbitrary $S$, each multiply and divide would add a rounding error.

</details>

3. **the cost of a skipped step.** with growth every 2,000 clean steps and backoff by 2 on overflow, a run settles into a cycle: the scale grows, overflows once, backs off. what fraction of steps is skipped?

<details><summary>solution</summary>

once the scale sits just below the overflow threshold, each growth overshoots by a factor of 2, the next step overflows and backs off by 2, and the cycle repeats: one skipped step per 2,001 steps, about 0.05%. the cost is negligible, which is why the defaults grow so slowly.

</details>

## experiment

the lab looks at the gradients of your chapter 18 GPT: what fraction would underflow in float16, with and without a loss scale, measured with your `round_to_format`. then it trains the GPT on the CPU three ways, float32, float16 with your `GradScaler` started at a deliberately oversized scale, and bfloat16, and plots the scale as it backs off, step by step: fp16 overflow, seen live. last, it measures matmul throughput in each dtype on your CPU and GPU.

## check your understanding

1. why is bfloat16 preferred for training when the hardware supports it, even though it's less precise than float16?

<details><summary>answer</summary>

it has float32's exponent range, so gradients don't underflow and activations don't overflow: no loss scaling, no skipped steps, no tuning. the missing precision matters little for the inputs of a matmul whose products are accumulated in float32, and the master weights handle the update precision.

</details>

2. a colleague keeps the weights in float16 and drops the master copy to save memory. what goes wrong, and when?

<details><summary>answer</summary>

updates smaller than half the float16 spacing at each weight round away, so learning stalls for weights whose updates are small relative to their size, typically late in training when the learning rate decays. the loss curve flattens early for no visible reason.

</details>

3. why must `GradScaler.step` skip `optimizer.step()` entirely instead of just zeroing the bad gradients?

<details><summary>answer</summary>

an infinite gradient means the scale was too large, not that the gradient is meaningful; zeroing it would still step the other parameters with a partially valid gradient and, worse, update Adam's moment estimates (and step count) with garbage. skipping leaves the parameters and the optimizer state exactly as if the batch never happened.

</details>

4. why does autocast keep the softmax and the loss in float32?

<details><summary>answer</summary>

they reduce over many elements and take exponentials and logs. a float16 sum of thousands of terms loses most of its precision (and can overflow), and $\log p$ for small probabilities needs range. they're memory-bound and cheap, so running them in float32 costs little.

</details>

5. you clip gradients at norm 1.0 before calling `scaler.unscale_`. what happens?

<details><summary>answer</summary>

the gradients are still multiplied by $S \approx 65{,}536$, so their norm is huge and every step clips them to norm 1, which after unscaling is a true norm of $1/S$: training barely moves. always unscale first.

</details>

6. what does the dynamic scale tell you when it keeps falling over the course of training?

<details><summary>answer</summary>

overflows keep happening: some gradients or activations are growing towards float16's max, often a sign of instability (exploding logits, attention scores, or a learning rate that's too high). a healthy run's scale stays roughly flat.

</details>

## common bugs

- **clipping before unscaling.** the clip threshold is effectively divided by the scale; training crawls.
- **calling `optimizer.step()` directly with a scaler.** it applies scaled (65,536× too large) gradients, or steps on infinities.
- **forgetting `scaler.update()`.** the scale never adapts: a too-large scale skips every step forever.
- **computing the loss outside autocast but the model inside it.** fine for the model, but a float16 loss computed by hand (say, a custom log) can overflow; keep losses in float32.
- **casting the model with `.half()` instead of using autocast.** that removes the master weights and puts normalizations and losses in float16 too.
- **comparing float16 and float32 losses bit for bit.** mixed precision changes the numbers slightly every step; compare loss *curves* and final metrics, with seeds.

## going deeper

- Micikevicius et al. (2018), "Mixed precision training", *ICLR*: master weights, loss scaling, float32 accumulation. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Kalamkar et al. (2019), "A study of BFLOAT16 for deep learning training". [arXiv:1905.12322](https://arxiv.org/abs/1905.12322)
- Micikevicius et al. (2022), "FP8 formats for deep learning": E4M3 and E5M2, per-tensor scaling. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
- Goldberg (1991), "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1). [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
- the PyTorch automatic mixed precision documentation lists which ops autocast runs in which dtype. [pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)

## connections

- ← builds on: floating point and stability (ch 00 · 03), the training loop (ch 15), the roofline's bytes (01).
- → used in: memory accounting counts 16-bit copies and float32 master weights (05); int8 quantization is the same rounding idea taken further (06); the capstone trains your GPT in float16 with your `GradScaler`. chapter 27's large models are trained in bfloat16.
