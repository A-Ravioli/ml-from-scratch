# initialization

## why this matters

take the network you trained on MNIST in chapter 14, make it twenty layers deep instead of three, and train it exactly the same way. nothing crashes. the loss starts at 2.30 and stays there, epoch after epoch, to three decimal places. the gradients aren't `nan`; they're simply around $10^{-19}$ in the first layer, so the optimizer has nothing to work with. the code is correct. the architecture is reasonable. what's wrong is the *starting point*: with the default initialization, the signal that enters the first layer has faded to nothing long before it reaches the twentieth, and the error signal travelling back fades the same way.

for most of the 1990s and 2000s this was simply how deep networks behaved, and the folklore was that they couldn't be trained without elaborate layer-by-layer pretraining. then Glorot and Bengio (2010) and He et al. (2015) showed that much of the problem was a scale that compounds over depth, and that choosing the variance of the initial weights with one line of arithmetic fixes the scale. this topic derives that arithmetic, builds the initializers every framework ships (with `torch.nn.init`'s names and conventions), and gives you an instrument for watching the signal layer by layer. the lab then shows you what the arithmetic *doesn't* fix.

## warm-up: predict first

1. $z = \sum_{j=1}^{n} w_j x_j$ with all $w_j, x_j$ independent, mean 0, variance 1. what's $\operatorname{Var}(z)$ for $n = 784$?
2. a ReLU receives a symmetric, zero-mean input with variance 2. what's the mean of its squared output?
3. a 50-layer network multiplies the scale of its activations by 0.9 per layer. by how much has the scale shrunk at the end? and at 1.1 per layer?
4. you initialize every weight of a network to the same constant (say 0.01). what goes wrong, even if the scale is perfect?
5. a convolution weight has shape `(64, 3, 3, 3)`. how many inputs feed each output?

## prerequisites check

- [ ] you can compute the mean and variance of a sum and of a product of independent random variables (→ ch 04 · topic 03)
- [ ] you know $\operatorname{Var}(U(-a, a)) = a^2/3$ (→ ch 04 · topic 02)
- [ ] you have a working `mlfs.nn` (`Linear`, `ReLU`, `Tanh`, `Sequential`, `fit`) with torch-layout weights `(out, in)` (→ ch 14 · topic 04)
- [ ] you know why backpropagation multiplies by $W^\top$ on the way back (→ ch 14 · topics 01 and 06)
- [ ] you know `as_rng` and why one generator should be threaded through a whole model (→ ch 00 · topic 05)

## core ideas

### idea 1: the variance of a sum, and the two numbers that set it

every unit in a linear layer computes $z_i = \sum_{j=1}^{n_\text{in}} W_{ij}h_j$ (plus a bias, which we'll set to zero). suppose the weights are independent of the inputs and of each other, with mean 0 and variance $\operatorname{Var}(W)$. each term $W_{ij}h_j$ then has mean 0 and variance $E[W^2]E[h^2] = \operatorname{Var}(W)\,E[h^2]$, and the terms are uncorrelated, so their variances add:

$$\operatorname{Var}(z_i) = n_\text{in}\,\operatorname{Var}(W)\,E[h^2].$$

that's the whole law. read it as: *every layer multiplies the signal's second moment by $n_\text{in}\operatorname{Var}(W)$ (times whatever the activation does)*. it uses $E[h^2]$, not $\operatorname{Var}(h)$, because the inputs to a layer needn't have mean zero (a ReLU's outputs never do).

the backward pass obeys the same law with the roles swapped. the gradient with respect to a layer's input is $\bar h_j = \sum_i W_{ij}\bar z_i$, a sum over the $n_\text{out}$ units the input feeds, so $\operatorname{Var}(\bar h) = n_\text{out}\operatorname{Var}(W)\operatorname{Var}(\bar z)$.

so the two numbers that matter are how many terms each sum has. they're called the **fans**:

- **fan_in**: inputs per output. for a `Linear` weight of shape `(out, in)` it's `in`.
- **fan_out**: outputs per input, `out`.

for a convolution weight `(out, in, kh, kw)` (chapter 16), each output sums over `in × kh × kw` inputs, and each input feeds `out × kh × kw` outputs, so both fans pick up the kernel's size, the **receptive field**. that's `calculate_fan`: a weight shape in, `(fan_in, fan_out)` out, refusing shapes with fewer than two dimensions (a bias has no fan).

the second number is the **gain**: a factor that compensates for what an activation does to the second moment. a linear (or sigmoid, near 0) activation leaves it alone: gain 1. a ReLU zeroes half of its symmetric input and so halves the second moment (idea 3): gain $\sqrt 2$. a leaky ReLU with negative slope $a$ keeps a bit more: gain $\sqrt{2/(1 + a^2)}$. `tanh` gets $5/3$, an empirical value (PyTorch's) that sits slightly above 1 to offset tanh's squashing of large inputs. `calculate_gain` returns these, with PyTorch's names and leaky ReLU's default slope of 0.01.

> **▶ build: stage 1.** implement `calculate_fan(shape)` and `calculate_gain(nonlinearity, param=None)`.

### idea 2: xavier (glorot): keep both directions at scale 1

Glorot and Bengio (2010) asked for two things at once. forward, keep $\operatorname{Var}(z)$ constant from layer to layer: $n_\text{in}\operatorname{Var}(W) = 1$. backward, keep the gradients' variance constant: $n_\text{out}\operatorname{Var}(W) = 1$. unless the layer is square, these disagree, so they took the compromise

$$\operatorname{Var}(W) = \frac{2}{n_\text{in} + n_\text{out}},$$

the harmonic mean of the two requirements. their analysis assumed an activation that is roughly linear around zero, like tanh for small inputs, so the gain is 1 (a gain multiplies the standard deviation, so the variance becomes $\text{gain}^2 \cdot 2/(n_\text{in} + n_\text{out})$).

the same variance can come from two distributions. **`xavier_normal_`** draws $N(0, \sigma^2)$ with $\sigma = \text{gain}\sqrt{2/(n_\text{in} + n_\text{out})}$. **`xavier_uniform_`** draws $U(-a, a)$; since $\operatorname{Var}(U(-a, a)) = a^2/3$, matching the variance needs $a = \sqrt 3\,\sigma = \text{gain}\sqrt{6/(n_\text{in} + n_\text{out})}$. the uniform version has bounded weights; the normal one is the more common default in papers. their behaviour in a network is almost identical, because only the variance enters the law.

conventions, which the tests hold you to: an initializer fills the weight **in place** (`arr[...] = draws`), so anything already holding the array (an optimizer!) sees the new values; it **returns the parameter**, so you can write `w = xavier_normal_(Parameter(np.empty((3, 4))))`; it accepts an `mlfs` `Parameter` (filling `.data`) or a plain ndarray; and it draws the whole weight with **one** call, `as_rng(rng).uniform(-a, a, size=shape)` or `.normal(0, std, size=shape)`, so seeded models are reproducible draw for draw. the trailing underscore is PyTorch's mark for "modifies its argument".

> **▶ build: stage 2.** implement `xavier_uniform_` and `xavier_normal_` (`weight_array`, which picks the array to fill, is given).

### idea 3: kaiming (he): account for the relu

a ReLU sets negative inputs to zero. if $z$ is symmetric around zero, half of its mass is cut, and

$$E[\operatorname{relu}(z)^2] = \int_0^\infty z^2 p(z)\,dz = \tfrac12 E[z^2] = \tfrac12\operatorname{Var}(z).$$

put that into the law of idea 1. with $h_l = \operatorname{relu}(z_l)$,

$$\operatorname{Var}(z_{l+1}) = n_\text{in}\operatorname{Var}(W)\,\tfrac12\operatorname{Var}(z_l),$$

so xavier's $\operatorname{Var}(W) = 1/n$ (on a square layer) *halves* the signal at every ReLU layer: after 50 layers it's $2^{-50} \approx 10^{-15}$ of what it was. He, Zhang, Ren and Sun (2015) fixed the factor:

$$\operatorname{Var}(W) = \frac{2}{n_\text{in}},\qquad \sigma = \frac{\text{gain}}{\sqrt{\text{fan}}},\ \text{gain} = \sqrt2.$$

with that, $\operatorname{Var}(z_{l+1}) = \operatorname{Var}(z_l)$ exactly, at any depth. **`kaiming_normal_`** draws $N(0, \sigma^2)$, **`kaiming_uniform_`** draws $U(-b, b)$ with $b = \sqrt3\,\sigma$. the `nonlinearity` argument picks the gain (so `kaiming_normal_(w, nonlinearity='tanh')` uses $5/3$, and `'linear'` gives $1/\sqrt{n_\text{in}}$, LeCun's classic init). `mode='fan_in'` (the default) preserves the forward signal; `mode='fan_out'` divides by $\sqrt{n_\text{out}}$ instead and preserves the backward signal. He et al. showed that for a network of similar-width layers either choice works, because the two products differ only by the ratio of the first and last widths.

> **🔮 predict:** with kaiming init, the *variance* of a 50-layer ReLU network's activations stays constant. does the network still map two different images to two different-looking representations at layer 50? (experiment 2 of the lab measures it.)

> **▶ build: stage 3.** implement `kaiming_uniform_` and `kaiming_normal_` (`kaiming_std`, which computes $\sigma$ and checks `mode`, is given).

### idea 4: depth turns small factors into huge ones

the law makes a deep network a product. if every layer multiplies the second moment by a factor $c$, then after $L$ layers it's $c^L$, and exponentials are unforgiving: $0.9^{50} \approx 0.005$, $1.1^{50} \approx 117$, $0.5^{50} \approx 10^{-15}$. at depth 3 (chapter 14's capstone) almost any sensible init works; at depth 50 only $c \approx 1$ does.

the classic failures read straight off the law, for a layer of width $n = 256$:

- **"small random weights", $N(0, 0.01^2)$**: $c = 256 \times 10^{-4} = 0.0256$ per layer (for tanh, which is linear near zero). by layer 10 the signal is $10^{-8}$ of the input; by layer 50, below $10^{-40}$.
- **$N(0, 1)$**: $c = 256$ (halved by a ReLU). the activations overflow to `inf` within a few dozen layers, or saturate every tanh at $\pm1$, where its derivative is zero.
- **PyTorch's `Linear` default**, $U(-1/\sqrt n, 1/\sqrt n)$, has variance $1/(3n)$: $c = 1/3$ for a linear layer and $1/6$ through a ReLU. it was chosen for shallow networks. its random biases add a constant floor, so the activations don't vanish; they stop *depending on the input* (on paper 4).

to experiment with this you need deep networks built to order. **`deep_mlp(sizes, activation=..., init=...)`** stacks `Linear(sizes[i], sizes[i+1])` layers with the activation between them (none after the last; `'linear'` means no activation modules at all) and initializes every weight with the chosen scheme: `'xavier'` is `xavier_normal_` with gain 1, Glorot's original; `'kaiming'` is `kaiming_normal_` with `nonlinearity=activation`; a float is $N(0, \text{std}^2)$; `'default'` keeps `Linear`'s own init. every scheme except `'default'` sets the biases to zero, the standard choice (a zero bias adds nothing to the variance). draw everything from one generator, layer after layer: each `Linear(..., rng=rng)` first, then its re-initialization with the same `rng`.

> **▶ build: stage 4.** implement `deep_mlp(sizes, *, activation='relu', init='kaiming', rng=None)`.

### idea 5: watching the signal

derivations assume independence and infinite width; real networks have neither. so measure. **`activation_stats(model, X)`** runs a batch through a `Sequential` one layer at a time, under `no_grad` (it's an instrument; it shouldn't build a graph), and records for every layer's output its **mean**, its **standard deviation** (over all entries, examples and units together) and its **fraction of exact zeros**. it returns one dict per layer, named `'<index>.<layer class>'` (`'0.Linear'`, `'1.ReLU'`, …), in order.

each statistic diagnoses something. a std that shrinks or grows geometrically with depth is the variance law at work. a mean drifting away from zero is a shift that later layers have to undo (ReLU outputs always have a positive mean). `frac_zero` after a ReLU should sit near one half; near 1 means **dead units**, which pass neither signal nor gradient (topic 05 builds a logger around this). for tanh, the useful extra statistic is the fraction near $\pm1$ (saturated), which a histogram shows at a glance, and the lab plots them.

there's a limit to what these numbers can tell you, and it's the lab's central surprise. they're all *per-layer* statistics of one batch; they say nothing about whether two different inputs still look different. a network can keep a perfect scale while squeezing every input towards the same direction, and then its last layer has nothing to work with. the study of how *correlations* between inputs evolve with depth is called **signal propagation** (Poole et al., 2016; Schoenholz et al., 2017), and it's why very deep networks need more than a good init: residual connections (chapter 16) and normalization (topic 02).

> **🔮 predict:** in a 50-layer ReLU network with xavier init, what's the std of the activations at layer 50, as a power of 2? and with kaiming?

> **▶ build: stage 5.** implement `activation_stats(model, X)`.

## worked example

the first layer of chapter 14's MNIST network has a weight of shape `(256, 784)`: fan_in $= 784$, fan_out $= 256$.

| initializer | formula | value |
|---|---|---|
| `xavier_uniform_` | $a = \sqrt{6/(784 + 256)} = \sqrt{6/1040}$ | $a = 0.075955$ |
| `xavier_normal_` | $\sigma = \sqrt{2/1040}$ | $\sigma = 0.043853$ |
| `kaiming_normal_` (relu, fan_in) | $\sigma = \sqrt2/\sqrt{784}$ | $\sigma = 0.050508$ |
| `kaiming_uniform_` (relu, fan_in) | $b = \sqrt3 \cdot 0.050508 = \sqrt{6/784}$ | $b = 0.087482$ |
| `Linear` default | $U(-1/\sqrt{784}, 1/\sqrt{784})$ | bound $0.035714$, $\sigma = 0.020620$ |

notice that both xavier variants have the same $\sigma$ ($0.075955/\sqrt3 = 0.043853$), and that kaiming's std is $2.45\times$ the default's, so its variance is $(0.050508/0.020620)^2 = 6$ times larger: the factor of 6 from idea 4.

a convolution weight `(64, 3, 3, 3)` has fan_in $= 3 \cdot 9 = 27$ and fan_out $= 64 \cdot 9 = 576$, so its kaiming std is $\sqrt{2/27} = 0.2722$.

now the law, for a 3-layer ReLU stack of width 256 with standardized inputs ($E[x^2] = 1$). with kaiming, each layer's pre-activation has variance $256 \cdot (2/256) \cdot E[h^2]$: layer 1 gives $\operatorname{Var}(z_1) = 2$, the ReLU halves it back to $E[h_1^2] = 1$, and so on: 1, 1, 1. with xavier on square layers ($\operatorname{Var}(W) = 1/256$), $E[h_l^2] = \frac12, \frac14, \frac18$. the tests check your initializers' exact draws and these statistics, and `deep_mlp` against the law at depth 20 ($2^{-20}$ for xavier).

## on paper

1. **the variance of a product.** $W$ and $h$ are independent, $E[W] = 0$. show that $\operatorname{Var}(Wh) = \operatorname{Var}(W)\,E[h^2]$, and explain why the law can't use $\operatorname{Var}(h)$.

<details><summary>solution</summary>

$E[Wh] = E[W]E[h] = 0$, so $\operatorname{Var}(Wh) = E[W^2h^2] = E[W^2]E[h^2] = \operatorname{Var}(W)E[h^2]$. if $h$ has mean $\mu \ne 0$, $E[h^2] = \operatorname{Var}(h) + \mu^2 > \operatorname{Var}(h)$: the mean of the input contributes to the output's variance, because $W\mu$ is a random variable with variance $\mu^2\operatorname{Var}(W)$. a ReLU's output has $\mu > 0$, so using $\operatorname{Var}(h)$ would underestimate the next layer's variance.

</details>

2. **the relu halves the second moment.** for $z$ with a density symmetric about 0, show $E[\operatorname{relu}(z)^2] = \frac12 E[z^2]$, and for $z \sim N(0, s^2)$ compute $E[\operatorname{relu}(z)]$ and $\operatorname{Var}(\operatorname{relu}(z))$.

<details><summary>solution</summary>

$E[\operatorname{relu}(z)^2] = \int_0^\infty z^2p(z)\,dz$, and by symmetry $\int_0^\infty = \int_{-\infty}^0$, each half of $E[z^2]$. for the Gaussian, $E[\operatorname{relu}(z)] = \int_0^\infty z\,\frac{e^{-z^2/2s^2}}{\sqrt{2\pi}s}dz = \frac{s}{\sqrt{2\pi}} \approx 0.399s$, so $\operatorname{Var} = \frac{s^2}{2} - \frac{s^2}{2\pi} = s^2\frac{\pi - 1}{2\pi} \approx 0.341s^2$: a std of $0.584s$. (with kaiming, $s^2 = 2$ for unit-scale inputs, so the activations' std is about $0.83$: the lab's ~0.8–0.97.)

</details>

3. **fan_out mode.** in a ReLU network, the backward signal at layer $l$ is $\bar h_l = W_{l+1}^\top(\bar h_{l+1} \odot \operatorname{relu}'(z_{l+1}))$. assuming independence and that the derivative is 0 or 1 with probability ½ each, show that keeping $\operatorname{Var}(\bar h)$ constant needs $\operatorname{Var}(W) = 2/n_\text{out}$.

<details><summary>solution</summary>

each entry of $\bar h_l$ is a sum over the $n_\text{out}$ units of layer $l+1$ of $W_{ij}\,\bar h_{l+1,i}\,\operatorname{relu}'(z_{l+1,i})$. each term has mean 0 and variance $\operatorname{Var}(W)\cdot\operatorname{Var}(\bar h_{l+1})\cdot\frac12$ (the derivative squared is 1 half of the time). summing $n_\text{out}$ uncorrelated terms: $\operatorname{Var}(\bar h_l) = \frac{n_\text{out}}{2}\operatorname{Var}(W)\operatorname{Var}(\bar h_{l+1})$, which is constant when $\operatorname{Var}(W) = 2/n_\text{out}$: `kaiming_normal_(w, mode='fan_out')`.

</details>

4. **where the default's floor comes from.** a ReLU network of width $n$ with PyTorch's default init: $W \sim U(-1/\sqrt n, 1/\sqrt n)$, $b \sim U(-1/\sqrt n, 1/\sqrt n)$. write the recursion for $q_l = E[z_l^2]$, find its fixed point, and evaluate the std of the activations at $n = 256$.

<details><summary>solution</summary>

$\operatorname{Var}(W) = \operatorname{Var}(b) = 1/(3n)$, so $q_{l+1} = n\cdot\frac{1}{3n}\cdot\frac{q_l}{2} + \frac{1}{3n} = \frac{q_l}{6} + \frac{1}{3n}$. the fixed point solves $q^* = q^*/6 + 1/(3n)$: $q^* = \frac{2}{5n} = 0.00156$ at $n = 256$, a pre-activation std of $0.040$, and (on paper 2) an activation std of about $0.584 \times 0.040 = 0.023$, the lab's floor near 0.02. the trouble is what $q^*$ is made of: the input's contribution shrinks by $1/6$ per layer, while the biases' contribution is renewed at every layer. after ten layers the activations are almost entirely bias, the same for every image.

</details>

## experiment

`lab.py` uses your initializers, `deep_mlp` and `activation_stats` for three experiments:

1. activation statistics, histograms and gradient norms across 50 layers, for five combinations of activation and init.
2. the same networks, asked a different question: do two different images still look different at layer 50?
3. 20-layer networks trained on MNIST with four of the inits, three seeds each.

## check your understanding

1. why do xavier's formulas average fan_in and fan_out, while kaiming's default uses only fan_in?

<details><summary>answer</summary>

xavier asks for both the forward and the backward variance to be preserved, which conflict unless $n_\text{in} = n_\text{out}$, and splits the difference. He et al. observed that preserving only one direction is enough in practice: for a stack of similar-width layers the other direction's scale changes only by the ratio of the first and last widths, not exponentially in depth. so they keep the forward pass exact (fan_in) by default.

</details>

2. why is the bias usually initialized to zero, and why don't initializers need to care about it?

<details><summary>answer</summary>

a zero bias contributes nothing to the variance, so the law (and the scheme) is unaffected; there's also no symmetry problem, since the weights already differ between units. a random bias adds a constant, input-independent vector to every example, which is exactly what drowns the input in the torch default at depth (on paper 4).

</details>

3. initializing all weights to the same value $c$ has a perfect scale for a suitable $c$. why does it still fail?

<details><summary>answer</summary>

symmetry. every unit in a layer computes the same function of the same inputs, so they get the same gradient and stay identical forever: a width-256 layer behaves like a single unit. random initialization breaks the symmetry; the variance only sets its scale.

</details>

4. a 50-layer tanh network with xavier init isn't variance-preserving (its std falls from 0.68 to 0.10 in the lab), yet the decay is slow, not geometric. why?

<details><summary>answer</summary>

tanh is linear near zero with slope 1, so for small activations xavier's factor is close to 1; the loss comes from the curvature of tanh, which squashes large values. as the signal gets smaller, tanh gets more linear and the loss per layer gets smaller, so the decay slows down (it's roughly like $1/\sqrt l$, not $c^l$). gain $5/3$ was chosen to counteract this squashing.

</details>

5. why does `activation_stats` use `no_grad`?

<details><summary>answer</summary>

it's a measurement, not part of training. without `no_grad`, the forward pass would record a graph for every layer and keep every intermediate activation alive until the result is dropped: for 50 layers of 1,000 × 256 that's 100 MB for nothing. (it also can't accidentally leave gradients behind.)

</details>

6. with kaiming init, the activations' std in the lab stays near 0.8–1.0 rather than exactly 1. why isn't it exactly the input's?

<details><summary>answer</summary>

kaiming preserves the *second moment* $E[h^2]$, not the standard deviation. a ReLU's output has mean $s/\sqrt{2\pi}$, so its std is smaller than its root-mean-square ($0.584s$ versus $0.707s$, on paper 2). finite width adds random fluctuations around the law, which compound slowly over depth.

</details>

7. a colleague's 100-layer ReLU network with kaiming init doesn't train. the activation statistics look perfect. what might be wrong?

<details><summary>answer</summary>

the per-layer statistics can be perfect while the representations of different inputs collapse towards one direction (the lab's experiment 2), so the network's output barely depends on its input and gradients carry little useful information. variance preservation is necessary, not sufficient. the fixes are architectural: residual connections, normalization layers, or initializations designed for correlations (orthogonal weights, "dynamical isometry").

</details>

## common bugs

- **(in, out) fans.** symptom: your fans are swapped, so kaiming with `mode='fan_in'` uses fan_out; nothing fails on square layers and everything is subtly off on the first and last ones. torch layout is `(out, in)`: fan_in is `shape[1]`.
- **xavier's sum in kaiming.** symptom: kaiming std of $\sqrt{2/1040}$ instead of $\sqrt{2/784}$ for a `(256, 784)` weight; deep ReLU networks shrink by $\sim 0.75$ per layer instead of staying flat.
- **replacing instead of filling.** `param.data = draws` creates a new array; an optimizer created earlier still holds the old one and keeps updating it. fill in place: `arr[...] = draws`.
- **forgetting the uniform $\sqrt3$.** symptom: weights from $U(-\sigma, \sigma)$ have variance $\sigma^2/3$; a deep network shrinks by a factor 3 per layer.
- **the same seed for every layer.** `Linear(..., rng=0)` for every layer gives every square layer the *same* weights. pass one `Generator` through the whole model.
- **standardizing nothing.** the law assumes the input's second moment is about 1. raw pixels in $[0, 255]$ have $E[x^2]$ in the thousands, and the first layer's outputs are huge whatever the init.

## going deeper

- Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks", *AISTATS*. the variance analysis and xavier init. [proceedings.mlr.press/v9/glorot10a](https://proceedings.mlr.press/v9/glorot10a.html)
- He, Zhang, Ren & Sun (2015), "Delving deep into rectifiers: surpassing human-level performance on ImageNet classification", *ICCV*. kaiming init and the fan_in/fan_out argument. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Saxe, McClelland & Ganguli (2014), "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", *ICLR*. orthogonal initialization and dynamical isometry. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)
- Poole, Lahiri, Raghu, Sohl-Dickstein & Ganguli (2016), "Exponential expressivity in deep neural networks through transient chaos", *NeurIPS*. how correlations between inputs propagate. [arXiv:1606.05340](https://arxiv.org/abs/1606.05340)
- Schoenholz, Gilmer, Ganguli & Sohl-Dickstein (2017), "Deep information propagation", *ICLR*. depth scales for trainability, and the edge of chaos. [arXiv:1611.01232](https://arxiv.org/abs/1611.01232)
- Mishkin & Matas (2016), "All you need is a good init", *ICLR*. LSUV: set each layer's scale by measuring it, not by formula. [arXiv:1511.06422](https://arxiv.org/abs/1511.06422)

## connections

- ← builds on: variances of sums and products (ch 04), `mlfs.nn`'s `Linear`, `Sequential` and `fit` (ch 14 · topic 04), backpropagation through $W^\top$ (ch 14), `as_rng` (ch 00 · topic 05).
- → used in: normalization layers, which make the scale self-correcting (topic 02); dead-unit diagnostics (topic 05); the capstone's planted init bug; conv fans (ch 16 · topic 01); residual networks and why they need zero-initialized branches (ch 16 · topic 04); initialization in transformers (ch 18); signal propagation and the NTK (ch 19).
