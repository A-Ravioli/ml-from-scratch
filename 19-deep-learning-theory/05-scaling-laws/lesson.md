# scaling laws

## why this matters

in 2020 Kaplan and colleagues at OpenAI trained hundreds of transformer language models, from a few thousand to over a billion parameters, and found that the test loss fell along straight lines on log-log plots: a power law in the number of parameters, in the amount of data, and in the compute spent, over seven orders of magnitude, with almost nothing else (depth, width, number of heads) mattering much. two years later Hoffmann et al. at DeepMind redid the measurement more carefully and concluded that the field had been training models that were too large on too little data: their 70-billion-parameter Chinchilla, trained on four times more tokens than the 280-billion-parameter Gopher, beat it on almost every benchmark, at the same compute. and OpenAI's GPT-4 report showed the final loss of the full model predicted in advance from runs using up to 10,000 times less compute. scaling laws turned "how big should the next model be, and how long should we train it" from a guess into a fit.

a fit is also where they can go wrong. a power law is an empirical regularity, not a theorem; its exponent depends on the data, the tokenizer and the training recipe; and extrapolating a curve fitted to a handful of points is a classic way to be confidently wrong. chapter 18's capstone ended with exactly such a curve: three GPTs whose losses fell with size with local slopes of 0.097 and then 0.043. the curve bends. this topic builds the tools to fit such curves properly, to reason about compute, and to say what a fit can and can't tell you, and the lab uses them on your own GPTs.

## warm-up: predict first

1. a loss falls from 3.0 to 2.7 when a model grows from 1M to 10M parameters. if it follows a pure power law $L = aN^{-b}$, what loss do you expect at 100M parameters?
2. on a log-log plot of loss against parameters, a pure power law is a straight line. what does a curve that bends upward (flattens) suggest?
3. roughly how many floating-point operations does it take to train a model with $N$ parameters on $D$ tokens?
4. you have a fixed compute budget. is it better to train a bigger model on fewer tokens or a smaller model on more?
5. a language model's vocabulary embedding has 50,000 × 4,096 parameters. should those count as "model size" when you fit a scaling law?

## prerequisites check

- [ ] you can fit a straight line by least squares and know what its residuals mean (→ ch 08 · topic 01)
- [ ] you know logarithms turn products and powers into sums and multiples (→ ch 01)
- [ ] you can solve a constrained minimization with a Lagrange multiplier (→ ch 06 · topic 05)
- [ ] you trained GPTs of several sizes with `train_lm` and know bits per character (→ ch 18 · topic 04 and capstone)
- [ ] you know how many multiply-adds a matrix-vector product costs (→ ch 02 · topic 01)

## core ideas

### idea 1: power laws, and the slope between two points

a **power law** is $y = a x^{-b}$. take logarithms and it becomes a straight line, $\log y = \log a - b\log x$, so on log-log axes every power law is a line with slope $-b$. that's what makes it special: it has no characteristic scale. multiplying $x$ by 10 multiplies $y$ by $10^{-b}$ wherever you start, so a doubling of model size buys the same *relative* improvement at 1M parameters as at 1B. for language models, Kaplan et al. measured $b \approx 0.076$ for parameters: every 10× in parameters reduces the loss by about 16%.

real losses can't fall forever: text has an entropy, and no model can predict it better than that. so the usual form adds an offset,

$$L(N) = aN^{-b} + c,$$

where $c$ is the **irreducible loss** (the entropy of the data, plus whatever the setup makes unreachable). now the curve is a line on log-log axes only far from $c$; as $L$ approaches $c$ it flattens.

a quick diagnostic needs no fit at all: the **local exponent** between two neighbouring points,

$$b_{\text{loc}} = -\frac{\log y_2 - \log y_1}{\log x_2 - \log x_1}.$$

for a pure power law it's the same for every pair. if it shrinks as $x$ grows, the curve is bending: either towards a floor $c$, or because something other than $x$ is limiting the loss (on paper 3 shows how fast an offset makes it shrink). you can also compute it against a guessed offset, $\log(y - c)$, and see whether that straightens the curve.

> **spec.** `power_law(x, a, b, c=0.0)` returns `a * x**(-b) + c` elementwise for floats or arrays. `local_exponents(x, y, c=0.0)` (x increasing) returns the `len(x) - 1` slopes $-\Delta\log(y - c) / \Delta\log x$ between consecutive points, natural logarithms.

> **▶ build: stage 1.** implement `power_law` and `local_exponents`.

### idea 2: fitting in log space

which least-squares problem? the points of a scaling study are spread over decades: losses from 4 down to 2, parameters from $10^3$ to $10^9$. fitting $\sum (y_i - \hat y_i)^2$ in linear space weights the large losses of the smallest models most and treats an error of 0.05 the same at loss 4 and at loss 2.05. what you usually care about is *relative* error, which is what residuals in log space measure:

$$\min_{a, b, c}\ \sum_i \big(\log y_i - \log(a x_i^{-b} + c)\big)^2, \qquad a > 0,\ 0 \le c < \min_i y_i.$$

without the offset this is ordinary least squares of $\log y$ on $\log x$, solved exactly: $b$ is minus the slope, $a$ is $e^{\text{intercept}}$. with the offset it's nonlinear, and a general-purpose optimizer started from a fixed guess can stall (the loss surface in $(a, b, c)$ has long curved valleys: a smaller $c$ can be traded for a larger $a$ and a smaller $b$). a robust recipe uses the structure: **for a fixed $c$, the problem is linear** in $(\log a, b)$ after subtracting $c$. so profile over $c$: try a grid of values in $[0, \min y)$, solve the linear fit for each, keep the best, then polish all three parameters together with a local optimizer (`scipy.optimize.least_squares`, with bounds). the constraint $c < \min y$ is necessary (you can't take the log of a negative number) and $c \ge 0$ says the irreducible loss isn't negative.

three warnings come with any fit. with as many points as parameters (three points, three parameters) the fit passes through every point and tells you nothing about how well the form describes the data. with a few more, the offset is often poorly determined: very different $(b, c)$ pairs fit the observed range equally well and disagree wildly beyond it. and a fitted curve is a summary of the measurements, not of the mechanism: if the larger models were all trained in a way that held them back (fewer steps per parameter, a learning rate tuned on small models), the fit learns the handicap and extrapolates it.

> **spec.** `fit_power_law(x, y, *, offset=True)` returns `(a, b, c)` minimizing the log-space squared error above, with $a > 0$ and $0 \le c < \min y$. with `offset=False`, $c = 0.0$ exactly and $(a, b)$ come from ordinary least squares of $\log y$ on $\log x$.

> **🔮 predict:** chapter 18's three GPTs had validation losses of 2.567, 2.190 and 2.062 bits per character at 0.35M, 1.8M and 7.5M parameters. fit with an offset: what irreducible loss $c$ do you think three points will give you, and would you trust it?

> **▶ build: stage 2.** implement `fit_power_law`.

### idea 3: compute, and the 20-tokens rule

how much computation does training take? a dense network with $N$ parameters does about one multiply and one add per parameter per token in the forward pass: $2N$ floating-point operations per token. the backward pass computes two products per layer (the gradient with respect to the activations, to pass backwards, and with respect to the weights, to update them), about twice the forward cost: $4N$. so training on $D$ tokens costs

$$C \approx 6ND$$

FLOPs (Kaplan et al., 2020; attention adds a term proportional to the context length, small for short contexts). a fixed budget $C$ can be spent on a big model trained briefly or a small model trained long. Kaplan et al. concluded that most of an increased budget should go to model size ($N \propto C^{0.73}$). Hoffmann et al. (2022) found, with three different methods, that $N$ and $D$ should grow in equal proportion ($N \propto C^{0.5}$, $D \propto C^{0.5}$), with about **20 training tokens per parameter** at the optimum; much of the difference came from Kaplan's use of a fixed learning-rate schedule length, so that the intermediate checkpoints of a long run, cut short in mid-schedule, were compared with finished small runs. Chinchilla itself is the rule in action: 70B parameters and 1.4T tokens, $C = 6 \times 70\text{B} \times 1.4\text{T} = 5.9 \times 10^{23}$.

the rule of thumb is two equations: $C = 6ND$ and $D = 20N$, so $N = \sqrt{C/120}$. it was measured for subword-token language models from about 70M to 16B parameters; it's a starting point, not a law, and later models are deliberately trained far past it (Llama 3's 8B model saw about 15T tokens, almost 2,000 per parameter) because a smaller model is cheaper to serve.

> **spec.** `compute_optimal(compute, *, tokens_per_param=20.0)` returns `(n_params, n_tokens)` with `compute = 6 * N * D` and `D = tokens_per_param * N` (floats or arrays, elementwise).

> **▶ build: stage 3.** implement `compute_optimal`.

### idea 4: a loss surface in two variables

the 20-tokens rule is the optimum of a fitted model of the loss as a function of both size and data. Hoffmann et al.'s third method fits

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}:$$

an irreducible term $E$, a term that falls as the model grows (the error of the best model of size $N$ with infinite data), and a term that falls with data (the error of a finite training run). on the constraint $6ND = C$, minimizing $A N^{-\alpha} + B D^{-\beta}$ has a closed form (on paper 2):

$$N_{\text{opt}} = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \quad D_{\text{opt}} = G^{-1}\left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}, \quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}.$$

the exponents tell you how to grow: if the data term decays more slowly ($\beta < \alpha$), extra compute should go mostly to data. with the published fit ($E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$), $N_{\text{opt}} \propto C^{0.45}$. plug Chinchilla's own budget into it, though, and you get about 32B parameters, not the 70B of their other two methods; Besiroglu et al. (2024) reconstructed the data from the paper's figures, found that the published parametric fit doesn't match it, and refitted it with the two exponents much closer to equal. a fit is only as trustworthy as the procedure behind it.

> **spec.** `chinchilla_loss(n_params, n_tokens, *, E, A, B, alpha, beta)` returns $E + A/N^\alpha + B/D^\beta$ elementwise. `optimal_allocation(compute, *, A, B, alpha, beta)` returns `(N_opt, D_opt)`, the closed form above, which satisfies $6 N_{\text{opt}} D_{\text{opt}} = C$.

> **▶ build: stage 4.** implement `chinchilla_loss` and `optimal_allocation`.

### idea 5: what counts as a parameter

scaling laws need a definition of $N$. a character-level GPT with $d = 16$ has about a third of its parameters in its token and position embeddings, which are lookup tables: each token touches one row, so they cost almost no compute per token and behave differently from the matrices that process every token. Kaplan et al. found that counting **non-embedding parameters** gave much cleaner laws (models of different shapes collapse onto one curve), and they used non-embedding $N$ in $C = 6ND$ too. so: count every parameter except the token embedding and the learned position embedding. two details: a tied output head *is* the token embedding (the same tensor), so it's excluded with it and must not be counted twice; an untied head is a real $d \times V$ matrix applied to every token and counts.

> **spec.** `non_embedding_params(model)` returns the number of elements in `model.parameters()` (each tensor once, even if tied) excluding `model.tok_emb.weight` and `model.pos_emb.weight` when those modules exist (`pos_emb` may be `None`). buffers never count.

> **🔮 predict:** in the lab, five GPTs from 3 thousand to 340 thousand non-embedding parameters are trained for the same 150 steps. how close will a power law fitted to them come to the loss of a 790-thousand-parameter model?

> **▶ build: stage 5.** implement `non_embedding_params`.

## worked example

chapter 18's capstone: S (354,240 parameters), M (1,816,896) and L (7,460,160) reached 2.567, 2.190 and 2.062 validation bits per character after the same 2,000 steps of 64 windows of 128 characters.

- **local exponents.** S to M: $-\ln(2.190/2.567)/\ln(1{,}816{,}896/354{,}240) = 0.1589/1.6349 = 0.097$. M to L: $0.0602/1.4125 = 0.043$. the curve bends: the second doubling-equivalent bought less than half the relative improvement of the first.
- **compute.** each model saw $D = 2{,}000 \times 64 \times 128 = 16{,}384{,}000$ tokens. L spent $C = 6 \times 7{,}460{,}160 \times 16{,}384{,}000 = 7.33 \times 10^{14}$ FLOPs.
- **the 20-tokens rule for that budget.** $N = \sqrt{7.33 \times 10^{14}/120} = 2.47$ million parameters, trained on $D = 20N = 49.4$ million tokens: three times the data L saw. by this rule, L was a model three times too large for its compute, trained on a third of the data it should have had (and the rule wasn't calibrated for characters, so treat that as a direction, not a number).

your `local_exponents` must return `[0.097, 0.043]` for these numbers, and `compute_optimal` must return Chinchilla's (70B, 1.4T) for its compute.

## on paper

1. **$C \approx 6ND$.** for one linear layer $y = Wx$ with $W \in \mathbb R^{m \times n}$, count the multiply-adds of the forward pass, of the gradient with respect to $x$ and of the gradient with respect to $W$, per token. sum over layers to get the total per token, and multiply by $D$.

<details><summary>solution</summary>

forward: $mn$ multiply-adds = $2mn$ FLOPs. backward: $\partial L/\partial x = W^\top \partial L/\partial y$, another $mn$ multiply-adds; $\partial L/\partial W = (\partial L/\partial y)\,x^\top$, another $mn$. total $6mn$ FLOPs per token for $mn$ parameters. summing over all the layers' weight matrices gives $6N$ per token and $6ND$ for training. the embedding lookup costs no multiplications, which is one more reason to leave it out of $N$; attention's score computation ($QK^\top$ and the weighted sum) adds about $6 \cdot n_{\text{layers}} \cdot T \cdot d$ per token, which matters only when the context $T$ is comparable to $d_{ff}$ or larger.

</details>

2. **the compute-optimal split.** minimize $f(N, D) = AN^{-\alpha} + BD^{-\beta}$ subject to $6ND = C$. derive $N_{\text{opt}}$ and $D_{\text{opt}}$.

<details><summary>solution</summary>

substitute $D = C/(6N)$: $f(N) = AN^{-\alpha} + B(6N/C)^{\beta}$. setting $f'(N) = 0$: $\alpha A N^{-\alpha - 1} = \beta B (6/C)^\beta N^{\beta - 1}$, so $N^{\alpha+\beta} = \frac{\alpha A}{\beta B}(C/6)^\beta$ and $N_{\text{opt}} = G(C/6)^{\beta/(\alpha+\beta)}$ with $G = (\alpha A/(\beta B))^{1/(\alpha+\beta)}$. then $D_{\text{opt}} = (C/6)/N_{\text{opt}} = G^{-1}(C/6)^{\alpha/(\alpha+\beta)}$. at the optimum the two terms are in a fixed ratio, $\alpha A N^{-\alpha} = \beta B D^{-\beta}$: you spend compute where the marginal return is equal.

</details>

3. **why an offset bends the curve.** for $y = ax^{-b} + c$, compute the local exponent $-\frac{d\log y}{d\log x}$ as a function of $y$. what happens as $y \to c$?

<details><summary>solution</summary>

$\frac{dy}{dx} = -abx^{-b-1}$, so $-\frac{d\log y}{d\log x} = \frac{x}{y}\,abx^{-b-1} = b\,\frac{y - c}{y}$. far above the floor it's close to $b$; as $y \to c$ it goes to 0. a measured slope that falls from 0.097 to 0.043 is consistent with a floor, but equally with anything else that slows the larger models, such as a fixed step budget (chapter 18's case); the local exponent alone can't tell the two apart.

</details>

4. **three points, three parameters.** show that through three points with $x_1 < x_2 < x_3$ and $y$ decreasing and convex in $\log x$ (bending upward), a curve $ax^{-b} + c$ with $c \ge 0$ can usually be fitted exactly. why does that make the fit useless as a test of the power-law form, and how many points would you want?

<details><summary>solution</summary>

for a fixed $b$, $a$ and $c$ solve a 2×2 linear system from two of the points; the third gives one equation in $b$, which (for a convex decreasing sequence) has a solution, so the three free parameters absorb the three numbers. a zero residual is guaranteed and says nothing about the form. you need several more points than parameters, spread over a wide range, and ideally a held-out larger point (the lab's protocol) to test extrapolation. for chapter 18's numbers the exact fit gives $c \approx 1.97$ bits with $b \approx 0.61$: a huge exponent and a floor just under the last point, which is what three bending points force, not a measurement of anything.

</details>

## experiment

the lab trains six of your chapter-18 GPTs, from 3 thousand to 790 thousand non-embedding parameters, for the same number of steps with two seeds each, fits power laws with and without an offset to the five smallest, and checks each fit's prediction for the largest. then it fixes one compute budget and spends it on five sizes, small models for many steps and large ones for few, to see where the best trade-off lies and how it compares with the 20-tokens rule. finally it refits chapter 18's capstone numbers and asks what Chinchilla's rule would have done with their compute.

## check your understanding

1. two fits describe your five points equally well, one with an offset and one without. they disagree by 0.1 nats at 100× the largest size. what's the cheapest experiment that tells them apart?

<details><summary>answer</summary>

train one model that is a few times larger than your largest (not 100×) and check which fit predicted it. the disagreement between the fits grows with the extrapolation distance, so a point just beyond the range already separates them a little, and it's the only kind of evidence that tests extrapolation rather than interpolation.

</details>

2. why are scaling-law fits done in log space?

<details><summary>answer</summary>

the data span decades in $x$ and the quantity of interest is relative improvement. in log space a power law is linear (so the no-offset fit is ordinary least squares with an exact solution), every point counts by its relative error, and the residuals are closer to homoscedastic (the noise in a loss is roughly proportional to its size). in linear space the smallest models' large losses dominate.

</details>

3. Kaplan et al. recommended spending most extra compute on parameters; Hoffmann et al. recommended equal growth. name one methodological reason they disagreed.

<details><summary>answer</summary>

Kaplan et al. trained every model with a learning-rate schedule of the same length and read off intermediate checkpoints as if they were finished runs of shorter length. a run cut short in the middle of its cosine schedule hasn't had its learning rate decayed, so its loss is higher than a properly finished short run's. this penalizes small-data configurations, i.e. it makes long training look less useful and large models look relatively better. Hoffmann et al. matched the schedule to each run's length.

</details>

4. why exclude embeddings from $N$, but count an untied output head?

<details><summary>answer</summary>

an embedding is a lookup: each token reads one row, so it adds parameters without adding compute per token, and models with very different vocabularies or widths fall onto one curve only when it's excluded. an untied head is a dense $d \times V$ matrix multiplied for every token, exactly like any other layer. a tied head shares the embedding's tensor, so it's excluded with it.

</details>

5. your lab's IsoFLOP profile is monotone: the smallest model wins. what are two different explanations, and what run would separate them?

<details><summary>answer</summary>

(1) the optimum really is below the smallest size at this budget (small character-level models want many more tokens per parameter than 20): adding an even smaller model would show the loss rising again. (2) the large models were handicapped by the recipe: a learning rate and warmup tuned for small models, too few steps to get past the early transient. tuning the learning rate per size (or using μP, topic 06) at the same budget would separate the two.

</details>

6. Chinchilla-optimal models are compute-optimal for *training*. why do practitioners train smaller models far longer than 20 tokens per parameter?

<details><summary>answer</summary>

because training happens once and inference happens billions of times, and inference cost scales with $N$. a smaller model trained on more data reaches the same loss at a higher training cost and a much lower serving cost. the scaling law still tells you the price of that trade: how much extra training compute buys the loss of a model twice as large.

</details>

## common bugs

- **fitting in linear space** (`curve_fit(lambda x, a, b, c: a * x**-b + c, ...)`) and reporting the exponent: the small models' points dominate, and the answer changes with the units of $y$.
- **letting $c$ go negative or above $\min y$**: the log of a negative prediction is NaN, or the fit "explains" the data with a floor below zero.
- **a fixed starting guess for the offset fit**: the optimizer stops in a valley far from the optimum. profile over $c$ first.
- **counting a tied head twice** (iterating over `state_dict()` instead of `parameters()`), or counting buffers.
- **comparing runs at intermediate checkpoints** of a fixed-length schedule as if they were finished shorter runs.
- **mixing log bases** in local exponents: fine as long as numerator and denominator use the same one.

## going deeper

- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu & Amodei (2020), "Scaling laws for neural language models". [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022), "Training compute-optimal large language models" (Chinchilla), *NeurIPS*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Besiroglu, Erdil, Barnett & You (2024), "Chinchilla scaling: a replication attempt". [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
- Hestness, Narang, Ardalani, Diamos, Jun, Kianinejad, Patwary, Yang & Zhou (2017), "Deep learning scaling is predictable, empirically". [arXiv:1712.00409](https://arxiv.org/abs/1712.00409)
- Henighan et al. (2020), "Scaling laws for autoregressive generative modeling". [arXiv:2010.14701](https://arxiv.org/abs/2010.14701)
- OpenAI (2023), "GPT-4 technical report", section 3 (predictable scaling). [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
- Bahri, Dyer, Kaplan, Lee & Sharma (2024), "Explaining neural scaling laws", *PNAS* 121(27). [arXiv:2102.06701](https://arxiv.org/abs/2102.06701)
- Sardana, Portes, Doubov & Frankle (2024), "Beyond Chinchilla-optimal: accounting for inference in language model scaling laws", *ICML*. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)

## connections

- ← builds on: least squares (ch 08 · 01), constrained optimization (ch 06 · 05), your GPT, `train_lm` and chapter 18's capstone (ch 18), FLOP counting (ch 02).
- → used in: topic 06 (μP makes the hyperparameters of a scaling study transfer across sizes), the capstone (fitting curves to a theory figure); FLOPs and memory budgets (ch 26), the LLM pipeline (ch 27), and the research milestones, where a scaling curve is often the cheapest convincing experiment.
