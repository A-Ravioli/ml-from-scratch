# your first ML model

## why this matters

here's the whole idea of machine learning in one sentence: **choose a family of functions, measure how badly each one fits the data, and search for the one that fits best.** everything else, from logistic regression to GPT, is a more elaborate version of those three steps: a bigger family of functions (the model), a cleverer measure of badness (the loss), and a faster search (the optimizer).

in this topic you do all three with the simplest model there is: a straight line through points. given past houses' sizes and prices, predict a new house's price. you'll measure badness with the **mean squared error**, derive its **gradient** by hand with the rules from topic 02, and search with **gradient descent**, the algorithm that trains essentially every neural network. you'll also find the best line *exactly*, by algebra, and use that to check your gradient descent. finally you'll meet the first practical lesson of training models: the units you measure your data in can make gradient descent fast or hopelessly slow, and a one-line fix, **standardizing**, solves it. the chapter capstone then scales all of this up to real house prices with eight features.

## warm-up: predict first

1. a line predicts 3, 3 and 4 for three houses whose true prices are 2, 4 and 5. what's its mean squared error?
2. the loss gets bigger when you increase the slope $w$ a little. should gradient descent increase or decrease $w$?
3. you double the learning rate. does training always get twice as fast?
4. is there a formula for the best line, or do you have to search for it?
5. you measure house sizes in square centimeters instead of square meters (numbers $10{,}000$ times bigger). does the best line change? does gradient descent behave the same?

## prerequisites check

- [ ] you know a line is $y = wx + b$, with slope $w$ and intercept $b$ (high school algebra)
- [ ] you can differentiate with the power rule and the chain rule (→ ch 01 · topic 02)
- [ ] you know a smooth function's minimum is where its derivative is 0, and what the second derivative says about curvature (→ ch 01 · topic 02, idea 5)
- [ ] you can compute means and standard deviations with NumPy (→ ch 01 · topic 05)

## core ideas

### idea 1: a model and a loss

you have $n$ examples: inputs $x_1, \ldots, x_n$ (house sizes) and targets $y_1, \ldots, y_n$ (prices). the **model** is a line with two **parameters**, a slope $w$ and an intercept $b$:

$$\hat y = wx + b.$$

(the hat means "predicted".) different $(w, b)$ give different lines. to choose among them, measure how badly a line fits. the **residual** $r_i = \hat y_i - y_i = wx_i + b - y_i$ is how far off the prediction for example $i$ is. the **mean squared error** averages the squared residuals:

$$L(w, b) = \frac1n\sum_{i=1}^n (wx_i + b - y_i)^2.$$

$L$ is the **loss**: a single number that says how bad the line $(w, b)$ is, with $0$ meaning perfect. for warm-up 1, the residuals are $1, -1, -1$, so $L = \frac{1 + 1 + 1}{3} = 1$.

why square the residuals? squaring makes every miss count as positive (so overestimates and underestimates can't cancel), punishes big misses much more than small ones, and gives a smooth function of $w$ and $b$ that you can differentiate. (absolute values would also work, but have a corner at zero, like $|x|$ in topic 02's lab.) and why average rather than sum? so the loss doesn't grow just because you have more data, which keeps the learning rate meaningful across dataset sizes.

the loss is a function of the *parameters*, not of $x$: the data is fixed, and you're asking which $(w, b)$ makes $L$ smallest. picture $L$ as a landscape over the $(w, b)$ plane. for the mean squared error that landscape is a bowl, and the best line sits at its bottom.

> **▶ build: stage 1.** implement `predict(x, w, b)` and `mse_loss(x, y, w, b)`.

### idea 2: the gradient of the loss

to walk downhill on the loss landscape, you need to know which way is down. $L$ depends on two variables, so ask about one at a time. the **partial derivative** $\frac{\partial L}{\partial w}$ is the ordinary derivative of $L$ with respect to $w$, treating $b$ (and the data) as constants: how fast the loss changes when you nudge only the slope. $\frac{\partial L}{\partial b}$ is the same for the intercept.

each term of $L$ is a square of something, $(r_i)^2$ with $r_i = wx_i + b - y_i$. by the chain rule, its derivative is $2r_i$ times the derivative of $r_i$. the derivative of $r_i$ with respect to $w$ is $x_i$, and with respect to $b$ it's $1$. so

$$\frac{\partial L}{\partial w} = \frac2n\sum_{i=1}^n r_i\,x_i, \qquad \frac{\partial L}{\partial b} = \frac2n\sum_{i=1}^n r_i.$$

the pair $\nabla L = \left(\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}\right)$ is the **gradient**. it points in the direction in which $L$ increases fastest (uphill), and its size says how steep the slope is. read the formulas: if the line predicts too high on average ($r_i > 0$), $\frac{\partial L}{\partial b} > 0$, so increasing $b$ makes things worse. if the big residuals are at big $x$, $\frac{\partial L}{\partial w}$ is big, and the slope is what needs fixing. (warm-up 2: if the loss increases with $w$, the partial derivative is positive, and you should *decrease* $w$.)

at the best line, the bottom of the bowl, both partial derivatives are zero. and because you derived these by hand, check them: topic 02's `check_derivative` idea applies to each partial derivative separately, and the tests do exactly that.

> **▶ build: stage 2.** implement `mse_gradient(x, y, w, b)`, returning $\left(\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}\right)$. derive it on paper first (on paper, problem 1).

### idea 3: gradient descent

the gradient points uphill, so step the other way. **gradient descent** starts from a guess $(w_0, b_0)$ and repeats

$$w \leftarrow w - \eta\,\frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b},$$

where $\eta$ (eta) is the **learning rate**, the step size. both partial derivatives are computed at the *current* point, then both parameters move together. record the loss after each step: plotted against the step number, that's the **loss curve**, the first thing anyone looks at when training a model.

how big should $\eta$ be? look at one parameter at a time. near its minimum $w^*$, the loss is a parabola, $L \approx L^* + \frac a2(w - w^*)^2$, whose curvature $a$ is its second derivative. its derivative is $a(w - w^*)$, so one step gives

$$w_{\text{new}} - w^* = (w - w^*) - \eta\,a\,(w - w^*) = (1 - \eta a)(w - w^*).$$

every step multiplies the distance to the minimum by $(1 - \eta a)$. so:

- $\eta$ **too small** ($\eta a \ll 1$): the factor is just below 1, and progress is painfully slow.
- $\eta a = 1$: one step lands exactly on the minimum.
- $1 < \eta a < 2$: the factor is negative, so $w$ overshoots to the other side every step, but by less each time. it zigzags in.
- $\eta a > 2$: the factor is bigger than 1 in size, so every step overshoots by *more*. the loss explodes.

the maximum safe learning rate is $\eta < 2/a$, set by the curvature. doubling a learning rate that's already near this limit doesn't double the speed; it makes training diverge (warm-up 3). with two parameters, each direction has its own curvature, and the *steepest* one sets the limit for both.

> **🔮 predict:** for data whose inputs are standardized (mean 0, standard deviation 1), the curvature of the mean squared error is $a = 2$ in both directions. which learning rates converge without zigzagging? which zigzag but converge? which diverge?

> **▶ build: stage 3.** implement `gradient_descent(x, y, *, lr=0.1, n_steps=100, w0=0.0, b0=0.0)`, returning the final $(w, b)$ and the loss curve.

### idea 4: the exact answer

for this model you don't actually have to search (warm-up 4). at the bottom of the bowl both partial derivatives are zero, which gives two equations for two unknowns. setting $\frac{\partial L}{\partial b} = 0$:

$$\frac2n\sum_i(wx_i + b - y_i) = 0 \quad\Longrightarrow\quad w\bar x + b - \bar y = 0 \quad\Longrightarrow\quad b = \bar y - w\bar x,$$

where $\bar x$ and $\bar y$ are the averages. so the best line always passes through the point $(\bar x, \bar y)$. substituting into $\frac{\partial L}{\partial w} = 0$ and simplifying (on paper, problem 2) gives

$$w = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}.$$

this is the **least-squares** or **closed-form** solution. the denominator is zero only when all the $x_i$ are equal, and then no single slope is best: every line through $(\bar x, \bar y)$ fits equally well.

if there's a formula, why bother with gradient descent? because formulas like this are the exception. for logistic regression (chapter 08), neural networks (chapter 14) and almost every other model, setting the gradient to zero gives equations nobody can solve, while gradient descent still works. even for linear models with millions of features, the exact solution means solving a huge linear system (chapter 02 builds the tools), and gradient descent is often cheaper. here, the closed form plays a different role: it's the **ground truth** that tells you whether your gradient descent got it right.

> **▶ build: stage 4.** implement `closed_form_fit(x, y)`.

### idea 5: why units matter, and standardizing

the best line doesn't care what units you use (warm-up 5). measure sizes in square centimeters and the best slope is just 10,000 times smaller. but gradient descent cares enormously. here's why.

the curvature of $L$ in the $w$ direction is its second derivative, $\frac{\partial^2L}{\partial w^2} = \frac2n\sum_i x_i^2$, and in the $b$ direction it's $\frac{\partial^2L}{\partial b^2} = 2$. house sizes around 150 m² make the $w$-curvature about $2 \times 150^2 = 45{,}000$, so idea 3 says the learning rate must stay below $2/45{,}000 \approx 4 \times 10^{-5}$. at that learning rate, the $b$ direction (curvature 2) moves by a factor of $1 - 4 \times 10^{-5} \times 2 \approx 0.99992$ per step: it would take about a hundred thousand steps to make real progress on the intercept. (that's the simplified picture. with raw $x$ the two directions are also coupled, because $\frac{\partial^2L}{\partial w\,\partial b} = 2\bar x \neq 0$: the valley is tilted, and its flattest direction is flatter still, with a curvature of about 0.27 instead of 2 for this kind of data. experiment 3 of the lab draws it.) the loss landscape is a long, narrow valley, steep across and nearly flat along, and gradient descent crawls.

the fix is to **standardize** the input: subtract its mean and divide by its standard deviation,

$$z = \frac{x - \bar x}{s_x}.$$

now $z$ has mean 0 and standard deviation 1, both curvatures are 2, the valley is a round bowl, and a learning rate like $0.1$ works in a few dozen steps. the model hasn't lost anything: a line in $z$ is a line in $x$. if $\hat y = w_zz + b_z$, substitute $z = (x - \bar x)/s_x$ and expand:

$$\hat y = \frac{w_z}{s_x}\,x + \left(b_z - \frac{w_z\bar x}{s_x}\right),$$

so $w = w_z/s_x$ and $b = b_z - w_z\bar x/s_x$. standardizing features is the first step of almost every ML pipeline (chapter 07 builds it as a reusable transformer), and this is the reason.

> **🔮 predict:** you switch from square meters to square centimeters, making every $x$ ten thousand times bigger. by what factor must the largest safe learning rate shrink? what happens to the number of steps the intercept needs?

> **▶ build: stage 5.** implement `standardize(x)` and `unstandardize_params(w_z, b_z, mean, std)`.

## worked example

three houses: $x = (1, 2, 3)$, $y = (2, 4, 5)$.

**the loss at $(w, b) = (1, 1)$.** predictions $2, 3, 4$; residuals $0, -1, -1$; $L = \frac{0 + 1 + 1}{3} = 0.6667$.

**the gradient there.**

$$\frac{\partial L}{\partial w} = \frac23(0 \cdot 1 + (-1) \cdot 2 + (-1) \cdot 3) = -\frac{10}3, \qquad \frac{\partial L}{\partial b} = \frac23(0 - 1 - 1) = -\frac43.$$

both are negative: increasing $w$ or $b$ lowers the loss, which makes sense, since the line predicts too low.

**one gradient descent step** with $\eta = 0.1$: $w = 1 + 0.1 \cdot \frac{10}3 = 1.3333$ and $b = 1 + 0.1 \cdot \frac43 = 1.1333$. the loss drops from $0.6667$ to $0.0919$.

**the exact answer.** $\bar x = 2$, $\bar y = \frac{11}3$. deviations $x_i - \bar x = -1, 0, 1$ and $y_i - \bar y = -\frac53, \frac13, \frac43$. so $w = \frac{\frac53 + 0 + \frac43}{1 + 0 + 1} = \frac32 = 1.5$ and $b = \frac{11}3 - 1.5 \cdot 2 = \frac23 \approx 0.6667$, with loss $\frac1{18} \approx 0.0556$. gradient descent from $(0, 0)$ with $\eta = 0.1$ gets within $10^{-4}$ of this in about 200 steps.

the tests check the loss $0.6667$, the gradient $(-\frac{10}3, -\frac43)$, a first step from $(0, 0)$ by hand, and the exact line $(1.5, 0.6667)$.

## on paper

1. **the gradient.** derive $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ for $L = \frac1n\sum_i(wx_i + b - y_i)^2$, term by term with the chain rule.

<details><summary>solution</summary>

the derivative of a sum is the sum of the derivatives, and the $\frac1n$ is a constant. for one term, $\frac{\partial}{\partial w}(wx_i + b - y_i)^2 = 2(wx_i + b - y_i) \cdot x_i$ (outer derivative $2u$, inner derivative $x_i$), and $\frac{\partial}{\partial b}(\ldots)^2 = 2(wx_i + b - y_i) \cdot 1$. so $\frac{\partial L}{\partial w} = \frac2n\sum_ir_ix_i$ and $\frac{\partial L}{\partial b} = \frac2n\sum_ir_i$.

</details>

2. **the closed form.** substitute $b = \bar y - w\bar x$ into $\sum_i(wx_i + b - y_i)x_i = 0$ and solve for $w$.

<details><summary>solution</summary>

with $b = \bar y - w\bar x$, the residual is $w(x_i - \bar x) - (y_i - \bar y)$. so $\sum_i[w(x_i - \bar x) - (y_i - \bar y)]x_i = 0$. since $\sum_i (x_i - \bar x) = 0$ and $\sum_i(y_i - \bar y) = 0$, you can replace the final $x_i$ by $(x_i - \bar x)$ without changing the sum: $w\sum_i(x_i - \bar x)^2 = \sum_i(x_i - \bar x)(y_i - \bar y)$. divide.

</details>

3. **the learning-rate limit.** for $L(w) = \frac a2(w - w^*)^2$, show that gradient descent converges exactly when $0 < \eta < 2/a$. what happens at $\eta = 2/a$?

<details><summary>solution</summary>

$w_{\text{new}} - w^* = (1 - \eta a)(w - w^*)$, so after $t$ steps the distance is $(1 - \eta a)^t$ times the start. it shrinks to 0 exactly when $|1 - \eta a| < 1$, i.e. $0 < \eta a < 2$. at $\eta = 2/a$ the factor is $-1$: $w$ bounces between $w^* \pm (w_0 - w^*)$ forever, never converging and never diverging.

</details>

4. **undoing standardization.** check the conversion $w = w_z/s_x$, $b = b_z - w_z\bar x/s_x$ for $w_z = 2$, $b_z = 5$, $\bar x = 10$, $s_x = 4$, at $x = 18$.

<details><summary>solution</summary>

$z = (18 - 10)/4 = 2$, so $\hat y = 2 \cdot 2 + 5 = 9$. converted: $w = 0.5$, $b = 5 - 2 \cdot 10/4 = 0$, so $\hat y = 0.5 \cdot 18 + 0 = 9$. ✓

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the loss curve for learning rates from tiny to too large, on standardized data, with a slider.
2. the same house-price data with sizes in square meters, trained raw and standardized.
3. the loss landscape itself, as a contour map, with gradient descent's path drawn on it, for raw and standardized inputs.

## check your understanding

1. why is the loss a function of $(w, b)$ and not of $x$?

<details><summary>answer</summary>

the data is fixed during training. what you're choosing is the line, so the loss measures how bad each possible $(w, b)$ is on that fixed data. (at prediction time it's the other way around: $(w, b)$ is fixed and $x$ varies.)

</details>

2. at the start of training, $\frac{\partial L}{\partial b} = 3.2$. what will gradient descent do to $b$, and what does that say about the current predictions?

<details><summary>answer</summary>

decrease it ($b \leftarrow b - 3.2\eta$). $\frac{\partial L}{\partial b} = \frac2n\sum r_i > 0$ means the residuals are positive on average: the line predicts too high.

</details>

3. why must both partial derivatives be computed before either parameter changes?

<details><summary>answer</summary>

the gradient is a property of one point. if you update $w$ first and then compute $\frac{\partial L}{\partial b}$ at the new $w$, you're taking a step in a direction that isn't the gradient anywhere. it often still works, but it's a different algorithm with different behavior, and it won't match the hand calculation or the theory.

</details>

4. (the predict callout in idea 3.) with curvature $a = 2$ in both directions, which learning rates converge smoothly, zigzag, or diverge?

<details><summary>answer</summary>

the factor per step is $1 - 2\eta$. smooth convergence for $0 < \eta < 0.5$ (fastest near $0.5$, where one step lands on the minimum), zigzagging convergence for $0.5 < \eta < 1$, and divergence for $\eta > 1$. the lab's slider shows all three.

</details>

5. your loss curve goes down for 50 steps and then shoots up to `inf`. what's the most likely cause?

<details><summary>answer</summary>

a learning rate above the stability limit $2/a$ for the steepest direction. an unstable direction starts tiny (so the loss falls at first, driven by the other directions) and grows geometrically until it dominates. lower the learning rate, or standardize the inputs.

</details>

6. standardizing changes $x$. why doesn't it change the predictions of the fitted model?

<details><summary>answer</summary>

a line in $z$ is a line in $x$, so the set of possible prediction functions is exactly the same, and the best one is the same. standardizing only changes how the parameters are *written* (and how easy they are to find by gradient descent). `unstandardize_params` converts between the two descriptions.

</details>

7. why does the best line pass through $(\bar x, \bar y)$?

<details><summary>answer</summary>

$\frac{\partial L}{\partial b} = 0$ says the residuals average to zero: $w\bar x + b = \bar y$. so the prediction at $\bar x$ is exactly $\bar y$.

</details>

8. (the predict callout in idea 5.) sizes in cm² instead of m²: how does the safe learning rate change, and what happens to the intercept?

<details><summary>answer</summary>

the $w$-curvature $\frac2n\sum x_i^2$ grows with the *square* of the scale, so it becomes $10^8$ times bigger, and the largest safe learning rate $10^8$ times smaller. the intercept's curvature is still 2, so it now moves $10^8$ times more slowly per step: effectively never. (standardized, the two versions are identical: $z$ doesn't care about units.)

</details>

## common bugs

- **summing instead of averaging.** symptom: the loss and gradient are $n$ times too big, and a learning rate that works on 10 examples diverges on 1,000.
- **a residual with the wrong sign.** using $y - \hat y$ in the gradient but subtracting it anyway. symptom: gradient descent climbs, and the loss grows from the first step.
- **forgetting the factor 2.** symptom: the gradient is half the finite-difference value, and training is mysteriously half as fast as the theory says.
- **updating $w$, then computing $\frac{\partial L}{\partial b}$ with the new $w$.** symptom: the first step doesn't match the hand calculation.
- **an off-by-one loss curve.** recording the loss only after each step loses the starting loss. symptom: `losses` has `n_steps` entries instead of `n_steps + 1`.
- **training on raw, large-valued inputs.** symptom: any learning rate big enough to move the intercept makes the loss explode, and any small enough one barely changes it.

## going deeper

- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, §4.3 (gradient-based optimization) and §5.1.4 (linear regression as a first learning algorithm). free online. [deeplearningbook.org](https://www.deeplearningbook.org)
- James, Witten, Hastie & Tibshirani (2021), *An Introduction to Statistical Learning*, 2nd ed., Springer, chapter 3 (simple linear regression and least squares). free online. [doi:10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
- Cauchy (1847), "Méthode générale pour la résolution des systèmes d'équations simultanées", *Comptes Rendus de l'Académie des Sciences* 25: 536–538. the first description of gradient descent.
- Goh (2017), "Why momentum really works", *Distill*. an interactive tour of gradient descent on narrow valleys like the one in experiment 3, and how to fix them. [doi:10.23915/distill.00006](https://doi.org/10.23915/distill.00006)

## connections

- ← builds on: derivatives, the chain rule and curvature (topic 02), means and standard deviations (topic 05), dot products (topic 04: with many features the prediction is $w \cdot x + b$).
- → used in: the chapter capstone (multivariate gradient descent on house prices), least squares via QR and the normal equations (ch 02 · topic 04), gradients with respect to vectors (ch 02 · topic 07), gradient descent, momentum and learning-rate theory (ch 06), standardization as a preprocessing step (ch 07), and linear and logistic regression (ch 08). every training loop in chapters 14–28 is this loop with a bigger model.
