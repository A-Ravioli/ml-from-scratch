# from linear models to neurons

## why this matters

every model in the classical part of this course computes something linear in a fixed set of features: $w^\top x + b$, maybe passed through a sigmoid. that's powerful when someone has designed the right features, and helpless when they haven't. the textbook example is tiny: two binary inputs, and the label is 1 when exactly one of them is 1 (XOR). no line in the plane puts $(0, 1)$ and $(1, 0)$ on one side and $(0, 0)$ and $(1, 1)$ on the other. Minsky and Papert's 1969 book made this limitation of single-layer perceptrons famous, and it's often blamed for a decade of lost interest in neural networks.

the fix is to **learn the features**: put a layer of simple units between input and output, each computing a linear function followed by a nonlinearity, and let the output be linear in *those*. that's a multilayer perceptron (MLP), the network every later chapter builds on. this topic builds one from NumPy parts, shows exactly why the nonlinearity is essential (without it, ten layers are one layer), wires an XOR network by hand, and trains one with gradients you derive yourself. that last part is deliberately a bit painful: by the end you'll want a machine that derives gradients for you, which is what topics 02 and 03 build.

## warm-up: predict first

1. you stack five linear layers (no activation functions) of width 100. can this model represent anything a single linear layer can't?
2. how many hidden ReLU units does a network need to compute XOR exactly on the four corners?
3. you initialize every weight of a network to zero and train with gradient descent. what happens?
4. the network has enough hidden units to solve XOR. does gradient descent always find a solution?
5. a hidden layer of 2 tanh units maps the four XOR corners to four points in $\mathbb R^2$. what has to be true of those four points for the output layer to work?

## prerequisites check

- [ ] you can multiply matrices and know $(AB)^\top = B^\top A^\top$ (→ ch 02 · topic 02)
- [ ] you know the logistic sigmoid, the binary cross-entropy, and why "prediction minus target" is its gradient (→ ch 02 · topic 07)
- [ ] you can compute softplus and sigmoid without overflow (→ ch 00 · topic 03)
- [ ] you can check a gradient with `mlfs.numgrad` (→ ch 02 · topic 07)
- [ ] you know gradient descent and how the learning rate matters (→ ch 06 · topic 02)

## core ideas

### idea 1: a layer of neurons

a **neuron** computes a weighted sum of its inputs plus a bias, then applies a fixed nonlinear function: $a = \phi(w^\top x + b)$. a **layer** is many neurons reading the same input, so the weights stack into a matrix. for a batch $X$ of shape $(n, \text{in})$ and $W$ of shape $(\text{out}, \text{in})$ (row $j$ holds neuron $j$'s weights),

$$Z = XW^\top + b \quad (n \times \text{out}), \qquad A = \phi(Z).$$

storing $W$ as (out, in) is PyTorch's convention, and the course uses it everywhere from here on (`mlfs.nn.Linear` in topic 04 too). the bias $b$ of shape (out,) is broadcast over the batch.

the two activation functions you need now:

- **ReLU**, $\phi(z) = \max(z, 0)$: piecewise linear, cheap, and the default in modern networks;
- **sigmoid**, $\sigma(z) = 1/(1 + e^{-z})$: squashes to $(0, 1)$, used as the output of a binary classifier. compute it as $\exp(-\operatorname{logaddexp}(0, -z))$ so that $z = \pm 1000$ doesn't overflow.

(tanh, $\sigma$'s centred cousin with range $(-1, 1)$, is `np.tanh` and appears in stage 4.)

> **▶ build: stage 1.** implement `linear_layer(X, W, b)`, `relu(z)` and `sigmoid(z)`.

### idea 2: stacking layers, and why the nonlinearity matters

an MLP applies layers in sequence: $h_1 = \phi(XW_1^\top + b_1)$, $h_2 = \phi(h_1W_2^\top + b_2)$, …, with **no** activation after the last layer (its output is a score, a logit, or a regression value). the widths in between are free choices.

now remove the activations. two linear layers compose to

$$(XW_1^\top + b_1)W_2^\top + b_2 = X(W_2W_1)^\top + (W_2b_1 + b_2),$$

which is one linear layer with $W = W_2W_1$ and $b = W_2b_1 + b_2$. by induction, any stack of linear layers is a single linear layer (warm-up 1). depth adds parameters but not expressive power: the decision boundary of a 100-layer linear classifier is still a straight line. (stacking can even *reduce* what's representable: a narrow middle layer forces the product $W_2W_1$ to have low rank.) every bit of the extra power of a deep network comes from the nonlinearities between its layers.

> **🔮 predict:** you compute a deep linear network's output directly, and through its collapsed $(W, b)$. how large will the difference be for 10 layers?

> **▶ build: stage 2.** implement `mlp_forward(X, params, activation=relu)` for `params = [(W1, b1), ..., (WL, bL)]`, and `collapse_linear(params)`, returning the single `(W, b)` equivalent to the stack *without* activations.

### idea 3: XOR by hand

why can't a linear classifier do XOR? suppose $w^\top x + b$ is positive at $(0, 1)$ and $(1, 0)$ and negative at $(0, 0)$ and $(1, 1)$. adding the first two conditions gives $w_1 + w_2 + 2b > 0$; adding the last two gives $w_1 + w_2 + 2b < 0$. contradiction (on paper 1).

a hidden layer fixes it by **changing the representation**. take two ReLU units that both look at $s = x_1 + x_2$, the number of ones:

$$h_1 = \operatorname{relu}(x_1 + x_2), \qquad h_2 = \operatorname{relu}(x_1 + x_2 - 1).$$

$h_1$ counts the ones; $h_2$ fires only when both are on. the output $y = h_1 - 2h_2$ is then $0, 1, 1, 0$ on the four corners (the worked example). in the hidden space $(h_1, h_2)$ the corners land at $(0, 0)$, $(1, 0)$, $(1, 0)$ and $(2, 1)$: the two "1" corners have merged into one point, and a straight line now separates the classes (warm-up 5). that's the general story of a hidden layer: it maps the data to a space where the output layer's linear rule works. two hidden units are enough (warm-up 2), and one isn't: a single hidden unit's output is a monotone function of one linear combination $w^\top x$, so its sign pattern can't single out the middle values of $s$.

> **▶ build: stage 3.** implement `xor_params()`, returning `[(W1, b1), (W2, b2)]` with shapes (2, 2), (2,), (1, 2), (1,) such that `mlp_forward(X, params)` gives exactly 0, 1, 1, 0 on the four corners.

### idea 4: gradients by hand

hand-wiring doesn't scale; we want to *learn* $W_1, b_1, W_2, b_2$ from data by gradient descent. take a network with one hidden layer of tanh units and a single logit output, trained with the mean binary cross-entropy:

$$Z_1 = XW_1^\top + b_1,\quad H = \tanh(Z_1),\quad z_2 = HW_2^\top + b_2,\quad L = \frac1n\sum_i\big[\operatorname{softplus}(z_{2,i}) - y_iz_{2,i}\big].$$

(softplus$(z) = \log(1 + e^z)$, computed with `np.logaddexp(0, z)`; this is the cross-entropy of $\sigma(z)$ written so it never takes a log of 0.) the chain rule, applied from the loss backwards, one step per line:

$$\bar z_2 = \frac{\sigma(z_2) - y}{n}, \qquad \bar W_2 = \bar z_2^\top H,\quad \bar b_2 = \textstyle\sum_i \bar z_{2,i},$$

$$\bar H = \bar z_2W_2, \qquad \bar Z_1 = \bar H \odot (1 - H^2), \qquad \bar W_1 = \bar Z_1^\top X,\quad \bar b_1 = \textstyle\sum_i \bar Z_{1,i}.$$

here $\bar v$ means $\partial L/\partial v$, with $v$'s shape. each line uses only quantities from the forward pass and the line above: the output error flows backwards through the output layer, through the tanh (multiplying by its derivative $1 - \tanh^2$), and into the first layer. that's **backpropagation**, written out for one specific network. note the pattern of the weight gradients, $\bar W = (\text{error at the layer's output})^\top(\text{the layer's input})$: chapter 02's rank-one $ab^\top$, summed over the batch.

the same function also supports `activation="identity"` ($H = Z_1$, so the $1 - H^2$ factor becomes 1), which the lab uses to train a network with no nonlinearity.

> **▶ build: stage 4.** implement `two_layer_loss_and_grads(X, y, params, activation="tanh")`, returning the loss and `[(dW1, db1), (dW2, db2)]`. check every gradient against finite differences before you trust it (the tests do).

### idea 5: training, and breaking symmetry

gradient descent repeats $\theta \leftarrow \theta - \eta\nabla L(\theta)$ for every parameter. two things about the **initialization** matter even in this tiny network:

- **not zero.** with $W_1 = 0$ and $W_2 = 0$, every hidden unit computes the same thing, $H = 0$, and the gradients of $W_1$ and $W_2$ are zero (the stage 4 hand example): gradient descent can't move them, ever (warm-up 3). more generally, any two hidden units that start identical get identical gradients and stay identical forever. random initialization **breaks the symmetry** so that units can learn different features.
- **reproducible.** draw $W_1, b_1, W_2, b_2$ from a standard normal, in that order, from one `rng`, so a seed determines the run.

even with a good initialization, the loss of a neural network isn't convex: gradient descent can stall in a poor local minimum or a flat region. with exactly enough hidden units for the task it often does; with a few more it almost never does (warm-up 4, and the lab's surprise).

> **🔮 predict:** with 2 hidden units (enough to represent XOR exactly), what fraction of random initializations will gradient descent turn into a solution?

> **▶ build: stage 5.** implement `train_two_layer(X, y, hidden, *, lr=1.0, n_steps=2000, activation="tanh", rng=None)`: full-batch gradient descent from the standard normal init, returning the final parameters and the list of losses (one per step, each computed *before* that step's update).

## worked example

the hand-wired XOR network, evaluated on all four corners. $W_1 = \begin{pmatrix}1&1\\1&1\end{pmatrix}$, $b_1 = (0, -1)$, $W_2 = (1, -2)$, $b_2 = 0$.

| $x$ | $Z_1 = W_1x + b_1$ | $h = \operatorname{relu}(Z_1)$ | $y = W_2h + b_2$ |
|---|---|---|---|
| $(0, 0)$ | $(0, -1)$ | $(0, 0)$ | $0$ |
| $(0, 1)$ | $(1, 0)$ | $(1, 0)$ | $1$ |
| $(1, 0)$ | $(1, 0)$ | $(1, 0)$ | $1$ |
| $(1, 1)$ | $(2, 1)$ | $(2, 1)$ | $2 - 2 = 0$ |

the outputs are exactly XOR. now drop the ReLU: `collapse_linear` gives $W = W_2W_1 = (-1, -1)$ and $b = W_2b_1 + b_2 = 2$, so the "same" network without its nonlinearity computes $2 - x_1 - x_2$: the values $2, 1, 1, 0$, which aren't XOR. the tests check both: your `xor_params` must produce exactly $(0, 1, 1, 0)$, and its collapsed linear version must not.

## on paper

1. **XOR is not linearly separable.** show that no $(w, b)$ has $w^\top x + b > 0$ exactly on $\{(0, 1), (1, 0)\}$ among the four corners.

<details><summary>solution</summary>

the conditions are $w_2 + b > 0$, $w_1 + b > 0$, $b < 0$ (at the origin, $\le 0$ works too) and $w_1 + w_2 + b < 0$ (or $\le 0$). adding the first two: $w_1 + w_2 + 2b > 0$. adding the last two: $w_1 + w_2 + 2b < 0$. contradiction. geometrically: the midpoint of $(0, 1)$ and $(1, 0)$ is $(\tfrac12, \tfrac12)$, which is also the midpoint of $(0, 0)$ and $(1, 1)$. a linear function's value at a midpoint is the average of its values at the endpoints, so it can't be positive when averaged one way and negative the other.

</details>

2. **collapsing $L$ layers.** prove by induction that $L$ linear layers equal one, and give the recursion for $(W, b)$.

<details><summary>solution</summary>

base case: one layer is one layer. step: if layers $1..k$ equal $(W, b)$, then applying layer $k + 1$ gives $(XW^\top + b)W_{k+1}^\top + b_{k+1} = X(W_{k+1}W)^\top + (W_{k+1}b + b_{k+1})$. so $W \leftarrow W_{k+1}W$ and $b \leftarrow W_{k+1}b + b_{k+1}$. order matters: the later layer multiplies on the left.

</details>

3. **derive the backward pass of idea 4.** start from $\partial L/\partial z_{2,i} = (\sigma(z_{2,i}) - y_i)/n$ and use differentials (ch 02 · topic 07) to get $\bar W_2$, $\bar H$, $\bar Z_1$ and $\bar W_1$.

<details><summary>solution</summary>

$dL = \sum_i\bar z_{2,i}\,dz_{2,i} = \bar z_2^\top dz_2$. from $z_2 = HW_2^\top + b_2$: $dz_2 = dH\,W_2^\top + H\,dW_2^\top + db_2$. the $dW_2$ term gives $\bar z_2^\top H\,dW_2^\top = \operatorname{tr}(dW_2\,H^\top\bar z_2)$, so $\bar W_2 = \bar z_2^\top H$ (shape (1, hidden)). the $dH$ term gives $\bar H = \bar z_2W_2$ (shape $(n, \text{hidden})$). $H = \tanh(Z_1)$ elementwise, so $dH = (1 - H^2)\odot dZ_1$ and $\bar Z_1 = \bar H\odot(1 - H^2)$. finally $Z_1 = XW_1^\top + b_1$ gives $\bar W_1 = \bar Z_1^\top X$ and $\bar b_1 = \sum_i\bar Z_{1,i}$, by the same argument as for $W_2$.

</details>

4. **symmetry.** two hidden units start with identical incoming weights, biases and outgoing weights. show that they receive identical gradients, and hence stay identical under gradient descent.

<details><summary>solution</summary>

if units $j$ and $k$ have the same incoming weights and bias, they have the same $Z_1$ column, the same $H$ column, and, with equal outgoing weights, the same column of $\bar H = \bar z_2W_2$. so their columns of $\bar Z_1$ are equal, and so are their rows of $\bar W_1 = \bar Z_1^\top X$, their entries of $\bar b_1$, and their entries of $\bar W_2 = \bar z_2^\top H$. equal parameters and equal gradients give equal parameters after the step. by induction they're identical forever: the network effectively has one unit fewer.

</details>

## experiment

`lab.py` runs three experiments on your functions:

1. how close a deep linear network is to its collapsed single layer, and what a network without an activation learns on a "ring" dataset.
2. how often gradient descent solves XOR with 1, 2, 3, 4 and 8 hidden units, over 50 random initializations each.
3. where the four XOR corners land in hidden space, for your hand-wired network and for a trained one.

## check your understanding

1. why does the last layer of an MLP usually have no activation?

<details><summary>answer</summary>

its output is a score: a logit (for a sigmoid or softmax inside the loss) or a real-valued prediction. an activation would restrict its range (ReLU can't output negatives, which a regression may need) or duplicate what the loss already applies (a sigmoid followed by a cross-entropy that applies another sigmoid).

</details>

2. a deep linear network and a single linear layer represent the same functions. are they equivalent in every respect?

<details><summary>answer</summary>

no: they represent the same set of functions (if no middle layer is narrower than the input and output), but they're different parameterizations, so gradient descent follows different paths and can behave differently (deep linear networks are a standard model for studying training dynamics). and a narrow middle layer restricts the product to low rank, so the deep version can represent *fewer* functions.

</details>

3. why is ReLU's output at the hand-wired network's hidden point for $(1, 1)$ equal to $(2, 1)$, and what would go wrong with $b_1 = (0, -0.5)$?

<details><summary>answer</summary>

$Z_1 = (2, 1)$ and both are positive. with $b_1 = (0, -0.5)$, the corners $(0, 1)$ and $(1, 0)$ would give $h_2 = 0.5$ instead of 0, so their output would be $1 - 2 \cdot 0.5 = 0$: wrong. the bias $-1$ is chosen so that $h_2$ switches on only when both inputs are on.

</details>

4. the loss is a *mean* over the $n$ examples. what changes in the gradients if you use the sum instead?

<details><summary>answer</summary>

every gradient is $n$ times larger, so the same learning rate takes $n$-times-larger steps, and a learning rate tuned for one dataset size breaks on another. with the mean, the gradient's scale doesn't depend on $n$.

</details>

5. why compute the cross-entropy as $\operatorname{softplus}(z) - yz$ rather than $-y\log\sigma(z) - (1 - y)\log(1 - \sigma(z))$?

<details><summary>answer</summary>

for a confidently wrong prediction ($z = 800$, $y = 0$) the second form computes $\log(1 - \sigma(800)) = \log 0 = -\infty$ in floating point. the first gives $\operatorname{softplus}(800) = 800$, the correct (large but finite) loss, and its gradient $\sigma(z) - y$ is exact.

</details>

6. with 1 hidden unit, what's the best XOR accuracy you can get, and why?

<details><summary>answer</summary>

3 out of 4. the output is a monotone function of a single linear combination $w^\top x + b$ (tanh and the output layer are monotone), so its decision region is a half-plane, like a linear classifier's, and a half-plane gets at most 3 corners right.

</details>

## common bugs

- **$(in, out)$ weights.** computing `X @ W + b` with $W$ stored as (out, in). symptom: a shape error for non-square layers, or silently wrong outputs for square ones.
- **an activation after the last layer.** symptom: a ReLU output can't go negative, so a regression can't fit negative targets, and logits are clipped at 0.
- **collapsing in the wrong order.** symptom: `collapse_linear` works for two layers of compatible shapes in either order, but disagrees with the stack for square layers. the later layer multiplies on the left: $W_2W_1$.
- **sum instead of mean in the loss.** symptom: gradients $n$ times too large; finite differences of the mean loss disagree by exactly a factor of $n$.
- **forgetting the activation's derivative.** symptom: the gradients of $W_1$ and $b_1$ are wrong (they'd be right for a linear network), and gradient checks fail only for the first layer.
- **zero initialization.** symptom: the loss never moves from $\log 2$.

## going deeper

- Minsky & Papert (1969), *Perceptrons: An Introduction to Computational Geometry*, MIT Press. the analysis of what single-layer perceptrons can't compute, XOR included.
- Rumelhart, Hinton & Williams (1986), "Learning internal representations by error propagation", in Rumelhart, McClelland & the PDP Research Group (eds.), *Parallel Distributed Processing*, vol. 1, chapter 8, MIT Press. backpropagation for MLPs, with XOR as the first worked example.
- Rumelhart, Hinton & Williams (1986), "Learning representations by back-propagating errors", *Nature* 323: 533–536. the short version that made backpropagation widely known. [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, §6.1 (learning XOR) and §6.5 (back-propagation). [deeplearningbook.org](https://www.deeplearningbook.org)
- Nair & Hinton (2010), "Rectified linear units improve restricted Boltzmann machines", *ICML*. the paper that popularized ReLU.
- Saxe, McClelland & Ganguli (2014), "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", *ICLR*. why deep *linear* networks still train differently from shallow ones. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)

## connections

- ← builds on: matrix products (ch 02 · topic 02), the logistic loss and its gradient (ch 02 · topic 07), stable sigmoid and softplus (ch 00 · topic 03), gradient descent (ch 06 · topic 02), `as_rng` (ch 00 · topic 05).
- → used in: the scalar autodiff engine that derives these gradients automatically (topic 02), the tensor engine (topic 03), `mlfs.nn.Linear` with the same layout (topic 04), universal approximation (topic 05), and every network in the rest of the course.
