# 19 · checkpoint: deep learning theory

pass this to finish the chapter, or to skip it if you already know the material. no notes, no code from the chapter, no internet. a pen, paper and a calculator are fine for parts 1 and 2.

## 1. concept check

answer each in two or three sentences, then check.

1. a one-hidden-layer ReLU network with $m$ units and a scalar input: how many linear pieces can it have at most, and how many units does it need to interpolate $n$ points exactly?

<details><summary>answer</summary>

at most $m + 1$ pieces (each unit adds one kink). interpolating $n$ points with the piecewise-linear interpolant needs $n - 1$ units: $f(x) = y_0 + \sum_j c_j\,\text{relu}(x - x_j)$ with $c_j$ the change of slope at knot $j$ (topic 01).

</details>

2. what does Telgarsky's sawtooth show about depth, and what did training on it show?

<details><summary>answer</summary>

the tent map composed $k$ times has $2^k$ pieces and is computed exactly by $k$ layers of width 2 ($6k + 1$ weights), while a shallow network needs about $2^{k-1}$ units even to approximate it: an exponential depth separation in representation. trained from random starts, neither deep nor shallow networks found it (topic 01's lab: relative error 1.0 at $k = 6$): representable is not learnable.

</details>

3. why does a loss-surface plot along random directions need filter normalization?

<details><summary>answer</summary>

layers and filters have very different norms, and ReLU networks are invariant to rescaling one layer by $c$ and the next by $1/c$. a raw Gaussian direction perturbs small-norm filters by a lot and large ones by little, so the picture depends on the parameterization. scaling each filter's direction to that filter's norm makes the plot invariant to those rescalings (Li et al., 2018).

</details>

4. you multiply a ReLU network's first layer by 5 and divide its second by 5. what happens to its predictions, its top Hessian eigenvalue, and its filter-normalized surface?

<details><summary>answer</summary>

predictions are unchanged; the top Hessian eigenvalue changes (topic 02's lab: 4.15 to 23.2), because the Hessian is a derivative with respect to the parameters; the filter-normalized surface is identical. raw sharpness is not a property of the function (Dinh et al., 2017).

</details>

5. what is the edge of stability, and why does full-batch gradient descent end up there?

<details><summary>answer</summary>

gradient descent with step $\eta$ can't stay where the top Hessian eigenvalue exceeds $2/\eta$ (the component along it gets multiplied by $1 - \eta\lambda < -1$). Cohen et al. (2021) found that the sharpness rises during training until it reaches $2/\eta$ and then hovers there while the loss keeps decreasing; small-batch SGD stops well below it.

</details>

6. define the empirical neural tangent kernel and say how it governs training with squared loss.

<details><summary>answer</summary>

$\Theta(x, x') = \langle\nabla_\theta f(x), \nabla_\theta f(x')\rangle$. one gradient step on $\frac12\sum_i(f(x_i) - y_i)^2$ changes predictions by $-\eta\,\Theta(x, X)(f(X) - y)$, so if $\Theta$ stays fixed the training predictions follow $f_{t+1} - y = (I - \eta K)(f_t - y)$ and the converged network is kernel regression with kernel $\Theta$ on the residual $y - f_0$.

</details>

7. why does the NTK stay fixed as the width goes to infinity in the NTK parameterization, and what changes in the mean-field parameterization?

<details><summary>answer</summary>

with the $1/\sqrt m$ output scaling each weight needs to move only $O(1/\sqrt m)$ to fit the data, so activation patterns and gradients, hence the kernel, change by $O(1/\sqrt m)$: lazy training. with $1/m$ scaling the output is an average over units, each must move $O(1)$, the kernel changes by $O(1)$ (topic 03's lab: about 7× its initial size), and features are learned.

</details>

8. a network reaches 100% training accuracy on random labels. what does that imply for VC- or Rademacher-based generalization bounds?

<details><summary>answer</summary>

the class can fit any labeling of the sample, so its empirical Rademacher complexity is maximal (1) and every bound that depends only on the class is vacuous. whatever explains generalization on real labels must depend on the data and on which solution training finds (Zhang et al., 2017).

</details>

9. explain why the test error of minimum-norm random-features regression peaks at $p = n$.

<details><summary>answer</summary>

at $p = n$ the feature matrix is square and nearly singular, so the unique interpolant has huge coefficients along directions with tiny singular values, which amplify label noise and anything the features can't represent. for $p > n$ the minimum-norm interpolant spreads over many directions and shrinks; for $p < n$ there's no exact fit and noise is averaged. a ridge caps the amplification and removes the peak.

</details>

10. from which starting point does gradient descent on an underdetermined least-squares problem converge to the minimum-norm solution, and why?

<details><summary>answer</summary>

from $w_0 = 0$ (or any start in the row space). every gradient $X^\top(Xw - y)$ lies in the row space of $X$, so the null-space component of $w$ never changes; the only interpolant in the row space is the minimum-norm one. from a random start the limit is $X^+y$ plus the null-space part of $w_0$.

</details>

11. what is $C \approx 6ND$, and what does the "20 tokens per parameter" rule say?

<details><summary>answer</summary>

training a dense model with $N$ parameters on $D$ tokens costs about $2N$ FLOPs per token forward and $4N$ backward. Hoffmann et al. (2022) found that for a fixed $C$ the loss is minimized with $N$ and $D$ growing in equal proportion, about 20 tokens per parameter for their subword-token models, so $N \approx \sqrt{C/120}$.

</details>

12. why fit scaling laws in log space, and why is the offset $c$ the least trustworthy parameter?

<details><summary>answer</summary>

the data span decades and the quantity of interest is relative error; log-space residuals weight every point by its relative error and make the no-offset fit linear. the offset trades off against the exponent: very different $(b, c)$ pairs fit a short range equally well and diverge beyond it (topic 05's lab: the offset fit's prediction moved by 0.045 nats between seeds).

</details>

13. what are $q^*$ and $\chi_1$ in signal-propagation theory, and what happens on either side of $\chi_1 = 1$?

<details><summary>answer</summary>

$q^*$ is the fixed point of the variance map $q \mapsto \sigma_w^2\mathbb E[\phi(\sqrt q z)^2] + \sigma_b^2$; $\chi_1 = \sigma_w^2\mathbb E[\phi'(\sqrt{q^*}z)^2]$ is the slope of the correlation map at $c = 1$. below 1 (ordered) inputs converge and gradients vanish; above 1 (chaotic) inputs decorrelate and gradients explode; at 1 information propagates through many layers. for ReLU the edge is $\sigma_w^2 = 2$, He initialization.

</details>

14. under the standard parameterization with Adam, why does the best learning rate shrink with width, and what does μP change?

<details><summary>answer</summary>

Adam moves each weight by about $\eta$, and those moves add coherently over a fan-in of $n$, so a unit's input changes by $\Theta(\eta n)$: wider networks take larger steps at the same $\eta$. μP divides the hidden layers' learning rate by the width multiplier and scales the readout's input by $1/m$ (zero-initialized), so every layer's change is width-independent and the best learning rate transfers across widths.

</details>

15. name one thing each of the chapter's theories explains well and one thing it doesn't.

<details><summary>answer</summary>

approximation: what can be represented compactly, not what training finds. loss landscapes: why GD sits at $2/\eta$ and why some solutions are connected, not a reparameterization-invariant link from flatness to generalization. NTK: training of very wide networks as kernel regression, not feature learning. double descent and implicit bias: why interpolation needn't hurt, not a full account of deep networks. scaling laws: predictable improvements within a recipe, not why the exponents take their values. signal propagation and μP: trainable initializations and transferable learning rates, not what the trained network computes.

</details>

## 2. by hand

1. **the interpolant.** write the one-hidden-layer ReLU network through $(0, 1), (2, 3), (3, 0), (5, 2)$, and evaluate it at $x = 1$, $x = 4$ and $x = 6$.

<details><summary>answer</summary>

slopes 1, −3, 1; output weights 1, $-3 - 1 = -4$, $1 - (-3) = 4$; $f(x) = 1 + \text{relu}(x) - 4\,\text{relu}(x - 2) + 4\,\text{relu}(x - 3)$. $f(1) = 2$, $f(4) = 1 + 4 - 8 + 4 = 1$, $f(6) = 1 + 6 - 16 + 12 = 3$ (the last slope, 1, continues past 5: $f(5) = 2$).

</details>

2. **an infinite-width kernel value.** compute $\Theta_\infty(x, x')$ for the two-layer ReLU network of topic 03 with $x = (1, 0)$, $x' = (0, 2)$, and with $x' = (2, 0)$.

<details><summary>answer</summary>

$\Theta_\infty = \frac{\lVert x\rVert\lVert x'\rVert}{2\pi}(\sin\vartheta + (\pi - \vartheta)\cos\vartheta) + x\cdot x'\,\frac{\pi - \vartheta}{2\pi}$. orthogonal ($\vartheta = \pi/2$): $\frac{2}{2\pi}(1 + 0) + 0 = 1/\pi \approx 0.318$. parallel ($\vartheta = 0$): $\frac{2}{2\pi}\pi + 2 \cdot \frac12 = 1 + 1 = 2$, which is $x\cdot x'$ as it must be for parallel inputs.

</details>

3. **compute and allocation.** a 400M-parameter model is trained on 8B tokens. how many FLOPs did it cost, and what model size and token count would the 20-tokens rule have chosen for the same compute? with $L = E + A/N^{\alpha} + B/D^{\beta}$ and $\alpha = \beta$, $A = B$, what's the optimal $N$?

<details><summary>answer</summary>

$C = 6 \times 4 \times 10^8 \times 8 \times 10^9 = 1.92 \times 10^{19}$. $N = \sqrt{C/120} = \sqrt{1.6 \times 10^{17}} = 4 \times 10^8$ and $D = 8 \times 10^9$: this model already follows the rule. with $\alpha = \beta$ and $A = B$, $G = 1$ and $N_{\text{opt}} = D_{\text{opt}} = \sqrt{C/6} \approx 1.79 \times 10^9$: equal parameters and tokens, a very different split, which shows how much the rule depends on the fitted constants.

</details>

## 3. blind rebuild (60 minutes)

with no notes, open a fresh notebook (or a copy of `03-infinite-width-ntk-and-mean-field/exercise.py` with its function bodies deleted) and implement, from the specs you remember:

- `empirical_ntk(model, x1, x2=None)`: per-example gradients over the trainable parameters, $K = J_1 J_2^\top$, no `.grad` side effects;
- `kernel_regression_predict(K_train, y_train, K_test_train, *, ridge=0.0)`, for tensors and arrays;
- `NTKNet` and `relu_ntk`: the two-layer network in NTK/mean-field parameterization and its infinite-width kernel;
- `ntk_gd_predictions(...)`: gradient descent on the linearized network, in closed form, robust to a singular kernel.

check it with topic 03's tests: `pytest 19-deep-learning-theory/03-infinite-width-ntk-and-mean-field/test_exercise.py -k "stage1 or stage2 or stage3 or stage4"`.

## 4. readiness

move on when you score at least 80% on parts 1 and 2 and your blind rebuild passes stages 1–4. the same checkpoint works as a placement test: if you pass it cold, skim the labs (their results are the part you can't derive) and go on to chapter 20.
