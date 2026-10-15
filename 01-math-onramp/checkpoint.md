# 01 · math onramp · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. what's the difference between $(f \circ g)(x)$ and $(g \circ f)(x)$? give an example where they differ.

<details><summary>answer</summary>

$(f \circ g)(x) = f(g(x))$ applies $g$ first; $(g \circ f)(x) = g(f(x))$ applies $f$ first. with $f(x) = x^2$ and $g(x) = x + 1$: $f(g(2)) = 9$ but $g(f(2)) = 5$.

</details>

2. why does the raw series $\sum x^k/k!$ give garbage for $e^{-30}$, and how does range reduction fix it?

<details><summary>answer</summary>

the terms alternate in sign and reach about $10^{12}$ before shrinking, while the answer is about $10^{-13}$. each big term carries a rounding error around $10^{-4}$, which swamps the answer (catastrophic cancellation). range reduction writes $x = k + r$ with $|r| \le \frac12$ and uses $e^x = e^k e^r$, so the series only sees a small $r$, where every term is small.

</details>

3. Newton's method has an error of $10^{-3}$. roughly what is it after two more steps, and why?

<details><summary>answer</summary>

about $10^{-12}$ (more precisely, for $\log$: $10^{-3} \to 5 \times 10^{-7} \to 10^{-13}$). near the answer the error is roughly squared at every step (quadratic convergence), so the number of correct digits doubles.

</details>

4. a finite-difference derivative with $h = 10^{-15}$ is worse than with $h = 10^{-5}$. why?

<details><summary>answer</summary>

rounding error. function values carry relative errors around $10^{-16}$; subtracting two nearly equal values cancels their correct digits, and dividing by a tiny $h$ amplifies what's left. the total error is truncation ($\propto h$ or $h^2$) plus rounding ($\propto \varepsilon/h$), which is smallest at an intermediate $h$: about $10^{-8}$ for forward and $10^{-5}$ for central differences.

</details>

5. why is the central difference more accurate than the forward difference?

<details><summary>answer</summary>

the curvature term $\frac12f''h^2$ appears with the same sign in $f(x + h)$ and $f(x - h)$ and cancels when you subtract them, leaving an error $\propto h^2$ instead of $\propto h$.

</details>

6. state the chain rule, and name the two most common ways of getting it wrong.

<details><summary>answer</summary>

$(f(g(x)))' = f'(g(x))\,g'(x)$. the classic bugs: forgetting the inner derivative $g'(x)$, and evaluating the outer derivative at $x$ instead of at $g(x)$.

</details>

7. a derivative check at $x = 0$ and $x = 1$ passes. why isn't that convincing?

<details><summary>answer</summary>

formulas simplify at special points, so a wrong derivative can agree with the right one there (for example, $2x\cos x$ and $2x\cos(x^2)$ agree at $0$ and $1$). check at several random-looking points.

</details>

8. what does the fundamental theorem of calculus say, in words?

<details><summary>answer</summary>

the running area under $f$ has slope $f$: if $F(x) = \int_a^xf$, then $F' = f$. consequently, $\int_a^bf = G(b) - G(a)$ for any antiderivative $G$.

</details>

9. doubling the number of strips cuts the trapezoid rule's error by 4 and Simpson's by 16 on smooth functions. why do both collapse to about $2^{1.5}$ on $\sqrt x$ over $[0, 1]$?

<details><summary>answer</summary>

the error formulas need bounded derivatives, but $\sqrt x$ has infinite slope (and curvature) at $0$. almost all the error comes from the first strip, where the true area is $\frac23h^{3/2}$, so the error is $\propto h^{1.5}$ whatever the rule.

</details>

10. what are the columns of a $2 \times 2$ matrix, geometrically, and what does $AB$ mean as a transformation?

<details><summary>answer</summary>

the columns are the images of $e_1 = (1, 0)$ and $e_2 = (0, 1)$. $AB$ is "first apply $B$, then $A$", and its column $j$ is $A$ times column $j$ of $B$.

</details>

11. what does the determinant of a $2 \times 2$ matrix tell you about the transformation? what does $\det = 0$ mean?

<details><summary>answer</summary>

$|\det A|$ is the factor by which every area is multiplied, and a negative sign means the plane is flipped over (orientation reversed). $\det A = 0$ means the plane is squashed onto a line or a point, so the transformation can't be undone.

</details>

12. why does a standard error shrink like $1/\sqrt n$, and what does that cost?

<details><summary>answer</summary>

averaging $n$ independent draws divides the variance by $n$, so the standard deviation of the average is $\sigma/\sqrt n$. one more decimal digit (10× smaller error) costs 100× more samples.

</details>

13. a simulation sees a rare event 0 times in 50 trials and reports $0 \pm 0$. what's wrong?

<details><summary>answer</summary>

with all values 0, the sample standard deviation is 0, so the "error bar" claims certainty. the $\pm 1.96$ standard errors rule needs a bell-shaped estimate, which fails when the expected number of events is small. a 95% upper bound is about $3/n$ (the rule of three).

</details>

14. gradient descent on a 1-D quadratic with curvature $a$: for which learning rates does it converge, and when does it zigzag?

<details><summary>answer</summary>

each step multiplies the distance to the minimum by $1 - \eta a$. it converges for $0 < \eta < 2/a$, zigzags (overshoots but converges) for $1/a < \eta < 2/a$, lands in one step at $\eta = 1/a$, and diverges beyond $2/a$.

</details>

15. why does standardizing features speed up gradient descent without changing the fitted model?

<details><summary>answer</summary>

the loss's curvature in each weight's direction grows with the square of that feature's scale. very different scales make a long narrow valley: the steepest direction caps the learning rate and the flat directions barely move. standardizing makes the curvatures comparable (a round bowl). a linear model in standardized features is still a linear model in the original features, so the best predictions are the same.

</details>

## 2. by hand

1. **Newton and series.** (a) write Newton's update for solving $e^x = 5$ and take one step from $x_0 = 2$ (use $e^{-2} \approx 0.1353$). (b) how many terms of $\sum x^k/k!$ make the tail for $x = 1$ smaller than $10^{-6}$?

<details><summary>solution</summary>

(a) $x_{\text{new}} = x - 1 + 5e^{-x}$, so $x_1 = 2 - 1 + 5 \times 0.1353 = 1.6767$ (the true $\log 5 = 1.6094$). (b) the tail after $N$ terms is below $2/N!$; $2/N! < 10^{-6}$ needs $N! > 2 \times 10^6$: $9! = 362{,}880$ is too small, $10! = 3{,}628{,}800$ works, so 10 terms.

</details>

2. **derivatives, areas, and a step.** (a) differentiate $f(x) = x^2e^{-x}$ and find where its slope is 0. (b) compute $\int_0^2 x^3\,dx$ exactly, then with Simpson's rule and $n = 2$. (c) for the data $(0, 1), (1, 3), (2, 2)$, compute the gradient of the mean squared error at $(w, b) = (0, 0)$ and one gradient descent step with $\eta = 0.1$.

<details><summary>solution</summary>

(a) product rule: $f'(x) = 2xe^{-x} - x^2e^{-x} = x(2 - x)e^{-x}$, which is 0 at $x = 0$ and $x = 2$. (b) $\int_0^2x^3\,dx = \frac{2^4}{4} = 4$. Simpson with $h = 1$: $\frac13(0 + 4 \cdot 1 + 8) = 4$, exact (Simpson is exact for cubics). (c) at $(0, 0)$ the residuals are $r = (0 - 1, 0 - 3, 0 - 2) = (-1, -3, -2)$. $\frac{\partial L}{\partial w} = \frac23(-1 \cdot 0 - 3 \cdot 1 - 2 \cdot 2) = -\frac{14}{3}$ and $\frac{\partial L}{\partial b} = \frac23(-6) = -4$. the step gives $w = 0.4667$, $b = 0.4$.

</details>

3. **matrices and probability.** (a) for $A = \begin{pmatrix}3&1\\0&2\end{pmatrix}$, where do $e_1$ and $e_2$ go, what's $\det A$, and what's the area of the image of the unit circle? (b) you estimate a probability near $0.2$ from 2,500 simulated trials. what's the standard error, and how many trials make it $0.002$?

<details><summary>solution</summary>

(a) $e_1 \to (3, 0)$ and $e_2 \to (1, 2)$ (the columns). $\det A = 3 \cdot 2 - 1 \cdot 0 = 6$, so the image of the unit circle has area $6\pi$. (b) $\sqrt{0.2 \times 0.8/2500} = 0.4/50 = 0.008$. to get $0.002$, 4 times smaller, you need $4^2 = 16$ times more trials: 40,000.

</details>

## 3. blind rebuild (60 minutes)

with no notes, no lessons and no hints, in a fresh copy of each notebook (`git stash` your work first, or copy the stub files somewhere), rebuild the core of `mlfs.numerics` and the training loop:

1. `02-derivatives/exercise.py`: `secant_slope`, `derivative` (all three methods) and `check_derivative`.
2. `03-integrals/exercise.py`: `trapezoid` and `simpson`.
3. `05-probability-by-simulation/exercise.py`: `mean_and_se` and `mc_estimate`.
4. `06-your-first-ml-model/exercise.py`: `predict`, `mse_loss`, `mse_gradient` and `gradient_descent`.

then run the checks:

```bash
pytest 01-math-onramp/02-derivatives -k "stage1 or stage2 or stage4"
pytest 01-math-onramp/03-integrals -k "stage2 or stage3"
pytest 01-math-onramp/05-probability-by-simulation -k "stage3 or stage4"
pytest 01-math-onramp/06-your-first-ml-model -k "stage1 or stage2 or stage3"
```

you pass if every selected test is green within the hour. derive the MSE gradient on paper first: it's the step most people get wrong under time pressure.

## 4. readiness

move on to chapter 02 if you scored at least 80% on sections 1 and 2 together (at least 15 of 18) and passed section 3. as a placement test: if you pass all of it before starting the chapter, skip to chapter 02, and come back to individual topics (the labs especially) when a later chapter points to them.
