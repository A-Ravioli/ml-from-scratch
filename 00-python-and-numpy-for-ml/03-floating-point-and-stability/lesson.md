# floating point & stability

## why this matters

you're training a classifier. the loss goes down nicely: 2.3, 1.1, 0.6, 0.31... then it prints `nan`, and every number in the model turns into `nan`. your math is fine. the model got confident, one of its scores (a "logit") reached about 710, and your code computed `np.exp(710)`. the largest number a 64-bit float can hold is about $1.8 \times 10^{308}$, and $e^{710}$ is bigger. the result was `inf`, then `inf / inf` gave `nan`, and `nan` spreads through every calculation it touches.

or quieter: the school formula for a quadratic's small root returns `-7.45e-09` instead of `-1.00e-08`, a 25% error with no warning.

computers do **floating point** arithmetic: every number keeps a fixed number of significant digits, and every operation rounds. usually that's invisible, because 16 digits is a lot. but some computations **amplify** rounding errors until they swamp the answer, and some produce numbers too big or too small to store. this topic shows how floats work, the two big ways computations go wrong (cancellation, and overflow/underflow), and the standard fixes. the stable `sigmoid`, `logsumexp` and `softmax` you build here go into `mlfs.utils`, and every classifier and language model in the rest of the course uses them.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. does `0.1 + 0.2 == 0.3` evaluate to `True` in Python?
2. what is `(1e16 + 1.0) - 1e16`? and `(1e16 - 1e16) + 1.0`?
3. how many significant decimal digits does a 64-bit float (`float64`) hold? a 32-bit one?
4. what does `np.exp(1000.0)` return? and `np.exp(-1000.0)`?
5. softmax turns scores into probabilities: $p_i = e^{x_i} / \sum_j e^{x_j}$. what does the code `np.exp(x) / np.exp(x).sum()` return for `x = [1000, 1001, 1002]`?

## prerequisites check

- [ ] you can vectorize with NumPy: elementwise operations, `np.where`, reductions with `axis` and `keepdims` (→ ch 00 · topic 02, ideas 1 and 3)
- [ ] you know what a `with` block does (→ ch 00 · topic 01, the python toolkit)
- [ ] you know powers and their rules: $2^{a}\cdot 2^{b} = 2^{a+b}$, $2^{-3} = 1/8$ (high school)
- [ ] you know logarithms and exponentials: $\log(ab) = \log a + \log b$, $\log e^x = x$, $e^{a+b} = e^ae^b$ (high school; chapter 01 reviews them)
- [ ] you remember the quadratic formula $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ (high school)

## core ideas

### idea 1: what a floating point number is

in **scientific notation**, $6.02 \times 10^{23}$ is a few significant digits (the *mantissa*, 6.02) times a power of ten (the *exponent*, 23): the decimal point "floats".

a floating point number is scientific notation in base 2. every nonzero `float64` is

$$x = (-1)^{s} \times m \times 2^{e}, \qquad 1 \le m < 2,$$

stored in 64 bits: 1 bit for the **sign** $s$, 11 bits for the **exponent** $e$ (from $-1022$ to $1023$), and 52 bits for the **fraction** of the mantissa $m = 1.f_1f_2\dots f_{52}$ in binary. (the leading "1." is always there, so it isn't stored: a free extra bit.) for example,

$$6 = +1.5 \times 2^2, \qquad -0.375 = -1.5 \times 2^{-2}, \qquad 0.1 \approx +1.6 \times 2^{-4}.$$

the last one is only approximate. $0.1 = 1/10$, and in binary, $1/10$ is a repeating fraction ($0.000110011001100\ldots_2$), just as $1/3 = 0.333\ldots$ repeats in decimal. it gets cut off after 52 bits, so the stored value is $0.1000000000000000055511\ldots$. the same happens to $0.2$ and $0.3$, and the rounding errors don't happen to line up: `0.1 + 0.2` is `0.30000000000000004`, so `0.1 + 0.2 == 0.3` is `False` (warm-up 1). **never compare floats with `==`**; compare with a tolerance (`np.isclose`, or the `assert_close` you'll write in topic 05).

**machine epsilon.** with 52 fraction bits, the next number after $1$ is $1 + 2^{-52}$. that gap,

$$\varepsilon = 2^{-52} \approx 2.22 \times 10^{-16},$$

is called **machine epsilon**. between $2$ and $4$ the gap is $2\varepsilon$, between $4$ and $8$ it's $4\varepsilon$: the spacing scales with the number, so the **relative** spacing is always about $\varepsilon$. any real number gets rounded to the nearest float, with a relative error of at most $\varepsilon / 2 \approx 1.1 \times 10^{-16}$. that's about **16 significant decimal digits**, wherever the decimal point is (warm-up 3).

**other sizes.** `float32` has 23 fraction bits, so $\varepsilon = 2^{-23} \approx 1.2 \times 10^{-7}$: about **7 digits**. `float16` has 10, so $\varepsilon \approx 10^{-3}$: barely 3 digits. deep learning uses these smaller formats all the time, because they halve or quarter the memory and are much faster on GPUs. so the problems in this topic are much closer to the surface there.

**the range.** the largest `float64` is about $1.8 \times 10^{308}$; anything bigger **overflows** to `inf`. below about $2.2 \times 10^{-308}$ come "subnormal" numbers with fewer digits, down to $5 \times 10^{-324}$; anything smaller **underflows** to `0.0`. `nan` ("not a number") is what `0/0`, `inf - inf` and `0 * inf` give. it's contagious: any arithmetic with `nan` gives `nan`, and `nan == nan` is `False`.

> **▶ build: stage 1.** implement `decompose(x)`, which finds $(s, m, e)$ with a loop that halves or doubles (exact operations in binary), and `machine_epsilon(dtype)`, which finds $\varepsilon$ by halving until $1 + \varepsilon/2$ rounds back to $1$, for `float64`, `float32` and `float16`.

### idea 2: rounding errors, and how cancellation amplifies them

every arithmetic operation on floats computes the exact result and then rounds it to the nearest float. so for any operation $\circ$ (add, subtract, multiply, divide),

$$\text{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad |\delta| \le \varepsilon/2.$$

one operation is accurate to 16 digits. that's the good news, and it's why floating point works at all. the bad news comes in two forms.

**rounding can swallow small numbers.** `1e16 + 1.0` is `1e16`: the spacing between floats near $10^{16}$ is $2$, so the exact answer $10^{16} + 1$ sits exactly halfway between two floats, and the tie goes to the "even" one, $10^{16}$. so `(1e16 + 1.0) - 1e16` is `0.0`, while `(1e16 - 1e16) + 1.0`, the same three numbers grouped differently, is `1.0`. **floating point addition isn't associative** (warm-up 2), which is why summing in a different order (as a GPU does across threads) can change the last digits.

**subtraction can expose earlier errors.** suppose $a$ and $b$ each carry a tiny relative error from earlier rounding, say $10^{-16}$. if $a \approx 1.0000001$ and $b \approx 1.0000000$, their difference is about $10^{-7}$, but the *absolute* errors they carried, around $10^{-16}$ each, are still there. relative to the small difference, they're now $10^{-16}/10^{-7} = 10^{-9}$: seven digits have been lost. in general, subtracting nearly equal numbers multiplies the relative error by $\frac{|a|}{|a - b|}$. this is called **catastrophic cancellation**. the subtraction itself is exact; it just reveals that the leading digits of $a$ and $b$, which were right, cancel, leaving only the trailing digits, which were noise. you met it in topic 02's lab, where the fast distance formula subtracted two numbers near $3 \times 10^{16}$ to get an answer near 5.

**the quadratic formula.** the roots of $ax^2 + bx + c = 0$ are $\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$. take $a = 1$, $b = 10^8$, $c = 1$. then $\sqrt{b^2 - 4ac} = \sqrt{10^{16} - 4} \approx 10^8 - 2\times 10^{-8}$, which rounds to a number extremely close to $b$. the "$+$" root computes $-b + \sqrt{\cdots}$: two numbers near $10^8$ subtracted to get something near $-2 \times 10^{-8}$. cancellation leaves essentially one or two correct digits, and the formula returns `-7.45e-09` instead of `-1.00e-08`.

the cure is to **rearrange the algebra so that nothing nearly equal gets subtracted.** the "$-$" root, $-b - \sqrt{\cdots}$, adds two numbers of the same sign, which is perfectly accurate. and the two roots multiply to $c/a$ (Vieta's formula, on paper 2), so the small root is $c / (a \cdot \text{big root})$, a division, which is also accurate. the standard way to write this, which handles both signs of $b$, is

$$q = -\tfrac12\Big(b + \operatorname{sign}(b)\sqrt{b^2 - 4ac}\Big), \qquad x_1 = \frac{q}{a}, \qquad x_2 = \frac{c}{q}.$$

`sign(b)` makes the square root take the same sign as $b$, so the bracket is always a sum of two same-signed numbers. (in code, `math.copysign(math.sqrt(disc), b)`.) this pattern, **find the subtraction that cancels, then rewrite the formula to avoid it**, is the core skill of numerical stability.

> **▶ build: stage 2.** implement `quadratic_roots(a, b, c)` with the stable formulas, returning the roots smallest first.

> **🔮 predict:** the variance of numbers $x_1, \dots, x_n$ can be computed as $\overline{x^2} - \bar{x}^2$ (the mean of the squares minus the square of the mean). for a million measurements around $10^8$ with spread 1, how accurate is this formula? (the lab tries it.)

### idea 3: overflow, underflow, and the sigmoid family

`np.exp` overflows quickly. $e^{709} \approx 8 \times 10^{307}$ is the largest power of $e$ that fits; `np.exp(710)` returns `inf` with a `RuntimeWarning: overflow`. at the other end, `np.exp(-746)` underflows to `0.0` (warm-up 4). ML formulas are full of exponentials, so you need to know how to keep them in range.

**the sigmoid** $\sigma(x) = \frac{1}{1 + e^{-x}}$ squashes any number into $(0, 1)$ and is used to turn a score into a probability (chapter 08's logistic regression is built on it). the direct code, `1 / (1 + np.exp(-x))`, is fine for positive $x$, but for $x = -1000$ it computes $e^{1000}$, overflows and warns. (the answer, 0, happens to come out right, but in `float32` the overflow starts at $x \approx -89$, and a warning in a training loop usually means worse is coming.) the fix: never exponentiate a large positive number. multiplying top and bottom by $e^{x}$ gives an equivalent formula,

$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^{x}}{1 + e^{x}},$$

and the first form only needs $e^{-x}$ while the second only needs $e^{x}$. so use the first for $x \ge 0$ and the second for $x < 0$. in both cases the exponential is $e^{-|x|}$, a number between 0 and 1 that can never overflow. vectorized:

```python
e = np.exp(-np.abs(x))
np.where(x >= 0, 1 / (1 + e), e / (1 + e))
```

a trap to avoid: `np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))` looks the same but still overflows. `np.where` doesn't skip anything: both branches are computed for **every** element, and only then is one picked.

**`log1p` and the softplus.** the function $\operatorname{softplus}(x) = \log(1 + e^x)$ is a smooth version of $\max(0, x)$ that appears in losses and activations. it has two problems. for big $x$, $e^x$ overflows, though the answer is about $x$. rewrite it the same way as the sigmoid:

$$\log(1 + e^{x}) = \max(x, 0) + \log\big(1 + e^{-|x|}\big).$$

(for $x \ge 0$, factor $e^x$ out of the bracket: $\log(e^x(e^{-x} + 1)) = x + \log(1 + e^{-x})$.) the second problem is at the other end. for $x = -40$, $e^{-40} \approx 4.2 \times 10^{-18}$ is smaller than $\varepsilon$, so `1 + 4.2e-18` rounds to exactly `1.0`, and `np.log(1.0)` is `0.0`: every digit of the true answer, $4.2 \times 10^{-18}$, is gone. NumPy has a function for exactly this case: **`np.log1p(t)` computes $\log(1 + t)$ accurately even when $t$ is tiny**, without ever forming $1 + t$. (its partner `np.expm1(t)` computes $e^t - 1$ accurately for tiny $t$.) whenever you see $\log(1 + \text{something small})$, reach for `log1p`.

**log-sigmoid.** in losses you usually need $\log \sigma(x)$, not $\sigma(x)$. computing `np.log(sigmoid(x))` fails at both ends: for $x = -1000$, $\sigma(x)$ underflows to 0 and its log is `-inf` (the true value is $-1000$); for $x = 40$, $\sigma(x)$ rounds to exactly 1 and its log is 0 (the true value is $-4.2 \times 10^{-18}$). but

$$\log\sigma(x) = \log\frac{1}{1 + e^{-x}} = -\log(1 + e^{-x}) = -\operatorname{softplus}(-x),$$

so a stable softplus gives a stable log-sigmoid for free.

> **▶ build: stage 3.** implement `sigmoid`, `softplus` and `log_sigmoid`, elementwise on arrays of any shape, with no overflow warnings and full accuracy at both ends.

### idea 4: working in log space, and logsumexp

probabilities multiply. a 1,000-character sentence under a language model has a probability of about $0.01^{1000} = 10^{-2000}$, which underflows to 0. so you store **log-probabilities**: products become sums, $\log(p_1\cdots p_n) = \sum_i \log p_i \approx -4600$, an ordinary number.

but sometimes you need to **add** probabilities held as logs (to normalize, or to combine mixture components). given $a_i = \log p_i$, you want $\log\sum_i p_i = \log\sum_i e^{a_i}$. computed directly, `np.log(np.sum(np.exp(a)))`, this underflows for $a_i$ around $-1000$ (every $e^{a_i}$ is 0, so the answer is `-inf`) and overflows for $a_i$ around $+1000$.

the trick is to factor out the largest term. with $m = \max_i a_i$,

$$\log\sum_i e^{a_i} = \log\Big(e^{m}\sum_i e^{a_i - m}\Big) = m + \log\sum_i e^{a_i - m}.$$

now every exponent $a_i - m$ is $\le 0$, so every term is at most 1 (no overflow), and the largest term is exactly $e^0 = 1$, so the sum is between 1 and $n$ (no underflow to 0, and the log is safe). since the sum is between 1 and $n$,

$$\max_i a_i \;\le\; \operatorname{logsumexp}(a) \;\le\; \max_i a_i + \log n,$$

which is why it's often called a "smooth maximum".

one edge case needs care. if every $a_i = -\infty$ (all probabilities are zero, which happens with masks), then $m = -\infty$ and $a_i - m = -\infty - (-\infty) = $ `nan`. the right answer is $\log 0 = -\infty$. the fix: if $m$ isn't finite, shift by 0 instead. then the sum is 0, its log is `-inf` (NumPy warns about `log(0)`; silence that one warning with `np.errstate(divide="ignore")`, since here it's the correct answer).

your `logsumexp` takes an `axis`, like NumPy's reductions: `logsumexp(A, axis=1)` gives one value per row. compute the max with `keepdims=True` (shape `(n, 1)`) so it broadcasts back against `A`; your own `keepdims` argument lets callers do the same.

> **▶ build: stage 4.** implement `logsumexp(x, axis=None, keepdims=False)`, matching `scipy.special.logsumexp`: finite for huge inputs, $-\infty$ when every entry is $-\infty$, no warnings.

### idea 5: softmax and log-softmax

a classifier with $k$ classes produces $k$ scores (logits) $x_1, \dots, x_k$, and turns them into probabilities with the **softmax**:

$$\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^k e^{x_j}}.$$

the outputs are positive and sum to 1. the naive code, `np.exp(x) / np.exp(x).sum()`, overflows as soon as a logit passes about 709: for `x = [1000, 1001, 1002]` it computes `inf / inf` and returns `[nan, nan, nan]` (warm-up 5).

softmax has a property that makes the fix easy: **adding the same constant to every logit doesn't change it.** for any $c$,

$$\frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^{c}e^{x_i}}{e^{c}\sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}.$$

choose $c = -\max_j x_j$. then the largest shifted logit is 0, every exponential is at most 1, and the denominator is at least 1. with a batch of logit vectors (one row per example), take the max **per row** (`axis=-1, keepdims=True`). a logit of $-\infty$ (a "masked" class) gets probability exactly 0.

**log-softmax.** classification losses need $\log \operatorname{softmax}(x)_y$, the log-probability of the correct class $y$ (chapter 08's cross-entropy loss is its negative). taking `np.log(softmax(x))` throws away accuracy, and for a very unlikely class returns $\log 0 = -\infty$, so the loss becomes infinite and its gradient `nan`. but the log of a softmax simplifies:

$$\log\operatorname{softmax}(x)_i = x_i - \log\sum_j e^{x_j} = x_i - \operatorname{logsumexp}(x).$$

so `log_softmax` is one subtraction and one call to your `logsumexp` (with `keepdims=True`, so it broadcasts). `log_softmax([0, -1000])` returns `[0, -1000]` exactly, where `np.log(softmax(...))` gives `[0, -inf]`. this is why every deep learning library computes cross-entropy from log-softmax, never from softmax.

> **🔮 predict:** language models often sample from $\operatorname{softmax}(x / T)$ for a "temperature" $T > 0$. what does the distribution look like as $T \to 0$? as $T \to \infty$? and what happens to the naive formula at $T = 0.01$ with ordinary logits around 10?

> **▶ build: stage 5.** implement `softmax(x, axis=-1)` and `log_softmax(x, axis=-1)`.

## worked example

compute $\operatorname{logsumexp}$, softmax and log-softmax of $x = (1000, 1001, 1002)$ by hand.

**shift.** $m = 1002$, so $x - m = (-2, -1, 0)$.

**exponentials.** $e^{-2} \approx 0.135335$, $e^{-1} \approx 0.367879$, $e^{0} = 1$. their sum is $S \approx 1.503214$.

**logsumexp.** $\log S \approx 0.407606$, so $\operatorname{logsumexp}(x) = 1002 + 0.407606 = 1002.407606$. check the bounds: it lies between $\max = 1002$ and $\max + \log 3 \approx 1003.0986$. ✓

**softmax.** divide each exponential by $S$: $(0.135335, 0.367879, 1)/1.503214 \approx (0.090031, 0.244728, 0.665241)$. they sum to 1. ✓ these are exactly the softmax probabilities of $(0, 1, 2)$, or of $(1, 2, 3)$: only the differences between logits matter.

**log-softmax.** $x - \operatorname{logsumexp}(x) = (1000, 1001, 1002) - 1002.407606 = (-2.407606, -1.407606, -0.407606)$. check: $e^{-0.407606} \approx 0.665241$. ✓

meanwhile the naive code computes $e^{1000}$, which is `inf`, and returns `nan` everywhere. the tests check that your `softmax([1000, 1001, 1002])` equals `softmax([0, 1, 2])`, and that `softmax([1, 2, 3])` equals the probabilities above.

## on paper

1. **shift invariance.** show that $\operatorname{softmax}(x + c\mathbf{1}) = \operatorname{softmax}(x)$ and $\operatorname{logsumexp}(x + c\mathbf 1) = \operatorname{logsumexp}(x) + c$ for any constant $c$. what does $\operatorname{softmax}(x)$ look like when one $x_i$ is much bigger than all the others?

<details><summary>solution</summary>

the first is in idea 5: $e^{c}$ factors out of the top and bottom and cancels. for the second, $\log\sum_j e^{x_j + c} = \log(e^{c}\sum_j e^{x_j}) = c + \operatorname{logsumexp}(x)$. if $x_1$ exceeds every other $x_j$ by a lot, then after shifting by $x_1$ the others' exponentials are nearly 0, so softmax is nearly $(1, 0, \dots, 0)$: a "soft" version of picking the maximum (an argmax as a one-hot vector). that's where the name comes from.

</details>

2. **Vieta's formula.** show that the two roots of $ax^2 + bx + c$ satisfy $x_1x_2 = c/a$, and use it to explain why $x_2 = c/q$ in idea 2's stable formula.

<details><summary>solution</summary>

if the roots are $x_1, x_2$, then $ax^2 + bx + c = a(x - x_1)(x - x_2) = ax^2 - a(x_1 + x_2)x + ax_1x_2$. matching the constant terms gives $ax_1x_2 = c$. with $x_1 = q/a$ (one of the two roots from the formula, $\frac{-b - \operatorname{sign}(b)\sqrt{\cdot}}{2a}$), $x_2 = \frac{c}{a x_1} = \frac{c}{q}$. a division of two accurate numbers is accurate.

</details>

3. **the relative error of a subtraction.** suppose $\hat a = a(1 + \delta_a)$ and $\hat b = b(1 + \delta_b)$ with $|\delta_a|, |\delta_b| \le \delta$. show that the relative error of $\hat a - \hat b$ as an approximation of $a - b$ is at most $\delta\,\frac{|a| + |b|}{|a - b|}$. evaluate the bound for $a = 1.0000001$, $b = 1$, $\delta = 10^{-16}$.

<details><summary>solution</summary>

$(\hat a - \hat b) - (a - b) = a\delta_a - b\delta_b$, whose size is at most $\delta(|a| + |b|)$. dividing by $|a - b|$ gives the bound. here it's $10^{-16} \cdot 2.0000001 / 10^{-7} \approx 2 \times 10^{-9}$: from 16 good digits to about 9. the factor $\frac{|a| + |b|}{|a - b|}$ is the **condition number** of subtraction; chapter 02 generalizes it to linear systems.

</details>

## experiment

`lab.py` breaks the naive versions on purpose, using your stable ones as the reference:

1. at what logit size does naive softmax start returning `nan`, in `float64` and in `float32`?
2. the variance of measurements with a large mean, computed with the "mean of squares minus square of mean" formula.
3. the cross-entropy loss of a confident wrong prediction, computed from `softmax` and from `log_softmax`.

record your predictions first.

## check your understanding

1. why is the spacing between consecutive floats near $10^{16}$ equal to 2?

<details><summary>answer</summary>

$10^{16}$ lies between $2^{53} \approx 9.007 \times 10^{15}$ and $2^{54}$, so its exponent is 53. the mantissa has 52 fraction bits, so the spacing is $2^{53} \cdot 2^{-52} = 2$.

</details>

2. which of these lose accuracy through cancellation: (a) $x + y$ for $x, y > 0$; (b) $\sqrt{x + 1} - \sqrt{x}$ for $x = 10^{12}$; (c) $\log(1 + x)$ for $x = 10^{-20}$; (d) $xy$?

<details><summary>answer</summary>

(b) and (c). (b) subtracts two nearly equal numbers; rewrite it as $\frac{1}{\sqrt{x + 1} + \sqrt{x}}$ (multiply by the conjugate). (c) loses $x$ entirely when forming $1 + x$; use `log1p`. (a) adds same-signed numbers and (d) multiplies, which are both accurate to about $\varepsilon$.

</details>

3. why does $\operatorname{logsumexp}$ shift by the **maximum**, rather than by the mean or by any other constant?

<details><summary>answer</summary>

the shift must make every exponent $\le 0$ (no overflow) *and* keep at least one term from underflowing (so the log is finite). shifting by the max does both: every $x_i - m \le 0$, and the largest term is exactly $e^0 = 1$. shifting by the mean can leave exponents far above 0.

</details>

4. (the predict callout in idea 2.) how accurate is $\overline{x^2} - \bar x^2$ for data around $10^8$ with spread 1?

<details><summary>answer</summary>

useless. $\overline{x^2} \approx 10^{16}$ and $\bar x^2 \approx 10^{16}$, each rounded to a spacing of about 2, and their difference, about 1, is pure rounding noise; it can even come out negative. computing the mean first and then averaging $(x_i - \bar x)^2$ (the "two-pass" formula) avoids the cancellation. the lab shows it.

</details>

5. (the predict callout in idea 5.) what does $\operatorname{softmax}(x/T)$ do as $T \to 0$ and $T \to \infty$?

<details><summary>answer</summary>

dividing by a small $T$ stretches the gaps between logits, so as $T \to 0$ all the probability moves to the largest logit (a one-hot argmax: "greedy" sampling). as $T \to \infty$ the gaps shrink to nothing and the distribution becomes uniform. at $T = 0.01$, logits around 10 become logits around 1,000, and the naive formula overflows, even though nothing about the model is unusual. a stable softmax handles any $T$.

</details>

## common bugs

- **naive `exp` in sigmoid, softmax or softplus.** symptom: `RuntimeWarning: overflow encountered in exp`, then `nan` or `inf` losses as a model becomes confident. subtract the max (softmax, logsumexp) or use $e^{-|x|}$ (sigmoid, softplus).
- **`np.log(1 + t)` for tiny `t`.** symptom: a result of exactly 0 where you expected something like $10^{-18}$, or a loss that stops changing. use `np.log1p(t)`.
- **taking the max over the whole array instead of per row.** symptom: softmax works on a single vector but rows of a batch with very different scales come out as `nan` or all zeros. use `axis=-1, keepdims=True`.
- **`log(softmax(x))`.** symptom: `-inf` losses and `nan` gradients for confident wrong predictions. use `log_softmax`.

## going deeper

- Goldberg (1991), "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1): 5–48. the classic introduction: representation, rounding, cancellation, and why IEEE 754 is designed the way it is. [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, chapters 1–4: the definitive treatment of rounding error, including the variance formulas and summation. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- Blanchard, Higham & Higham (2021), "Accurately computing the log-sum-exp and softmax functions", *IMA Journal of Numerical Analysis* 41(4): 2311–2330. a careful error analysis of exactly the functions you built, showing that the shifted versions are accurate. [doi:10.1093/imanum/draa038](https://doi.org/10.1093/imanum/draa038)
- Micikevicius et al. (2018), "Mixed precision training", ICLR. training in `float16`, with loss scaling against underflow (chapter 26). [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

## connections

- ← builds on: NumPy arrays, broadcasting, `axis` and `keepdims` (topic 02); the cancellation you met in topic 02's lab.
- → used in: tolerances in tests (`assert_close`, topic 05); conditioning and the stability of algorithms (ch 02 · topic 08); `log1p`/`expm1` and Monte Carlo error (ch 01, ch 04); entropy and KL divergence in log space (ch 05); `sigmoid` and `log_sigmoid` in logistic regression and `log_softmax` in softmax regression (ch 08); `logsumexp` in mixture models and HMMs (ch 11, ch 12); cross-entropy losses in every neural network (ch 14 on); online softmax in attention (ch 18); mixed precision and loss scaling (ch 26).
