# floating point & stability · lab notes

## experiment 1: where does naive softmax break?

**what people usually predict:** somewhere around "very large" logits, maybe thousands, and only on the positive side, since only big numbers overflow. most people don't have a number in mind for `float32` at all.

**what happens:** naive `float64` works for offsets from about **−747 to 707.5**, and returns `nan` outside that window. naive `float32` works only from about **−105.5 to 86.5**. your `softmax` gives exactly the same probabilities (a difference of 0) over the whole range from −1200 to 1200: after the shift, every input is $(-2, -1, 0)$.

**why:** the largest `float64` is about $1.8 \times 10^{308} = e^{709.78}$, so `exp` overflows to `inf` once its argument passes 709.78. the biggest logit is $c + 2$, so the formula breaks at $c \approx 707.8$: the numerator and denominator are both `inf`, and `inf / inf` is `nan`. for `float32` the largest number is about $3.4 \times 10^{38} = e^{88.72}$, so it breaks at $c \approx 86.7$. **logits of 90 are completely ordinary in a neural network**, which is why no deep learning library computes softmax this way. at the negative end nothing overflows; instead every exponential **underflows** to 0 once $c + 2$ drops below about $-745$ (`float64`, the edge of the subnormal range) or $-103$ (`float32`), and $0 / 0$ is `nan` too. subtracting the max fixes both ends at once: the largest shifted logit is exactly 0, so the largest exponential is exactly 1 (no overflow) and the denominator is at least 1 (no division by zero).

## experiment 2: the variance of big numbers

**what people usually predict:** both formulas are fine, perhaps with errors in the last digit or two at $c = 10^8$.

**what happens:** the two-pass formula stays within about $10^{-10}$ of 1 at every offset up to $10^{10}$. the one-pass formula happens to be exactly right for small offsets (on this data), then falls apart: **0.984 at $c = 10^7$, 2.0 at $c = 10^8$, −128 at $10^9$, and −16,384 at $10^{10}$**. a negative variance is impossible, and `np.sqrt` of it would give `nan` for the standard deviation.

**why:** at $c = 10^8$, every $x^2 \approx 10^{16}$, and your `decompose` shows that floats near $10^{16}$ are 2 apart. so $\overline{x^2}$ and $\bar x^2$ are each known only to within about 2, and their difference, the answer, is about 1: it's pure rounding noise (2.0 here). at $10^9$ the spacing near $10^{18}$ is 128, which is exactly the size of the garbage you got. this is catastrophic cancellation in its most common form. the two-pass formula subtracts the mean from each $x_i$ first. that subtraction, $x_i - \bar x$, also involves nearly equal numbers, but it happens *before* squaring, while the numbers are about $10^8$ (spacing $10^{-8}$), so it loses only about 8 of its 16 digits and leaves the small differences accurate enough. squaring and averaging those is harmless. when you can't afford two passes (streaming data), use Welford's (1962) update, which keeps a running mean and a running sum of squared deviations and is as stable as the two-pass formula. you'll build it in chapter 04.

reference: Welford (1962), "Note on a method for calculating corrected sums of squares and products", *Technometrics* 4(3): 419–420. [doi:10.1080/00401706.1962.10490022](https://doi.org/10.1080/00401706.1962.10490022)

## experiment 3: the loss of a confident mistake

**what people usually predict:** the naive version breaks at some point. most expect a stable softmax followed by a log to be fine, since the softmax itself never overflows.

**what happens:** all three agree (the loss grows almost exactly like $m$) until $m \approx 710$, where the naive version becomes `inf`. your stable softmax followed by `np.log` lasts a little longer, then also becomes **`inf` from $m \approx 745.5$**. your `log_softmax` gives the right answer everywhere: $1000.0$ at $m = 1000$.

**why:** the naive version overflows: $e^{710}$ is `inf`, so $p_{\text{true}} = 1/(e^{m} + 2)$ is $1/\infty = 0$, and $-\log 0 = \infty$. the stable softmax computes $p_{\text{true}} = e^{-m}/(1 + 2e^{-m})$ correctly as long as $e^{-m}$ is representable, but at $m \approx 745$ it drops below the smallest subnormal ($5 \times 10^{-324}$) and underflows to 0. a probability that small is meaningless, but its log, $-745$, is a perfectly ordinary number that you need. `log_softmax` never forms the probability: it computes $x_{\text{true}} - \operatorname{logsumexp}(x) = 0 - (m + \log(1 + 2e^{-m}))$, a subtraction of ordinary numbers. in training, these confident mistakes are exactly the examples the model most needs to learn from; an infinite loss produces `inf` and `nan` gradients that destroy every weight in one step. so the rule is: **compute losses from log-probabilities, never from probabilities.** every deep learning library's cross-entropy takes logits for this reason.
