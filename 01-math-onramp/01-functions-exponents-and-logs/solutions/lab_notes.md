# functions, exponents & logs · lab notes

## experiment 1: how many terms does the series need?

**what people usually predict:** something like 10–15 terms for $x = 1$, a few more for $x = 20$, and about the same for $x = -20$ as for $x = 20$, since the series "works for every $x$". hardly anyone predicts *never*.

**what happens:** for a target of $10^{-10}$, $x = 1$ needs **13** terms and $x = 20$ needs **55**: the count grows roughly in proportion to $|x|$. on the negative side the count also grows at first ($x = -8$ needs 43), and then, from $x = -9$ down, the raw series **never** reaches $10^{-10}$, not even with 300 terms. tighten the target to $10^{-14}$ and it already fails at $x = -4$. range reduction needs **11** terms for every $x$ (14 for $10^{-14}$).

**why:** term $k$ is term $k-1$ times $x/k$, so the terms only start shrinking once $k > |x|$, and then you need another stretch of terms for them to become negligible. that's why the count grows with $|x|$. for negative $x$ the terms alternate in sign, and the biggest has size about $e^{|x|}/\sqrt{2\pi|x|}$. each term is rounded with relative error about $10^{-16}$, so the sum carries an absolute error of roughly $10^{-16}e^{|x|}$, while the answer is $e^{-|x|}$. the relative error is therefore about $10^{-16}e^{2|x|}$: at $x = -9$ that's $10^{-16} \times 6.6 \times 10^{7} \approx 7 \times 10^{-9}$, already worse than the target, and at $x = -20$ it's about 200%. more terms can't help, because the damage is done by the big terms in the middle. range reduction sidesteps both problems at once: the series only ever sees $|r| \le \frac12$, where the terms shrink from the start and nothing cancels badly, and the exponent rule $e^{k + r} = e^ke^r$ does the rest. this is how real math libraries work, with $k\ln 2$ in place of $k$ so that $e^k$ becomes an exact power of two.

## experiment 2: bisection against Newton

**what people usually predict:** that Newton is "faster", perhaps two or three times faster than bisection.

**what happens:** after 5 steps bisection's error is still about $0.04$ (one correct digit), while Newton's is already at the limit of float precision, $4 \times 10^{-16}$. to reach $10^{-12}$, bisection needs **39** steps and Newton needs **4**. on the log-scale plot, bisection's error zigzags around a straight line that falls by a factor of 2 per step. Newton's errors go $5 \times 10^{-2},\ 1.3 \times 10^{-3},\ 8 \times 10^{-7},\ 3 \times 10^{-13}$, a curve that plunges.

**why:** bisection halves the bracket every step, so its error shrinks by a constant factor: a straight line on a log scale with slope $\log_{10}2 \approx 0.3$ digits per step. (the zigzag is because the midpoint can land closer to or farther from the answer than the bracket width suggests.) Newton's error obeys $t_{\text{new}} \approx t^2/2$: squaring doubles the *number of digits*, so the gain per step grows every step. check it: $(1.3 \times 10^{-3})^2/2 \approx 8 \times 10^{-7}$ and $(8 \times 10^{-7})^2/2 \approx 3 \times 10^{-13}$. the price is that Newton needs the derivative (here $e^x$ again) and a good start, while bisection only needs a bracket and never fails. robust solvers often combine the two: bisect until close, then switch to Newton. you'll see Newton's method again in chapter 06, minimizing functions instead of solving equations, with the same quadratic convergence.

## experiment 3: who wins, and when?

**what people usually predict:** that $1.01^n$ overtakes $n^5$ somewhere in the hundreds, and that $n^{0.1}$ overtakes $\log n$ at some modest $n$, maybe a few thousand.

**what happens:** $1.01^n$ overtakes $n^5$ at about **$n = 4{,}200$**. before that, $n^5$ is ahead by as much as a factor of about $2 \times 10^{11}$ (near $n = 500$). after it, $1.01^n$ runs away: at $n = 100{,}000$ it's about $10^{432}$, against $10^{25}$. and $n^{0.1}$ only overtakes $\log n$ at about **$n = 3.4 \times 10^{15}$**. moving the base toward 1 or raising the power pushes the crossover out dramatically: $1.001^n$ against $n^5$ crosses near $n = 55{,}000$.

**why:** compare logs: $n\log b$ grows in proportion to $n$, while $a\log n$ grows only like $\log n$. any straight line eventually passes any log curve, but a shallow line (small $\log b \approx b - 1$) takes a long time. for $\log n$ against $n^{0.1}$, write $n = e^t$: it's $t$ against $e^{0.1t}$, an exponential against a power *in disguise*, and $e^{0.1t}$ wins only once $0.1t > \log t$, at $t \approx 36$. we compare logs because the numbers themselves overflow: `1.01 ** 100000` is `inf` in floating point, but its log, $995$, is an ordinary number, the same trick as the log-likelihoods in the lesson's opening. the practical lesson: "exponential beats polynomial" is a statement about the far future. for the sizes you actually run, constants and crossover points decide which algorithm is faster.
