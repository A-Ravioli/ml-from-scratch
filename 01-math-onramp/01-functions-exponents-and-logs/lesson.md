# functions, exponents & logs

## why this matters

open almost any machine learning paper and you'll find $e^x$ and $\log x$ on the first page. a classifier turns scores into probabilities with $e^x$ (the softmax). a model is trained by maximizing a *log*-likelihood. learning rates are searched on a log scale ($10^{-4}, 10^{-3}, 10^{-2}$), and "exponential growth" is how people describe everything from compute budgets to the number of possible sentences.

here's a concrete reason logs matter. suppose a model assigns probability $0.01$ to each of 1,000 independent events. the probability of all of them together is $0.01^{1000} = 10^{-2000}$. try that on a computer: `0.01 ** 1000` gives `0.0`, because the smallest positive float is about $10^{-308}$. the answer has vanished. its logarithm, $1000 \cdot \log 0.01 \approx -4605$, is a perfectly ordinary number. logs turn products of tiny numbers into sums of reasonable ones, and that's why ML works in log space.

but there's a puzzle underneath. your computer's processor can only add, subtract, multiply and divide. so how does `np.exp(2.3)` come up with $9.974182454814718$? in this topic you answer that by building $e^x$ and $\log x$ yourself, from nothing but arithmetic. along the way you'll meet three ideas that come back again and again in this course: **functions you can combine and undo**, **infinite sums that you cut off when they're accurate enough**, and **Newton's method**, the fastest way to solve an equation.

## warm-up: predict first

write down a quick guess for each. you'll find the answers in the lesson.

1. $2^{10}$ is about a thousand. about how big is $2^{40}$?
2. which is bigger when $n = 1000$: $n^5$ or $1.01^n$? what about when $n = 10{,}000$?
3. $e = 1 + 1 + \frac12 + \frac16 + \frac1{24} + \dots$. how many terms do you need to get $e$ right to 10 digits?
4. you're trying to pin down a number between 0 and 1 by repeatedly halving an interval. how many halvings until you know it to within $10^{-12}$?
5. what is $\log(1000 \times 0.000001)$, if $\log$ means $\log_{10}$?

## prerequisites check

- [ ] you can simplify expressions like $2^3 \cdot 2^4$, $(3^2)^3$ and $10^{-2}$ (high school algebra)
- [ ] you know what a function is: a rule that turns each input into one output (high school algebra)
- [ ] you can write a Python function and loop over a range (→ ch 00 · topic 01)
- [ ] you can do arithmetic on whole NumPy arrays at once, like `2 * x + 1` (→ ch 00 · topic 02)
- [ ] you know that floats have about 16 significant digits (→ ch 00 · topic 03)

## core ideas

### idea 1: functions you can combine and undo

a **function** is a machine: put a number in, get one number out. $f(x) = x + 1$ adds one. $g(x) = 2x$ doubles. in Python a function is something you can pass around, store in a variable, and even return from another function.

**composition** means feeding one machine's output into the next. "$f$ after $g$", written $f \circ g$, is the function

$$(f \circ g)(x) = f(g(x)).$$

order matters, and it reads right to left: $g$ runs *first*. with $f(x) = x + 1$ and $g(x) = 2x$, $(f \circ g)(5) = f(10) = 11$, but $(g \circ f)(5) = g(6) = 12$. ML models are long compositions: a neural network is $f_3(f_2(f_1(x)))$, one layer after another. when you get to the chain rule in topic 02, it'll be the rule for taking compositions apart.

an **inverse** undoes a function. if $f$ turns $x$ into $y$, then $f^{-1}$ turns $y$ back into $x$:

$$f^{-1}(f(x)) = x \quad\text{and}\quad f(f^{-1}(y)) = y.$$

the inverse of "add one" is "subtract one". the inverse of $x^3$ is the cube root. not every function has one: $x^2$ sends both $3$ and $-3$ to $9$, so "undo" is ambiguous. a function that is **increasing** (bigger inputs always give bigger outputs) never has that problem, so it always has an inverse on its range.

**finding an inverse by computing.** even when you can't write a formula for $f^{-1}$, you can compute $f^{-1}(y)$: it's the $x$ that solves $f(x) = y$. for an increasing $f$, **bisection** finds it. start with a bracket $[lo, hi]$ where $f(lo) \le y \le f(hi)$, so the answer must be inside. look at the midpoint $mid$. if $f(mid) < y$, the answer is to the right, so the new bracket is $[mid, hi]$. otherwise it's $[lo, mid]$. every step halves the bracket, and the answer never escapes.

how many steps does it take? after $n$ halvings, a bracket of width $W$ has width $W/2^n$. to get below a tolerance $\tau$ you need $2^n \ge W/\tau$. for $W = 1$ and $\tau = 10^{-12}$, that's $2^n \ge 10^{12}$, and since $2^{10} \approx 10^3$, $n \approx 40$ (warm-up 4). each step buys one binary digit, about 0.3 decimal digits. that's reliable, but slow. by idea 4 you'll have something much faster.

one practical catch: a float can't be halved forever. near $10^6$, neighbouring floats are about $10^{-10}$ apart, so a bracket there can never shrink to $10^{-12}$. that's why the stage caps the loop at 200 halvings.

> **▶ build: stage 1.** implement `compose(f, g)`, which returns a new function, and `bisect_inverse(f, y, lo, hi, *, tol=1e-12)`, which solves $f(x) = y$ by bisection.

### idea 2: exponents, $e$, and an infinite sum

**exponent rules.** $a^n$ means $n$ copies of $a$ multiplied together. everything else follows from counting copies:

$$a^m \cdot a^n = a^{m+n}, \qquad (a^m)^n = a^{mn}, \qquad a^0 = 1, \qquad a^{-n} = \frac{1}{a^n}, \qquad a^{1/2} = \sqrt a.$$

the first rule is the important one for this topic: **multiplying powers adds exponents**. ($2^{40} = (2^{10})^4 \approx (10^3)^4 = 10^{12}$, warm-up 1.)

**the number $e$.** put \$1 in a bank paying 100% interest a year. compounded once, you get \$2. compounded monthly, $(1 + \frac1{12})^{12} \approx 2.613$. daily, $(1 + \frac1{365})^{365} \approx 2.7146$. as you compound more and more often, the total approaches a limit, $e = 2.718281828\ldots$. the function $e^x$ is continuous growth at rate $x$, and it has a unique property that you'll prove in topic 02: **its slope at every point equals its value.** that's why it appears every time something grows (or shrinks) in proportion to its current size.

**$e^x$ as a sum.** here is how a computer can actually evaluate $e^x$:

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \dots = \sum_{k=0}^{\infty} \frac{x^k}{k!},$$

where $k! = 1 \cdot 2 \cdots k$ (and $0! = 1$). why this sum? think of "slope equals value". the slope of $x^k/k!$ is $k x^{k-1}/k! = x^{k-1}/(k-1)!$, the term before it. so taking the slope of the whole sum shifts every term down one place, and you get the same sum back. a function whose slope is itself, and which equals 1 at $x = 0$, is $e^x$.

an infinite sum can't be computed, but it can be **cut off**. the $k$-th term is the previous one times $x/k$. once $k$ is bigger than $|x|$, each term is smaller than the last, and soon the terms are too tiny to matter. this ratio is also how you should compute the terms: `term = term * x / k`. no factorials (which overflow: $171!$ is bigger than the largest float), no powers, one multiplication and one division per term.

> **🔮 predict:** how many terms of the series do you need for $e^1$ to 10 correct digits? for $e^{20}$? for $e^{-20}$?

> **▶ build: stage 2.** implement `exp_series(x, n_terms)`: the sum of the first `n_terms` terms, for a float or every entry of an array.

### idea 3: range reduction, an exponent rule to the rescue

the series converges for every $x$, but "converges" hides two problems.

**problem 1: big $x$ needs many terms.** the terms only start shrinking once $k > |x|$. for $x = 1$, 13 terms give 10 digits. for $x = 20$ you need 55.

**problem 2: negative $x$ is a disaster.** for $x = -30$ the terms alternate in sign and grow huge before they shrink: the biggest is $30^{30}/30! \approx 7.8 \times 10^{11}$. the answer, $e^{-30} \approx 9.4 \times 10^{-14}$, is what's left after those giants cancel each other out. a float carries about 16 significant digits, so each giant term carries a rounding error around $10^{12} \times 10^{-16} = 10^{-4}$, about a billion times bigger than the answer. no number of extra terms fixes this. it's **catastrophic cancellation**, which you met in chapter 00.

**the fix is an exponent rule.** split $x$ into a whole number $k$ and a small remainder $r$:

$$x = k + r,\quad k = \operatorname{round}(x),\quad |r| \le \tfrac12 \qquad\Longrightarrow\qquad e^x = e^{k + r} = e^k \cdot e^r.$$

now $e^r$ comes from the series with tiny $r$: no cancellation to speak of, and 20 terms are far more than enough. $e^k$ is just $e$ multiplied by itself $k$ times, which a computer does accurately (`E**k` with $E = e$). and $e$ itself? that's `exp_series(1.0, 20)`. real math libraries do the same thing with $x = k \ln 2 + r$, so that $e^k$ becomes $2^k$, which a computer can apply exactly by changing a float's exponent. this trick, turning a hard input into an easy one using an identity, is called **range reduction**, and you'll use it again for $\log$.

> **▶ build: stage 3.** implement `exp(x)` with range reduction. it should match `np.exp` to about 13 digits for any $x$ between $-700$ and $700$.

### idea 4: logarithms, and undoing $e^x$ with Newton's method

the **natural logarithm** is the inverse of $e^x$: $\log y$ is the number $x$ with $e^x = y$.

$$\log(e^x) = x, \qquad e^{\log y} = y \quad (y > 0).$$

since $e^x$ is always positive, $\log$ only takes positive inputs. every exponent rule becomes a log rule, read backwards. multiplying powers adds exponents, so logs turn products into sums:

$$\log(ab) = \log a + \log b, \qquad \log(a^p) = p \log a, \qquad \log 1 = 0, \qquad \log\frac1a = -\log a.$$

numbers below 1 have negative logs. logs in other bases are rescaled natural logs: $\log_{10} y = \log y / \log 10$, and $\log_2 y = \log y / \log 2$. (warm-up 5: $\log_{10}(10^3 \cdot 10^{-6}) = 3 - 6 = -3$.)

**computing $\log y$.** you could use your `bisect_inverse` on $e^x$: about 40 steps for 12 digits. Newton's method does it in about 5. the idea: to solve $g(x) = 0$, stand at your current guess $x$, replace the curve by its **tangent line** there, and jump to where the tangent line hits zero. a line through $(x, g(x))$ with slope $g'(x)$ hits zero at

$$x_{\text{new}} = x - \frac{g(x)}{g'(x)}.$$

for $\log y$ the equation is $g(x) = e^x - y = 0$, and the slope of $e^x$ is $e^x$ itself (idea 2). so

$$x_{\text{new}} = x - \frac{e^x - y}{e^x} = x - 1 + y\,e^{-x}.$$

**why it's so fast.** call the error $t = x - \log y$. then $y\,e^{-x} = e^{\log y - x} = e^{-t}$, so the update changes the error to

$$t_{\text{new}} = t - 1 + e^{-t} = \frac{t^2}{2} - \frac{t^3}{6} + \dots \approx \frac{t^2}{2},$$

using the series for $e^{-t}$ from idea 2. the new error is about the *square* of the old one. an error of $10^{-3}$ becomes about $10^{-6}$, then $10^{-12}$: **the number of correct digits doubles every step.** this is called quadratic convergence.

there's a catch: the squaring only kicks in once you're close. far away it can be slow or wild. if $x$ is much too big ($t$ large and positive), $t - 1 + e^{-t} \approx t - 1$, so the guess creeps down by one per step. if $x$ is much too small ($t$ very negative), $e^{-t}$ is enormous and the first step flies off to a huge guess. good starting guesses matter.

> **🔮 predict:** Newton for $\log 2$, starting from $x_0 = 1$. the true answer is $0.693147\ldots$. how many correct digits do you expect after one step? after two? after three?

> **▶ build: stage 4.** implement `newton_log_steps(y, x0, n_steps)`, returning every iterate $[x_0, x_1, \ldots]$ so you can watch the digits double. use your own `exp`.

### idea 5: an accurate log, and how fast things grow

to give Newton a good start for *any* $y$, shrink $y$ first, using the log rules. every float is stored as a **mantissa** times a power of two, $y = m \cdot 2^p$, with $\frac12 \le m < 1$ and $p$ a whole number. NumPy hands you both: `m, p = np.frexp(y)`. that isn't cheating: you could find them yourself by halving $y$ while it's at least 1, or doubling it while it's below $\frac12$, counting the steps in $p$. every halving or doubling is exact in floating point, and since floats span about $2^{-1074}$ to $2^{1024}$, it never takes more than about 1,075 steps. `frexp` just reads the answer straight out of the way the float is stored. then

$$\log y = \log m + p \log 2.$$

now Newton only ever has to find $\log m$ for $m \in [\frac12, 1)$, where $\log m \in [-0.69, 0]$. a good first guess is $x_0 = m - 1$, because $\log(1 + u) \approx u$ for small $u$. the worst case is $m = \frac12$, where the starting error is $-0.5 - \log 0.5 \approx 0.19$. squaring (and halving) each step: $0.19 \to 0.018 \to 1.5 \times 10^{-4} \to 1.2 \times 10^{-8} \to 7 \times 10^{-17}$. four steps reach full float accuracy, and the stage uses six for safety. $\log 2$ itself comes from `newton_log_steps(2.0, 0.7, 6)`.

**growth rates.** logs also let you compare numbers far too big to compute. which is bigger, $1.01^n$ or $n^5$? take logs of both: compare $n \log 1.01 \approx 0.00995\,n$ with $5 \log n$. at $n = 1000$: $9.95$ versus $34.5$, so $n^5$ wins by a factor of $e^{24.6} \approx 5 \times 10^{10}$. at $n = 10{,}000$: $99.5$ versus $46.1$, and now $1.01^n$ wins by $e^{53} \approx 10^{23}$ (warm-up 2). the crossover is near $n \approx 4200$.

the general pattern, which you'll lean on all through the course:

$$\log n \;\ll\; n^{a} \;\ll\; b^{n} \qquad \text{for large } n, \text{ any } a > 0,\ b > 1.$$

any exponential eventually beats any power, and any power eventually beats a log. "eventually" can be very late, though. the lab finds out how late.

> **▶ build: stage 5.** implement `log(y)` with frexp range reduction and six Newton steps. it should match `np.log` from $10^{-300}$ to $10^{300}$.

## worked example

**$e^{2.3}$ by range reduction.** $k = \operatorname{round}(2.3) = 2$ and $r = 0.3$. then

$$e^{2.3} = e^2 \cdot e^{0.3} = 7.389056 \times 1.349859 = 9.974182,$$

matching `np.exp(2.3)`. (the tests also check `exp_series(1.0, 4)`: $1 + 1 + \frac12 + \frac16 = \frac83 \approx 2.6667$. four terms means $k = 0, 1, 2, 3$.)

**$\log 2$ by Newton, from $x_0 = 1$.** the update is $x_{\text{new}} = x - 1 + 2e^{-x}$.

| step | $x$ | error $x - \log 2$ |
|---|---|---|
| 0 | $1$ | $3.1 \times 10^{-1}$ |
| 1 | $1 - 1 + 2/e = 0.7357588823$ | $4.3 \times 10^{-2}$ |
| 2 | $0.7357588823 - 1 + 2e^{-0.7357588823} = 0.6940422999$ | $9.0 \times 10^{-4}$ |
| 3 | $0.6931475811$ | $4.0 \times 10^{-7}$ |
| 4 | $0.6931471805600$ | $8.0 \times 10^{-14}$ |

each error is roughly half the square of the one before: $(0.043)^2/2 \approx 9 \times 10^{-4}$, $(9 \times 10^{-4})^2/2 \approx 4 \times 10^{-7}$. your `newton_log_steps(2.0, 1.0, 2)` must give the first three rows exactly.

## on paper

1. **how many terms?** the series for $e^1$, cut off after $N$ terms, leaves out $\frac1{N!} + \frac1{(N+1)!} + \dots$. show this tail is less than $\frac{2}{N!}$, and use it to find how many terms give an error below $10^{-10}$.

<details><summary>solution</summary>

$\frac1{(N+1)!} = \frac1{N!}\cdot\frac1{N+1} \le \frac1{N!}\cdot\frac12$ for $N \ge 1$, and each later term is at most half the one before. so the tail is at most $\frac1{N!}(1 + \frac12 + \frac14 + \dots) = \frac2{N!}$. you need $\frac2{N!} < 10^{-10}$, i.e. $N! > 2 \times 10^{10}$. $13! \approx 6.2 \times 10^9$ is too small; $14! \approx 8.7 \times 10^{10}$ works. so 14 terms are guaranteed, and in practice 13 already give a relative error of $6 \times 10^{-11}$ (warm-up 3). the lab measures this for other $x$.

</details>

2. **Newton's update.** derive $x_{\text{new}} = x - 1 + y\,e^{-x}$ from the tangent line to $g(x) = e^x - y$.

<details><summary>solution</summary>

the tangent line at the guess $x$ is $\ell(z) = g(x) + g'(x)(z - x) = (e^x - y) + e^x(z - x)$. setting $\ell(z) = 0$: $z = x - (e^x - y)/e^x = x - 1 + y e^{-x}$.

</details>

3. **digits double.** with $t = x - \log y$, show that the update gives $t_{\text{new}} = t - 1 + e^{-t}$, and use the series for $e^{-t}$ to show $t_{\text{new}} \approx t^2/2$ for small $t$.

<details><summary>solution</summary>

$y e^{-x} = e^{\log y} e^{-x} = e^{-(x - \log y)} = e^{-t}$. so $x_{\text{new}} - \log y = (x - \log y) - 1 + e^{-t} = t - 1 + e^{-t}$. with $e^{-t} = 1 - t + \frac{t^2}{2} - \frac{t^3}{6} + \dots$, this is $\frac{t^2}{2} - \frac{t^3}{6} + \dots$. for small $t$ the first term dominates.

</details>

4. **bisection's cost.** a bracket of width $W$ is halved until its width is at most $\tau$. how many halvings is that, as a formula? evaluate it for $W = 10$, $\tau = 10^{-12}$.

<details><summary>solution</summary>

after $n$ halvings the width is $W/2^n$, so you need $2^n \ge W/\tau$, i.e. $n \ge \log_2(W/\tau) = \log(W/\tau)/\log 2$. for $W/\tau = 10^{13}$: $13 \log 10 / \log 2 = 13 \times 3.32 \approx 43.2$, so 44 halvings.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. how many series terms 10-digit accuracy takes, for different $x$, with and without range reduction.
2. a race between your bisection and your Newton's method, both computing $\log 10$: error after each step.
3. growth rates: when does an exponential like $1.01^n$ overtake a power like $n^5$, and when does $n^{0.1}$ overtake $\log n$?

## check your understanding

1. with $f(x) = x^2$ and $g(x) = x + 3$, what are $(f \circ g)(1)$ and $(g \circ f)(1)$?

<details><summary>answer</summary>

$(f \circ g)(1) = f(4) = 16$ and $(g \circ f)(1) = g(1) = 4$. composition isn't commutative.

</details>

2. why does bisection need $f$ to be increasing (or at least continuous with $f(lo) \le y \le f(hi)$)?

<details><summary>answer</summary>

the step "if $f(mid) < y$, the answer is to the right" assumes that values below $y$ are on the left and values above are on the right. for an increasing function that's true. for a wiggly one, the answer could be on either side (or there could be several answers), and bisection might throw away the half that contains the one you wanted.

</details>

3. in `exp_series`, why compute each term as `term * x / k` instead of `x**k / math.factorial(k)`?

<details><summary>answer</summary>

it's cheaper (one multiply and one divide per term) and it never overflows. $x^k$ and $k!$ separately can both exceed the largest float ($171!$ already does) even when their ratio is tiny. the ratio form only ever stores the term itself.

</details>

4. the raw series for $e^{-30}$ gives garbage. why doesn't the same problem hit $e^{+30}$?

<details><summary>answer</summary>

for $x = +30$ every term is positive, so nothing cancels: the rounding error in each term is tiny *relative to the answer*, which is the biggest number around. for $x = -30$ the terms alternate in sign and the answer is a tiny leftover after huge terms cancel, so their rounding errors swamp it. (another fix: $e^{-30} = 1/e^{30}$.)

</details>

5. what goes wrong if you run Newton for $\log(10^6)$ starting from $x_0 = 0$?

<details><summary>answer</summary>

the error is $t = -13.8$, and the first step gives $x_1 = 0 - 1 + 10^6 e^0 \approx 10^6$. now the guess is far too big, and each step only lowers it by about 1 ($t_{\text{new}} \approx t - 1$), so it takes about a million steps to come back. this is why `log` first reduces $y$ to $m \in [\frac12, 1)$ and starts at $m - 1$.

</details>

6. using only $\log 2 \approx 0.693$ and $\log 10 \approx 2.303$, estimate $\log 80$.

<details><summary>answer</summary>

$80 = 2^3 \cdot 10$, so $\log 80 = 3 \log 2 + \log 10 \approx 2.079 + 2.303 = 4.382$. (the true value is $4.3820$.)

</details>

7. Newton's method has an error of $10^{-4}$. roughly what is it after one more step?

<details><summary>answer</summary>

about $t^2/2 = 5 \times 10^{-9}$. (in floats it can't go below about $10^{-16}$.)

</details>

8. how long does it take $n^{0.1}$ to overtake $\log n$? guess first, then compare with the lab.

<details><summary>answer</summary>

with $t = \log n$, you're comparing $e^{0.1 t}$ with $t$, i.e. $0.1t$ with $\log t$. that crossover is at $t \approx 35.8$, so $n \approx e^{35.8} \approx 3.4 \times 10^{15}$. powers do beat logs, but a small power takes its time.

</details>

## common bugs

- **composing in the wrong order.** `compose(f, g)` must compute `f(g(x))`. symptom: right answers only when $f$ and $g$ happen to commute.
- **moving the wrong end of the bracket.** if $f(mid) < y$ the answer is to the *right*, so `lo = mid`. symptom: bisection converges to one of the bracket's ends.
- **an infinite bisection loop.** `while hi - lo > tol` never ends when the bracket is so far from zero that floats can't get that close. symptom: the cell hangs. cap the number of halvings.
- **an off-by-one in the number of terms.** `n_terms = 4` means $k = 0, 1, 2, 3$. symptom: `exp_series(1.0, 4)` gives $2.7083$ or $2.5$ instead of $2.6667$.
- **forgetting the $k!$.** updating `term = term * x` gives $x^k$, not $x^k/k!$. symptom: the series blows up for $|x| > 1$.
- **Newton with the raw $y$.** inside `log`, the Newton steps must use the reduced $m$, not $y$. symptom: right for $y$ near 1, wildly wrong (or overflowing) for $y = 10^{100}$.

## going deeper

- Muller (2016), *Elementary Functions: Algorithms and Implementation*, 3rd ed., Birkhäuser. how real math libraries compute $e^x$, $\log x$ and friends: range reduction, polynomial approximation, and correct rounding. [doi:10.1007/978-1-4899-7983-4](https://doi.org/10.1007/978-1-4899-7983-4)
- Tang (1989), "Table-driven implementation of the exponential function in IEEE floating-point arithmetic", *ACM Transactions on Mathematical Software* 15(2): 144–157. the $x = k\ln 2 + r$ reduction in a production algorithm. [doi:10.1145/63522.214389](https://doi.org/10.1145/63522.214389)
- Süli & Mayers (2003), *An Introduction to Numerical Analysis*, Cambridge University Press, chapter 1 (solving nonlinear equations: bisection and Newton's method, with proofs of their convergence rates). [doi:10.1017/CBO9780511801181](https://doi.org/10.1017/CBO9780511801181)
- Goldberg (1991), "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1): 5–48. why cancellation destroys the raw series for negative $x$. [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)

## connections

- ← builds on: Python functions and NumPy arrays (ch 00 · topics 01–02), floating point and cancellation (ch 00 · topic 03).
- → used in: the slope of $e^x$ and the chain rule (topic 02), the normal distribution's $e^{-x^2/2}$ (topic 05), log-likelihoods and entropy (ch 04–05), Newton's method for optimization (ch 06), logistic regression and softmax (ch 08), and log-space numerics everywhere from chapter 12 on.
