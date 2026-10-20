# probability by simulation

## why this matters

machine learning is reasoning under uncertainty. a spam filter doesn't know an email is spam, it estimates a probability. a model's accuracy on a test set isn't its true accuracy, it's an estimate from a random sample, and if you test on 100 examples, "92% accurate" could easily be 87% or 97% in truth. a training run starts from random weights and sees the data in a random order, so running it again gives a slightly different model. you can't do ML well without being able to say how much a number would change if you rolled the dice again.

this topic gives you two ways to answer probability questions. the first is **counting**: when all outcomes are equally likely, a probability is a fraction. the second, far more general, is **simulation**: make the computer play out the random process many times and average what happens. this is called the **Monte Carlo method** (after the casino), and it works even when the math is hopeless. the price is that a simulated answer is itself random, so it must come with an **error bar**, and the size of that error bar follows one of the most important laws in statistics: it shrinks like $1/\sqrt n$. `mc_estimate`, which returns an estimate together with its standard error, goes into your library (`mlfs.numerics`).

## warm-up: predict first

1. you roll two dice. which total is most likely, and how likely is it?
2. how many people do you need in a room for a 50% chance that two of them share a birthday?
3. you flip a fair coin 100 times. roughly how far from 50 heads would you typically be?
4. a simulation with 10,000 samples estimates something to about 2 decimal places. how many samples would you need for 3 decimal places?
5. a game pays \$10 with probability 10% and nothing otherwise. what's it worth on average?

## prerequisites check

- [ ] you know a probability is a number between 0 and 1, and that the probabilities of all possible outcomes add up to 1 (high school)
- [ ] you can compute averages and square roots, and sum with `np.sum` (→ ch 00 · topic 02)
- [ ] you know what a function that returns a function or takes a function looks like (→ ch 01 · topic 01)
- [ ] you know integrals are areas (→ ch 01 · topic 03), and $\pi$ is the area of a circle of radius 1

## core ideas

### idea 1: counting outcomes

when an experiment has a finite number of **equally likely outcomes**, the probability of an **event** (a set of outcomes) is

$$P(\text{event}) = \frac{\text{number of outcomes in the event}}{\text{total number of outcomes}}.$$

roll two dice. there are $6 \times 6 = 36$ equally likely outcomes, as *ordered* pairs: (first die, second die). the total is 7 for six of them, $(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)$, so $P(\text{total} = 7) = \frac6{36} = \frac16$ (warm-up 1). the total is 2 only for $(1, 1)$, so $P(2) = \frac1{36}$. a classic mistake is to count $(3, 4)$ and $(4, 3)$ as one outcome. they're different outcomes (paint one die red and you'll see), and treating them as one breaks the "equally likely" assumption.

listing, for each possible value, its probability gives the **probability distribution**, or **probability mass function** (pmf), of the total:

| total | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ways | 1 | 2 | 3 | 4 | 5 | 6 | 5 | 4 | 3 | 2 | 1 |

a computer can count by brute force: `itertools.product(range(1, 7), repeat=k)` lists all $6^k$ outcomes of $k$ dice. that works for 3 dice (216 outcomes), but not for 30 ($6^{30} \approx 2 \times 10^{23}$). which is why you also need simulation. to simulate, you need random numbers: in NumPy, always through a **generator**, `rng = np.random.default_rng(seed)`. the same seed gives the same "random" numbers every time, which makes experiments reproducible, the habit chapter 00 started.

> **▶ build: stage 1.** implement `dice_sum_pmf(n_dice=2)` by counting, and `roll_dice(n, n_dice=2, *, rng=None)` by simulation.

### idea 2: frequencies, expectation and variance

roll two dice 100 times and count how often each total appears. the **frequencies** (counts divided by 100) won't match the probabilities exactly, but roll 100,000 times and they get very close. that's the **law of large numbers**: in the long run, the fraction of times an event happens approaches its probability. it's what makes simulation work, and what makes probabilities meaningful for things that happen repeatedly.

the **expected value** of a random quantity $X$ is its probability-weighted average:

$$E[X] = \sum_k x_k\,P(X = x_k).$$

for a fair die, $E[X] = (1 + 2 + \dots + 6) \cdot \frac16 = 3.5$, a value the die never actually shows. it's what the *average* of many rolls approaches. for warm-up 5, $E = 0.1 \times 10 + 0.9 \times 0 = 1$: a fair price for that game is \$1. note it's not the plain average of the possible payouts (\$5): rare outcomes count less.

expected value says where a distribution is centered. **variance** says how spread out it is: the expected squared distance from the center,

$$\operatorname{Var}[X] = E\big[(X - E[X])^2\big] = \sum_k P(X = x_k)\,(x_k - E[X])^2.$$

its square root, the **standard deviation** $\sigma$, is in the same units as $X$ and measures a "typical" distance from the mean. for a die, $\operatorname{Var} = \frac{35}{12} \approx 2.92$ and $\sigma \approx 1.71$.

> **🔮 predict:** you roll two dice and add them. the variance of one die is $\frac{35}{12}$. what do you think the variance of the total is?

> **▶ build: stage 2.** implement `empirical_pmf(samples, values)`, `expectation(values, probs)` and `variance(values, probs)`.

### idea 3: the sample mean and its standard error

suppose you don't know $E[X]$ and want to estimate it. draw $n$ independent samples $x_1, \ldots, x_n$ and average them:

$$\bar x = \frac1n\sum_{i=1}^n x_i.$$

$\bar x$ is itself random: draw another $n$ samples and you get a different average. how much does it vary? here's the key fact (you'll prove it in chapter 04): averaging $n$ independent draws divides the variance by $n$, so the standard deviation of $\bar x$ is

$$\frac{\sigma}{\sqrt n}.$$

this is the **standard error** of the mean. it shrinks like $1/\sqrt n$, which is slow: to make it 10 times smaller you need **100 times** more samples (warm-up 4). for 100 coin flips counted as 0 or 1, $\sigma = 0.5$, so the fraction of heads has standard error $0.5/\sqrt{100} = 0.05$, i.e. a typical miss of about 5 heads out of 100 (warm-up 3).

you don't know $\sigma$ either, but you can estimate it from the same samples with the **sample standard deviation**

$$s = \sqrt{\frac{1}{n - 1}\sum_{i=1}^n (x_i - \bar x)^2}.$$

why $n - 1$ and not $n$? the deviations are measured from $\bar x$, which was fitted to these very samples, so they come out slightly too small on average. dividing by $n - 1$ exactly corrects for that (chapter 05 proves it). in NumPy that's `np.std(x, ddof=1)`. the estimated standard error is $s/\sqrt n$.

> **▶ build: stage 3.** implement `mean_and_se(x)`, returning the sample mean and $s/\sqrt n$.

### idea 4: Monte Carlo: estimate anything that's an average

here's the general recipe. many quantities you'd like to know are expected values $E[f(X)]$ of something: a probability is the expected value of an **indicator** (1 if the event happens, 0 if not), an average payout is an expected value, and as you'll see in idea 5, an area is too. to estimate $E[f(X)]$:

1. draw $n$ samples $x_1, \ldots, x_n$ of $X$;
2. compute $f(x_1), \ldots, f(x_n)$;
3. report their mean, **plus or minus its standard error**.

that's **Monte Carlo estimation**. the error bar is what makes it science rather than guesswork. a common rule: the true value is within about **2 standard errors** of the estimate about 95% of the time. (why 2, and why 95%? because averages of many draws have a bell-shaped distribution, whatever the shape of the draws themselves: the central limit theorem, chapter 04. for now, take it as a rule of thumb, and watch the lab test it.)

**the birthday problem.** in a room of $k$ people with random birthdays, what's the chance that at least two share one? counting is possible but fiddly. simulation is easy: draw $k$ random days from 365, check for a repeat, and repeat that 10,000 times. the answer surprises nearly everyone: with only **23** people the chance is about 51% (warm-up 2). there are $\binom{23}{2} = 253$ *pairs* of people, and each pair has a $1/365$ chance of matching.

in code, you'll pass the sampler as a function `sample(n, rng)` that returns $n$ draws, and the quantity as a function `f`. the estimator itself doesn't need to know anything about birthdays, dice or darts, which is what makes it reusable.

> **🔮 predict:** you estimate a probability near $0.5$ with 10,000 simulated trials. roughly how big is the standard error?

> **▶ build: stage 4.** implement `mc_estimate(f, sample, n=10000, *, rng=None)`, returning `(estimate, standard error)`.

### idea 5: areas by throwing darts

throw a dart uniformly at random at the square $[-1, 1] \times [-1, 1]$, which has area 4. the circle of radius 1 inside it has area $\pi$, so the dart lands in the circle with probability $\pi/4$. a dart at $(x, y)$ is in the circle when $x^2 + y^2 \le 1$. so

$$\pi = 4\,P(\text{in circle}) = E\big[4 \cdot \mathbf{1}\{x^2 + y^2 \le 1\}\big],$$

where $\mathbf 1\{\cdot\}$ is 1 when the condition holds and 0 otherwise. that's an expected value, so your `mc_estimate` can compute it: each dart scores 4 or 0, and the average score estimates $\pi$. each score has variance $4^2 \cdot \frac\pi4(1 - \frac\pi4) = \pi(4 - \pi) \approx 2.70$, so with $n$ darts the standard error is about $\sqrt{2.70/n} = 1.64/\sqrt n$: about $0.016$ for 10,000 darts. two correct digits.

compare that with topic 03: the trapezoid rule's error shrinks like $1/n^2$, which crushes Monte Carlo's $1/\sqrt n$ in one dimension. so why would anyone throw darts? because of **dimension**. a grid with 10 points per side needs $10^d$ points in $d$ dimensions, which is hopeless by $d = 20$. Monte Carlo's $1/\sqrt n$ doesn't depend on the dimension at all. averages over thousands of dimensions (all the weights of a neural network, all the pixels of an image) are exactly what Bayesian ML (chapter 12) and generative models (chapters 21–22) need, and Monte Carlo is how they get them.

> **▶ build: stage 5.** implement `estimate_pi(n=10000, *, rng=None)` using your `mc_estimate`.

## worked example

**two dice, by counting.** the total 7 occurs in 6 of 36 outcomes: $P(7) = \frac16 \approx 0.167$. by the table in idea 1, the expected total is $\sum_s s \cdot P(s) = \frac{2 \cdot 1 + 3 \cdot 2 + \dots + 12 \cdot 1}{36} = \frac{252}{36} = 7$.

**a mean and its standard error, by hand.** the samples $1, 2, 3, 4$ have mean $\bar x = 2.5$. the squared deviations are $2.25, 0.25, 0.25, 2.25$, adding to $5$. so

$$s = \sqrt{\frac{5}{4 - 1}} = \sqrt{1.667} = 1.291, \qquad \text{se} = \frac{1.291}{\sqrt 4} = 0.645.$$

(dividing by $n = 4$ instead would give $s = 1.118$ and $\text{se} = 0.559$.)

**$\pi$ with 10,000 darts.** suppose 7,860 of them land in the circle. the estimate is $4 \times 0.786 = 3.144$. each score is 4 (7,860 times) or 0 (2,140 times), with sample standard deviation about $4\sqrt{0.786 \times 0.214} = 1.64$, so the standard error is $1.64/\sqrt{10{,}000} = 0.016$. report $\pi \approx 3.144 \pm 0.016$, which comfortably contains $3.1416$.

the tests check the dice distribution, the hand computation $(2.5, 0.645)$, and that your $\pi$ is within a few standard errors of the truth with a standard error near $\sqrt{\pi(4 - \pi)/n}$.

## on paper

1. **three dice.** how many of the $216$ outcomes of three dice total 10? (hint: count the ways to write 10 as an ordered sum of three numbers from 1 to 6.)

<details><summary>solution</summary>

the unordered combinations are $\{1,3,6\}, \{1,4,5\}, \{2,2,6\}, \{2,3,5\}, \{2,4,4\}, \{3,3,4\}$. combinations with three different numbers can be ordered in $3! = 6$ ways and those with one repeat in 3 ways: $6 + 6 + 3 + 6 + 3 + 3 = 27$. so $P(10) = \frac{27}{216} = 0.125$.

</details>

2. **the birthday problem, exactly.** compute the probability that $k$ people all have *different* birthdays, and use it for $k = 23$.

<details><summary>solution</summary>

the first person can have any birthday; the second must avoid one day ($\frac{364}{365}$), the third two days ($\frac{363}{365}$), and so on. so $P(\text{all different}) = \prod_{i=0}^{k-1}\frac{365 - i}{365}$. for $k = 23$ this is about $0.493$, so $P(\text{shared}) \approx 0.507$.

</details>

3. **the standard error of a probability.** a Monte Carlo estimate of a probability $p$ averages $n$ indicators (0 or 1). show that one indicator has variance $p(1 - p)$, so the standard error is $\sqrt{p(1 - p)/n}$. what's the largest it can be for $n = 10{,}000$?

<details><summary>solution</summary>

$E[\mathbf 1] = p$ and $E[\mathbf 1^2] = E[\mathbf 1] = p$, so $\operatorname{Var} = E[(\mathbf 1 - p)^2] = p(1 - p)^2 + (1 - p)p^2 = p(1 - p)$. it's largest at $p = \frac12$: $\sqrt{0.25/10{,}000} = 0.005$. (that answers the predict callout in idea 4.)

</details>

4. **how many samples?** your estimate has standard error $0.02$ with $n = 2{,}500$ samples. how many samples give a standard error of $0.001$?

<details><summary>solution</summary>

the standard error must shrink by a factor of 20, so $n$ must grow by $20^2 = 400$: $n = 1{,}000{,}000$.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the birthday problem by simulation, with error bars, for any room size.
2. how the error of your `estimate_pi` shrinks as the number of darts grows from 10 to a million.
3. how often a "95% error bar" (estimate ± 1.96 standard errors) actually contains the true value, for a common and a rare event, with few and many samples.

## check your understanding

1. why is $P(\text{two dice total } 7)$ equal to $\frac{6}{36}$ and not $\frac{3}{21}$?

<details><summary>answer</summary>

there are 21 unordered pairs, but they aren't equally likely: $\{3, 4\}$ can happen two ways, $(3, 4)$ and $(4, 3)$, while $\{3, 3\}$ only one way. the 36 ordered pairs are equally likely, so you count those.

</details>

2. (the predict callout in idea 2.) what's the variance of the total of two dice?

<details><summary>answer</summary>

$2 \times \frac{35}{12} = \frac{35}{6} \approx 5.83$. for independent quantities, variances add (chapter 04). you can check it with your `variance` and `dice_sum_pmf(2)`.

</details>

3. a die's expected value is 3.5, which it can never show. what does $E[X] = 3.5$ actually mean?

<details><summary>answer</summary>

it's the long-run average: the average of many rolls approaches 3.5 (law of large numbers). it's a property of the distribution, not a prediction for a single roll.

</details>

4. you halve the standard error of a simulation. how much longer does it take to run?

<details><summary>answer</summary>

four times as long: the standard error is $\sigma/\sqrt n$, so halving it needs $4n$ samples.

</details>

5. why must `mc_estimate` pass the *generator*, and not the seed, on to `sample`?

<details><summary>answer</summary>

if every function re-created a generator from the same seed, separate calls would reuse the same "random" numbers, and an experiment calling `sample` twice would get identical draws. passing one generator along means each draw continues its stream: reproducible from the starting seed, but independent from call to call.

</details>

6. a Monte Carlo estimate says $0.031 \pm 0.004$. is the true value 0.031?

<details><summary>answer</summary>

almost certainly not exactly. it's probably within about $\pm 0.008$ (two standard errors) of 0.031, so somewhere around 0.023 to 0.039. the error bar is the honest summary.

</details>

7. your simulation of a rare event ($p \approx 0.001$) with 500 trials sees it zero times. what does `mean_and_se` report, and what's wrong with that?

<details><summary>answer</summary>

$(0, 0)$: every value is 0, so $s = 0$ and the standard error is 0. it claims perfect certainty that the probability is 0, which is false. the $s/\sqrt n$ standard error is unreliable when the event is rarely or never seen. the lab shows how often this bites, and chapter 05 gives better intervals for proportions.

</details>

## common bugs

- **`rng.integers(1, 6)`.** the upper end is excluded, so 6 never comes up. symptom: dice averages near 3 instead of 3.5. use `rng.integers(1, 7)`.
- **global randomness.** `np.random.randint` or `np.random.seed` ignore your `rng`. symptom: results that don't reproduce, or don't change when you change the seed.
- **dividing by $n$ for the standard deviation.** symptom: standard errors slightly too small (by a factor $\sqrt{(n - 1)/n}$), noticeable for small $n$.
- **forgetting the $\sqrt n$.** reporting $s$ instead of $s/\sqrt n$. symptom: error bars that don't shrink as you add samples.
- **re-seeding inside a loop.** calling `default_rng(0)` on every repetition gives the same draws every time. symptom: an experiment repeated 1,000 times shows zero variation.
- **forgetting the factor 4.** the fraction of darts in the circle estimates $\pi/4$. symptom: $\pi \approx 0.785$.

## going deeper

- Blitzstein & Hwang (2019), *Introduction to Probability*, 2nd ed., CRC Press, chapters 1–4 (counting, conditional probability, random variables, expectation). free online. [doi:10.1201/9780429428357](https://doi.org/10.1201/9780429428357)
- Metropolis & Ulam (1949), "The Monte Carlo method", *Journal of the American Statistical Association* 44(247): 335–341. the paper that named the method. [doi:10.1080/01621459.1949.10483310](https://doi.org/10.1080/01621459.1949.10483310)
- Owen (2013), *Monte Carlo Theory, Methods and Examples*, chapters 1–2 (simple Monte Carlo and its error). a free online draft is available from the author's website
- O'Neill (2014), "PCG: a family of simple fast space-efficient statistically good algorithms for random number generation", Harvey Mudd College technical report HMC-CS-2014-0905. the algorithm behind NumPy's default generator.

## connections

- ← builds on: functions as arguments (topic 01), integrals as areas (topic 03), NumPy arrays and seeds (ch 00 · topics 02, 05).
- → used in: random variables, the central limit theorem and samplers built from uniforms (ch 04), confidence intervals and the bootstrap (ch 05), honest evaluation with error bars from chapter 07 on, Monte Carlo inference in Bayesian ML (ch 12), and stochastic gradient descent, which estimates a gradient from a random sample of the data (ch 06, ch 14).
