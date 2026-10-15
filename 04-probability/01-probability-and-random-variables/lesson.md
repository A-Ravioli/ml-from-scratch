# probability & random variables

## why this matters

a screening test for a rare disease is 90% sensitive (it catches 90% of sick people) and has a 9% false-positive rate. you test positive. how worried should you be? when Gigerenzer and colleagues put exactly these numbers (with a 1% prevalence) to 160 gynecologists in a training session, only about one in five picked the right answer; most picked 81% or 90% (Gigerenzer et al., 2007). the right answer is about 9%. nobody did any hard arithmetic wrong. they skipped a step: they never asked how rare the disease was in the first place.

machine learning is built out of exactly this kind of reasoning. a classifier outputs $P(y \mid x)$. a spam filter combines how often words appear in spam with how common spam is. a language model is a machine for producing conditional distributions over the next token. a Bayesian model updates beliefs as data arrives. every one of those systems is three operations applied over and over: **combine** random quantities, **condition** on what you observed, and **marginalize** away what you don't care about. get those three right and the rest of probability is bookkeeping. get them subtly wrong and you'll be confidently wrong, like the doctors.

in this topic you'll build those operations as code, on distributions with finitely many outcomes, where everything can be computed exactly. you'll end up with Bayes' rule as a two-line function, and you'll use it to settle the base-rate puzzle above and the Monty Hall problem, which has started more arguments than any other three doors in history.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. the screening test above: 1% of people have the disease, the test is 90% sensitive, and 9% of healthy people test positive anyway. what is $P(\text{sick} \mid \text{positive})$?
2. you roll two fair dice and someone tells you they sum to 8. what's the probability you rolled doubles?
3. Monty Hall: you pick door 1 of 3. the host, who knows where the car is, opens door 3 to show a goat and offers you door 2. should you switch? does it matter if the host *didn't* know and just happened to open a goat door?
4. two random variables have zero correlation. are they independent?
5. can two events, each with positive probability, be both disjoint and independent?

## prerequisites check

- [ ] you can work with sets: union, intersection, complement, and the size of a finite set (→ ch 01 · topic 01)
- [ ] you can write a Python class with `__init__` and methods, and loop over a dict's items (→ ch 00 · topic 01)
- [ ] you know a function can be passed as an argument (`lambda x: x % 2 == 0`) (→ ch 00 · topic 01)
- [ ] you can add fractions without a calculator

## core ideas

### idea 1: distributions as objects

start with an experiment whose result you can't predict: roll a die, flip a coin, draw a card. the set of all possible results is the **sample space** $\Omega$. an **event** is a set of results you can ask about, like "the die shows an even number" $= \{2, 4, 6\}$. a **probability** assigns each event a number, following three rules (Kolmogorov's axioms):

1. $P(A) \ge 0$ for every event $A$;
2. $P(\Omega) = 1$;
3. if $A_1, A_2, \dots$ are pairwise disjoint, $P(A_1 \cup A_2 \cup \cdots) = P(A_1) + P(A_2) + \cdots$.

everything else is a consequence. complements: $P(A^c) = 1 - P(A)$. inclusion–exclusion: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (you'll prove it on paper).

when $\Omega$ is finite (or countable), you never need to think about events directly. give each single outcome $\omega$ a probability $p(\omega) \ge 0$ with $\sum_\omega p(\omega) = 1$, the **probability mass function** (pmf), and then any event's probability is a sum:

$$P(A) = \sum_{\omega \in A} p(\omega).$$

axiom 3 is what makes that sum legitimate. so a discrete distribution *is* a table of outcomes and probabilities, and that's exactly what your `Discrete` class stores: a dict from outcomes to probabilities. outcomes can be anything hashable: numbers, strings, or tuples (which is how you'll represent several random quantities at once in idea 4).

a **random variable** is a number (or any value) determined by the outcome: a function $X: \Omega \to \mathbb{R}$. "the die shows an even number" and "the sum of two dice" are random variables. in code, the distinction between "outcome" and "random variable" mostly disappears: a `Discrete` object describes the *distribution* of some random variable, $P(X = x)$ for each value $x$.

two kinds of question come up constantly: $P(X = x)$ (the pmf at a point) and $P(X \in A)$ (the probability of an event). your class answers both. an event can be given as a set of outcomes, or as a predicate, a function that says yes or no to each outcome. predicates are often easier: `lambda x: x % 2 == 0` is the event "even".

> **▶ build: stage 1.** implement `Discrete.__init__` (validate: probabilities finite, non-negative, summing to 1 within $10^{-9}$; drop zeros; store a new dict in `self.probs`), `Discrete.pmf(x)`, and `Discrete.prob(event)` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: functions of random variables, independence, and sums

if $X$ is a random variable and $f$ any function, $f(X)$ is a random variable too. its distribution is

$$P(f(X) = y) = \sum_{x:\ f(x) = y} P(X = x).$$

the sum is the whole point: several values of $x$ can land on the same $y$, and their probabilities pile up. for a fair die, $X \bmod 3$ takes the value $0$ for $x \in \{3, 6\}$, so $P(X \bmod 3 = 0) = 2/6$. this operation is called the **pushforward** of the distribution through $f$. it's the discrete version of the change-of-variables formula from calculus (ch 03 · topic 06), and it's the most common operation in this whole topic: marginals in idea 4 are pushforwards, and so are sums.

two random variables $X$ and $Y$ are **independent** if knowing one tells you nothing about the other. formally, their joint distribution factors:

$$P(X = x, Y = y) = P(X = x)\,P(Y = y) \quad \text{for all } x, y.$$

two dice rolled separately are independent. when you have independent $X$ and $Y$, their joint distribution, a distribution over pairs $(x, y)$, is the **product** of the two pmfs. your `product(X, Y)` builds it.

and now sums come for free. $X + Y$ is a function of the pair $(X, Y)$, so its distribution is the pushforward of the product through $(x, y) \mapsto x + y$:

$$P(X + Y = s) = \sum_x P(X = x)\,P(Y = s - x).$$

this formula is a **convolution**. for two dice, it produces the familiar triangle $1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1$ (over 36) on the sums $2, \dots, 12$. add more dice and the triangle rounds into a bell, which is your first glimpse of the central limit theorem (topic 05).

> **🔮 predict:** you add up 4 fair coin flips (each 0 or 1). which total is most likely, and what's its probability? (stage 2's tests check it.)

> **▶ build: stage 2.** implement `pushforward(X, f)`, `product(X, Y)`, and `add(X, Y)`. build `add` out of the other two, not with its own double loop.

### idea 3: conditioning is restricting and renormalizing

you learn that an event $B$ happened. what's the probability of $A$ now? the outcomes outside $B$ are ruled out, so throw them away. the outcomes inside $B$ keep their *relative* probabilities (nothing you learned favors one over another), but they now have to sum to 1. so divide by $P(B)$:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0.$$

on a pmf, that's a two-step recipe: **restrict** to the outcomes in $B$, then **renormalize**. that's all `condition` does. it returns a new distribution, and you can keep conditioning it, take pushforwards of it, or anything else.

take warm-up 2. the pairs of dice that sum to 8 are $(2,6), (3,5), (4,4), (5,3), (6,2)$: five outcomes, each with probability $1/36$ before conditioning, so each gets $1/5$ after. one of them is doubles. $P(\text{doubles} \mid \text{sum } 8) = 1/5$, compared with $1/6$ unconditionally.

two consequences you'll use constantly. rearranging the definition gives the **multiplication rule** $P(A \cap B) = P(A \mid B)\,P(B)$. and if $B_1, \dots, B_k$ split $\Omega$ into disjoint pieces, adding up the multiplication rule over the pieces gives the **law of total probability**:

$$P(A) = \sum_i P(A \mid B_i)\,P(B_i).$$

conditioning on an event of probability zero is undefined: you'd divide by zero. your code should refuse rather than return garbage. (for continuous variables, every single point has probability zero, and you'll still want to condition on $X = 1.3$. topic 04 shows how that works for Gaussians.)

> **▶ build: stage 3.** implement `condition(X, event)`. raise `ValueError` when $P(\text{event}) = 0$.

### idea 4: joints and marginals

real problems involve several random quantities at once: the weather and whether the grass is wet, a hypothesis and the data, a word and the word after it. their **joint distribution** gives a probability to every combination. in code, you don't need a new class: a joint distribution over $(X, Y)$ is a `Discrete` whose outcomes are tuples $(x, y)$.

to get back the distribution of $X$ alone, add up over everything else:

$$P(X = x) = \sum_y P(X = x, Y = y).$$

this is the **marginal** of $X$ (from the habit of writing the row sums of a table in its margin). notice that it's just a pushforward through the projection $(x, y) \mapsto x$. marginalizing is how you "forget" a variable without ignoring its effect.

with joints and marginals you can test independence: compute both marginals and check $P(x, y) = P(x)P(y)$ at every pair of values $x, y$ in their supports. if $X$ and $Y$ each take value 0 with positive probability, but the pair $(0, 0)$ never happens, they can't be independent. (it turns out you could get away with checking only the pairs the joint lists. check your understanding, question 8, asks why.)

that also answers warm-up 4. let $X$ be uniform on $\{-1, 0, 1\}$ and $Y = X^2$. then $\mathbb{E}[XY] = \mathbb{E}[X^3] = 0 = \mathbb{E}[X]\,\mathbb{E}[Y]$, so their covariance (topic 03) is zero: they're **uncorrelated**. but $Y$ is a function of $X$, as dependent as two variables can be. zero correlation only rules out *linear* relationships; independence rules out *all* of them.

> **🔮 predict:** in the weather table used by the tests, $P(\text{sun}, \text{dry}) = 0.5$, $P(\text{sun}, \text{wet}) = 0.1$, $P(\text{rain}, \text{dry}) = 0.1$, $P(\text{rain}, \text{wet}) = 0.3$. compute $P(\text{dry} \mid \text{sun})$ and $P(\text{dry})$. are weather and wetness independent?

> **▶ build: stage 4.** implement `marginal(J, idx)` (an int index gives plain values; a tuple of indices gives tuples, in that order) and `is_independent(J, tol=1e-9)` for joints over pairs.

### idea 5: the chain rule and Bayes' rule

most models are written forward, as a story of how the data came about: first nature picks a hypothesis $H$ with probability $P(H)$ (the **prior**), then the data $D$ is generated with probability $P(D \mid H)$ (the **likelihood**). the multiplication rule turns that story into a joint distribution, the **chain rule**:

$$P(H = h, D = d) = P(H = h)\,P(D = d \mid H = h).$$

your `chain(prior, likelihood)` builds exactly this, with `likelihood` a function that returns a distribution over data for each hypothesis.

but you usually want to go backward: having seen $D = d$, which hypothesis is responsible? that's a conditional in the other direction. and you already have all the tools: condition the joint on $D = d$, then marginalize out $D$. written as a formula, that's **Bayes' rule**:

$$P(H = h \mid D = d) = \frac{P(D = d \mid H = h)\,P(H = h)}{\sum_{h'} P(D = d \mid H = h')\,P(H = h')}.$$

in words: **posterior ∝ likelihood × prior**, normalized so it sums to 1. the denominator is the law of total probability: the overall chance of seeing $d$.

now warm-up 1 is mechanical. per 1000 people, 10 are sick and 9 of them test positive. of the 990 healthy, 9% test positive: about 89. so of the roughly 98 positives, only 9 are sick, about 9%. the doctors' 90% was the likelihood $P(+ \mid \text{sick})$, and confusing it with the posterior $P(\text{sick} \mid +)$ is the **base-rate fallacy**. the base rate (the prior) matters because healthy people vastly outnumber sick ones, so even a small false-positive rate produces more false alarms than true ones.

two habits make Bayes' rule easier to think with. first, counting in natural frequencies ("10 out of 1000") as above, which people find far more intuitive than conditional probabilities. second, the **odds form**: posterior odds = likelihood ratio × prior odds. for the test, the prior odds are $1:99$ and the likelihood ratio is $0.9/0.09 = 10$, so the posterior odds are $10:99$, about 9%. one positive test multiplies the odds by 10. a second independent positive test would multiply them by 10 again.

> **▶ build: stage 5.** implement `chain(prior, likelihood)` and `bayes(prior, likelihood, observed)`. `bayes` should be two lines built from `chain`, `condition` and `marginal`.

## worked example

**two dice.** let $S$ be the product distribution of two fair dice: 36 pairs, each $1/36$.

- $P(\text{sum} = 7) = 6/36 = 1/6$, since six pairs sum to 7.
- conditioning on "sum is 8" leaves the 5 pairs listed in idea 3, each with probability $(1/36)/(5/36) = 1/5$. so $P(\text{doubles} \mid \text{sum } 8) = 1/5$.
- conditioning on "first die is 6" leaves 6 pairs, each $1/6$, and exactly one of them, $(6, 1)$, sums to 7. so $P(\text{sum } 7 \mid \text{first die } 6) = 1/6 = P(\text{sum } 7)$: the sum being 7 is independent of the first die. (try "sum is 8" instead: it isn't.)

**the screening test.** prior $P(\text{sick}) = 0.01$. the likelihood is $P(+ \mid \text{sick}) = 0.9$ and $P(+ \mid \text{healthy}) = 0.09$. the chain rule gives the joint:

| | $+$ | $-$ |
|---|---|---|
| sick | $0.01 \times 0.9 = 0.009$ | $0.001$ |
| healthy | $0.99 \times 0.09 = 0.0891$ | $0.9009$ |

condition on the $+$ column: $P(+) = 0.009 + 0.0891 = 0.0981$, so

$$P(\text{sick} \mid +) = \frac{0.009}{0.0981} \approx 0.0917.$$

**Monty Hall.** you pick door 0 (doors are 0, 1, 2 in code). the hypotheses are the car's location, each with prior $1/3$. the host knows where the car is and opens a goat door other than yours: if the car is behind 0 he picks door 1 or 2 at random, if it's behind 1 he must open 2, and if it's behind 2 he must open 1. he opens door 2. the likelihoods of that observation are $1/2, 1, 0$ for car $= 0, 1, 2$. multiplying by the prior gives $1/6, 1/3, 0$, which sums to $1/2$, so the posterior is

$$P(\text{car} = 0 \mid \text{opens } 2) = 1/3, \qquad P(\text{car} = 1 \mid \text{opens } 2) = 2/3.$$

switch. the host's choice carries information because it was constrained by the car's location.

the tests check every number here: your code must reproduce $1/5$, $1/6$, $0.0917$, and $(1/3, 2/3)$.

## on paper

1. **inclusion–exclusion from the axioms.** using only the three axioms, prove $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

<details><summary>solution</summary>

split into disjoint pieces: $A \cup B = (A \setminus B) \cup (A \cap B) \cup (B \setminus A)$, and also $A = (A \setminus B) \cup (A \cap B)$ and $B = (B \setminus A) \cup (A \cap B)$, all disjoint unions. by axiom 3, $P(A \cup B) = P(A \setminus B) + P(A \cap B) + P(B \setminus A)$, $P(A) = P(A\setminus B) + P(A \cap B)$, $P(B) = P(B \setminus A) + P(A \cap B)$. substitute the last two into the first: $P(A \cup B) = [P(A) - P(A\cap B)] + P(A \cap B) + [P(B) - P(A \cap B)] = P(A) + P(B) - P(A \cap B)$.

</details>

2. **disjoint vs independent (warm-up 5).** show that if $P(A) > 0$, $P(B) > 0$ and $A \cap B = \emptyset$, then $A$ and $B$ are *not* independent. explain in words why that makes sense.

<details><summary>solution</summary>

independence would need $P(A \cap B) = P(A)P(B) > 0$, but $P(A \cap B) = P(\emptyset) = 0$. contradiction. in words: disjoint events are extremely *dependent*: learning that $A$ happened tells you for certain that $B$ didn't.

</details>

3. **the odds form.** starting from Bayes' rule for two hypotheses $h_1, h_2$, show that $\frac{P(h_1 \mid d)}{P(h_2 \mid d)} = \frac{P(d \mid h_1)}{P(d \mid h_2)}\cdot\frac{P(h_1)}{P(h_2)}$. then use it to find the posterior probability of disease after **two** independent positive tests with the warm-up's numbers.

<details><summary>solution</summary>

write Bayes' rule for $h_1$ and for $h_2$ and divide: the denominators $P(d)$ cancel. for the test, each positive multiplies the odds by $0.9 / 0.09 = 10$. prior odds $1:99$, after two positives $100:99$, so $P(\text{sick} \mid ++) = 100/199 \approx 0.50$. even two positives leave it a coin flip. (this assumes the two test results are independent *given* the disease status, which repeat tests on the same person often aren't.)

</details>

4. **the ignorant host.** in Monty Hall, suppose the host doesn't know where the car is. he opens door 1 or door 2 (never yours) uniformly at random, and it *happens* to show a goat. what is $P(\text{car behind door 1} \mid \text{he opened door 2 and it showed a goat})$? set up the hypotheses, the observation, and the likelihoods.

<details><summary>solution</summary>

the observation is the pair (door opened = 2, goat shown). likelihoods: car $= 0$: he opens 2 with probability $1/2$ and it's a goat, so $1/2$. car $= 1$: opens 2 with probability $1/2$, goat, so $1/2$. car $= 2$: opens 2 with probability $1/2$, but it shows the car, so the observation has probability $0$. posterior $\propto (1/2, 1/2, 0) \times (1/3, 1/3, 1/3)$, so $(1/2, 1/2, 0)$. switching doesn't help. same doors, same goat, different answer: what matters is the *process* that produced the observation, not just the observation.

</details>

## experiment

`lab.py` runs three experiments with your code:

1. the base-rate fallacy, with a slider for how rare the disease is, checked against a simulated population.
2. Monty Hall, exactly (with your `bayes`) and by simulating thousands of games, for a host who knows and a host who doesn't.
3. the sum of $k$ dice, built with your `add`, as $k$ grows.

write down a prediction before each one. at least one result surprises almost everyone.

## check your understanding

1. why does `Discrete` reject a pmf whose probabilities sum to 1.1, instead of quietly normalizing it?

<details><summary>answer</summary>

a pmf that doesn't sum to 1 is almost always a bug upstream (a missing outcome, a double-counted one, a forgotten normalization). silently fixing it hides the bug. the one place normalization is *meant* to happen, conditioning, does it explicitly.

</details>

2. what's the difference between $P(A \mid B)$ and $P(A \cap B)$? give an example where one is large and the other small.

<details><summary>answer</summary>

$P(A \cap B)$ is the probability that both happen, measured against everything. $P(A \mid B)$ measures against $B$ only. "a person is over 2.1 m tall and plays professional basketball" is rare, but $P(\text{plays pro basketball} \mid \text{over 2.1 m})$ is far higher than for the population at large.

</details>

3. your `bayes` divides by $P(D = d)$. what happens if you skip that step, and when can you get away with it?

<details><summary>answer</summary>

you get an unnormalized posterior: the right *ratios* between hypotheses, but values that don't sum to 1. for choosing the most probable hypothesis (MAP), or comparing two hypotheses, the ratios are enough. for anything that needs actual probabilities (expected costs, calibrated predictions), you need the normalization. much of chapter 12 is about approximating that denominator when it can't be computed.

</details>

4. in the worked example, "sum is 7" turned out to be independent of the first die, but "sum is 8" wasn't. why the difference?

<details><summary>answer</summary>

whatever the first die shows, exactly one value of the second die makes the sum 7, so $P(\text{sum } 7 \mid \text{first} = k) = 1/6$ for every $k$. for a sum of 8, a first die of 1 makes it impossible, while 2–6 each leave one way. so the conditional probability depends on the first die.

</details>

5. why is the marginal of a joint distribution a pushforward?

<details><summary>answer</summary>

$P(X = x) = \sum_y P(x, y)$ adds up the probability of every outcome $(x, y)$ that maps to $x$ under the projection $(x, y) \mapsto x$. that's the definition of a pushforward through the projection.

</details>

6. in the Monty Hall problem, the host's door choice changes your beliefs about the car. in the ignorant-host version, why does it change them less?

<details><summary>answer</summary>

the informed host's choice depends on where the car is (he *must* avoid it), so it carries information about the car. the ignorant host's choice doesn't depend on the car at all. the only information comes from seeing a goat, which rules out door 2 and nothing more, leaving doors 0 and 1 symmetric.

</details>

7. a test with 99% sensitivity and a 1% false-positive rate is used on a disease with prevalence 1 in 10,000. roughly what is $P(\text{sick} \mid +)$?

<details><summary>answer</summary>

prior odds $1 : 9999$, likelihood ratio $99$, posterior odds $99 : 9999 \approx 1 : 101$. so about 1%. 99% "accuracy" still means 99 out of 100 positives are false alarms at this base rate.

</details>

8. to test independence, you check $P(x, y) = P(x)P(y)$ only for the pairs $(x, y)$ that the joint distribution lists (positive probability), and every one of them passes. can an unlisted pair still break independence?

<details><summary>answer</summary>

no. if every listed pair satisfies $P(x, y) = P(x)P(y)$, then summing over the listed pairs gives $\sum_{\text{listed}}P(x)P(y) = \sum_{\text{listed}}P(x, y) = 1$. but $\sum_{\text{all } x, y}P(x)P(y) = 1$ too, so the products over the *unlisted* pairs sum to 0: each is 0, which is exactly what the joint says about them. a dependent joint always fails at some listed pair (for $Y = X^2$ above, $P(0, 0) = 1/3 \ne P(X = 0)P(Y = 0) = 1/9$). checking every pair is still the clearest way to write it, and the argument needs exact arithmetic: with a tolerance, small violations could hide in the unlisted pairs.

</details>

## common bugs

- **overwriting instead of accumulating.** `out[f(x)] = p` instead of `out[f(x)] = out.get(f(x), 0) + p`. symptom: a pushforward that sums to less than 1 (which your validation will catch), or a two-dice sum with $P(7) = 1/36$.
- **forgetting to renormalize after conditioning.** symptom: `Discrete` raises "probabilities sum to 0.33" inside `condition`.
- **aliasing the caller's dict.** `self.probs = pmf` means later edits to the caller's dict silently change your distribution. symptom: a distribution that changes between two lines of code that never touched it.
- **double-counting in a list event.** `sum(probs[x] for x in event)` with `event = [1, 1, 2]` counts outcome 1 twice. an event is a set.
- **confusing $P(D \mid H)$ with $P(H \mid D)$.** the base-rate fallacy, in code: returning the normalized likelihood instead of likelihood × prior.

## going deeper

- Blitzstein & Hwang (2019), *Introduction to Probability*, 2nd ed., CRC Press, chapters 1–3 and 7: probability, conditioning, random variables, and joint distributions, with many puzzles like the ones here. free online at [probabilitybook.net](https://probabilitybook.net).
- Kolmogorov (1933), *Grundbegriffe der Wahrscheinlichkeitsrechnung*, Springer (English translation: *Foundations of the Theory of Probability*, Chelsea, 1950). where the axioms come from. short and readable.
- Tversky & Kahneman (1974), "Judgment under uncertainty: heuristics and biases", *Science* 185(4157): 1124–1131. the classic catalogue of probability misjudgments, including neglect of base rates. [doi:10.1126/science.185.4157.1124](https://doi.org/10.1126/science.185.4157.1124)
- Gigerenzer, Gaissmaier, Kurz-Milcke, Schwartz & Woloshin (2007), "Helping doctors and patients make sense of health statistics", *Psychological Science in the Public Interest* 8(2): 53–96. the gynecologist survey quoted at the top of this lesson, and what to do about it. [doi:10.1111/j.1539-6053.2008.00033.x](https://doi.org/10.1111/j.1539-6053.2008.00033.x)
- Gigerenzer & Hoffrage (1995), "How to improve Bayesian reasoning without instruction: frequency formats", *Psychological Review* 102(4): 684–704. why "10 out of 1000" works better than "1%". [doi:10.1037/0033-295X.102.4.684](https://doi.org/10.1037/0033-295X.102.4.684)
- Jaynes (2003), *Probability Theory: The Logic of Science*, Cambridge University Press, chapters 1–4. probability as extended logic, with Bayes' rule at the center.

## connections

- ← builds on: sets and functions (ch 01), Python classes and dicts (ch 00).
- → used in: samplers for these distributions (topic 02), expectation as a weighted sum over a pmf (topic 03), Markov chains as repeated conditioning (topic 07), the n-gram capstone (conditional distributions of the next character), naive Bayes and generative classifiers (ch 08), Bayesian ML and graphical models (ch 12), and every language model from chapter 17 on.
