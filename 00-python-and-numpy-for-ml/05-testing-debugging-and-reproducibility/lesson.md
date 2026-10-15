# testing, debugging & reproducibility

## why this matters

machine learning code fails quietly. a sorting function with a bug crashes or returns an obviously unsorted list. a training script with a bug usually runs to the end and prints a number. a wrong sign in a gradient, an off-by-one in a data split, a `(n,)` array broadcast against an `(n, 1)` one: each of these produces a model that is merely *worse*, and "worse" looks just like "this idea doesn't work" or "I need to tune it more". people have spent weeks tuning models whose real problem was a one-line bug.

and ML code is random. weights are initialized randomly, data is shuffled randomly, dropout drops units randomly. run the same script twice and you get two different numbers. that's fine, until you change something and the number goes up, and you can't tell whether your change helped or you just got a luckier draw. or until a bug shows up once in ten runs, and you can never make it happen again when you're watching.

the tools in this topic are the defence. **controlled randomness**: every random choice flows from a seed you chose, so any run can be repeated exactly. **tests with tolerances**: automatic checks that compare numbers the way floating point requires, and that say precisely what went wrong. **random testing**: let the computer hunt for the inputs that break your code. and a **debugging workflow** that finds the bug systematically rather than by staring, including binary search through your project's history. you've been on the receiving end of all of this already: every green check in this course is a test. now you'll build the tools yourself, and `as_rng`, `set_seed` and `assert_close` complete `mlfs.utils`.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. you call `np.random.seed(0)` at the top of a script. then a library you import also calls `np.random.seed(1)` somewhere inside. what happens to the reproducibility of your script?
2. `np.allclose(a, b)` with `a` of shape `(3,)` and `b` of shape `(3, 1)`: what does it compare?
3. you test your `softmax` on 1,000 random inputs drawn from a standard normal distribution, and all pass. what kind of bug could still be hiding?
4. a bug appeared somewhere in the last 1,000 commits. each test run takes a minute. what's the fewest number of test runs that is guaranteed to find the commit that introduced it?
5. a model scores 91.3% with your change and 90.8% without it, one run each. did your change help?

## prerequisites check

- [ ] you can raise exceptions, catch them with `try`/`except`, and pass a function as an argument to another function (→ ch 00 · topic 01, the python toolkit)
- [ ] you know why `0.1 + 0.2 != 0.3` and what a relative error is (→ ch 00 · topic 03)
- [ ] you can index arrays, use `np.argmax`, and know what broadcasting does to mismatched shapes (→ ch 00 · topic 02)
- [ ] you've used the live checks in the earlier notebooks and run `pytest -k stage1` once from a terminal

## core ideas

### idea 1: randomness you can repeat

computers can't flip coins. a **pseudo-random number generator** (PRNG) is a deterministic algorithm that produces a sequence of numbers that *look* random: they pass every statistical test you'd care to run, but the whole sequence is determined by one starting value, the **seed**. same seed, same sequence, every time, on every machine. that's not a weakness; it's the feature that makes experiments repeatable.

NumPy has two interfaces, and the difference matters:

```python
np.random.seed(0)                 # legacy: one hidden GLOBAL generator, shared by everyone
x = np.random.normal(size=3)

rng = np.random.default_rng(0)    # modern: a generator OBJECT you hold and pass around
x = rng.normal(size=3)
```

the global generator is shared state. any code that draws from it, including code inside libraries you call, advances the same hidden stream. so whether your result is reproducible depends on exactly how many random numbers *every other piece of code* drew before yours. and if a library reseeds the global generator (warm-up 1), your own seed is silently overridden: everything after that point is reproducible, but not by your seed, and changing your seed no longer changes anything.

the modern pattern avoids shared state entirely: **functions that need randomness take a generator as an argument.** the course's convention (from `CONTRACTS.md`) is a keyword argument `rng=None` that accepts three things:

- `None`: "I don't care about reproducibility", so make a fresh, unpredictable generator;
- an `int`: a seed, so make `np.random.default_rng(seed)`, the same stream every time;
- a `Generator`: use **that very object**, so the caller's single stream flows through every function it calls.

the third case is subtle and important. if `as_rng(g)` returned a *copy* of `g`, then a function that calls two helpers, `a = f(rng=g); b = h(rng=g)`, would give both helpers identical "random" numbers, and the caller's generator would never advance. returning the same object makes the program behave like one long, reproducible random stream. `as_rng` is the three-line function that normalizes the argument, and every random function in the course starts with `rng = as_rng(rng)`.

**`set_seed`** is for the other situation: the top of a script or notebook, where you want *everything* reproducible, including any code (yours or a library's) that still uses the global generators. it seeds Python's `random` module, NumPy's global generator, and PyTorch's if it's installed, and returns a fresh `Generator` for you to pass around. use it once, at the top. inside library code, never touch global state; take an `rng`.

> **▶ build: stage 1.** implement `as_rng(rng=None)` and `set_seed(seed)` (open the notebook with `marimo edit exercise.py`).

### idea 2: comparing numbers the way floating point demands

a **test** is a piece of code that checks another piece of code and complains loudly if something's wrong. the simplest test is an `assert`:

```python
assert mean([1, 2, 3, 6]) == 3.0, "mean of (1, 2, 3, 6) should be 3"
```

for numerical code, `==` is almost always wrong. topic 03 showed why: `0.1 + 0.2` isn't `0.3`, and two correct implementations of the same formula, doing operations in a different order, typically differ in the last few digits. so numerical tests compare with a **tolerance**: actual value $a$ and expected value $e$ are "close" if

$$|a - e| \;\le\; \text{atol} + \text{rtol}\cdot|e|.$$

the **relative tolerance** `rtol` scales with the size of the expected value: `rtol=1e-6` means "the first six significant digits agree", whether $e$ is $10^{-3}$ or $10^{9}$. the **absolute tolerance** `atol` handles values at or near zero, where any relative test is impossible ($10^{-17}$ is infinitely far from 0 in relative terms, but obviously close enough). note the asymmetry: the tolerance scales with $|e|$, the *expected* value, the one you trust. this is the same rule as `np.isclose` and `np.testing.assert_allclose`.

special values need explicit rules. two `nan`s in the same place count as matching (a function that should return `nan` there did), while `nan` against a number is a mismatch. infinities match only an identical infinity. and since an infinite expected value would make the tolerance infinite, closeness only applies when $e$ is finite.

**shapes.** here's a trap that has burned many ML researchers (warm-up 2). `np.allclose(a, b)` broadcasts its arguments. if `a` has shape `(3,)` and `b` has shape `(3, 1)`, they broadcast to `(3, 3)`, and `allclose` compares every entry of `a` with every entry of `b`. it returns `False` with a mystifying message, or, worse, if all entries happen to be equal, `True`, hiding the fact that your function returned the wrong *shape*. a shape bug in an ML pipeline usually means a broadcast is computing something entirely different from what you intended. so your `assert_close` checks shapes **exactly**, before anything else.

**failure messages that teach.** when a test fails, the message is the first, and often the only, debugging information you get. compare

```
AssertionError
```

with

```
logits: 2 of 12 values differ (rtol=1e-06, atol=1e-08)
  worst at index (2, 1): actual 2.75, expected 2.25
  abs error 0.5, rel error 0.222
```

the second one tells you the problem is localized (2 of 12), where it is, and how big. that's a five-minute fix instead of an hour of printing arrays. the "worst" element should be the one **furthest outside its own tolerance**, the largest ratio $|a - e| / (\text{atol} + \text{rtol}|e|)$, not simply the largest absolute error: an error of 0.5 on a value of 1,000 may be fine while an error of 0.01 on a value of 1 is not. `np.argmax` finds the worst position in the flattened array, and `np.unravel_index(flat_index, shape)` converts it back to a `(row, column, ...)` index.

the tests in this course follow the same rule: every failure message says what was expected, what came back, and the most likely cause.

> **▶ build: stage 2.** implement `assert_close(actual, expected, *, rtol=1e-06, atol=1e-08, name="value")`, which returns `None` when the values match and raises an `AssertionError` with the message format in its docstring when they don't.

> **🔮 predict:** a function returns `np.float32` results that are correct to `float32` precision. you compare them with `float64` reference values using the default `rtol=1e-6`. pass or fail? (topic 03 told you `float32`'s machine epsilon.)

### idea 3: tests that hunt for bugs

hand-written examples test the cases you thought of. bugs live in the cases you didn't. **random testing** (also called property-based testing) flips the approach around. instead of "on input $x$, the answer is $y$", you state a **property** that should hold for *every* input, and let the computer try many random inputs:

- softmax: every output row sums to 1, and adding a constant to a row doesn't change it;
- `pairwise_distances(X)`: symmetric, zero diagonal, non-negative, no `nan`;
- a sort: the output is ordered, and has the same elements as the input;
- `decompose(x)`: $(-1)^s\, m\, 2^e$ reconstructs $x$ exactly.

properties are powerful because they don't require knowing the answer. the test loop is simple: generate a random input, check the property, repeat; stop at the first failure and report the input that caused it, the **counterexample**. it must also be reproducible: if a random test fails once and you can't make it fail again, you've learned only that a bug exists somewhere. so the random test takes an `rng` like everything else, draws every input from that one generator, and the same seed always produces the same counterexample. a crash (an exception inside the property) is a failure too: "the function raised `ZeroDivisionError` on this input" is exactly the kind of thing you want to find.

the catch (warm-up 3): random inputs are only as good as their distribution. inputs drawn from a standard normal never contain `-inf`, never contain two identical rows, never have values of size $10^{8}$, and are never empty. a bug that only shows up in those cases survives any number of standard-normal tests. good generators deliberately mix in **edge cases**: zeros, duplicates, huge and tiny magnitudes, infinities, length-0 and length-1 inputs. the lab shows a bug that passes a thousand random tests with a naive generator and fails within a handful with a better one. (libraries like [Hypothesis](https://hypothesis.readthedocs.io/) do this, and also *shrink* a failing input to the smallest example that still fails.)

> **▶ build: stage 3.** implement `find_counterexample(prop, generate, *, n_trials=100, rng=None)`.

### idea 4: a debugging workflow, and binary search through history

when a test fails or a number looks wrong, the temptation is to stare at the code and guess. guessing works for easy bugs. for hard ones, a systematic loop works better:

1. **reproduce.** make the failure happen on demand, with a fixed seed and a fixed input. a bug you can't reproduce, you can't confirm you've fixed.
2. **minimize.** shrink the failing case: fewer data points, a smaller model, fewer steps, until removing anything else makes the failure go away. small cases are fast to run and easy to reason about.
3. **hypothesize and test.** form one specific hypothesis ("the gradient of the bias has the wrong sign") and design a check that distinguishes it from the alternatives. print shapes, compare against a slow but obviously correct version, check one step by hand.
4. **fix, and keep the test.** once found, turn the minimal failing case into a permanent test, so the bug can never come back silently. this is called a **regression test**.

**version control as a debugging tool.** this course assumes you keep your work in **git**, a version control system that records snapshots ("commits") of your project. the handful of commands you need: `git status` (what's changed), `git diff` (exactly how), `git add` and `git commit -m "message"` (save a snapshot), `git log` (the history). committing whenever something works gives you a trail of known-good states.

that trail makes one powerful technique possible. suppose your tests passed a month ago and fail today, with 1,000 commits in between. checking them one by one could take 1,000 test runs. but "bad" is monotone in time: once the bug is introduced, every later commit has it. so you can **binary search**: test the middle commit. if it's bad, the bug came in at or before it; if good, after it. either way you've halved the range. after $\lceil\log_2 1000\rceil = 10$ tests you've pinpointed the commit (warm-up 4), and reading a single commit's diff usually makes the bug obvious. `git bisect` automates exactly this: you mark one good and one bad commit, it checks out midpoints, you (or a script) say "good" or "bad", and it reports the first bad commit.

in code, binary search keeps an interval $[lo, hi]$ that is known to contain the first bad version, starting at $[0, n - 1]$ (the newest version must be bad, or there's nothing to find). each step tests $mid = \lfloor (lo + hi)/2 \rfloor$: if it's bad, the answer is $\le mid$, so $hi \leftarrow mid$; otherwise the answer is $> mid$, so $lo \leftarrow mid + 1$. the interval shrinks by about half each time, and when $lo = hi$, that's the answer. off-by-one mistakes are the classic binary-search bug (an interval that never shrinks loops forever; one that shrinks too far skips the answer), which is exactly why the tests try every possible answer for every $n$ up to 40.

**judging improvements** (warm-up 5) is the last piece of the workflow. two single runs that differ by 0.5% tell you nothing unless you know how much a run varies with its seed. run each setting with several seeds and compare the spread with the difference. the lab makes this concrete, and chapter 07 builds the proper statistical tools.

> **▶ build: stage 4.** implement `first_bad(n, is_bad)` with binary search, using at most $\lceil\log_2 n\rceil + 1$ calls to `is_bad`.

> **🔮 predict:** you train the same small model with 20 different seeds. how much do you think the final test accuracy varies: by 0.01%, 0.1%, 1% or 5%? (the lab measures a version of this.)

## worked example

**a failure message by hand.** expected values $e = \begin{pmatrix}0 & 0.25 & 0.5 & 0.75\\ 1 & 1.25 & 1.5 & 1.75\\ 2 & 2.25 & 2.5 & 2.75\end{pmatrix}$ (that's `np.arange(12).reshape(3, 4) / 4`). the actual array equals $e$ except that entry $(2, 1)$ is $2.75$ instead of $2.25$, and entry $(0, 3)$ is $0.751$ instead of $0.75$. defaults: `rtol=1e-6`, `atol=1e-8`.

- entry $(2, 1)$: $|a - e| = 0.5$; tolerance $10^{-8} + 10^{-6} \cdot 2.25 \approx 2.25 \times 10^{-6}$; ratio about $2.2 \times 10^{5}$. fails.
- entry $(0, 3)$: $|a - e| = 0.001$; tolerance $\approx 7.5 \times 10^{-7}$; ratio about $1.3 \times 10^{3}$. fails, but much less badly.
- every other entry: $|a - e| = 0$. passes.

so 2 of 12 values differ, and the worst is $(2, 1)$, with abs error $0.5$ and rel error $0.5 / 2.25 = 0.222$. the message is

```
logits: 2 of 12 values differ (rtol=1e-06, atol=1e-08)
  worst at index (2, 1): actual 2.75, expected 2.25
  abs error 0.5, rel error 0.222
```

**a bisection by hand.** ten versions, $0$ to $9$, and the bug arrived in version $6$. start with $[lo, hi] = [0, 9]$ (after confirming version 9 is bad). $mid = 4$: good, so $lo = 5$. $mid = 7$: bad, so $hi = 7$. $mid = 6$: bad, so $hi = 6$. $mid = 5$: good, so $lo = 6$. now $lo = hi = 6$. that's five calls in total (counting the initial check of version 9), within the limit $\lceil \log_2 10 \rceil + 1 = 5$.

the tests check both: this exact message content for your `assert_close`, and every possible bug location for your `first_bad`.

## on paper

1. **how many bisection steps?** show that binary search over $n$ versions needs at most $\lceil \log_2 n\rceil$ tests after the initial check of the newest version. how many tests for a million commits?

<details><summary>solution</summary>

the interval $[lo, hi]$ starts with $n$ candidates. each test leaves at most $\lceil k/2 \rceil$ of $k$ candidates (the bad half keeps $mid - lo + 1 = \lceil k/2 \rceil$, the good half $hi - mid = \lfloor k/2 \rfloor$). after $t$ tests at most $\lceil n / 2^t\rceil$ remain, which is 1 once $2^t \ge n$, that is $t = \lceil\log_2 n\rceil$. for $n = 10^6$: $2^{20} = 1{,}048{,}576 \ge 10^6$, so 20 tests (21 with the initial check).

</details>

2. **relative versus absolute.** for which values of $e$ does the absolute tolerance dominate the relative one, with the defaults `rtol=1e-6`, `atol=1e-8`? what happens if you compare results near $10^{-12}$ (say, tiny probabilities) with the defaults?

<details><summary>solution</summary>

$\text{atol} > \text{rtol}\cdot|e|$ when $|e| < 10^{-2}$. for $|e| \approx 10^{-12}$ the tolerance is essentially $10^{-8}$, ten thousand times bigger than the values themselves: $10^{-12}$ and $5 \times 10^{-12}$ "match", and so does 0. for very small quantities, either set `atol=0` and rely on `rtol`, or compare their logarithms.

</details>

3. **seeds and streams.** a function `f(rng)` draws 3 numbers, and `g(rng)` draws 2. what does `g` see if you call `f(as_rng(0))` and then `g(as_rng(0))`, compared with `r = as_rng(0); f(r); g(r)`?

<details><summary>solution</summary>

in the first version both calls start fresh streams from seed 0, so `g` sees the *same* first two numbers that `f` saw: their randomness is correlated (identical!), which is almost never what you want. in the second, `g` sees the 4th and 5th numbers of the stream, independent of `f`'s. that's why `as_rng` returns a `Generator` unchanged instead of copying it.

</details>

## experiment

`lab.py` plants bugs and asks you to catch them:

1. three implementations of the same function, plotted side by side, two of them buggy. can you tell which from the picture? then `assert_close` gets a look.
2. a function with a bug that only shows up on special inputs, tested with your `find_counterexample` and two different input generators.
3. a "better" training setting that wins on one seed. does it win on twenty?

record your predictions first.

## check your understanding

1. why should library functions take an `rng` argument instead of calling `np.random.seed`?

<details><summary>answer</summary>

reseeding the global generator inside a library silently overrides the caller's seed and couples unrelated code through shared state. an `rng` argument lets the caller decide: reproducible (pass a seed or a generator) or not (pass nothing), and keeps every random stream explicit.

</details>

2. why does the tolerance scale with $|e|$, not with $|a|$ or $\max(|a|, |e|)$?

<details><summary>answer</summary>

the expected value is the trusted reference; the actual value is the thing under test. scaling with $|a|$ would let a wildly wrong answer loosen its own tolerance: with `rtol=0.6`, $a = 2$, $e = 1$ would pass because $|2 - 1| \le 0.6 \cdot 2$. (`math.isclose` uses the symmetric $\max$ version; NumPy and this course use the reference version.)

</details>

3. (the predict callout in idea 2.) do `float32` results pass a comparison with `rtol=1e-6` against `float64` references?

<details><summary>answer</summary>

often not. `float32` has a machine epsilon of about $1.2 \times 10^{-7}$, and any computation with more than a few operations accumulates errors of several times that, easily above $10^{-6}$. compare `float32` results with `rtol` around $10^{-5}$ to $10^{-4}$. tolerances should match the precision of the computation being tested.

</details>

4. your random tests for `logsumexp` draw inputs from a standard normal. name three inputs they will never produce, and that you should test anyway.

<details><summary>answer</summary>

all entries `-inf` (should give `-inf`, not `nan`); huge entries like $10^4$ (overflow); an empty array or a single element; mixtures of `inf` and finite values; reductions over an axis of length 1.

</details>

5. what goes wrong in a binary search that sets `hi = mid - 1` when `mid` is bad?

<details><summary>answer</summary>

it can skip the answer: if `mid` itself is the first bad version, the interval no longer contains it. and setting `lo = mid` (instead of `mid + 1`) when `mid` is good can loop forever once `hi = lo + 1`, because `mid` stays equal to `lo`.

</details>

6. (the predict callout in idea 4.) how much does a model's test accuracy vary across seeds?

<details><summary>answer</summary>

it depends on the model and the data, but for small models and test sets, a spread of 0.5–2% is common, and on test sets of a few hundred examples, sampling noise alone is about $\sqrt{p(1-p)/n}$, which is 2% for $p = 0.9$ and $n = 225$. that's why one run each is not evidence of an improvement of 0.5%.

</details>

## common bugs

- **a function that copies the caller's generator.** symptom: two "independent" random helpers produce identical numbers, or results don't change when you change the seed. `as_rng` must return a `Generator` unchanged.
- **`np.random.seed` inside library code.** symptom: changing the seed at the top of your script has no effect on part of the run. take an `rng` argument instead.
- **`==` on floats in tests.** symptom: a test that fails on one machine (or one NumPy version) and passes on another. compare with tolerances.
- **broadcasting in comparisons.** symptom: a comparison that passes or fails for reasons unrelated to the values, or a test that passes even though the function returned the wrong shape. check shapes first, exactly.
- **random tests without edge cases.** symptom: 1,000 passing random tests and a bug in production. mix zeros, duplicates, infinities, huge values and tiny sizes into your generators.
- **binary search off by one.** symptom: an infinite loop, or an answer one too high or one too low. keep the invariant "the answer is in `[lo, hi]`" and check it on every branch.

## going deeper

- Claessen & Hughes (2000), "QuickCheck: a lightweight tool for random testing of Haskell programs", *ICFP 2000*, 268–279. the origin of property-based testing. [doi:10.1145/351240.351266](https://doi.org/10.1145/351240.351266)
- O'Neill (2014), "PCG: a family of simple fast space-efficient statistically good algorithms for random number generation", Harvey Mudd College technical report HMC-CS-2014-0905. the PCG64 generator behind `np.random.default_rng`. [pcg-random.org](https://www.pcg-random.org/paper.html)
- Sculley et al. (2015), "Hidden technical debt in machine learning systems", *NeurIPS 2015*. why ML systems are especially prone to silent failures. [proceedings](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", *MLSys 2021*. how much results move with seeds and data splits, and how to compare methods honestly. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Chacon & Straub (2014), *Pro Git*, 2nd ed., Apress: chapters 2 (git basics) and 7.10 (debugging with `git bisect`). [free online](https://git-scm.com/book/en/v2)

## connections

- ← builds on: exceptions and classes (topic 01), array indexing and broadcasting (topic 02), floating point tolerances (topic 03).
- → used in: every test in the course; `as_rng` in every function that samples (ch 01 Monte Carlo, ch 04 samplers, ch 07 splitters and bootstraps, ch 11 k-means initialization, ...); `assert_close` and gradient checks (ch 02 · topic 07); honest comparisons with baselines and uncertainty (ch 07 onward); multi-seed experiments and ablations (ch 30's `mlfs.research`).
