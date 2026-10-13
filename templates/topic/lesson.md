# __TITLE__

<!-- TEMPLATE: this file was scaffolded. replace every section below, then delete this comment.
plan from CURRICULUM.md (chapter __CHAPTER_NUM__, topic __NUM__):
  core ideas: __PLAN_LEARN__
  you build:  __PLAN_BUILD__
  lab:        __PLAN_LAB__
the mean/variance example shows the format; it is not real content. -->

## why this matters

a concrete problem or failure this topic solves. no definitions yet.

## warm-up: predict first

1. you average 10 numbers, then add an 11th that's 1,000× bigger. what happens to the mean?
2. …

## prerequisites check

- [ ] you can … (→ ch NN · topic MM)

## core ideas

### idea 1: the mean

intuition first, then the formal statement:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

> **▶ build: stage 1.** implement `mean` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** is the mean of a sample always one of the sample's values?

### idea 2: the variance

$$s^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

> **▶ build: stage 2.** implement `variance`.

> **🔮 predict:** what happens to the variance if every value is doubled?

## worked example

$x = (1, 2, 3, 6)$: $\bar{x} = 3$, $s^2 = (4 + 1 + 0 + 9)/4 = 3.5$. your code should reproduce this exactly.

## on paper

1. show that $\frac{1}{n}\sum (x_i - \bar{x})^2 = \overline{x^2} - \bar{x}^2$.

<details><summary>solution</summary>

expand the square and use $\sum x_i = n\bar{x}$.

</details>

## experiment

what `lab.py` investigates and what to record. no spoilers.

## check your understanding

1. why might you divide by $n-1$ instead of $n$?

<details><summary>answer</summary>

it makes the estimator unbiased (see ch 05 · 01).

</details>

## common bugs

- **integer division / dtype surprises**: symptom: … fix: …

## going deeper

- author (year), *title*. link.
- textbook, section.

## connections

- ← builds on: …
- → used in: …
