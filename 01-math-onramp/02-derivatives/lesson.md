# derivatives

## why this matters

every model in this course is trained the same way: measure how wrong it is (the *loss*), work out which direction to nudge each parameter to make the loss smaller, nudge, repeat. "which direction, and how much" is a **derivative**. a derivative tells you how fast a function's output changes when you change its input a little. if increasing a weight by $0.001$ increases the loss by $0.003$, the derivative is $3$, and you should decrease that weight.

in topic 06 you'll derive the derivative of a loss by hand and use it to fit your first model. hand derivations go wrong, though, constantly, even for experts. a missing factor of 2 or a forgotten minus sign doesn't crash anything. the model just trains badly, and you may never find out why. so this topic gives you two things: what a derivative *is* and the rules for computing one, and a way to **check** any derivative you derive, using nothing but two function values. that checker, `check_derivative`, goes into your library (`mlfs.numerics`), and its big brother in chapter 02 tests every gradient for the rest of the course.

## warm-up: predict first

1. the curve $y = x^2$ passes through $(3, 9)$. what's the slope of the curve right at that point?
2. you estimate a slope as $(f(x + h) - f(x))/h$. is $h = 10^{-15}$ a better choice than $h = 10^{-5}$?
3. the derivative of $f(x)g(x)$: is it $f'(x)g'(x)$?
4. what's the slope of $|x|$ at $x = 0$?
5. a function's derivative is positive at $x = 2$. is the function bigger at $x = 2.1$ than at $x = 2$?

## prerequisites check

- [ ] you know that a line $y = mx + c$ has slope $m$ = rise/run (high school algebra)
- [ ] you can expand $(x + h)^2$ and $(x + h)^3$ (high school algebra)
- [ ] you can compose functions and you know $e^x$ and $\log x$ (→ ch 01 · topic 01)
- [ ] you know floats have about 16 significant digits, and that subtracting nearly equal numbers loses digits (→ ch 00 · topic 03)

## core ideas

### idea 1: a derivative is a slope

a straight line has one slope everywhere: rise over run. a curve doesn't. $y = x^2$ is nearly flat near $0$ and steep near $3$. but zoom in on any point of a smooth curve and it looks more and more like a straight line. the **derivative** $f'(x)$ is the slope of that line, the slope "right at" $x$.

you can't compute the slope of a single point. you need two. so take a second point a little to the right, at $x + h$, and compute the slope of the line through both, the **secant** line:

$$\text{slope} = \frac{\text{rise}}{\text{run}} = \frac{f(x + h) - f(x)}{h}.$$

for $f(x) = x^2$ at $x = 3$:

| $h$ | $\frac{(3 + h)^2 - 9}{h}$ |
|---|---|
| $1$ | $7$ |
| $0.1$ | $6.1$ |
| $0.01$ | $6.01$ |
| $0.001$ | $6.001$ |

the slopes close in on $6$ (warm-up 1). algebra shows why: $\frac{(3 + h)^2 - 9}{h} = \frac{6h + h^2}{h} = 6 + h$, which becomes $6$ as $h$ shrinks to zero. "what a quantity approaches as $h$ shrinks to zero" is called a **limit**, and the derivative is defined as one:

$$f'(x) = \lim_{h \to 0}\frac{f(x + h) - f(x)}{h}.$$

the same computation at a general $x$ gives $\frac{(x + h)^2 - x^2}{h} = 2x + h \to 2x$, so the derivative of $x^2$ is the function $2x$. you'll also see it written $\frac{df}{dx}$, read "the change in $f$ per change in $x$".

what a derivative tells you: if $f'(x) > 0$ the function is going up at $x$, if $f'(x) < 0$ it's going down, and if $f'(x) = 0$ it's flat, which is what happens at the bottom of a valley. near $x$, $f(x + h) \approx f(x) + f'(x)\,h$ for small $h$ (warm-up 5: yes, for a small enough step). that last formula is the most useful sentence in calculus: **the derivative predicts how the output responds to a small change in the input**.

> **▶ build: stage 1.** implement `secant_slope(f, a, b)`, the slope of the line through $(a, f(a))$ and $(b, f(b))$.

### idea 2: finite differences, and why smaller isn't better

a computer can't take a limit, but it can use a small $h$. that's a **finite difference**. there are three natural choices of the two points:

$$\underbrace{\frac{f(x + h) - f(x)}{h}}_{\text{forward}}, \qquad \underbrace{\frac{f(x) - f(x - h)}{h}}_{\text{backward}}, \qquad \underbrace{\frac{f(x + h) - f(x - h)}{2h}}_{\text{central}}.$$

for $x^2$ these give $2x + h$, $2x - h$ and exactly $2x$. the forward difference is too steep and the backward too shallow, both by $h$, and the central one averages them and the errors cancel. that's not luck. near $x$ a smooth function looks like $f(x + h) \approx f(x) + f'(x)h + \frac12 f''(x)h^2$, where $f''$ measures how fast the slope itself changes (idea 5). so

- forward: $\frac{f(x + h) - f(x)}{h} \approx f'(x) + \tfrac12 f''(x)\,h$. the error is proportional to $h$: halve $h$, halve the error.
- central: the $h^2$ terms of $f(x + h)$ and $f(x - h)$ are equal and cancel in the subtraction, leaving an error proportional to $h^2$: halve $h$, quarter the error.

this error, from using a finite $h$ instead of a limit, is called **truncation error**. it says: make $h$ small.

**but floats push back.** $f(x)$ is only stored to about 16 significant digits, so each function value is off by roughly $\varepsilon|f|$, with $\varepsilon \approx 10^{-16}$. when $h$ is tiny, $f(x + h)$ and $f(x)$ agree in almost all their digits, and subtracting them cancels the good digits and keeps the noise. then you divide that noise by $h$. this **rounding error** is about $\varepsilon|f|/h$, and it *grows* as $h$ shrinks.

so the total error of a forward difference is roughly

$$\underbrace{c\,h}_{\text{truncation}} + \underbrace{\varepsilon/h}_{\text{rounding}},$$

big for large $h$, big for tiny $h$, smallest in between. balancing the two terms gives a best step around $h \approx \sqrt{\varepsilon} \approx 10^{-8}$ and a best error around $10^{-8}$. for the central difference ($c\,h^2 + \varepsilon/h$) the best step is around $\varepsilon^{1/3} \approx 10^{-5}$, with an error around $10^{-11}$, a thousand times better. that's why `derivative` defaults to central differences with $h = 10^{-5}$. (warm-up 2: $h = 10^{-15}$ is a terrible choice. the lab plots the whole curve.)

> **🔮 predict:** on a log-log plot of error against $h$, from $h = 10^{-1}$ down to $10^{-15}$, what shape do you expect for the forward difference? where is its lowest point?

> **▶ build: stage 2.** implement `derivative(f, x, *, h=1e-5, method="central")` using your `secant_slope`. it must work on arrays of points, and raise `ValueError` for an unknown method.

### idea 3: the rules

finite differences give numbers. for formulas, a handful of rules does almost everything. each comes from the limit definition.

**powers.** $(x^n)' = n\,x^{n-1}$. for $n = 3$: $(x + h)^3 = x^3 + 3x^2h + 3xh^2 + h^3$, so $\frac{(x + h)^3 - x^3}{h} = 3x^2 + 3xh + h^2 \to 3x^2$. it holds for every power, including $x^{-1}$ (derivative $-x^{-2}$) and $\sqrt x = x^{1/2}$ (derivative $\frac{1}{2\sqrt x}$).

**sums and constants.** $(f + g)' = f' + g'$ and $(c\,f)' = c\,f'$. a constant on its own has derivative 0. so $(3x^2 - x + 7)' = 6x - 1$.

**exponentials and logs.** $(e^x)' = e^x$: differentiate the series $1 + x + \frac{x^2}{2!} + \dots$ term by term with the power rule and you get the same series back (topic 01, idea 2). and $(\log x)' = \frac1x$, which you'll derive on paper below. for reference, $(\sin x)' = \cos x$ and $(\cos x)' = -\sin x$.

**products.** the derivative of a product is *not* the product of the derivatives (warm-up 3: $(x \cdot x)' = 2x$, but $x' \cdot x' = 1$). picture $f(x)g(x)$ as the area of a rectangle with sides $f$ and $g$. nudge $x$: the sides grow by about $f'h$ and $g'h$, and the area gains a strip along each side, $f'h \cdot g$ and $f \cdot g'h$, plus a tiny corner $f'g'h^2$ that vanishes faster than $h$. so

$$(fg)' = f'g + fg'.$$

**compositions: the chain rule.** for $c(x) = f(g(x))$, nudging $x$ by $h$ changes $g$ by about $g'(x)\,h$. that change is the nudge to $f$'s input, at the point $g(x)$, so $f$ changes by about $f'(g(x)) \cdot g'(x)\,h$. rates multiply, like gears:

$$\big(f(g(x))\big)' = f'(g(x)) \cdot g'(x).$$

the outer derivative is evaluated **at the inner value** $g(x)$, not at $x$. for example, $(\sin(x^2))' = \cos(x^2) \cdot 2x$, and $(e^{-x^2/2})' = e^{-x^2/2} \cdot (-x)$. forgetting the $g'(x)$ factor, or evaluating $f'$ at $x$, are the two most common derivative bugs in all of ML. the chain rule is also the whole idea behind training neural networks: a network is a long composition, and backpropagation (chapter 14) is the chain rule applied layer by layer.

> **▶ build: stage 3.** implement `product_rule(f, df, g, dg)` and `chain_rule(f, df, g, dg)`. each takes functions and their derivatives and *returns a function*: the derivative of the product, or of the composition.

### idea 4: checking a derivative

here's the situation you'll be in all through this course: you've derived a formula `df` for the derivative of some `f`, and you want to know whether it's right. finite differences give an independent answer that's slow and slightly inexact but almost never *wrong*. compare the two at a set of points:

$$\text{error}(x) = \frac{|df(x) - fd(x)|}{\max(1, |fd(x)|)},$$

where $fd(x)$ is the central difference. dividing by $|fd|$ makes the error *relative*, so a derivative of size $10^6$ isn't judged by the same absolute yardstick as one of size $10^{-3}$. the $\max(1, \cdot)$ stops the error from exploding when the true derivative is near zero (then two tiny numbers like $10^{-9}$ and $2 \times 10^{-9}$ would look "100% different"). report the *largest* error over all the points: one bad point is enough to show a bug.

how big is "wrong"? with central differences at $h = 10^{-5}$, a correct derivative typically gives $10^{-10}$ or less. a bug usually gives $10^{-2}$ or more. anything in between deserves a closer look.

**where to check** matters as much as how. a wrong derivative can agree with the right one at special points. the buggy $\cos(x^2)$ for $(\sin x^2)'$ equals the correct $2x\cos(x^2)$ exactly at $x = \frac12$ (where $2x = 1$) and wherever $\cos(x^2) = 0$. so check at several "random-looking" points, never just $0$ or $1$. the lab shows a bug sailing through a badly chosen check.

> **▶ build: stage 4.** implement `check_derivative(f, df, xs, *, h=1e-5)`, returning the largest relative error over the points `xs`.

### idea 5: the slope of the slope

the derivative $f'$ is itself a function, so it has a derivative too: the **second derivative** $f''$. it measures how fast the slope changes, which you can see as **curvature**. for $x^2$: $f' = 2x$ and $f'' = 2$, a constant upward bend. where $f'' > 0$ the curve bends up like a bowl, and where $f'' < 0$ it bends down like a dome. this matters for training: the bottom of a bowl is a minimum, and how sharply the bowl curves decides how big a step you can safely take (topic 06, and chapter 06).

to estimate $f''$, take a slope of slopes. the forward slope at $x$ is $\frac{f(x + h) - f(x)}{h}$ and the backward slope is $\frac{f(x) - f(x - h)}{h}$. their difference, divided by $h$:

$$f''(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}.$$

like the central difference, its truncation error is proportional to $h^2$, and it's exact for quadratics. but it divides by $h^2$, so its rounding error is about $4\varepsilon|f|/h^2$, which grows much faster as $h$ shrinks. balancing $c\,h^2$ against $\varepsilon/h^2$ puts the best step near $\varepsilon^{1/4} \approx 10^{-4}$, and that's the default here.

> **🔮 predict:** you estimate $f''$ for $f = \sin$ at $x = 1$ with $h = 10^{-8}$, a perfectly good step for a *first* derivative. roughly how big is the error?

> **▶ build: stage 5.** implement `second_derivative(f, x, *, h=1e-4)`.

## worked example

$f(x) = x^2$ at $x = 3$ with $h = 0.1$. the true derivative is $2 \cdot 3 = 6$.

- forward: $\frac{3.1^2 - 3^2}{0.1} = \frac{9.61 - 9}{0.1} = 6.1$, too big by $h$.
- backward: $\frac{3^2 - 2.9^2}{0.1} = \frac{9 - 8.41}{0.1} = 5.9$, too small by $h$.
- central: $\frac{3.1^2 - 2.9^2}{0.2} = \frac{9.61 - 8.41}{0.2} = 6.0$, exact.

**second derivative.** for $x^4$ at $x = 1$ with $h = 0.1$: $\frac{1.1^4 - 2 \cdot 1 + 0.9^4}{0.01} = \frac{1.4641 - 2 + 0.6561}{0.01} = 12.02$. the true value is $12x^2 = 12$, and the error $0.02$ is proportional to $h^2$: halve $h$ and it becomes $0.005$.

**the checker.** $f = x^3$, $df = 3x^2$, points $0,\ 0.1,\ 2$, and $h = 0.1$. the central difference of $x^3$ works out to exactly $3x^2 + h^2 = 3x^2 + 0.01$ (on paper, problem 1), so the errors are $\frac{0.01}{\max(1, 0.01)} = 0.01$, then $\frac{0.01}{\max(1, 0.04)} = 0.01$, then $\frac{0.01}{12.01} \approx 0.0008$. the largest is $0.01$.

the tests check all three. your code should reproduce them exactly (up to rounding).

## on paper

1. **central differences of a cubic.** show that $\frac{(x + h)^3 - (x - h)^3}{2h} = 3x^2 + h^2$. what does the forward difference of $x^3$ give?

<details><summary>solution</summary>

$(x \pm h)^3 = x^3 \pm 3x^2h + 3xh^2 \pm h^3$. subtracting, the even terms cancel: $(x + h)^3 - (x - h)^3 = 6x^2h + 2h^3$. dividing by $2h$ gives $3x^2 + h^2$. the forward difference is $\frac{3x^2h + 3xh^2 + h^3}{h} = 3x^2 + 3xh + h^2$: its error $3xh + h^2$ is proportional to $h$, the central one to $h^2$.

</details>

2. **the derivative of $\log x$.** $\log$ is the inverse of $\exp$: $e^{\log x} = x$. differentiate both sides with the chain rule and solve for $(\log x)'$.

<details><summary>solution</summary>

the left side is $\exp(g(x))$ with $g = \log$. by the chain rule its derivative is $e^{\log x} \cdot (\log x)' = x \cdot (\log x)'$. the right side's derivative is $1$. so $x \cdot (\log x)' = 1$ and $(\log x)' = 1/x$.

</details>

3. **rules in action.** differentiate (a) $x^2 e^x$, (b) $e^{-x^2/2}$, (c) $\log(1 + e^x)$.

<details><summary>solution</summary>

(a) product rule: $2x e^x + x^2 e^x = (2x + x^2)e^x$. (b) chain rule with outer $e^u$ and inner $u = -x^2/2$: $e^{-x^2/2} \cdot (-x)$. (c) chain rule with outer $\log u$ and inner $u = 1 + e^x$: $\frac{1}{1 + e^x} \cdot e^x = \frac{e^x}{1 + e^x}$. that's the **sigmoid** function, which turns scores into probabilities in logistic regression (chapter 08).

</details>

4. **the best step.** the forward difference error is about $E(h) = \frac{M}{2}h + \frac{\varepsilon}{h}$ (with $M = |f''|$). find the $h$ that minimizes it. what are $h^*$ and $E(h^*)$ for $M = 1$, $\varepsilon = 10^{-16}$?

<details><summary>solution</summary>

$E'(h) = \frac M2 - \frac{\varepsilon}{h^2} = 0$ gives $h^* = \sqrt{2\varepsilon/M}$. for $M = 1$: $h^* = \sqrt{2 \times 10^{-16}} \approx 1.4 \times 10^{-8}$, and $E(h^*) = \sqrt{2M\varepsilon} \approx 1.4 \times 10^{-8}$. (yes, you just used a derivative to find the minimum of a function. that's topic 06 in one line.)

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the error of your forward and central `derivative` (and your `second_derivative`) as the step $h$ shrinks from $10^{-1}$ to $10^{-15}$.
2. what your finite differences say about the slope of $|x|$ at its corner, and just next to it.
3. whether your `check_derivative` catches a chain-rule bug, depending on where you check.

## check your understanding

1. what's the difference between a secant slope and a derivative?

<details><summary>answer</summary>

a secant slope uses two points a distance $h$ apart. a derivative is the limit of secant slopes as $h \to 0$, the slope at a single point. finite differences are secant slopes with small $h$, used as estimates of the derivative.

</details>

2. why is the central difference more accurate than the forward difference, for the same $h$?

<details><summary>answer</summary>

the curvature term $\frac12f''h^2$ appears with the same sign in $f(x + h)$ and $f(x - h)$, so it cancels when you subtract them. the forward difference keeps it, giving an error proportional to $h$ instead of $h^2$.

</details>

3. (the predict callout in idea 5.) what's the error of the second difference for $\sin$ at $x = 1$ with $h = 10^{-8}$?

<details><summary>answer</summary>

rounding error is about $4\varepsilon|f|/h^2 \approx 4 \times 10^{-16} \times 0.84 / 10^{-16} \approx 3$. the estimate is pure noise: with errors of order 1, it can come out as $0$, $\pm 2$, or anything nearby, against a true value of $-0.84$. second differences need a much bigger $h$.

</details>

4. a learner writes the chain rule as `df(x) * dg(x)`. at which points would `check_derivative` miss the bug for $f = \sin$, $g(x) = x^2$?

<details><summary>answer</summary>

the buggy value is $\cos(x)\cdot 2x$ and the true one is $\cos(x^2) \cdot 2x$. they agree where $x = 0$ (both zero), where $x = 1$ ($x^2 = x$), and wherever $\cos x = \cos x^2$ by coincidence. checking at $0$ and $1$ only would miss it completely.

</details>

5. `check_derivative` returns $3 \times 10^{-6}$. is the derivative right?

<details><summary>answer</summary>

probably, but look closer. likely innocent causes: a function with large values (rounding error grows with $|f|$), a kink or a jump within $h$ of a test point, or a very curved function. rerun with a couple of different $h$ values: a genuine bug gives an error that stays put, while truncation error changes like $h^2$.

</details>

6. at the bottom of a smooth valley, what are the signs of $f'$ and $f''$?

<details><summary>answer</summary>

$f' = 0$ (flat) and $f'' > 0$ (bending up). at the top of a hill, $f' = 0$ and $f'' < 0$. this is the second-derivative test you'll use in topic 06 and chapter 06.

</details>

7. what does the chain rule say about $(e^{3x})'$, and why does it make sense?

<details><summary>answer</summary>

$e^{3x} \cdot 3$. the function $e^{3x}$ is $e^u$ run three times faster ($u = 3x$), so it changes three times as fast.

</details>

## common bugs

- **rise and run in opposite orders.** `(f(a) - f(b)) / (b - a)` flips the sign. symptom: every derivative comes out negated.
- **dividing the central difference by $h$.** the two points are $2h$ apart. symptom: central derivatives exactly twice too big. (using `secant_slope` with the actual points avoids this.)
- **ignoring the `h` argument.** symptom: results don't change when you pass `h=0.1`, and the step-size experiment shows a flat line.
- **$f'(x)g'(x)$ for a product.** symptom: the product rule check fails everywhere except where one function is constant.
- **evaluating the outer derivative at $x$.** `df(x) * dg(x)` instead of `df(g(x)) * dg(x)`. symptom: right for $g(x) = x$, wrong for anything else.
- **a tiny step for the second derivative.** symptom: `second_derivative` returns noise (often exactly 0, or huge values) for $h$ below about $10^{-6}$.

## going deeper

- Thompson (1914), *Calculus Made Easy*, 2nd ed., Macmillan. a friendly, famous introduction to derivatives by thinking about "little bits"; free from Project Gutenberg. [gutenberg.org/ebooks/33283](https://www.gutenberg.org/ebooks/33283)
- Strang (1991), *Calculus*, Wellesley-Cambridge Press, chapters 2–4 (derivatives, the rules, and the chain rule). free on MIT OpenCourseWare.
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, §8.1: finite-difference derivatives and how to choose $h$. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Griewank & Walther (2008), *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM. how to get exact derivatives automatically, where chapters 03 and 14 lead. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)

## connections

- ← builds on: slopes of lines (high school), composition and $e^x$, $\log x$ (topic 01), floating-point rounding and cancellation (ch 00 · topic 03).
- → used in: areas and the fundamental theorem of calculus (topic 03), gradient descent on your first model (topic 06), gradients and `mlfs.numgrad` (ch 02 · topic 07), multivariable calculus and forward-mode autodiff (ch 03), optimization (ch 06), and backpropagation (ch 14), which is the chain rule run in reverse.
