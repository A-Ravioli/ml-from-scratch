# integrals

## why this matters

a car's speedometer tells you its speed at every moment. how far did it go in an hour? if the speed were constant, distance = speed × time, the area of a rectangle on a speed-versus-time graph. when the speed changes, the distance is still the **area under the speed curve**. that area is an **integral**, and integrals are how you add up something that changes continuously.

in machine learning, integrals are everywhere probability is. a probability density (topic 05, chapter 04) gives probabilities as areas: the chance that a measurement lands between 1 and 2 is the area under the density between 1 and 2. an average over a continuous distribution, the *expected value*, is an integral. the normalizing constant that makes a Bayesian posterior a valid distribution (chapter 12) is an integral, and most of the difficulty of Bayesian ML comes from integrals that can't be done by hand.

you can't always find a formula for an area, but you can always **compute** one: cut the region into thin strips, approximate each strip by a shape whose area you know, and add. this topic builds three such methods (rectangles, trapezoids, parabolas), and the most important skill in numerical computing: measuring how fast an approximation's error shrinks as you work harder. `trapezoid` and `simpson` go into your library (`mlfs.numerics`).

## warm-up: predict first

1. what's the area under $y = x^2$ between $x = 0$ and $x = 1$? (guess, if you don't know a formula.)
2. you approximate an area with 10 strips and get an error of $0.01$. with 20 strips, what error do you expect?
3. which do you think is more accurate for the same number of strips: rectangles whose height is taken at the strip's midpoint, or trapezoids?
4. Simpson's rule fits parabolas (degree-2 curves) to the function. for which functions do you think it's exactly right?
5. you know the area under $f$ from $0$ to every $x$. how could you get $f$ back?

## prerequisites check

- [ ] you know the area of a rectangle and of a trapezoid (high school geometry)
- [ ] you know what a derivative is and how to differentiate powers like $x^3$ (→ ch 01 · topic 02)
- [ ] you can make an evenly spaced grid with `np.linspace` and sum arrays (→ ch 00 · topic 02)
- [ ] you can read a log-log plot, where powers of $n$ show up as straight lines: if $y = C/n^p$ then $\log y = \log C - p\log n$, so on log-scaled axes the graph is a line with slope $-p$ (axes: ch 00 · topic 04; the log rules: ch 01 · topic 01)

## core ideas

### idea 1: area as a sum of rectangles

to find the area under a curve $f(x)$ between $x = a$ and $x = b$, cut $[a, b]$ into $n$ strips of equal width

$$h = \frac{b - a}{n}.$$

strip $i$ (counting from $0$) runs from $a + ih$ to $a + (i + 1)h$. replace the curve over each strip by a flat line, giving a rectangle of width $h$, and add up the rectangles. that's a **Riemann sum**:

$$\text{area} \approx h\big(f(x_0^*) + f(x_1^*) + \dots + f(x_{n-1}^*)\big),$$

where $x_i^*$ is where you measure the height of strip $i$: its **left** end $a + ih$, its **right** end $a + (i + 1)h$, or its **midpoint** $a + (i + \frac12)h$.

for an increasing function, left rectangles fall short of the curve and right rectangles overshoot it, so the true area is trapped between them. as $n$ grows, the gaps shrink and both sums close in on one number. that limit is the **definite integral**:

$$\int_a^b f(x)\,dx = \lim_{n \to \infty} h\sum_{i=0}^{n-1} f(x_i^*).$$

the $\int$ is a stretched S for "sum", and $dx$ is the infinitely thin strip width. how fast do the sums close in? for the left sum, each strip's error is a sliver of triangle with width $h$ and height about $f'h$, so area about $\frac12 f'h^2$. there are $n = (b - a)/h$ strips, so the total error is proportional to $h$: double $n$, halve the error. the midpoint sum is much better: on each strip, the curve sticks out above the flat top on one side and dips below it on the other, and to first order these cancel. its error is proportional to $h^2$.

> **▶ build: stage 1.** implement `riemann_sum(f, a, b, n=100, *, rule="left")` with the rules `"left"`, `"right"` and `"midpoint"`. no loops needed: build all $n$ sample points as one array.

### idea 2: trapezoids

instead of a flat top, join the two ends of each strip with a straight line. each strip becomes a **trapezoid**, with area $h \cdot \frac{f(x_i) + f(x_{i+1})}{2}$ (width times average height). adding all $n$ of them, every inner point is shared by two trapezoids and the two ends by one, so

$$\int_a^b f(x)\,dx \approx h\left(\frac{y_0}{2} + y_1 + y_2 + \dots + y_{n-1} + \frac{y_n}{2}\right), \qquad y_i = f(a + ih).$$

notice the grid has $n + 1$ points $x_0 = a, \ldots, x_n = b$ for $n$ strips, and the trapezoid sum is exactly the average of the left and right Riemann sums.

**how accurate?** a straight line is exactly right for straight-line functions, so the error comes from curvature. over one strip, the gap between a curve and its chord is a thin crescent with area about $\frac{1}{12}f''h^3$. adding $n$ of them gives a total error of about

$$\text{error}_{\text{trap}} = \text{approximation} - \text{true area} \approx \frac{(b - a)}{12}h^2 f'',$$

(with $f''$ an average value of the second derivative over $[a, b]$), proportional to $h^2$: **double $n$, quarter the error.** the sign says that for a curve bending upward ($f'' > 0$), chords lie above the curve, so trapezoids overestimate. the midpoint rule's error is about half as big and has the opposite sign, $-\frac{(b - a)}{24}h^2f''$ (warm-up 3: midpoint wins, by a factor of two).

> **▶ build: stage 2.** implement `trapezoid(f, a, b, n=100)`, the composite trapezoid rule.

### idea 3: parabolas: Simpson's rule

straight lines ignore curvature. so fit curves that bend: take the strips in **pairs** and fit a parabola through the three points of each pair (left, middle, right). the area under a parabola through three equally spaced points turns out to be

$$\frac{h}{3}\big(y_{\text{left}} + 4y_{\text{middle}} + y_{\text{right}}\big),$$

a weighted average that trusts the middle four times as much as the ends (you'll derive it on paper). adding up the pairs, the shared points between pairs get counted twice:

$$\int_a^b f(x)\,dx \approx \frac h3\big(y_0 + 4y_1 + 2y_2 + 4y_3 + 2y_4 + \dots + 2y_{n-2} + 4y_{n-1} + y_n\big).$$

since strips come in pairs, $n$ must be **even**. this is **Simpson's rule**.

parabolas are exact for any degree-2 polynomial, by construction. but Simpson's rule is also exact for every **cubic** (warm-up 4), which is a free bonus. on each pair of strips, write the cubic around the middle point: the $x^3$ part is odd, so it's positive on one side and negative on the other, and its area cancels exactly. its error over the whole interval is about

$$\text{error}_{\text{Simpson}} \approx \frac{(b - a)}{180}h^4 f'''',$$

proportional to $h^4$: **double $n$, and the error drops 16-fold.** for smooth functions that's a huge win. with $n = 100$, the trapezoid rule's error for $\int_0^1 e^x\,dx$ is about $10^{-5}$, while Simpson's is about $10^{-10}$, for the same 101 function values.

> **🔮 predict:** Simpson's rule beats the trapezoid rule by a factor of about a hundred thousand at $n = 100$ for $e^x$. do you think there's any smooth function for which the trapezoid rule beats Simpson? what about a function that isn't smooth, like $\sqrt x$ near $0$?

> **▶ build: stage 3.** implement `simpson(f, a, b, n=100)`, raising `ValueError` for odd $n$.

### idea 4: the fundamental theorem of calculus

topics 02 and 03 look like opposites: derivatives cut a function into slopes, integrals add a function up into an area. they are exactly opposites, and that's the **fundamental theorem of calculus**.

define the running area from $a$ up to a moving point $x$:

$$F(x) = \int_a^x f(t)\,dt.$$

what's the slope of $F$? move $x$ a little to $x + h$. the area grows by one thin strip of width $h$ and height about $f(x)$, so $F(x + h) - F(x) \approx f(x)\,h$, and

$$F'(x) = f(x).$$

**the slope of the running area is the height of the curve.** (warm-up 5: differentiate the area function.) it works the other way too. if $G$ is *any* function with $G' = f$ (an **antiderivative**), then $G$ and $F$ have the same slope everywhere, so they differ only by a constant, and

$$\int_a^b f(x)\,dx = G(b) - G(a).$$

that's how areas are done by hand: find an antiderivative by running the derivative rules backwards. for $f(x) = x^2$, $G(x) = x^3/3$ works (its derivative is $x^2$), so $\int_0^1 x^2\,dx = \frac13 - 0 = \frac13$ (warm-up 1). for $\cos$, $G = \sin$, so $\int_0^\pi \cos x\,dx = \sin\pi - \sin 0 = 0$: the positive and negative areas cancel. areas below the axis count as negative.

numerically, the running area is just the running total of the strip areas. keep a list that starts at $0$ and add each trapezoid as you go. its last entry is the whole trapezoid rule, and its slope across each strip is the average height of $f$ there, the fundamental theorem in discrete form.

> **▶ build: stage 4.** implement `cumulative_trapezoid(f, a, b, n=100)`, returning the grid and the running area at every grid point.

### idea 5: how fast does the error shrink?

"error proportional to $h^2$" is a claim you can test. with $h = (b - a)/n$, an error proportional to $h^p$ means

$$\text{error} \approx \frac{C}{n^p}$$

for some constant $C$. $p$ is the method's **order of convergence**: 1 for left and right sums, 2 for midpoint and trapezoid, 4 for Simpson. take logs of both sides (topic 01's log rules):

$$\log(\text{error}) \approx \log C - p\log n.$$

that's a straight line in the variables $(\log n, \log \text{error})$, with slope $-p$. so on a log-log plot, every method's error is a straight line, steeper for higher order. to *measure* $p$, compute the error for several $n$, and fit a straight line through the points $(\log n_i, \log \text{error}_i)$. the least-squares slope of points $(u_i, v_i)$ is

$$\text{slope} = \frac{\sum_i (u_i - \bar u)(v_i - \bar v)}{\sum_i (u_i - \bar u)^2},$$

where $\bar u, \bar v$ are the averages (chapter 08 explains where this formula comes from). the measured order is minus the slope. measuring orders is how numerical analysts check that a method is implemented correctly: a trapezoid rule with order 1 has a bug, even if its answers look roughly right.

> **🔮 predict:** a rule's errors at $n = 8$ and $n = 16$ are $1.2 \times 10^{-3}$ and $3 \times 10^{-4}$. what order will `convergence_order` report? how many strips would you need for an error of $10^{-8}$?

> **▶ build: stage 5.** implement `convergence_order(ns, errors)`, the measured order from a least-squares fit on log-log axes.

## worked example

$\int_0^1 x^2\,dx = \frac13 \approx 0.33333$ with $n = 4$ strips, so $h = \frac14$ and the grid is $0, \frac14, \frac12, \frac34, 1$, where $x^2 = 0, \frac1{16}, \frac4{16}, \frac9{16}, 1$.

| method | sum | value | error (approximation − true) |
|---|---|---|---|
| left | $\frac14(0 + \frac1{16} + \frac4{16} + \frac9{16})$ | $0.21875$ | $-0.115$ |
| right | $\frac14(\frac1{16} + \frac4{16} + \frac9{16} + 1)$ | $0.46875$ | $+0.135$ |
| midpoint | $\frac14(\frac1{64} + \frac9{64} + \frac{25}{64} + \frac{49}{64})$ | $0.328125$ | $-0.0052$ |
| trapezoid | $\frac14(\frac02 + \frac1{16} + \frac4{16} + \frac9{16} + \frac12)$ | $0.34375$ | $+0.0104$ |
| Simpson | $\frac{1/4}{3}(0 + 4 \cdot \frac1{16} + 2 \cdot \frac4{16} + 4 \cdot \frac9{16} + 1)$ | $0.333333$ | $0$ |

the trapezoid is the average of left and right. the midpoint error is about half the trapezoid error, with the opposite sign, just as idea 2 predicts. Simpson is exact, because $x^2$ is a parabola. (the tests also check $\int_0^1 x^4\,dx$ with Simpson and $n = 2$: $\frac{0.5}{3}(0 + 4 \cdot 0.0625 + 1) = 0.208333$, against the true $0.2$.)

## on paper

1. **the 1–4–1 weights.** find the area under the parabola through $(-h, y_L)$, $(0, y_M)$ and $(h, y_R)$. (hint: write it as $p(x) = \alpha + \beta x + \gamma x^2$, and note that $\int_{-h}^h x\,dx = 0$ and $\int_{-h}^{h} x^2\,dx = \frac{2h^3}{3}$.)

<details><summary>solution</summary>

the area is $\int_{-h}^h(\alpha + \beta x + \gamma x^2)\,dx = 2h\alpha + \frac{2h^3}{3}\gamma$. from the three points: $y_M = \alpha$, and $y_L + y_R = 2\alpha + 2\gamma h^2$, so $\gamma h^2 = \frac{y_L + y_R}{2} - y_M$. substituting: area $= 2hy_M + \frac{2h}{3}\left(\frac{y_L + y_R}{2} - y_M\right) = \frac h3(y_L + 4y_M + y_R)$.

</details>

2. **Simpson is exact for cubics.** show that the rule from problem 1 gives the exact area for $f(x) = x^3$ on $[-h, h]$.

<details><summary>solution</summary>

the exact area is $\int_{-h}^h x^3\,dx = 0$ (odd function: the left half cancels the right half). the rule gives $\frac h3((-h)^3 + 4 \cdot 0 + h^3) = 0$. ✓ since the rule is exact for $1, x, x^2$ (by construction) and $x^3$, it's exact for every cubic, and on any pair of strips (shift the middle point to 0).

</details>

3. **antiderivatives.** compute $\int_1^2 \frac1x\,dx$ and $\int_0^\pi \sin x\,dx$ using the fundamental theorem.

<details><summary>solution</summary>

$(\log x)' = \frac1x$, so $\int_1^2 \frac1x\,dx = \log 2 - \log 1 = \log 2 \approx 0.6931$. $(-\cos x)' = \sin x$, so $\int_0^\pi \sin x\,dx = -\cos\pi + \cos 0 = 1 + 1 = 2$.

</details>

4. **reading an order off a table.** errors at $n = 10, 20, 40$ are $3.2 \times 10^{-3}$, $2.0 \times 10^{-4}$, $1.25 \times 10^{-5}$. what's the order? which method could this be?

<details><summary>solution</summary>

each doubling divides the error by $16 = 2^4$, so $p = 4$: Simpson's rule (on a smooth function).

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the error of all four rules (left, midpoint, trapezoid, Simpson) as $n$ grows from 2 to 2048, with their measured orders.
2. the four rules on a smooth function that repeats itself, integrated over one full period.
3. the four rules on $\sqrt x$, which is continuous but has a vertical tangent at $0$.

## check your understanding

1. why does the trapezoid rule need $n + 1$ function values for $n$ strips?

<details><summary>answer</summary>

each strip needs a value at both ends, and neighbouring strips share an end. $n$ strips have $n + 1$ edges. (an off-by-one here, using `np.linspace(a, b, n)`, silently changes $h$.)

</details>

2. why is the trapezoid rule exactly the average of the left and right Riemann sums?

<details><summary>answer</summary>

each trapezoid's area is $h\frac{y_i + y_{i+1}}{2}$, the average of the left rectangle $hy_i$ and the right rectangle $hy_{i+1}$. averaging strip by strip and adding gives the average of the two sums.

</details>

3. the trapezoid rule overestimates $\int_0^1 e^x\,dx$. why, without computing anything?

<details><summary>answer</summary>

$e^x$ bends upward ($f'' > 0$), so every chord lies above the curve and every trapezoid includes a little extra area. the error formula $\frac{b - a}{12}h^2f''$ (approximation minus truth) is positive.

</details>

4. what's $\int_{-1}^{1} x^3\,dx$, and how many strips does Simpson's rule need to get it exactly?

<details><summary>answer</summary>

$0$, since the function is odd. Simpson's rule is exact for cubics with any even $n$, including $n = 2$. (so is the midpoint rule here, by symmetry, but only because the interval is symmetric.)

</details>

5. you measure an order of $1.0$ for your trapezoid rule on $e^x$. what's wrong?

<details><summary>answer</summary>

the implementation. on a smooth function the trapezoid rule has order 2. order 1 is what a left or right sum does, so likely the end points aren't being halved (the sum is then a Riemann sum plus an extra half-strip), or the grid has the wrong number of points.

</details>

6. by the fundamental theorem, what is the derivative of $F(x) = \int_0^x e^{-t^2}\,dt$?

<details><summary>answer</summary>

$F'(x) = e^{-x^2}$, even though $F$ itself has no formula in terms of familiar functions. (it's the heart of the normal distribution's cumulative probabilities, topic 05.)

</details>

7. halving $h$ costs about twice as many function evaluations. for a method of order $p$, how much does the error drop?

<details><summary>answer</summary>

by a factor $2^p$: 2 for left sums, 4 for trapezoid, 16 for Simpson. for a fixed budget of evaluations, a higher-order method wins enormously on smooth functions, which is the whole point of Simpson's rule.

</details>

8. (the predict callout in idea 5.) errors $1.2 \times 10^{-3}$ at $n = 8$ and $3 \times 10^{-4}$ at $n = 16$: what's the order, and how many strips give $10^{-8}$?

<details><summary>answer</summary>

doubling $n$ divided the error by 4, so the order is 2. to go from $3 \times 10^{-4}$ down to $10^{-8}$ the error must shrink by $3 \times 10^4$, so $n$ must grow by $\sqrt{3 \times 10^4} \approx 173$: about $16 \times 173 \approx 2800$ strips. (an order-4 rule would need only $16 \times (3 \times 10^4)^{1/4} \approx 210$.)

</details>

## common bugs

- **$n$ points instead of $n + 1$.** `np.linspace(a, b, n)` makes $n - 1$ strips. symptom: errors that don't shrink at the expected rate, or a constant offset.
- **forgetting to halve the ends.** symptom: the trapezoid rule is off by $\frac h2(y_0 + y_n)$ and its order drops to 1.
- **swapping Simpson's weights.** odd-numbered points get 4, inner even points get 2. symptom: Simpson's rule is *worse* than the trapezoid rule.
- **`h / 6` instead of `h / 3`.** (in $\frac h3$, $h$ is the width of one strip, not of a pair.) symptom: every Simpson answer is exactly half the right size.
- **`abs(b - a)`.** symptom: integrating from $b$ to $a$ gives a positive area instead of a negative one.
- **the order's sign.** the log-log slope is negative. symptom: `convergence_order` returns $-2$ for the trapezoid rule.

## going deeper

- Süli & Mayers (2003), *An Introduction to Numerical Analysis*, Cambridge University Press, chapter 7 (numerical integration: Newton–Cotes rules, including the trapezoid and Simpson rules, and their error formulas). [doi:10.1017/CBO9780511801181](https://doi.org/10.1017/CBO9780511801181)
- Trefethen & Weideman (2014), "The exponentially convergent trapezoidal rule", *SIAM Review* 56(3): 385–458. the full story behind lab experiment 2. [doi:10.1137/130932132](https://doi.org/10.1137/130932132)
- Davis & Rabinowitz (1984), *Methods of Numerical Integration*, 2nd ed., Academic Press. the classic reference on quadrature.
- Strang (1991), *Calculus*, Wellesley-Cambridge Press, chapter 5 (integrals and the fundamental theorem). free on MIT OpenCourseWare.

## connections

- ← builds on: derivatives and antiderivative rules (topic 02), log-log plots (topic 01), NumPy grids and sums (ch 00 · topic 02).
- → used in: probabilities as areas under densities and expectations (topic 05, ch 04), Monte Carlo integration as the alternative in many dimensions (topic 05, ch 12), normalizing constants in Bayesian inference (ch 12), likelihoods of continuous models and normalizing flows (ch 21), and solving differential equations step by step (ch 20).
