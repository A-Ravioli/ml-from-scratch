# forward-mode autodiff

## why this matters

you now have two ways to get derivatives, and both have a flaw. **by hand** (chapter 02 · topic 07) is exact but slow to derive and easy to get wrong, and it doesn't scale to a function that's a thousand lines of code. **finite differences** (topic 02) work on any code but are approximate (you fought the step-size U-curve twice already), and they can be spectacularly wrong when the function is sensitive.

there's a third way, and it's the one every deep-learning framework is built on: **automatic differentiation**. it computes derivatives that are exact up to rounding, for any function written as a program (loops, branches, function calls and all), at a cost of a small constant times running the program. it isn't symbolic algebra and it isn't numerical approximation. it's the chain rule, applied mechanically to each elementary operation as the program runs.

this topic builds the **forward mode** of autodiff, with a beautifully small idea: numbers that carry their own derivative along with them. you'll implement a `Dual` number class, and with it exact `derivative`, `grad`, `jacobian` and `jvp` functions that become **`mlfs.autodiff`**. you'll also measure the catch: forward mode's cost grows with the number of *inputs*, which for a neural network with a million weights is fatal. that's the motivation for reverse mode, which you'll build in chapter 14.

## warm-up: predict first

1. invent a number $\varepsilon \ne 0$ with $\varepsilon^2 = 0$. what is $(3 + \varepsilon)^2$? compare with $x^2$ and its derivative at $x = 3$.
2. a program computes $y = \sin(x^2)$ in two steps, $u = x^2$ then $y = \sin u$. if you know $du/dx$ after the first step, what do you need to know to get $dy/dx$ after the second?
3. how many evaluations of $f$ does a central-difference gradient of $f: \mathbb{R}^n \to \mathbb{R}$ take? how many would you guess an exact method needs?
4. a function is written as `if x == 1: return 1.0 else: return x * x`. what derivative should a tool that differentiates *programs* report at $x = 1$?
5. is it possible to get a second derivative with the same trick that gives a first derivative?

## prerequisites check

- [ ] you can apply the product, quotient and chain rules (→ ch 01 · topic 02)
- [ ] you can write a Python class with operator overloading (`__add__`, `__radd__`, `__mul__`, …) (→ ch 00 · topic 01)
- [ ] you know what a Jacobian is and that the chain rule multiplies Jacobians (→ topic 02)
- [ ] you can check a derivative against finite differences (→ ch 02 · topic 07)

## core ideas

### idea 1: numbers that carry their derivative

extend the real numbers with a new symbol $\varepsilon$ that is not zero but squares to zero: $\varepsilon^2 = 0$. a **dual number** is $a + b\varepsilon$, with a **real part** $a$ and a **dual part** $b$. (think of $\varepsilon$ as an infinitesimal so small that its square is negligible, made exact by decree.) arithmetic follows from ordinary algebra plus that one rule:

$$(a + b\varepsilon) \pm (c + d\varepsilon) = (a \pm c) + (b \pm d)\varepsilon,$$

$$(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon + bd\,\varepsilon^2 = ac + (ad + bc)\varepsilon,$$

$$\frac{a + b\varepsilon}{c + d\varepsilon} = \frac{(a + b\varepsilon)(c - d\varepsilon)}{c^2} = \frac{a}{c} + \frac{bc - ad}{c^2}\varepsilon.$$

look at the dual parts: the sum rule, the **product rule**, the **quotient rule**. that's no coincidence. warm-up 1: $(3 + \varepsilon)^2 = 9 + 6\varepsilon$, and $x^2$ has value 9 and derivative 6 at 3. in general, for any polynomial $p$,

$$p(a + \varepsilon) = p(a) + p'(a)\,\varepsilon,$$

because Taylor's theorem gives $p(a + \varepsilon) = p(a) + p'(a)\varepsilon + \tfrac12p''(a)\varepsilon^2 + \dots$ and every term after the second contains $\varepsilon^2$. so **evaluating a function on $x + 1\varepsilon$ returns $f(x)$ in the real part and $f'(x)$ in the dual part**, with no limits and no step size. the derivative is computed alongside the value, operation by operation, exactly (up to the same rounding as the value itself).

in code, a `Dual` is an object with `.real` and `.dual`, and Python's operator overloading (`__add__`, `__mul__`, …) makes `3 * x * x - 2 * x / (1 + x)` work unchanged when `x` is a `Dual`. plain numbers mix in as constants, duals with dual part 0. since `2 * x` calls `int.__mul__` first, which doesn't know about `Dual`s, Python falls back to `x.__rmul__(2)`: you need the reflected versions too. comparisons look only at the real part, so branches like `if x > 0` take the same path as they would on plain numbers.

NumPy cooperates: an `object` array of `Dual`s supports `+`, `*`, `@`, `np.sum` and friends by calling your methods entry by entry. so a function written for float arrays, like `x @ A @ x`, runs on an array of Duals without change.

one trap. in `x[0] - mu`, with `x[0]` a single Dual and `mu` a float *array*, Python calls `Dual.__sub__(mu)` first. a method that treats every non-Dual as "a number" happily builds `Dual(real - mu, dual)`: one Dual whose parts are arrays. nothing fails yet, but later code (`np.sum`, `dual_parts`, comparisons) breaks in confusing ways. the fix is one line per binary method: **if `other` is a NumPy array, return `NotImplemented`**. Python then asks the array, and NumPy applies your operation entry by entry, producing an object array of Duals, which is what you want. functions like `np.sum((x[0] - mu) ** 2)` are everywhere in later chapters (chapter 12's samplers differentiate log-densities written exactly like that).

two smaller conventions: comparisons use the real part (so branches take the same path as on floats), and so does truthiness, `bool(d) == bool(d.real)`, which the notebook provides for you.

> **▶ build: stage 1.** implement `Dual.__init__` and the arithmetic: `+`, `-`, `*`, `/` in both orders (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`), unary `-`, and the comparisons `<`, `<=`, `>`, `>=`, `==` (by real part). always return a new `Dual`, and return `NotImplemented` when the other operand is a NumPy array.

### idea 2: elementary functions, and derivatives of derivatives

for a smooth function $g$, Taylor's theorem again gives

$$g(a + b\varepsilon) = g(a) + g'(a)\,b\,\varepsilon.$$

so each elementary function needs one rule: compute $g(a)$, and multiply the incoming dual part $b$ by $g'(a)$.

| function | real part | dual part |
|---|---|---|
| $\exp$ | $e^a$ | $e^ab$ |
| $\log$ | $\log a$ | $b/a$ |
| $\sin$ | $\sin a$ | $b\cos a$ |
| $\cos$ | $\cos a$ | $-b\sin a$ |
| $\tanh$ | $\tanh a$ | $b(1 - \tanh^2a)$ |
| $\log(1 + x)$ (`log1p`) | $\log(1 + a)$ | $b/(1 + a)$ |
| $e^x - 1$ (`expm1`) | $e^a - 1$ | $e^ab$ |
| $\sqrt{\ }$ | $\sqrt a$ | $b/(2\sqrt a)$ |
| $x^p$ ($p$ a number) | $a^p$ | $pa^{p-1}b$ |
| $c^x$ ($c > 0$ a number) | $c^a$ | $c^a\log(c)\,b$ |

the factor $b$ is **the chain rule** (warm-up 2: you need $\cos u$ at the current $u$; multiply it by the incoming $du/dx$). when $\sin$ receives a Dual, its dual part already holds the derivative of everything computed so far, and $\sin$ multiplies on its own local derivative. composing functions composes the rules, so no function ever needs to know what came before it. for $x^y$ with both parts Duals, write $x^y = e^{y\log x}$ and let the rules above do the work. for $|x|$, return $-x$ when the real part is negative and $x$ otherwise (at $x = 0$ that picks the derivative $+1$, one valid subgradient).

NumPy's `np.exp(d)`, `np.sin(d)`, … on an object (anything that isn't a number) call a method of the same name, `d.exp()`, `d.sin()`. so naming your methods `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt` makes NumPy's elementary functions work on Duals and on object arrays of Duals.

**nesting** (warm-up 5). nothing above requires the real and dual parts to be *floats*. if they're Duals themselves, you get numbers $a + b\varepsilon_1 + c\varepsilon_2 + d\varepsilon_1\varepsilon_2$ with two independent infinitesimals ($\varepsilon_1^2 = \varepsilon_2^2 = 0$, but $\varepsilon_1\varepsilon_2 \ne 0$). evaluate $f$ at $x + \varepsilon_1 + \varepsilon_2$, which is `Dual(Dual(x, 1), Dual(1, 0))`, and Taylor's theorem gives

$$f(x + \varepsilon_1 + \varepsilon_2) = f(x) + f'(x)\varepsilon_1 + f'(x)\varepsilon_2 + f''(x)\,\varepsilon_1\varepsilon_2,$$

because $(\varepsilon_1 + \varepsilon_2)^2 = 2\varepsilon_1\varepsilon_2$ and the Taylor term is $\tfrac12f''\cdot2\varepsilon_1\varepsilon_2$. so `.dual.dual` holds the exact second derivative. this only works if your methods treat the parts generically: `np.exp(self.real)` works on a Dual (it calls `.exp()`), `math.exp(self.real)` and `float(self.real)` don't. the capstone uses nesting to compute exact Hessians for Newton's method.

**perturbation confusion.** nesting has one famous pitfall. a `Dual` doesn't record *which* infinitesimal its dual part belongs to, so two different derivatives that meet in one expression get their tangents mixed. try `derivative(lambda x: x * derivative(lambda y: x + y, 1.0), 1.0)`. the inner derivative of $x + y$ with respect to $y$ is 1, so the whole thing is $\frac{d}{dx}x = 1$. but inside, `x` is the outer `Dual(1, 1)` and `y` is the inner `Dual(1, 1)`, and `x + y` adds their dual parts as if they were the same $\varepsilon$: the inner derivative comes out as 2, and the answer as 2. no error, just a wrong number. production libraries tag each perturbation to keep them apart (Revels et al., 2016). with our untagged `Dual`, the rule is: **never close over an outer Dual inside an inner derivative.** when you need higher derivatives, build the nested inputs explicitly, as `Dual(Dual(x, 1), Dual(1, 0))` above (and the capstone's `hessian`) does, so each level of nesting is a distinct infinitesimal by construction.

**which NumPy functions work on Duals?** the arithmetic operators (`+ - * / **`, unary `-`, `abs`), `@` and `np.dot` on object arrays, `np.sum`, `np.prod`, `np.mean`, comparisons and the functions built on them (`np.maximum`, `np.minimum`), and the elementary functions whose names match your methods: `np.exp`, `np.log`, `np.log1p`, `np.expm1`, `np.sin`, `np.cos`, `np.tanh`, `np.sqrt`. `np.square`, `np.power`, `np.abs`, `np.where`, `np.clip` and `np.max` work too, because they reduce to these operations. these are known **not** to work: `np.logaddexp`, `np.tan`, `np.arctan`, `np.sinh`, `np.linalg.*`, `math.*`, and anything that converts to float (`float(x)`, `np.asarray(x, dtype=float)`). write $\log(1 + e^z)$ as `np.log1p(np.exp(-abs(z))) + np.maximum(z, 0.0)` rather than with `np.logaddexp`. at a point where a function isn't differentiable, you get the Python or NumPy behaviour of the formula: `Dual(0.0, 1.0) ** 0.5` raises `ZeroDivisionError` (it computes $0^{-1/2}$), while `np.sqrt` of the same Dual returns `inf` with a warning. both say the same thing: there is no finite derivative there.

> **🔮 predict:** you evaluate `np.sqrt(x * x)` on `Dual(0.0, 1.0)`. the function is $|x|$, which has one-sided derivatives $\pm1$ at 0. what do your rules return?

> **▶ build: stage 2.** implement `exp`, `log`, `log1p`, `expm1`, `sin`, `cos`, `tanh`, `sqrt`, `__pow__` (numeric or Dual exponent, $p = 0$ without dividing by zero, `NotImplemented` for an array exponent), `__rpow__` and `__abs__`, generically enough that nested Duals work.

### idea 3: from derivatives to gradients

for $f: \mathbb{R} \to \mathbb{R}$, one evaluation at `Dual(x, 1.0)` gives $f'(x)$: that's `derivative`.

for $f: \mathbb{R}^n \to \mathbb{R}$, give every input a dual part, $x_j + s_j\varepsilon$. by the chain rule, the dual part of the output is

$$\sum_j\frac{\partial f}{\partial x_j}s_j = \nabla f(x)\cdot s,$$

the directional derivative along the **seed** $s$. to get one partial derivative, seed with a basis vector: $s = e_i$ gives $\partial f/\partial x_i$. so the full gradient takes **$n$ forward passes**, one per input, each a complete evaluation of $f$ on Duals (warm-up 3: $n$ passes, against $2n$ for central differences). the value $f(x)$ comes free in the real part of any pass, so `value_and_grad` costs no more than `grad`.

it helps to picture what each pass does as an **evaluation trace**: the program broken into elementary steps $v_1, v_2, \dots$, each carrying its value and its **tangent** $\dot v_k = \partial v_k/\partial(\text{seeded input})$. the worked example below writes one out.

to collect results you need a helper, `dual_parts`, that splits whatever $f$ returned (a Dual, a plain number when $f$ ignores its input, or an array of them) into float real and dual parts.

> **▶ build: stage 3.** implement `dual_parts`, `derivative`, `value_and_grad` (exactly $n$ calls to $f$, each with an object array of Duals seeded with one $e_i$) and `grad`.

### idea 4: Jacobian-vector products and the cost of forward mode

for a vector-valued $F: \mathbb{R}^n \to \mathbb{R}^m$, seeding with $s$ puts in every output's dual part its directional derivative along $s$. together, that's the **Jacobian-vector product**:

$$\text{dual parts of } F(x + s\varepsilon) = J_F(x)\,s.$$

one pass, any $s$, and you never form $J_F$. that's `jvp`. a full Jacobian is $n$ JVPs, one per column ($s = e_j$): `jacobian` costs $n$ passes whatever $m$ is.

the cost of forward mode is therefore **proportional to the number of inputs**:

| shape | forward mode | what you'd want |
|---|---|---|
| $\mathbb{R}^1 \to \mathbb{R}^m$ (a curve) | 1 pass for the whole Jacobian | perfect |
| $\mathbb{R}^n \to \mathbb{R}^n$ (a solver's map) | $n$ passes | fine for small $n$ (the capstone) |
| $\mathbb{R}^n \to \mathbb{R}$ (a loss) | $n$ passes for the gradient | terrible: $n = 10^6$ weights means $10^6$ passes per step |

the last row is the one ML cares about. the fix is to apply the chain rule in the other order. forward mode computes $J_F\,s$, propagating tangents from the inputs forward. **reverse mode** computes $w^\top J_F$, propagating sensitivities $\partial(\text{output})/\partial v_k$ backwards from the output, so a scalar loss needs *one* backward pass for all $n$ partials (Jacobian rows instead of columns). that's backpropagation, and you'll build it in chapter 14. reverse mode pays for it with memory: it has to remember the whole forward trace. forward mode stores nothing, which is why it survives for JVPs, for few-input problems, and as the outer layer of forward-over-reverse Hessian-vector products.

there's a second, practical cost: each Dual operation is a Python method call creating a new object, while finite differences run on float arrays at NumPy speed. both make $O(n)$ passes, but an "exact" pass can be far slower than a "float" pass. experiment 1 measures both effects.

and there's a subtlety (warm-up 4): autodiff differentiates the **program**, not the mathematical function it computes. `1.0 if x == 1 else x * x` equals $x^2$ everywhere, but at $x = 1$ the Dual takes the first branch and returns the derivative of the constant `1.0`: zero. most of the time, programs and functions agree. experiment 2 shows where they don't.

> **🔮 predict:** you differentiate a loop that runs 5 steps of Newton's method for $\sqrt a$ with respect to $a$. does the derivative of the *loop's output* approximate $1/(2\sqrt a)$? how accurately?

> **▶ build: stage 4.** implement `jvp(f, x, v)` (one call to $f$, seeded with $v$) and `jacobian(f, x)` ($n$ JVPs, stacked as columns, shape $(m, n)$).

## worked example

$f(x_1, x_2) = \log x_1 + x_1x_2 - \sin x_2$ at $(2, 5)$. the evaluation trace, run twice with two seeds:

| step | value | tangent, seed $(1, 0)$ | tangent, seed $(0, 1)$ |
|---|---|---|---|
| $v_1 = x_1$ | $2$ | $1$ | $0$ |
| $v_2 = x_2$ | $5$ | $0$ | $1$ |
| $v_3 = \log v_1$ | $0.693147$ | $\dot v_1/v_1 = 0.5$ | $0$ |
| $v_4 = v_1v_2$ | $10$ | $\dot v_1v_2 + v_1\dot v_2 = 5$ | $2$ |
| $v_5 = \sin v_2$ | $-0.958924$ | $\dot v_2\cos v_2 = 0$ | $\cos 5 = 0.283662$ |
| $v_6 = v_3 + v_4$ | $10.693147$ | $5.5$ | $2$ |
| $v_7 = v_6 - v_5$ | $11.652071$ | $5.5$ | $1.716338$ |

so $f(2, 5) = 11.652071$ and $\nabla f = (5.5,\ 1.716338)$. check by hand: $\partial f/\partial x_1 = 1/x_1 + x_2 = 5.5$ and $\partial f/\partial x_2 = x_1 - \cos x_2 = 2 - 0.283662$. ✓ the value column is the same in both passes (that's the wasted work reverse mode avoids), and each pass produces exactly one partial derivative. your `value_and_grad` must reproduce these numbers to 14 digits; finite differences would agree to about 10.

## on paper

1. **polynomials.** show by induction that $(a + b\varepsilon)^k = a^k + ka^{k-1}b\,\varepsilon$, and conclude that $p(a + \varepsilon) = p(a) + p'(a)\varepsilon$ for every polynomial $p$.

<details><summary>solution</summary>

$k = 1$ is trivial. if it holds for $k$, then $(a + b\varepsilon)^{k+1} = (a^k + ka^{k-1}b\varepsilon)(a + b\varepsilon) = a^{k+1} + (a^kb + ka^kb)\varepsilon + ka^{k-1}b^2\varepsilon^2 = a^{k+1} + (k + 1)a^kb\,\varepsilon$. a polynomial is a sum of multiples of powers, and both $p \mapsto p(a + \varepsilon)$ and $p \mapsto p'$ are linear, so with $b = 1$, $p(a + \varepsilon) = \sum_kc_ka^k + \sum_kc_kka^{k-1}\varepsilon = p(a) + p'(a)\varepsilon$.

</details>

2. **composition is the chain rule.** suppose every elementary function obeys $g(a + b\varepsilon) = g(a) + g'(a)b\,\varepsilon$. show that $h = g\circ f$ then obeys $h(a + \varepsilon) = h(a) + h'(a)\varepsilon$.

<details><summary>solution</summary>

$f(a + \varepsilon) = f(a) + f'(a)\varepsilon$, so $g(f(a + \varepsilon)) = g(f(a) + f'(a)\varepsilon) = g(f(a)) + g'(f(a))f'(a)\varepsilon$, and $g'(f(a))f'(a) = h'(a)$ is the chain rule. by induction over the steps of an evaluation trace, the dual part of any program built from such operations is its derivative.

</details>

3. **two infinitesimals.** with $\varepsilon_1^2 = \varepsilon_2^2 = 0$, expand $f(x + \varepsilon_1 + \varepsilon_2)$ for $f(x) = x^3$ and confirm that the $\varepsilon_1\varepsilon_2$ coefficient is $f''(x) = 6x$. then explain how to get a mixed partial $\partial^2f/\partial x_i\partial x_j$ of $f: \mathbb{R}^n \to \mathbb{R}$ with the same trick.

<details><summary>solution</summary>

$(x + \varepsilon_1 + \varepsilon_2)^3 = x^3 + 3x^2(\varepsilon_1 + \varepsilon_2) + 3x(\varepsilon_1 + \varepsilon_2)^2 + (\varepsilon_1 + \varepsilon_2)^3$. $(\varepsilon_1 + \varepsilon_2)^2 = 2\varepsilon_1\varepsilon_2$ and the cube is 0, so the coefficient of $\varepsilon_1\varepsilon_2$ is $6x$. ✓ for a mixed partial, seed $x_i$ with $\varepsilon_1$ and $x_j$ with $\varepsilon_2$ (input $k$ is $x_k + \delta_{ki}\varepsilon_1 + \delta_{kj}\varepsilon_2$): the $\varepsilon_1\varepsilon_2$ coefficient of the output is $\partial^2f/\partial x_i\partial x_j$. a full Hessian takes $n(n+1)/2$ such passes, or $n^2$ without using symmetry.

</details>

4. **counting passes.** a network has a scalar loss and $10^6$ weights, and one forward evaluation costs 10 ms. how long does one gradient take by central differences, by forward mode, and by reverse mode (assuming a backward pass costs about 2 forward passes)?

<details><summary>solution</summary>

central differences: $2 \times 10^6 \times 10$ ms $\approx 5.6$ hours, and approximate. forward mode: $10^6$ passes, each a few times the cost of a float pass (the dual parts double the arithmetic), so at least 3 hours, though exact. reverse mode: one forward and one backward pass, about 30 ms. this ratio, independent of $n$, is the "cheap gradient principle" that makes training large models possible.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the wall-clock cost of your `grad`, `jvp` and `jacobian` against finite differences and a hand-derived gradient, as the number of inputs grows.
2. what your `derivative` returns for four programs: a loop, a branch, and a rewrite of $|x|$ at two points.
3. derivatives of a chaotic iterated map, by your dual numbers and by finite differences.

## check your understanding

1. why is forward-mode autodiff exact, while finite differences aren't, even though both evaluate $f$ about $n$ times?

<details><summary>answer</summary>

finite differences estimate a limit from two nearby function values, which brings truncation error ($h$ too big) or cancellation ($h$ too small). dual numbers never take a limit: each elementary operation applies its exact derivative rule to exact intermediate values, so the only error is the ordinary rounding of each operation, the same as in computing $f$ itself.

</details>

2. why does `Dual` need `__radd__` and `__rmul__`, and why doesn't it need `__rlt__`?

<details><summary>answer</summary>

for `2 * d`, Python first tries `int.__mul__(2, d)`, which returns `NotImplemented`, then calls `d.__rmul__(2)`. without it, mixing numbers and Duals in that order fails. comparisons are reflected automatically: `1 < d` falls back to `d.__gt__(1)`.

</details>

3. (the predict callout in idea 2.) what does `np.sqrt(x * x)` give at `Dual(0.0, 1.0)`?

<details><summary>answer</summary>

`x * x` is `Dual(0, 0)`: the product rule gives $0 \cdot 1 + 1 \cdot 0 = 0$. then `sqrt` divides the dual part 0 by $2\sqrt0 = 0$: `nan`. the composite has a kink, and the chain rule breaks at the inner point where $\sqrt{\ }$ isn't differentiable. `abs(x)` gives a sensible subgradient (+1); for norms, use `np.sqrt(np.sum(x**2) + tiny)` or handle zero separately (experiment 2).

</details>

4. you need $J_F(x)v$ for $F: \mathbb{R}^{1000} \to \mathbb{R}^{1000}$ and one vector $v$. how many passes? and for the full Jacobian?

<details><summary>answer</summary>

one pass for $J_Fv$ (seed with $v$); 1000 passes for $J_F$. matrix-free methods, such as Newton–Krylov solvers and Hessian-vector products, only ever need products, which is why JVPs are a primitive in JAX and PyTorch.

</details>

5. (the predict callout in idea 4.) does differentiating a 5-step Newton loop for $\sqrt a$ give $1/(2\sqrt a)$?

<details><summary>answer</summary>

yes, to about 16 digits at $a = 2$ (experiment 2a). the dual parts run their own recursion alongside the values: differentiating $x_{k+1} = \tfrac12(x_k + a/x_k)$ gives $\dot x_{k+1} = \tfrac12(\dot x_k + 1/x_k - a\dot x_k/x_k^2)$, which converges to $1/(2\sqrt a)$ as $x_k \to \sqrt a$, quadratically like the values. this is "differentiating through an iterative solver", used for implicit layers and hyperparameter optimization; whether the derivative of the loop equals the derivative of the solution is a real question in general, and here the answer is yes.

</details>

6. why can't `grad` take its $n$ passes "all at once", by seeding with $(1, 1, \dots, 1)$?

<details><summary>answer</summary>

with one infinitesimal, the output's dual part is the single number $\nabla f\cdot s$, the sum of all partials. you can't unmix $n$ numbers from one. to get them all in one pass you'd need $n$ independent infinitesimals, a vector dual part, which does the same arithmetic as $n$ passes, just batched. (that's a legitimate speedup in practice, called vector forward mode.)

</details>

7. when would you choose forward mode over reverse mode?

<details><summary>answer</summary>

when there are few inputs and many outputs (Jacobians of $\mathbb{R}^k \to \mathbb{R}^m$ with $k \ll m$, sensitivity of a simulation to a few parameters), when you need a JVP rather than a gradient, when memory is tight (forward mode stores nothing), or as the outer layer of forward-over-reverse Hessian-vector products.

</details>

## common bugs

- **forgetting the incoming dual part.** symptom: `Dual(0.7, 2).sin()` returns dual part $\cos 0.7$ instead of $2\cos 0.7$. every rule multiplies its local derivative by `self.dual`; tests with dual part 1 can't catch this.
- **$\varepsilon^2 \ne 0$.** symptom: the product's dual part is $bd$, or $ad + bc + bd$. the $bd$ term multiplies $\varepsilon^2$ and vanishes.
- **`math.exp` or `float()` inside the methods.** symptom: everything works until you nest Duals (second derivatives), then `TypeError: must be real number`. use `np.exp`, `np.log`, … and keep the parts as they are.
- **a `__rsub__` or `__rtruediv__` with the wrong orientation.** symptom: `2 - d` has the dual part of `d - 2`. the reflected method computes `other - self`.
- **seeding every input at once.** symptom: every gradient entry equals the sum of all the partials. seed one basis vector per pass.
- **a transposed Jacobian.** symptom: shape $(n, m)$. each JVP with $e_j$ is a *column*.
- **no array guard.** symptom: `Dual * np.array(...)` returns one Dual with array parts, and something downstream (`np.sum`, `dual_parts`, a comparison) fails mysteriously. return `NotImplemented` when `other` is an `np.ndarray`.
- **perturbation confusion.** symptom: the derivative of a function that itself calls `derivative` on a closure over the outer variable comes out wrong, silently (2 instead of 1 in idea 2's example). build nested Duals explicitly instead of nesting `derivative` calls over a shared variable.

## going deeper

- Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic differentiation in machine learning: a survey", *JMLR* 18(153): 1–43. the best single overview; its forward-mode example is the one used above. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- Griewank & Walther (2008), *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM. the reference text on forward and reverse modes, their costs, and higher derivatives. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
- Wengert (1964), "A simple automatic derivative evaluation program", *Communications of the ACM* 7(8): 463–464. the original forward-mode paper. [doi:10.1145/355586.364791](https://doi.org/10.1145/355586.364791)
- Revels, Lubin & Papamarkou (2016), "Forward-mode automatic differentiation in Julia". dual numbers as a production technique, including nested (higher-order) duals and perturbation confusion. [arXiv:1607.07892](https://arxiv.org/abs/1607.07892)

## connections

- ← builds on: the chain rule and Jacobians (topic 02), operator overloading (ch 00 · topic 01), finite-difference checks (ch 02 · topic 07).
- → used in: Taylor models with exact derivatives and the Newton capstone (this chapter), Newton's method and Hessian-vector products (ch 06), HMC's gradients via `mlfs.autodiff.grad` (ch 12), reverse-mode autodiff and backpropagation (ch 14), JVPs in neural ODEs and implicit layers (ch 20).
