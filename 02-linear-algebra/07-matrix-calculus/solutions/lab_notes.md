# matrix calculus · lab notes

## experiment 1: how small should the step $h$ be?

**what people usually predict:** the smaller the better, since derivatives are limits as $h \to 0$. maybe $h = 10^{-15}$ is best, or at least no worse than $10^{-10}$.

**what happens:** a clean U-curve. the central difference is most accurate near **$h \approx 10^{-5}$**, with a relative error around $3 \times 10^{-11}$. the forward difference is best near **$h \approx 10^{-7}$–$10^{-8}$**, with an error around $3 \times 10^{-8}$, about a thousand times worse. to the right of the minimum (large $h$), the central error falls with slope 2 and the forward error with slope 1. to the left (tiny $h$), both *rise* with slope $-1$. at $h = 10^{-15}$ both are useless: about 85% error for central and 160% for forward.

**why:** truncation error is $c_1h^2$ (central) or $c_1h$ (forward), from Taylor's theorem. rounding error is about $\varepsilon|f|/h$: the two function values agree in their leading digits, so the subtraction cancels them, and the remaining noise is divided by $h$. minimizing $c_1h^2 + \varepsilon/h$ gives $h^* \sim \varepsilon^{1/3} \approx 6 \times 10^{-6}$ and error $\sim\varepsilon^{2/3} \approx 10^{-11}$. minimizing $c_1h + \varepsilon/h$ gives $h^* \sim \sqrt\varepsilon \approx 10^{-8}$ and error $\sim\sqrt\varepsilon$. that's why `mlfs.numgrad` defaults to central differences with $h = 10^{-6}$, comfortably near the bottom of the U for well-scaled functions, and why gradient checks are done in float64 (in float32, $\varepsilon \approx 10^{-7}$ and the best achievable error is about $10^{-5}$). the same U-curve appeared in 1-D in chapter 01. here it governs every gradient check you'll ever run.

## experiment 2: can a gradient check miss a bug?

**what people usually predict:** a gradient check is a gradient check: the bug is caught everywhere, or at least at three of the four points.

**what happens:** the buggy $2Ax$ passes the check at (a) $A = I$, (b) symmetric $A$, and (c) $x = 0$, with errors around $10^{-11}$ (and exactly 0 at $x = 0$). only (d), a random non-symmetric $A$ at a random $x$, catches it, with error about 0.4. your correct `grad_quadratic` passes all four.

**why:** the bug is $2Ax - (A + A^\top)x = (A - A^\top)x$. it vanishes whenever $A$ is symmetric (cases a and b) or $x = 0$ (case c), so at those points the buggy formula *is* the true gradient, and no numerical check can tell them apart. a gradient check only tests the formula at the points you give it. rules for the rest of the course: test at random points (never zeros, ones or identities), use non-symmetric, non-square matrices with distinct dimensions (so a misplaced transpose is a shape error or a wrong number, never a coincidence), and check more than one point. every gradient test in this repo follows these rules. look back at `test_exercise.py`.

## experiment 3: what does a numerical gradient cost?

**what people usually predict:** a fixed slowdown, perhaps 10× or 100×, that doesn't depend on the problem size.

**what happens:** the numerical gradient's cost grows faster than the analytic one's, and the *ratio* grows with $n$ (asymptotically in proportion to $n$): on a typical laptop it is about 50× at n = 25 and about 150× at n = 400 (about 500× if you push on to n = 1600). in the plotted range both curves are partly overhead-dominated, so don't expect clean $n^2$ and $n^3$ slopes; the ratio is the robust signal. extrapolated to a network with $10^6$ parameters and a 10 ms forward pass, one numerical gradient takes $2 \times 10^6 \times 0.01\,\text{s} \approx 5.6$ hours. a training run needs hundreds of thousands of gradients.

**why:** finite differences learn one partial derivative per pair of evaluations. no amount of vectorization changes the count. it's information-theoretic for a black-box $f$. analytic gradients exploit the *structure* of $f$: $2A^\top(Ax - b)$ reuses the residual for every coordinate at once. chapter 03's forward-mode autodiff gets one directional derivative per pass, which still costs $O(n)$ passes for a full gradient. chapter 14's reverse mode (backpropagation) gets *all* $n$ partials for a small constant times the cost of one evaluation. that's the "cheap gradient principle" that makes deep learning possible. numerical gradients stay in the toolbox for what they're good at: checking those cleverer methods on small problems.
