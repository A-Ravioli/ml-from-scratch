# conditioning & stability · lab notes

## experiment 1: Hilbert systems lose digits

**what people usually predict:** a few digits lost at most. the entries are all simple fractions between 0 and 1, and the matrix is symmetric positive definite. or, if the answer goes bad, "`solve` must have a bug or be unstable".

**what happens:** κ grows about 30× per unit of $n$: $5 \times 10^2$ at n = 3, $1.5 \times 10^{10}$ at n = 8, $1.8 \times 10^{16}$ at n = 12. the number of correct digits tracks $16 - \log_{10}\kappa$ closely: about 7 digits at κ = 1e10 (n = 8), 3 digits at n = 11, and essentially none from n = 12 (20% error, then about 900% at n = 14). meanwhile the **backward error stays between $10^{-17}$ and $10^{-16}$ for every n**. (past n ≈ 12 the computed κ itself stops growing cleanly, at a few times $10^{17}$: $\sigma_{\min}$ is below the rounding level, and even the SVD can't resolve it.)

**why:** the forward error is bounded by κ × backward error, and partial-pivoting LU keeps the backward error at the rounding level (it's backward stable, and the Hilbert matrix causes no growth). so the entire loss of accuracy comes from the *problem*. the Hilbert matrix's columns $(1, 1/2, 1/3, \dots)$, $(1/2, 1/3, 1/4, \dots)$ are nearly parallel, and a 16-digit representation of $H$ already differs from the true $H$ by enough to move the solution completely once κ ≈ $10^{16}$. no algorithm working in double precision can do better. the remedies all change the problem: higher precision, reformulating (orthogonal polynomials instead of monomials, which is where the Hilbert matrix comes from), or regularization. this distinction, *problem* (κ) versus *algorithm* (η), is the main idea of the chapter.

## experiment 2: estimating the condition number cheaply

**what people usually predict:** maybe 2–5× faster, and only a rough estimate (one significant digit).

**what happens:** the estimate agrees with the SVD condition number to between 7 and 13 digits on these random matrices, and the speedup grows with n: about 8× at n = 20, 18× at n = 40, 50× at n = 80, and over 100× at n = 160 (the exact timings depend on your machine).

**why:** your SVD does roughly 6–10 Jacobi sweeps of $n^2/2$ rotations, each $O(n)$: that's $O(n^3)$ with a large constant (and many Python-level loop iterations). `cond_estimate` pays for one LU ($\tfrac23n^3$ flops, vectorized row operations), then each iteration is a few matrix-vector products and triangular solves, $O(n^2)$. the loops converge geometrically at the rate $(\sigma_2/\sigma_1)^2$ (or $(\sigma_n/\sigma_{n-1})^2$), so they stop early. the accuracy is better than you'd need: in practice κ is used for warnings ("you're losing 12 digits") and for choosing tolerances, where a factor of 2 is irrelevant. that's why LAPACK settles for Hager–Higham 1-norm estimates, about 5 solves accurate to within a small factor, and why a good solver can report a condition estimate for almost no extra cost.

## experiment 3: the worst-case direction

**what people usually predict:** any perturbation of a badly conditioned system is amplified by about κ. or, with a random perturbation, "on average about κ".

**what happens:** with $b \parallel u_{\max}$ and $\delta b \parallel u_{\min}$, the amplification is exactly **$10^6 = \kappa$**. the reverse ($b \parallel u_{\min}$, $\delta b \parallel u_{\max}$) gives $10^{-6}$: the perturbation is *shrunk*. $b \parallel u_{\max}$ with a random $\delta b$ still gives about $3 \times 10^5$. but 200 random pairs $(b, \delta b)$ give a median amplification of about **1.2** and a maximum around 14.

**why:** with $b = \sum\beta_iu_i$, the solution is $x = \sum(\beta_i/\sigma_i)v_i$. the bound is attained when $x$ is as *small* as possible relative to $b$ (all of $b$ along $u_1$, divided by the large $\sigma_1$), while $\delta x$ is as *large* as possible relative to $\delta b$ (all of $\delta b$ along $u_n$, divided by the tiny $\sigma_n$). a random $b$ has a component of about $1/\sqrt n$ along $u_n$, and dividing by $\sigma_n$ makes that component dominate $x$. so $x$ is already "large" in the weak direction, and a random perturbation changes it by a similar *relative* amount: amplification of order 1. random tests are therefore wildly optimistic. real data isn't random. a smooth signal $b$ typically lives in the top singular directions (like $u_1$ here), while noise spreads into all directions, including $u_n$. that's exactly the worst case, and it's why ill-conditioned regression problems are dangerous in practice (and why regularization, chapter 08, damps the small-σ directions).
