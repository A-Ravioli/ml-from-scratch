# neural ODEs

## why this matters

look at a residual block, $h_{l+1} = h_l + f(h_l, \theta_l)$, and squint. it's one step of Euler's method for the differential equation $dh/dt = f(h, \theta(t))$ with step size 1. a ResNet with 100 blocks integrates an ODE in 100 crude steps. so why not let the layer index become continuous time, define the network as the *solution* of $dh/dt = f(t, h, \theta)$ from $t = 0$ to $1$, and hand the work to a proper ODE solver? that's a **neural ODE** (Chen et al., 2018).

three things change when you do. depth becomes a solver setting: the same model can be evaluated with 10 steps when you're in a hurry or 500 when you need accuracy. memory can become independent of depth: instead of storing every step's activations for backprop, the **adjoint method** recovers the gradient by solving a second ODE backwards in time, a trick from optimal control. and the model inherits the mathematics of flows, including its limits: an ODE's trajectories can never cross, which means some functions are impossible for a neural ODE no matter how it's trained.

this is the chapter's last kind of structure: not a symmetry of the input, but a constraint on the *map* (it must be the flow of a smooth vector field). you'll build fixed-step solvers, a differentiable `odeint`, and the adjoint, and the lab finds the adjoint's hidden price: on a system that decays quickly, its gradient can be wrong by a factor of ten million. chapter 22 reuses your `odeint` to sample from diffusion models along the probability-flow ODE.

## warm-up: predict first

1. Euler's method on $dy/dt = y$ from $y(0) = 1$ with one step of size 1 gives $y(1) \approx 2$. the truth is $e = 2.718$. if you use 10 steps of 0.1, roughly how big is the error? with 100 steps?
2. RK4 evaluates $f$ four times per step. per evaluation of $f$, is it more or less accurate than Euler?
3. backprop through 1,000 solver steps stores the intermediate states of all 1,000 steps. could you get the same gradient while storing only the final state?
4. a neural ODE maps the real line to itself, $y(0) \mapsto y(1)$. can it learn $y \mapsto -y$?
5. you integrate $dy/dt = -10y$ forwards from $t = 0$ to $2$, then backwards from the result to $t = 0$. in exact arithmetic you get $y(0)$ back. what happens in float32?

## prerequisites check

- [ ] you can write a function's Taylor expansion to second order (→ ch 03 · topic 03)
- [ ] you know what a vector-Jacobian product is and that reverse-mode autodiff computes them (→ ch 14 · topics 02–03)
- [ ] you know what a residual connection is (→ ch 16)
- [ ] you can write a custom `torch.autograd.Function` with `forward` and `backward`, or are ready to read how (→ ch 15 · topic 07 for PyTorch basics; the hint shows the pattern)
- [ ] you know the matrix exponential solves $dy/dt = Ay$: $y(t) = e^{At}y_0$ (→ ch 02 · topic 05)

## core ideas

### idea 1: one step at a time

an ODE $dy/dt = f(t, y)$ with $y(t_0) = y_0$ has (for smooth $f$) a unique solution. a **one-step method** advances an approximation by a step $h$ using only evaluations of $f$:

- **Euler:** $y_{n+1} = y_n + h f(t_n, y_n)$. follow the tangent.
- **midpoint:** take half an Euler step to estimate the slope in the middle, then use that slope for the full step: $y_{n+1} = y_n + h f\big(t_n + \tfrac h2,\ y_n + \tfrac h2 f(t_n, y_n)\big)$.
- **classical Runge–Kutta (RK4):** four slopes, $k_1 = f(t_n, y_n)$, $k_2 = f(t_n + \tfrac h2, y_n + \tfrac h2 k_1)$, $k_3 = f(t_n + \tfrac h2, y_n + \tfrac h2 k_2)$, $k_4 = f(t_n + h, y_n + h k_3)$, combined as $y_{n+1} = y_n + \tfrac h6(k_1 + 2k_2 + 2k_3 + k_4)$.

how good are they? compare one step with the Taylor series of the true solution, $y(t + h) = y + hy' + \tfrac{h^2}{2}y'' + \dots$. Euler matches the first two terms, so its **local error** (one step) is $O(h^2)$ (on paper 1); midpoint matches through $h^2$ (local error $O(h^3)$), RK4 through $h^4$ (local error $O(h^5)$). integrating to a fixed time $T$ takes $T/h$ steps, and the local errors add up to a **global error** one power lower: $O(h)$ for Euler, $O(h^2)$ for midpoint, $O(h^4)$ for RK4. that power is the method's **order**, and it's what the checks measure: halve $h$ and RK4's one-step error should shrink about 32-fold, its global error 16-fold.

order is why RK4 is the workhorse. per evaluation of $f$ it costs four times what Euler costs, but halving the step of an order-4 method cuts the error 16-fold against Euler's 2-fold. for the accuracy RK4 gets from 10 steps (40 evaluations), Euler can need thousands.

one more property matters: **stability**. on $dy/dt = \lambda y$ with $\lambda < 0$ the true solution decays, but Euler multiplies by $1 + h\lambda$ each step, which blows up if $h|\lambda| > 2$. every explicit method has such a limit (about $2.8$ for RK4), so a system with a fast decay rate forces small steps whether you need the accuracy or not. that's **stiffness**, and it returns in the lab.

> **spec · stage 1.** `euler_step(f, t, y, h)`, `midpoint_step(f, t, y, h)`, `rk4_step(f, t, y, h)`: one step of each method as written above, for `f(t, y)` returning a tensor of `y`'s shape, `t` a 0-d tensor, `y` a tensor of any shape, `h` a float or 0-d tensor; returns $y_{n+1}$.

> **▶ build: stage 1.** implement the three steps in `exercise.py` (`marimo edit exercise.py`).

### idea 2: integrating over a time grid

a solver call takes the times at which you want the solution, $t_0, t_1, \dots, t_K$ (increasing, or decreasing to integrate backwards), and returns $y(t_0) = y_0, y(t_1), \dots, y(t_K)$. with a fixed step size $s$, each interval $[t_k, t_{k+1}]$ is split into $n_k = \lceil |t_{k+1} - t_k| / s \rceil$ equal substeps, so the grid always lands exactly on the requested times. compute $n_k$ with a little tolerance: $(0.1 + 0.2)/0.1$ is $3.0000000000000004$ in floating point, and a literal ceiling would take a needless fourth step. with no step size, take one step per interval.

because every operation in a step is an ordinary differentiable tensor operation, gradients come for free: backprop through the solver computes the exact gradient of the *discretized* solution with respect to $y_0$ and to any parameters inside $f$. this is "discretize, then optimize". its price is memory: autograd must keep every intermediate tensor of every evaluation of $f$ for the backward pass, so memory grows linearly with the number of steps. for a large model and an accurate solve, that's the bottleneck.

> **spec · stage 2.** `odeint(f, y0, t, *, method="rk4", step_size=None)`: `t` a 1-D tensor of increasing or decreasing times; `method` in `{"euler", "midpoint", "rk4"}` (`ValueError` otherwise); `step_size=None` → one step per interval of `t`, else each interval is split into $\lceil |\Delta t| / s \rceil$ equal substeps (with a tolerance of about $10^{-9}$ for rounding); returns a tensor `(len(t), *y0.shape)` whose first entry is `y0`, differentiable by backprop. the helper `steps_per_interval(t, step_size)` computes the substep counts. this is `mlfs.ode.odeint`.

> **▶ build: stage 2.** implement `steps_per_interval` and `odeint`.

> **🔮 predict:** RK4 with 10 steps on $dy/dt = y$ from $0$ to $1$: how many correct digits of $e$?

### idea 3: the adjoint method

here's the problem the adjoint solves: compute $\partial L/\partial y_0$ and $\partial L/\partial \theta$ for a loss $L(y(T))$, where $y$ solves $dy/dt = f(t, y, \theta)$, without storing the trajectory.

define the **adjoint** $a(t) = \partial L/\partial y(t)$: how much the loss would change if you nudged the state at time $t$. at the end, $a(T) = \partial L/\partial y(T)$, which you know. how does $a$ evolve? a small perturbation $\delta y$ of the state obeys the linearized ODE $d\,\delta y/dt = J\,\delta y$ with $J = \partial f/\partial y$. its effect on the loss, $a^\top \delta y$, can't depend on *when* you measure it (it's the same perturbation, propagated), so $\tfrac{d}{dt}(a^\top \delta y) = 0$, which forces (on paper 2)

$$\frac{da}{dt} = -a^\top \frac{\partial f}{\partial y}, \qquad \frac{\partial L}{\partial \theta} = \int_{t_0}^{T} a(t)^\top \frac{\partial f}{\partial \theta}\big(t, y(t)\big)\, dt.$$

the right-hand sides are **vector-Jacobian products**, exactly what one reverse-mode autodiff call through $f$ computes, without ever forming a Jacobian. so the backward pass is itself an ODE solve, from $T$ back to $t_0$, of the **augmented state** $(y, a, a_\theta)$:

$$\frac{d}{dt}\begin{pmatrix} y \\ a \\ a_\theta \end{pmatrix} = \begin{pmatrix} f(t, y) \\ -a^\top \partial f/\partial y \\ -a^\top \partial f/\partial \theta \end{pmatrix}, \qquad \text{starting from } \begin{pmatrix} y(T) \\ \partial L/\partial y(T) \\ 0 \end{pmatrix}.$$

at $t_0$, $a$ is $\partial L/\partial y_0$ and $a_\theta$ is $\partial L/\partial\theta$. note the first row: since the trajectory wasn't stored, $y(t)$ is **recomputed by integrating backwards** alongside the adjoint. if the loss depends on several output times $t_k$, integrate interval by interval and add $\partial L/\partial y(t_k)$ to $a$ whenever you reach $t_k$; you can also reset $y$ to the stored forward value there.

what you get: memory independent of the number of steps (only the outputs are kept). what you pay: the backward solve evaluates $f$ *and* a vector-Jacobian product at every stage, roughly two to three times the work of backprop. and something subtler: the adjoint gradient is the discretization of the *exact* gradient ("optimize, then discretize"), not the exact gradient of your discretized forward pass, so the two agree only up to solver error. worse, the backward reconstruction of $y$ runs every decaying mode of the forward dynamics in reverse, where it *grows*. on a stiff system, errors in $y$ (from truncation, or just float32 rounding) are amplified exponentially, and the gradient can be badly wrong. the lab measures how badly.

> **spec · stage 3.** `odeint_adjoint(f, y0, t, *, method="rk4", step_size=None)`: the same result as `odeint`; `f` must be an `nn.Module` (`TypeError` otherwise); gradients with respect to `y0` and `f`'s parameters come from solving the augmented system above backwards with the same method and step grid. the forward solve records no autograd graph, and nothing per step is stored for the backward pass (the checks measure saved bytes for 10 versus 100 steps). a `torch.autograd.Function` (which can be defined inside `odeint_adjoint`, closing over `f` and the solver settings) is the natural tool. this is `mlfs.ode.odeint_adjoint`.

> **▶ build: stage 3.** implement `odeint_adjoint`.

### idea 4: neural ODEs, and what a flow can't do

a **neural ODE** block takes an input $y_0$, integrates $dy/dt = f_\theta(t, y)$ from $0$ to $t_1$ with an MLP $f_\theta$, and returns $y(t_1)$. stacked with ordinary layers (an encoder before, a linear classifier after), it trains like any network, by backprop through the solver or by the adjoint. its depth is the number of solver steps, a knob you can turn after training.

the map $y_0 \mapsto y(t_1)$ is the **flow** of a smooth vector field, and flows are special: they're continuous, invertible, and their trajectories never cross (two solutions that met at some time would be the same solution, by uniqueness). in one dimension that already rules out $y \mapsto -y$: the trajectories from $-1$ and $+1$ would have to pass through each other (on paper 4). in two dimensions it rules out separating a disk from a ring around it with a line: continuous invertible maps preserve "inside". Dupont et al. (2019) proposed a cheap escape, **augmented neural ODEs**: append a few zero coordinates, flow in a higher dimension where the inner disk can be lifted out of the ring, and read off the answer. invertibility is also a gift: chapters 21–22 use flows as generative models precisely because they can be run backwards.

> **spec · stage 4.** `NeuralODE(func, *, t1=1.0, method="rk4", n_steps=10, adjoint=False)`: an `nn.Module` with `func` (an `nn.Module` $f(t, y)$) registered as a submodule; `forward(y0)` integrates from $0$ to `t1` in `n_steps` equal steps of `method` and returns $y(t_1)$, using `odeint_adjoint` when `adjoint=True`. this is `mlfs.ode.NeuralODE`.

> **▶ build: stage 4.** implement `NeuralODE`.

> **🔮 predict:** a 2-D neural ODE followed by a linear classifier is trained on two concentric circles. what accuracy can it reach?

## worked example

one step of each method on $dy/dt = y$, $y(0) = 1$, $h = 0.1$ (truth $e^{0.1} = 1.1051709181$):

- Euler: $1 + 0.1 \cdot 1 = 1.1$. error $5.2 \times 10^{-3}$.
- midpoint: slope at the middle $f(0.05, 1.05) = 1.05$, so $1 + 0.1 \cdot 1.05 = 1.105$. error $1.7 \times 10^{-4}$.
- RK4: $k_1 = 1$, $k_2 = 1.05$, $k_3 = 1.0525$, $k_4 = 1.10525$, so $y_1 = 1 + \tfrac{0.1}{6}(1 + 2.1 + 2.105 + 1.10525) = 1.1051708333$. error $8.5 \times 10^{-8}$.

(RK4 on $dy/dt = y$ reproduces the Taylor series of $e^h$ through $h^4/24$: $1 + 0.1 + 0.005 + 0.000167 + 0.0000042$.) your steps must match these to $10^{-12}$; they're stage 1's first check.

and the adjoint, by hand, for $dy/dt = \lambda y$ with $L = y(T)$. the forward solution is $y(t) = y_0 e^{\lambda t}$. the adjoint equation $da/dt = -\lambda a$ with $a(T) = 1$ gives $a(t) = e^{\lambda(T - t)}$, so $\partial L/\partial y_0 = a(0) = e^{\lambda T}$ ✓. the parameter gradient is $\int_0^T a(t)\,\partial f/\partial\lambda\,dt = \int_0^T e^{\lambda(T-t)}\, y_0 e^{\lambda t}\,dt = T y_0 e^{\lambda T}$, which is $\partial/\partial\lambda$ of $y_0 e^{\lambda T}$ ✓. stage 2's gradient check uses the same numbers ($\lambda = -0.7$, $T = 2$, $y_0 = 1.5$).

## on paper

1. show that Euler's local error is $\tfrac{h^2}{2}y''(\xi)$ for some $\xi$, and deduce that the global error at a fixed time $T$ is $O(h)$ (assume the errors simply add up).

<details><summary>solution</summary>

Taylor's theorem with remainder: $y(t + h) = y(t) + h y'(t) + \tfrac{h^2}{2}y''(\xi)$, and $y'(t) = f(t, y(t))$ is exactly Euler's increment, so one step from the exact $y(t)$ errs by $\tfrac{h^2}{2}y''(\xi) = O(h^2)$. reaching $T$ takes $T/h$ steps; if each contributes $O(h^2)$ and they add (propagation through a Lipschitz $f$ multiplies them by at most $e^{LT}$, a constant), the total is $(T/h)\,O(h^2) = O(h)$.

</details>

2. derive the adjoint equation. let $\delta y(t)$ solve $d\,\delta y/dt = J(t)\,\delta y$ with $J = \partial f/\partial y$ along the trajectory. require $a(t)^\top\delta y(t)$ to be constant in $t$ for every such $\delta y$, and find $da/dt$.

<details><summary>solution</summary>

$\tfrac{d}{dt}(a^\top \delta y) = \dot a^\top \delta y + a^\top J \delta y = (\dot a^\top + a^\top J)\,\delta y$. this vanishes for every $\delta y$ iff $\dot a^\top = -a^\top J$, i.e. $\dot a = -J^\top a$. with $a(T) = \partial L/\partial y(T)$, constancy gives $a(t)^\top\delta y(t) = \partial L/\partial y(T)^\top \delta y(T) = \delta L$: $a(t)$ is the sensitivity of the loss to the state at time $t$.

</details>

3. derive the parameter gradient. a parameter perturbation $\delta\theta$ adds a forcing term: $d\,\delta y/dt = J\,\delta y + (\partial f/\partial\theta)\,\delta\theta$, with $\delta y(t_0) = 0$. show that $\delta L = \int_{t_0}^T a^\top (\partial f/\partial \theta)\,dt\ \delta\theta$.

<details><summary>solution</summary>

now $\tfrac{d}{dt}(a^\top\delta y) = (\dot a^\top + a^\top J)\delta y + a^\top (\partial f/\partial\theta)\delta\theta = a^\top(\partial f/\partial\theta)\delta\theta$, using the adjoint equation. integrate from $t_0$ to $T$: $a(T)^\top\delta y(T) - a(t_0)^\top\delta y(t_0) = \int a^\top(\partial f/\partial\theta)\,dt\,\delta\theta$. the left side is $\delta L - 0$. so $\partial L/\partial\theta = \int_{t_0}^T a^\top \partial f/\partial\theta\,dt$, which the augmented system accumulates as $a_\theta$.

</details>

4. show that no flow of a 1-D ODE $dy/dt = f(t, y)$ (with $f$ smooth) maps $-1 \mapsto 1$ and $1 \mapsto -1$ at the same time $t_1$.

<details><summary>solution</summary>

let $u(t)$ and $w(t)$ be the solutions from $u(0) = -1$ and $w(0) = 1$. then $w(0) - u(0) = 2 > 0$ and $w(t_1) - u(t_1) = -2 < 0$; by continuity there's a time $s$ with $u(s) = w(s)$. but solutions are unique, so two solutions equal at time $s$ are equal at all times, contradicting $u(0) \ne w(0)$. the same argument shows a 1-D flow is monotone increasing. adding a dimension removes the obstruction: in 2-D the two points can go around each other.

</details>

## experiment

the lab runs your solvers on three problems. you'll measure memory and time of backprop through the solver against your adjoint for 10 to 500 RK4 steps; compare the gradients of both, in float32 and float64, with an accurate reference on a linear system with a slow and a fast decay rate; and train plain and augmented neural ODEs to classify concentric circles. predict first: the second experiment's numbers are the ones people remember.

## check your understanding

1. RK4 needs four evaluations of $f$ per step. when would you still choose Euler?

<details><summary>answer</summary>

when $f$ is so noisy or cheap-and-rough that higher order buys nothing (stochastic dynamics, a vector field that's only piecewise smooth), when you need the simplest possible discretization to reason about (a ResNet *is* Euler), or when stability, not accuracy, dictates the step (stiff systems, where all explicit methods need tiny steps and the cheapest per step wins, or better, an implicit method).

</details>

2. why does `odeint` split each interval into equal substeps instead of stepping with exactly `step_size` and taking a short last step?

<details><summary>answer</summary>

either works, but equal substeps land exactly on every requested time with steps no longer than `step_size`, keep the step grid symmetric so the adjoint's backward solve can reuse the same grid, and avoid a tiny last step that wastes evaluations (or a floating-point remainder like $10^{-17}$ that creates a pointless extra step).

</details>

3. the adjoint's backward pass recomputes $y(t)$ by integrating backwards. why not just store $y$ at every step?

<details><summary>answer</summary>

that's a valid choice, called checkpointing: storing $y$ at every step costs $O(\text{steps} \times \text{state size})$, much less than backprop's activations of every layer of $f$, and avoids the reconstruction error. the pure adjoint stores nothing, which is the point of $O(1)$ memory. practical libraries offer the spectrum: checkpoint every $m$ steps and recompute in between.

</details>

4. you use the adjoint with RK4 and 10 steps, and the gradients disagree with backprop through the same solver in the fourth digit. bug or expected?

<details><summary>answer</summary>

expected. backprop gives the exact gradient of the discretized forward computation; the adjoint gives a discretization of the exact continuous gradient. they differ by the solver's error, $O(h^4)$ for RK4, which with 10 steps can easily be $10^{-4}$. refine the step and they converge to each other (stage 3's check uses step 0.02 to agree to $10^{-6}$).

</details>

5. why are neural ODEs invertible, and why does that matter beyond classification?

<details><summary>answer</summary>

the flow from $0$ to $t_1$ can be undone by integrating from $t_1$ back to $0$. that makes neural ODEs *continuous normalizing flows*: invertible maps whose density change follows from the trace of $\partial f/\partial y$, used as generative models (FFJORD), and the probability-flow ODE of diffusion models (ch 22) is exactly such a flow from noise to data.

</details>

6. an augmented neural ODE appends zeros to the input. why zeros, and what does the classifier read?

<details><summary>answer</summary>

the initial state must be a fixed function of the input so the model stays deterministic; zeros are the simplest. the flow then uses the extra dimension as room to move points past each other. the classifier reads all coordinates of the final state (or a projection of them); the extra coordinates are where the inner circle was lifted to, in the lab.

</details>

## common bugs

- **evaluating the stages at the wrong time**: symptom: the time-dependent check (`dy/dt = t`) fails although `dy/dt = y` passes. $k_2, k_3$ use $t + h/2$, $k_4$ uses $t + h$.
- **`while t < t_end: t += h`**: symptom: an extra step (or a missing one) from rounding, or an infinite loop when integrating backwards. count integer steps per interval.
- **a literal ceiling**: symptom: the grid check finds 4 steps where 3 were asked for. allow a tolerance before rounding up.
- **building a graph in the adjoint's forward pass**: symptom: the memory check finds saved bytes growing with the number of steps. solve under `torch.no_grad()`.
- **forgetting the intermediate outputs' gradients**: symptom: adjoint gradients match backprop when the loss uses only $y(T)$, and fail otherwise. add `grad_ys[k]` to the adjoint state when the backward solve reaches $t_k$.
- **sign errors in the adjoint**: symptom: gradients with the right magnitude and the wrong sign, or growing instead of matching. the augmented system has *minus* signs and is integrated from $T$ down to $t_0$.

## going deeper

- Chen, Rubanova, Bettencourt & Duvenaud (2018), "Neural ordinary differential equations", *NeurIPS*. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- Dupont, Doucet & Teh (2019), "Augmented neural ODEs", *NeurIPS*. what flows can't represent, and the fix. [arXiv:1904.01681](https://arxiv.org/abs/1904.01681)
- Gholami, Keutzer & Biros (2019), "ANODE: unconditionally accurate memory-efficient gradients for neural ODEs", *IJCAI*. the adjoint's reconstruction instability, and checkpointing. [arXiv:1902.10298](https://arxiv.org/abs/1902.10298)
- Haber & Ruthotto (2018), "Stable architectures for deep neural networks", *Inverse Problems* 34(1). ResNets as discretized ODEs. [arXiv:1705.03341](https://arxiv.org/abs/1705.03341)
- Kidger (2022), *On Neural Differential Equations*, PhD thesis, University of Oxford. a thorough, readable survey. [arXiv:2202.02435](https://arxiv.org/abs/2202.02435)
- Hairer, Nørsett & Wanner (1993), *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed., Springer, chapter II (Runge–Kutta methods). [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)

## connections

- ← builds on: Taylor series (ch 03), reverse-mode autodiff and vector-Jacobian products (ch 14), the matrix exponential (ch 02), residual networks (ch 16) and PyTorch (ch 15).
- → used in: chapter 22 integrates the probability-flow ODE of diffusion models with your `mlfs.ode.odeint`, and flow matching trains a vector field for exactly this kind of solver; continuous normalizing flows (ch 21–22) use invertibility; ResNets are the Euler-discretized cousin.
