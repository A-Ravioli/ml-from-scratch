# hamiltonian monte carlo

## why this matters

topic 05 left random-walk Metropolis with a scaling law you can't tune your way out of: in $d$ dimensions the step size has to shrink like $1/\sqrt d$ to keep proposals acceptable, and a random walk with small steps needs $O(d)$ steps per independent draw. on a posterior with a thousand parameters, or with two parameters that are strongly correlated, it crawls. Gibbs sampling has the same problem along narrow valleys.

**Hamiltonian Monte Carlo** fixes this by using the one thing the random walk ignores: the gradient of the log density. it imagines the parameters as the position of a frictionless puck sliding on the surface $U(q) = -\log\pi(q)$, gives it a random kick, and lets it slide for a while. the physics carries it a long way, in a consistent direction, along the contours of the distribution, and a simple numerical integrator keeps the energy almost exactly constant, so the far-away endpoint is accepted almost every time. the cost per independent draw grows only like $d^{1/4}$ instead of $d$. HMC (and its adaptive descendant NUTS) is the engine of Stan, PyMC and NumPyro, and it's what you'll use for the eight-schools model in the capstone.

you'll build the integrator, the sampler on top of it, plug in your forward-mode autodiff from chapter 03 so you never write a gradient by hand, and finish with the idea behind NUTS: how long should a trajectory be?

## warm-up: predict first

1. a puck slides without friction on a bowl-shaped surface. if you simulate it with a numerical integrator, what should happen to its total energy?
2. HMC accepts or rejects a whole trajectory based on the change in energy. if the integrator were exact, what would the acceptance rate be?
3. on a standard Gaussian, HMC's acceptance rate as a function of the step size: does it decline smoothly, or does something abrupt happen?
4. is a longer trajectory always better?
5. forward-mode autodiff costs one pass per parameter. how many passes does one HMC trajectory of 20 steps cost for a 10-parameter model?

## prerequisites check

- [ ] you can run and diagnose random-walk Metropolis, and compute ESS and $\hat R$ (→ topic 05)
- [ ] you can compute gradients with dual numbers via `mlfs.autodiff.grad`, and know which operations it supports (→ ch 03 · topic 04)
- [ ] you know the multivariate Gaussian and its density (→ ch 04 · topic 04)
- [ ] you know what an ODE is and Euler's method for solving one (→ ch 03)

## core ideas

### idea 1: Hamiltonian dynamics and the leapfrog integrator

augment the parameters $q \in \mathbb R^d$ with a **momentum** $p \in \mathbb R^d$ and define the total energy

$$H(q, p) = U(q) + K(p), \qquad U(q) = -\log\pi(q), \qquad K(p) = \tfrac12p^\top M^{-1}p,$$

where $M$ is a **mass matrix** (identity for now). Hamilton's equations move $q$ and $p$ through time:

$$\frac{dq}{dt} = \frac{\partial H}{\partial p} = M^{-1}p, \qquad \frac{dp}{dt} = -\frac{\partial H}{\partial q} = \nabla\log\pi(q).$$

the position moves with the velocity; the momentum is pushed downhill on $U$, i.e. towards high density. three properties make this the right dynamics for sampling: it **conserves** $H$ (warm-up 1), it's **reversible** (flip the momentum and you retrace your path), and it **preserves volume** in $(q, p)$ space (Liouville's theorem). exact solutions are rarely available, so you integrate numerically, and the integrator has to keep those properties. Euler's method doesn't: its energy drifts steadily.

the **leapfrog** (Störmer–Verlet) integrator does, apart from conservation, which it gets *almost* right. one step of size $\varepsilon$:

$$p \leftarrow p + \tfrac\varepsilon2\nabla\log\pi(q), \qquad q \leftarrow q + \varepsilon M^{-1}p, \qquad p \leftarrow p + \tfrac\varepsilon2\nabla\log\pi(q).$$

a half kick, a full drift, a half kick. each piece is a shear (it changes $q$ using only $p$, or $p$ using only $q$), so each has Jacobian determinant 1, and so does their composition: volume is preserved exactly. the scheme is symmetric, so it's exactly reversible. and its energy error doesn't drift: it oscillates, with an amplitude of order $\varepsilon^2$, because leapfrog exactly conserves a slightly perturbed "shadow" Hamiltonian. when you chain $L$ steps, the second half kick of one step and the first half kick of the next merge into a single full kick, so $L$ steps cost $L + 1$ gradient evaluations.

there's a hard limit. on $U(q) = q^2/(2\sigma^2)$ one leapfrog step is a linear map with eigenvalues on the unit circle only if $\varepsilon < 2\sigma$ (on paper 2). beyond that, the eigenvalues become real, one of them larger than 1, and the trajectory explodes exponentially: the step size must be below twice the *smallest* scale of the target.

> **spec.** `hamiltonian(q, p, log_prob, inv_mass=None)` returns $-\log\pi(q) + \frac12\sum_i m_i p_i^2$ as a float, where `inv_mass` $= (m_i)$ is the diagonal of $M^{-1}$ (None means all ones). `leapfrog(q, p, grad_log_prob, step_size, n_steps, inv_mass=None)` returns new arrays `(q, p)` after `n_steps` steps (zero steps: unchanged copies), uses at most `n_steps + 1` gradient evaluations, and doesn't modify its inputs. the position update is `q += step_size * inv_mass * p`.

> **🔮 predict:** one leapfrog step of size 0.5 on a standard Gaussian from $(q, p) = (1, 0)$. where does it land, compared with the exact solution $(\cos 0.5, -\sin 0.5) = (0.878, -0.479)$? is the energy higher or lower than at the start?

> **▶ build: stage 1.** implement `hamiltonian` and `leapfrog`.

### idea 2: the HMC sampler

the joint density $\pi(q, p) \propto \exp(-H(q, p)) = \pi(q)\cdot\mathcal N(p; 0, M)$ has the target as its $q$-marginal and an independent Gaussian momentum. HMC samples this joint by alternating two moves, each of which leaves it invariant:

1. **refresh the momentum**: draw $p \sim \mathcal N(0, M)$, an exact Gibbs step for $p$.
2. **propose with physics**: run $L$ leapfrog steps from $(q, p)$ to $(q^*, p^*)$ and accept $q^*$ with probability $\min\big(1, \exp(H(q, p) - H(q^*, p^*))\big)$; otherwise keep $q$.

why is step 2 a valid Metropolis–Hastings step? think of the proposal as "leapfrog, then flip the momentum": that map is deterministic, its own inverse (reversibility), and has Jacobian 1 (volume preservation), so the proposal densities cancel in the MH ratio and only $\pi(q^*, p^*)/\pi(q, p) = e^{-\Delta H}$ is left. the flip doesn't matter because $K$ is symmetric and $p$ is resampled next anyway. if the integrator were exact, $\Delta H = 0$ and every proposal would be accepted (warm-up 2); leapfrog's small energy error means a high but imperfect acceptance rate, and the rule exactly corrects for the discretization. if the trajectory blew up (a **divergence**: $H$ is huge, infinite or NaN), reject.

the **mass matrix** is a preconditioner: with $M^{-1} = \operatorname{diag}(\sigma_1^2, \dots, \sigma_d^2)$ set to the posterior variances, every direction looks like scale 1 to the integrator, and one step size fits all. (Stan learns it during warm-up.) with $M^{-1}$ diagonal: draw $p_i = z_i/\sqrt{m_i}$ with $z_i \sim \mathcal N(0, 1)$, and use the same $m_i$ in $K$ and in the drift.

> **spec.** `hmc(log_prob, x0, n_samples, *, grad_log_prob=None, step_size=0.1, n_leapfrog=20, burn_in=0, inv_mass=None, rng=None)` runs `burn_in + n_samples` iterations. each one: `z = rng.standard_normal(d)` (rng from `as_rng`), $p = z/\sqrt m$; compute $H_0$; leapfrog; compute $H_1$ (infinite if the new position isn't finite); `u = rng.random()`; accept if $H_1$ is finite and $\log u < H_0 - H_1$. record the position after every iteration past the burn-in. return `(samples (n_samples, d), accept_rate)`, the rate over all iterations. (stage 3 covers `grad_log_prob=None`.)

> **▶ build: stage 2.** implement `hmc`.

### idea 3: gradients without writing them

HMC needs $\nabla\log\pi$ at every leapfrog step, and deriving it by hand for every model is tedious and error-prone. you built forward-mode automatic differentiation in chapter 03: `mlfs.autodiff.grad(f, x)` evaluates `f` on dual numbers, once per coordinate, and returns the exact gradient. so when `grad_log_prob` is `None`, `hmc` should use `lambda x: grad(log_prob, x)`.

two consequences. first, `log_prob` must be **Dual-friendly**: written with arithmetic, `**`, and NumPy's `np.exp`, `np.log`, `np.log1p`, `np.expm1`, `np.sqrt`, `np.tanh` on scalars or object arrays, but not `math.*`, `float()`, `np.logaddexp` or `np.linalg`. second, cost: forward mode needs $d$ passes per gradient, so a trajectory of $L$ steps on a $d$-parameter model costs about $(L + 1)d$ evaluations of `log_prob` (warm-up 5: 210 for $L = 20$, $d = 10$). reverse mode (backpropagation, chapter 14) gets the whole gradient in one backward pass, which is why real PPLs use it; the capstone shows a trick that gets forward mode most of the way there.

> **spec.** with `grad_log_prob=None`, `hmc` uses `mlfs.autodiff.grad(log_prob, x)`; with the same seed it must produce the same samples as when you pass the exact gradient.

> **▶ build: stage 3.** add the autodiff default to `hmc`.

### idea 4: how long to integrate, and the NUTS idea

the step size $\varepsilon$ is limited by stability and acceptance; the number of steps $L$ (the trajectory length $T = L\varepsilon$) is the other knob, and it's treacherous (warm-up 4). too short and HMC degenerates into a random walk (a single leapfrog step is a Langevin step). too long and the trajectory curves back on itself: on a Gaussian the exact dynamics are periodic, so with $T = 2\pi\sigma$ the puck returns exactly where it started and the sampler doesn't move at all, while $T = \pi\sigma$ sends $q$ to $-q$, which is perfect for estimating the mean and useless for the variance. the lab maps this out. real posteriors have many scales at once, so no single $T$ is right everywhere.

the **No-U-Turn Sampler** (Hoffman & Gelman, 2014) chooses $T$ per iteration: keep integrating until the trajectory starts to come back. the criterion: the trajectory from $q^-$ to $q^+$ is still stretching as long as both ends move away from each other,

$$(q^+ - q^-)\cdot p^- \ge 0 \quad\text{and}\quad (q^+ - q^-)\cdot p^+ \ge 0.$$

when either fails, a U-turn has begun. NUTS grows the trajectory by repeated doubling, randomly forwards or backwards in time (to keep the whole procedure reversible), stops at the first U-turn, and samples a point from the trajectory with the right probabilities. the details are intricate; the idea is the U-turn test, and you'll implement it and watch it adapt the trajectory to the local scale.

> **spec.** `no_u_turn(q_minus, q_plus, p_minus, p_plus)` returns the bool above. `u_turn_steps(q0, p0, grad_log_prob, step_size, max_steps=1000)` integrates one leapfrog step at a time from $(q_0, p_0)$ and returns the first step count $s \ge 1$ with $(q_s - q_0)\cdot p_s < 0$, or `max_steps` if that never happens (the one-sided version of the test, for a trajectory grown only forwards).

> **🔮 predict:** start at rest at $q = 1$ on a standard Gaussian and take leapfrog steps of size 0.1. after how many steps does the trajectory make a U-turn? and on a Gaussian with standard deviation 5, starting at the mode with $p = 1$?

> **▶ build: stage 4.** implement `no_u_turn` and `u_turn_steps`.

## worked example

**one leapfrog step.** standard Gaussian, so $\nabla\log\pi(q) = -q$; start at $(q, p) = (1, 0)$ with $\varepsilon = 0.5$.

- half kick: $p = 0 + 0.25\cdot(-1) = -0.25$.
- drift: $q = 1 + 0.5\cdot(-0.25) = 0.875$.
- half kick: $p = -0.25 + 0.25\cdot(-0.875) = -0.46875$.

the exact solution after time 0.5 is $(\cos 0.5, -\sin 0.5) = (0.8776, -0.4794)$: leapfrog is close, even with a big step. the energy went from $H_0 = \frac12(1^2 + 0^2) = 0.5$ to $H_1 = \frac12(0.875^2 + 0.46875^2) = 0.49268$: an error of $-0.0073$, so this proposal would be accepted with probability $\min(1, e^{0.0073}) = 1$. your `leapfrog` must reproduce these numbers.

**the stability limit.** for $\nabla\log\pi(q) = -q$, one step maps $(q, p)$ linearly by
$$\begin{pmatrix}q'\\p'\end{pmatrix} = \begin{pmatrix}1 - \frac{\varepsilon^2}{2} & \varepsilon\\-\varepsilon\big(1 - \frac{\varepsilon^2}{4}\big) & 1 - \frac{\varepsilon^2}{2}\end{pmatrix}\begin{pmatrix}q\\p\end{pmatrix}.$$
its determinant is 1 (volume preservation) and its trace is $2 - \varepsilon^2$. the eigenvalues are complex with modulus 1 (a rotation: bounded orbits) when $|2 - \varepsilon^2| < 2$, i.e. $\varepsilon < 2$; at $\varepsilon > 2$ they're real with product 1, one of them above 1 in magnitude, and the orbit grows geometrically.

## on paper

1. **the leapfrog step preserves volume.** show that each of the three sub-steps (kick, drift, kick) has Jacobian determinant 1.

<details><summary>solution</summary>

the kick $(q, p) \mapsto (q, p + \frac\varepsilon2g(q))$ has Jacobian $\begin{pmatrix}I & 0\\\frac\varepsilon2\nabla g & I\end{pmatrix}$, block lower-triangular with identity blocks on the diagonal: determinant 1. the drift $(q, p) \mapsto (q + \varepsilon M^{-1}p, p)$ is block upper-triangular with identities on the diagonal: determinant 1. the determinant of a composition is the product.

</details>

2. **the stability limit.** derive the matrix in the worked example and show that its eigenvalues have modulus 1 exactly when $\varepsilon \le 2$.

<details><summary>solution</summary>

kick: $p_{1/2} = p - \frac\varepsilon2q$. drift: $q' = q + \varepsilon p_{1/2} = (1 - \frac{\varepsilon^2}{2})q + \varepsilon p$. kick: $p' = p_{1/2} - \frac\varepsilon2q' = -\varepsilon(1 - \frac{\varepsilon^2}{4})q + (1 - \frac{\varepsilon^2}{2})p$. the characteristic polynomial is $\lambda^2 - (2 - \varepsilon^2)\lambda + 1$, with discriminant $(2 - \varepsilon^2)^2 - 4 = \varepsilon^2(\varepsilon^2 - 4)$. for $\varepsilon < 2$ it's negative: complex conjugate roots with product 1, so both have modulus 1. for $\varepsilon > 2$, two real roots with product 1, one of magnitude above 1. with a target sd $\sigma$, replace $\varepsilon$ by $\varepsilon/\sigma$.

</details>

3. **HMC is a valid MH step.** let $F$ be "$L$ leapfrog steps, then negate $p$". show $F(F(z)) = z$, and conclude that accepting $F(z)$ with probability $\min(1, e^{-\Delta H})$ satisfies detailed balance for $e^{-H}$.

<details><summary>solution</summary>

reversibility of leapfrog says: integrating from $(q^*, -p^*)$ for $L$ steps returns $(q, -p)$; negating gives $(q, p)$. so $F$ is an involution, with $|\det DF| = 1$. for a deterministic involutive proposal, the MH acceptance probability is $\min(1, \pi(F(z))/\pi(z)\cdot|\det DF(z)|) = \min(1, e^{H(z) - H(F(z))})$, and detailed balance follows exactly as for ordinary MH (the flows between $z$ and $F(z)$ match).

</details>

4. **the energy error.** for the harmonic oscillator and one leapfrog step from $(q, 0)$, compute $\Delta H$ to leading order in $\varepsilon$.

<details><summary>solution</summary>

$q' = (1 - \frac{\varepsilon^2}2)q$ and $p' = -\varepsilon(1 - \frac{\varepsilon^2}4)q$, so $2H' = q^2\big[(1 - \frac{\varepsilon^2}{2})^2 + \varepsilon^2(1 - \frac{\varepsilon^2}{4})^2\big] = q^2\big[1 - \frac{\varepsilon^4}{4} + O(\varepsilon^6)\big]$. so $\Delta H = -\frac{\varepsilon^4}{8}q^2 + \dots$ for one step from rest. (check: $\varepsilon = 0.5$, $q = 1$ gives $-0.0078$; the exact value in the worked example is $-0.0073$.) over a trajectory of fixed length the error is $O(\varepsilon^2)$, oscillating rather than accumulating.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how HMC and random-walk Metropolis compare at equal computational cost, on a correlated 2-D Gaussian and a 100-D Gaussian.
2. how HMC's acceptance rate and energy error depend on the step size, in 1 and 100 dimensions.
3. how the effective sample size depends on the trajectory length.

## check your understanding

1. why does the momentum have to be redrawn at every iteration?

<details><summary>answer</summary>

the Hamiltonian flow conserves $H$, so without refreshing, the chain would be stuck on one energy level set and couldn't sample the full distribution. resampling $p$ moves it between energy levels; it's an exact Gibbs step on the auxiliary variable.

</details>

2. what's the acceptance rate of HMC with an exact integrator? why use leapfrog rather than a more accurate high-order integrator like Runge–Kutta?

<details><summary>answer</summary>

1. leapfrog is less accurate per step than RK4, but it's exactly reversible and volume preserving, which is what makes the MH correction work with just $e^{-\Delta H}$; RK4 is neither, so its proposals would need a Jacobian correction and wouldn't be reversible. leapfrog's energy error also doesn't drift over long trajectories.

</details>

3. why must the step size be below $2\sigma_{\min}$, and what happens to HMC on a badly scaled target without a mass matrix?

<details><summary>answer</summary>

the narrowest direction sets the stability limit; beyond it, trajectories explode and everything is rejected. so on a target with scales 10 and 0.1 the step must be below 0.2, and a trajectory needs at least 10/0.2 = 50 steps just to move one standard deviation along the wide direction. a mass matrix rescales the directions so one step size suits all.

</details>

4. a trajectory diverges (the energy error is enormous). what does that tell you about the posterior?

<details><summary>answer</summary>

that somewhere along the trajectory the curvature was too high for the step size: a region where the posterior gets narrow very quickly, like the neck of a funnel (capstone). the chain can't enter that region, so estimates are biased against it; you need a smaller step or a reparameterization.

</details>

5. why does HMC's cost per independent sample grow like $d^{1/4}$ while random-walk Metropolis's grows like $d$?

<details><summary>answer</summary>

the energy error of a leapfrog trajectory adds up over $d$ roughly independent coordinates, with variance $\propto d\varepsilon^4$, so keeping the acceptance rate fixed needs $\varepsilon \propto d^{-1/4}$; a trajectory of fixed length then needs $L \propto d^{1/4}$ steps, and it moves a distance $O(1)$ in a consistent direction, giving nearly independent samples. a random walk's step must shrink like $d^{-1/2}$ and its displacement grows only like the square root of the number of steps (Neal, 2011).

</details>

6. NUTS doubles the trajectory randomly forwards *or backwards* in time. why not just integrate forwards until the U-turn?

<details><summary>answer</summary>

the stopping rule depends on the trajectory; integrating forwards only and stopping at a U-turn would make the proposal non-reversible (starting from the endpoint and integrating forwards would stop somewhere else), breaking detailed balance. building a trajectory symmetric in time, and sampling within it, restores it.

</details>

## common bugs

- **full kicks at both ends.** symptom: an energy error that halves (instead of quartering) when you halve the step, and wrong samples. the first and last kicks are half kicks.
- **inconsistent mass matrix.** symptom: wrong variances with a non-identity `inv_mass`. draw $p \sim \mathcal N(0, M)$ (i.e. divide by $\sqrt{m}$), and use $M^{-1}$ in both the kinetic energy and the drift.
- **keeping the proposal after a rejection.** symptom: biased samples. a rejected trajectory leaves $q$ unchanged.
- **accepting NaN energies.** symptom: NaN samples after a divergence. treat a non-finite $H$ as a rejection.
- **a log density that isn't Dual-friendly.** symptom: `TypeError: float() argument must be ...` from autodiff. use `np.exp`/`np.log` on the values, not `math.exp`, and avoid `float()` inside `log_prob`.
- **gradient of the wrong sign.** symptom: the puck flies away from the mode and every proposal is rejected. `grad_log_prob` is the gradient of $\log\pi$, i.e. $-\nabla U$.

## going deeper

- Neal (2011), "MCMC using Hamiltonian dynamics", in *Handbook of Markov Chain Monte Carlo*, CRC Press. the definitive tutorial. [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)
- Duane, Kennedy, Pendleton & Roweth (1987), "Hybrid Monte Carlo", *Physics Letters B* 195(2). the original algorithm, from lattice QCD. [doi:10.1016/0370-2693(87)91197-X](https://doi.org/10.1016/0370-2693(87)91197-X)
- Hoffman & Gelman (2014), "The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo", *JMLR* 15. [arXiv:1111.4246](https://arxiv.org/abs/1111.4246)
- Betancourt (2017), "A conceptual introduction to Hamiltonian Monte Carlo". geometry, typical sets, and divergences. [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)
- Leimkuhler & Reich (2004), *Simulating Hamiltonian Dynamics*, Cambridge University Press. symplectic integrators and shadow Hamiltonians.
- Carpenter et al. (2017), "Stan: a probabilistic programming language", *Journal of Statistical Software* 76(1). [doi:10.18637/jss.v076.i01](https://doi.org/10.18637/jss.v076.i01)

## connections

- ← builds on: Metropolis–Hastings, ESS and $\hat R$ (topic 05), forward-mode autodiff (ch 03 · topic 04), ODEs and Euler's method (ch 03), the multivariate Gaussian (ch 04).
- → used in: the capstone's PPL and the eight-schools model (divergences and the funnel), reverse-mode autodiff that makes HMC scale (ch 14), Langevin dynamics and score-based generative models (ch 22), neural ODEs and symplectic networks (ch 20).
