# linear state-space models (S4)

## why this matters

transformers compare every token with every other token, which costs $O(L^2)$ in the sequence length: fine for a paragraph, painful for a book, impossible for an hour of audio at 16 kHz. recurrent networks cost $O(L)$, but you train them one step at a time, and their gradients vanish across long gaps, so in practice they forget. in 2021 the Long Range Arena benchmark (Tay et al., 2021) collected tasks with dependencies over thousands of steps, and every efficient transformer variant scored near chance on its hardest task, Path-X (16,384 steps). then S4 (Gu, Goel & Ré, 2022) solved it.

S4's trick was to go back to the oldest idea in this block: a linear state-space model, $x_t = \bar Ax_{t-1} + \bar Bu_t$, the same recurrence as topic 06's Kalman filter, but with the input driving it instead of noise, no probabilistic interpretation, and the matrices *learned*. three things make it work. it's defined in continuous time and discretized, which gives it principled, stable parameters. because it's linear and time-invariant, the whole output is one long convolution, which you can train in parallel with FFTs, and then run as a recurrence at inference. and its state matrix is initialized from HiPPO, a theory of how to compress a signal's entire history into a fixed-size state. this topic builds the diagonal version, S4D (Gu, Gupta, Goel & Ré, 2022), which keeps nearly all of S4's performance with far less machinery. in the lab you'll see a 16-number state remember a 1,000-step signal, and your layer beat an LSTM on a task that needs memory across hundreds of steps.

## warm-up: predict first

1. a continuous-time system $x'(t) = -x(t) + u(t)$ is sampled every $\Delta$ seconds. if you halve $\Delta$, should the discrete recurrence's coefficient move toward 0 or toward 1?
2. a linear recurrence $x_k = \bar a x_{k-1} + \bar bu_k$, $y_k = cx_k$ is run on a sequence of length $L$. can you write $y$ as a convolution of $u$ with something? what?
3. convolving two length-$L$ sequences directly costs $O(L^2)$. how fast can you do it, and what goes wrong if you do it naively with an FFT of length $L$?
4. you have 16 numbers and a stream of 1,000 samples. can you design an update rule that lets you reconstruct the whole stream, roughly, from the 16 numbers at the end?
5. an LSTM and a linear SSM layer, same number of parameters, must remember the first of 256 tokens. which learns faster, and why?

## prerequisites check

- [ ] you know the linear recurrence $x_t = Fx_{t-1} + \dots$ and its unrolling (→ topic 06; topic 05's psi weights)
- [ ] you can compute eigenvalues and know that $e^{\lambda t}$ decays iff $\operatorname{Re}\lambda < 0$ (→ ch 02 · topic 05)
- [ ] you know the convolution theorem and the FFT (→ ch 16 · topic 01; ch 01)
- [ ] you've built an LSTM and seen gradients vanish through time (→ ch 17 · topics 04–05)
- [ ] you can write a PyTorch module with complex parameters' real and imaginary parts as real `nn.Parameter`s (→ ch 15 · topic 07)

## core ideas

### idea 1: continuous-time SSMs and discretization

a **continuous-time linear state-space model** maps an input signal $u(t)$ to an output $y(t)$ through a state $x(t) \in \mathbb C^N$:

$$x'(t) = Ax(t) + Bu(t), \qquad y(t) = Cx(t) + Du(t).$$

it's a linear ODE: the state integrates the input, $A$ sets how the memory decays and oscillates, $B$ how the input enters, $C$ how the state is read out. when $A$ is **diagonal**, $A = \operatorname{diag}(\lambda_1, \dots, \lambda_N)$, the system splits into $N$ independent modes $x_n' = \lambda_nx_n + B_nu$, each a damped oscillator with decay rate $-\operatorname{Re}\lambda_n$ and frequency $\operatorname{Im}\lambda_n$. it's stable when every $\operatorname{Re}\lambda_n < 0$.

sequences are discrete, so you **discretize** with a step $\Delta$: treat $u$ as a signal sampled every $\Delta$ and find the recurrence $x_k = \bar Ax_{k-1} + \bar Bu_k$ that matches the ODE. the **zero-order hold** (ZOH) assumes the input is constant over each step; then the ODE solves exactly (on paper 1):

$$\bar A = e^{A\Delta}, \qquad \bar B = A^{-1}(e^{A\Delta} - I)B \quad\Big(\text{diagonal: } \bar\lambda_n = e^{\lambda_n\Delta},\ \bar B_n = \tfrac{\bar\lambda_n - 1}{\lambda_n}B_n\Big).$$

the **bilinear** (Tustin) method approximates the integral by the trapezoid rule:

$$\bar A = (I - \tfrac\Delta2A)^{-1}(I + \tfrac\Delta2A), \qquad \bar B = (I - \tfrac\Delta2A)^{-1}\Delta B.$$

both map stable continuous modes to stable discrete ones ($|\bar\lambda| < 1$), for any $\Delta$; for small $\Delta$ both approach Euler's $\bar A \approx I + A\Delta$, $\bar B \approx \Delta B$. S4 used bilinear; S4D and Mamba use ZOH.

why bother with continuous time at all, instead of learning $\bar A$ directly? three reasons. the continuous parameters make stability trivial to enforce (keep $\operatorname{Re}\lambda < 0$). $\Delta$ becomes a separate, interpretable knob: the time scale of the memory, which a layer can learn per channel. and the same continuous model can be resampled: a model trained on audio at 16 kHz can run at 8 kHz by doubling $\Delta$.

> **spec.** `discretize(Lambda, B, dt, method="zoh")` takes torch tensors (complex `Lambda`, `B` broadcastable against it, `dt` a float or a tensor broadcastable against `Lambda`, e.g. shape `(H, 1)` for one step per channel) and returns `(Lambda_bar, B_bar)` by the diagonal formulas above, for `method` in `{"zoh", "bilinear"}`; any other method raises `ValueError`. use torch operations so gradients flow to `dt` and `Lambda`.

> **🔮 predict:** with $\lambda = -1$ and $\Delta = \ln 2$, what are $\bar\lambda$ and $\bar B$ under ZOH (with $B = 1$)?

> **▶ build: stage 1.** implement `discretize`.

### idea 2: two views of one layer: recurrence and convolution

an S4D layer processes $H$ channels independently (it's single-input, single-output per channel; mixing across channels happens elsewhere, idea 5). per channel, with diagonal $\bar A$, the recurrence is elementwise over the $N$ modes:

$$x_k = \bar\lambda \odot x_{k-1} + \bar B\,u_k, \qquad y_k = 2\operatorname{Re}\Big(\sum_nC_nx_{k,n}\Big), \qquad x_{-1} = 0.$$

why $2\operatorname{Re}$? a real system with complex eigenvalues has them in conjugate pairs, and the two members of a pair contribute complex-conjugate terms to $y$. S4D stores only one of each pair and doubles the real part, which halves the state.

unroll the recurrence: $x_k = \sum_{j=0}^k\bar\lambda^j\bar Bu_{k-j}$, so

$$y_k = \sum_{j=0}^kK_ju_{k-j}, \qquad K_j = 2\operatorname{Re}\Big(\sum_nC_n\bar B_n\bar\lambda_n^j\Big).$$

the layer is a **causal convolution** with the **SSM kernel** $K = (K_0, \dots, K_{L-1})$, the impulse response of the recurrence. the two views compute the same function and have opposite strengths:

| | recurrent view | convolutional view |
|---|---|---|
| training on a length-$L$ sequence | $L$ sequential steps | one kernel ($O(NL)$) + one FFT convolution ($O(L\log L)$), all parallel |
| generating one new token | $O(N)$ per step, constant memory | recompute over the whole prefix |
| works when | always | the system is time-invariant (same $\bar A$, $\bar B$, $C$ at every step) |

so you train in the convolutional view and deploy in the recurrent one, with the same parameters. the last row matters: topic 08 makes the parameters depend on the input, and the convolution view disappears.

the kernel's shape is where the memory lives. each mode contributes a damped oscillation $\bar\lambda_n^j = e^{\lambda_n\Delta j}$ with envelope $e^{\operatorname{Re}(\lambda_n)\Delta j}$: with $\operatorname{Re}\lambda = -1/2$, a mode's amplitude halves every $2\ln2/\Delta$ steps. small $\Delta$ means long memory.

> **spec.** `ssm_recurrent(Lambda_bar, B_bar, C, u)` runs the recurrence above: `Lambda_bar`, `B_bar`, `C` are complex `(H, N)`, `u` is real `(batch, L, H)`, and it returns the real `(batch, L, H)` outputs. `ssm_kernel(Lambda_bar, B_bar, C, L)` returns the real `(H, L)` kernel $K$ above, lags $0, \dots, L - 1$.

> **▶ build: stage 2.** implement `ssm_recurrent` and `ssm_kernel`.

### idea 3: fast causal convolution

a direct causal convolution of length-$L$ sequences costs $O(L^2)$: $L$ outputs, each a sum of up to $L$ products. the **convolution theorem** does it in $O(L\log L)$: transform both sequences with the FFT, multiply pointwise, transform back. one catch: the DFT computes a *circular* convolution, where indices wrap around modulo the transform length. with transform length $L$, output $k$ picks up products $K_ju_{k-j+L}$ from the end of the sequence, which is both wrong and non-causal (the start of the output sees the end of the input). the fix is to zero-pad both sequences to length at least $2L - 1$: then no product wraps into the first $L$ outputs, and you keep those. with real inputs, `torch.fft.rfft` / `irfft` do half the work.

this is the step that makes long sequences trainable: for $L = 16{,}384$, a direct convolution needs about $1.3\times10^8$ multiply-adds per channel, the FFT version about $10^6$.

> **spec.** `causal_conv(u, K)`: `u` real `(batch, L, H)`, `K` real `(H, L)`; returns `y` of shape `(batch, L, H)` with $y_{b,k,h} = \sum_{j=0}^kK_{h,j}u_{b,k-j,h}$, computed with `torch.fft` and zero padding. it must be differentiable in both arguments.

> **🔮 predict:** if you forget the padding and use FFTs of length $L$, which outputs are wrong: the first few, the last few, or all of them?

> **▶ build: stage 3.** implement `causal_conv`.

### idea 4: HiPPO: memory as online function approximation

what should $A$ be? a random stable $A$ gives a memory that forgets exponentially fast. HiPPO (Gu, Dao, Ermon, Rudra & Ré, 2020) asks a precise question instead: at every time $t$, I want my $N$-number state $c(t)$ to be the best approximation of the *whole history* $u(s)$, $s \le t$, by $N$ polynomials, with respect to some measure over the past. for each measure, the coefficients obey a linear ODE $c'(t) = A(t)c(t) + B(t)u(t)$, and you can derive $A$ and $B$ exactly.

with the uniform measure over $[0, t]$ (**LegS**, scaled Legendre: every moment of the past matters equally), the projection onto Legendre polynomials gives

$$c'(t) = \frac1t\big(Ac(t) + Bu(t)\big), \qquad A_{nk} = -\begin{cases}\sqrt{2n+1}\sqrt{2k+1} & n > k\\ n + 1 & n = k\\ 0 & n < k\end{cases}, \qquad B_n = \sqrt{2n+1}.$$

the history is then recovered as $u(s) \approx \sum_nc_n(t)\sqrt{2n+1}\,P_n(2s/t - 1)$, with $P_n$ the Legendre polynomials. discretized with step $1/k$ at sample $k$ (the bilinear rule, $c_k = (I - \frac{A}{2k})^{-1}\big[(I + \frac{A}{2k})c_{k-1} + \frac1kBu_k\big]$), this runs online in $O(N^2)$ per sample, and lab experiment 1 shows 16 numbers reconstructing 1,000 samples.

S4 dropped the $1/t$ (making the system time-invariant, with memory controlled by a learned $\Delta$ instead) and kept $A$. that $A$ isn't diagonalizable stably, but it is **normal plus low rank**: with $P_n = \sqrt{n + 1/2}$, $A + PP^\top$ is a skew-symmetric matrix minus $\frac12I$ (on paper 3), so it's unitarily diagonalizable with eigenvalues $-\frac12 + i\omega_n$. S4 used this structure with a Cauchy-kernel algorithm; **S4D** (Gu, Gupta, Goel & Ré, 2022) simply throws away the low-rank part and uses the eigenvalues of the normal part as a diagonal $A$, which works almost as well (Gupta, Gu & Berant, 2022, found the same empirically). S4D also offers two closed-form approximations:

- **S4D-LegS**: the eigenvalues of $A + PP^\top$ for the HiPPO-LegS matrix of size $2N$, keeping the $N$ with positive imaginary part (one of each conjugate pair);
- **S4D-Inv**: $\lambda_n = -\frac12 + i\frac N\pi\big(\frac N{2n+1} - 1\big)$, which approximates LegS's frequencies;
- **S4D-Lin**: $\lambda_n = -\frac12 + i\pi n$, frequencies on a uniform grid, like a damped Fourier basis.

> **spec.** `hippo_legs(N)` returns `(A, B)` as float64 NumPy arrays of shapes `(N, N)` and `(N,)`, with the signs above (A's entries are $\le 0$). `s4d_init(N, init="lin")` returns a complex64 tensor of $N$ eigenvalues for `init` in `{"lin", "inv", "legs"}` as defined above (`"legs"` sorted by increasing imaginary part); any other `init` raises `ValueError`.

> **▶ build: stage 4.** implement `hippo_legs` and `s4d_init`.

### idea 5: the S4D layer

put it together. an `S4DLayer` holds, for $H$ channels and $N$ modes per channel:

- `log_dt` $(H,)$: one step size per channel, $\Delta = e^{\texttt{log\_dt}}$, initialized **log-uniformly** in $[\Delta_{\min}, \Delta_{\max}]$ (default $[10^{-3}, 10^{-1}]$). different channels get different time scales, from memories of tens of steps to thousands, and training adjusts them.
- `log_neg_re` and `im` $(H, N)$: $\lambda = -e^{\texttt{log\_neg\_re}} + i\cdot\texttt{im}$, initialized from `s4d_init`. the exponential keeps every mode stable whatever the optimizer does.
- `C` $(H, N, 2)$: the complex readout, stored as real and imaginary parts, initialized $\mathcal N(0, 1/2)$ each (unit variance per complex entry).
- `D` $(H,)$: a skip connection, $y = K * u + Du$, initialized $\mathcal N(0, 1)$.
- $B$ is fixed to 1: with $C$ learned, a learned $B$ would be redundant (only the product $C_n\bar B_n$ enters the kernel).

discretize by ZOH, build the kernel, convolve. that's the whole layer, and it's linear. a full model wraps it in a block, as for a transformer: normalize, apply the SSM, add a nonlinearity and a position-wise linear layer that mixes channels (S4D uses a GLU), add a residual. stacking blocks gives depth; the SSM gives each block a global receptive field at every layer, where a CNN would need depth to see far.

why does this learn long dependencies where an LSTM struggles? the gradient path. for an LSTM to use token 1 at step $L$, the gradient must pass back through $L - 1$ nonlinear recurrent steps, multiplying Jacobians; it shrinks geometrically unless the forget gates stay near 1. in an SSM layer, $\partial y_L/\partial u_1 = K_{L-1}$: one kernel entry, with a gradient that depends on how $\Delta$ and $\lambda$ shape the kernel, not on a product of $L$ factors. initializing $\Delta$ small on some channels guarantees that some kernels reach far from the start.

> **spec.** `S4DLayer(d_model, d_state=64, *, dt_min=1e-3, dt_max=1e-1, init="lin")` with exactly the parameters above (names, shapes, initializations). `discrete()` returns `(Lambda_bar, B_bar, C)` as complex `(H, N)` tensors (ZOH, $B = 1$). `kernel(L)` returns the `(H, L)` kernel. `forward(u)` maps real `(batch, L, H)` to `causal_conv(u, kernel(L)) + D * u`. `step(u_t, state=None)` advances the recurrence by one step: `u_t` is `(batch, H)`, `state` is complex `(batch, H, N)` (zeros if `None`); it returns `(y_t, new_state)` with `y_t` including the `D` skip, so that stepping through a sequence reproduces `forward`.

> **▶ build: stage 5.** implement `S4DLayer`.

## worked example

**discretize.** $\lambda = -1$, $B = 1$, $\Delta = \ln 2$: ZOH gives $\bar\lambda = e^{-\ln 2} = 0.5$ and $\bar B = (0.5 - 1)/(-1) = 0.5$. bilinear gives $\bar\lambda = (1 - 0.347)/(1 + 0.347) = 0.485$ and $\bar B = 0.693/1.347 = 0.515$.

**kernel and recurrence.** one channel, one mode, $\bar\lambda = 0.5$, $\bar B = 1$, $C = 1$: $K_j = 2\cdot0.5^j = (2, 1, 0.5, 0.25)$. input $u = (1, 0, 2)$: the state is $x = (1, 0.5, 0.25 + 2) = (1, 0.5, 2.25)$ and $y = 2x = (2, 1, 4.5)$. by convolution: $y_2 = K_0u_2 + K_1u_1 + K_2u_0 = 4 + 0 + 0.5 = 4.5$. ✓.

**HiPPO-LegS, $N = 3$.** $A = -\begin{pmatrix}1 & 0 & 0\\ \sqrt3 & 2 & 0\\ \sqrt5 & \sqrt{15} & 3\end{pmatrix}$, $B = (1, \sqrt3, \sqrt5)$.

your code must reproduce the kernel, the recurrence and the HiPPO matrix.

## on paper

1. **ZOH.** solve $x' = \lambda x + bu$ over one step $[0, \Delta]$ with $u$ held constant at $u_k$, and read off $\bar\lambda$ and $\bar b$.

<details><summary>solution</summary>

variation of constants: $x(\Delta) = e^{\lambda\Delta}x(0) + \int_0^\Delta e^{\lambda(\Delta - s)}b\,u_k\,ds = e^{\lambda\Delta}x(0) + \frac{e^{\lambda\Delta} - 1}{\lambda}b\,u_k$. so $\bar\lambda = e^{\lambda\Delta}$, $\bar b = \frac{e^{\lambda\Delta} - 1}{\lambda}b$. for small $\Delta$, $\bar b \approx \Delta b$ (Euler). the matrix version is the top-right block of $\exp\left(\begin{pmatrix}A & B\\0 & 0\end{pmatrix}\Delta\right)$, which is what the tests use.

</details>

2. **bilinear preserves stability.** show that $\operatorname{Re}\lambda < 0$ implies $\left|\frac{1 + \lambda\Delta/2}{1 - \lambda\Delta/2}\right| < 1$ for every $\Delta > 0$.

<details><summary>solution</summary>

write $z = \lambda\Delta/2 = a + ib$ with $a < 0$. $|1 + z|^2 = (1 + a)^2 + b^2$ and $|1 - z|^2 = (1 - a)^2 + b^2$; their difference is $4a < 0$. so $|1 + z| < |1 - z|$. the bilinear map sends the left half-plane onto the open unit disk, so no step size can make a stable mode unstable (forward Euler can: $|1 + \lambda\Delta| > 1$ for large $\Delta$).

</details>

3. **the normal part of HiPPO-LegS.** with $P_n = \sqrt{n + 1/2}$, show that $A + PP^\top = S - \frac12I$ with $S$ skew-symmetric. what does that imply for its eigenvalues?

<details><summary>solution</summary>

$(PP^\top)_{nk} = \sqrt{(n + \frac12)(k + \frac12)} = \frac12\sqrt{(2n+1)(2k+1)}$. below the diagonal, $A_{nk} + (PP^\top)_{nk} = -\sqrt{(2n+1)(2k+1)} + \frac12\sqrt{(2n+1)(2k+1)} = -\frac12\sqrt{(2n+1)(2k+1)}$; above, $0 + \frac12\sqrt{(2n+1)(2k+1)}$; on the diagonal, $-(n+1) + n + \frac12 = -\frac12$. the off-diagonal part is antisymmetric. a real skew-symmetric matrix has purely imaginary eigenvalues in conjugate pairs and is unitarily diagonalizable, so $A + PP^\top$ has eigenvalues $-\frac12 + i\omega$: every S4D-LegS mode decays at the same rate, and they differ only in frequency.

</details>

4. **conv equals recurrence.** prove $y_k = \sum_{j \le k}K_ju_{k-j}$ for the diagonal recurrence with $x_{-1} = 0$, and say what changes if $\bar\lambda$ depends on $k$.

<details><summary>solution</summary>

by induction, $x_k = \sum_{j=0}^k\bar\lambda^j\bar Bu_{k-j}$ (true for $k = 0$; $x_{k+1} = \bar\lambda x_k + \bar Bu_{k+1}$ shifts every power up by one and adds the $j = 0$ term). multiply by $C$, sum the modes, take $2\operatorname{Re}$ (linear), and the coefficient of $u_{k-j}$ is $K_j$. if $\bar\lambda_k$ varies, the coefficient of $u_{k-j}$ becomes $\prod_{i=k-j+1}^{k}\bar\lambda_i$, which depends on $k$ and not just on $j$: there's no single kernel, so no convolution. that's topic 08's problem.

</details>

## experiment

the lab runs your `hippo_legs` online on a 1,000-sample signal and reconstructs its history from $N$ numbers; plots the kernels of your layer at three step sizes and times your convolutional view against your recurrent view; and trains two of your S4D blocks against an LSTM with the same parameter count on a task where the first token must be remembered across up to 256 steps, with 3 seeds per setting.

## check your understanding

1. why initialize $\Delta$ log-uniformly over two orders of magnitude rather than at a single value?

<details><summary>answer</summary>

$\Delta$ sets each channel's time scale (a mode's envelope halves every $2\ln2/\Delta$ steps for $\operatorname{Re}\lambda = -1/2$). a range gives the layer channels for short-range patterns and channels that reach thousands of steps back, before any training; log-uniform spreads them evenly across scales. training then adjusts each one.

</details>

2. a model trained with ZOH at $\Delta$ on audio sampled at 16 kHz is run on 8 kHz audio. what would you change?

<details><summary>answer</summary>

double every $\Delta$: the continuous system is the same, sampled half as often. the discrete parameters $\bar\lambda = e^{\lambda\Delta}$ change accordingly, and the layer computes (approximately) the same continuous-time filter. a model with directly learned discrete parameters has no such knob.

</details>

3. why must the kernel be recomputed at every training step, but not at inference in recurrent mode?

<details><summary>answer</summary>

the kernel is a function of the parameters ($\Delta$, $\lambda$, $C$), which change at every optimizer step, so training rebuilds it (cheaply, $O(HNL)$). at inference the parameters are fixed; recurrent mode uses $\bar\lambda$, $\bar B$, $C$ directly and never needs $K$. (in convolutional inference you'd compute it once and cache it.)

</details>

4. the S4D layer is linear in $u$. where does the model's nonlinearity come from, and why isn't one layer enough?

<details><summary>answer</summary>

from the block around it: the GELU and GLU after the SSM, applied position-wise. one linear time-invariant layer can only filter its input; stacking SSM layers with nonlinearities between them gives a deep sequence model that can compute nonlinear functions of the history, with each layer's SSM providing global, long-range mixing along time and the position-wise layers mixing channels.

</details>

5. why does S4D store one eigenvalue per conjugate pair and output $2\operatorname{Re}(\cdot)$?

<details><summary>answer</summary>

the layer must map real inputs to real outputs. a real state-space system's complex eigenvalues come in conjugate pairs whose contributions to $y$ are complex conjugates of each other, so their sum is twice the real part of one. storing one per pair and doubling gives the same real system with half the complex state.

</details>

6. what does the convolutional view need that a Kalman filter with time-varying $F_t$ doesn't have?

<details><summary>answer</summary>

time invariance: the same $\bar A$, $\bar B$, $C$ at every step, so that the response to an input $j$ steps ago is the same kernel entry $K_j$ wherever you are in the sequence. a time-varying system has a different impulse response at every position, and the output is not a convolution (on paper 4).

</details>

## common bugs

- **circular convolution.** symptom: the first outputs depend on the last inputs, the direct-convolution check fails at the start of the sequence. fix: pad to `2 * L` before the FFT and keep the first `L` outputs.
- **Euler instead of ZOH for $\bar B$.** symptom: the matrix-exponential check fails on $\bar B$ only. fix: $\bar B = (\bar\lambda - 1)/\lambda\cdot B$.
- **forgetting the factor 2.** symptom: kernels and outputs half as large as expected. fix: $y = 2\operatorname{Re}(\cdot)$ in both views.
- **a learned real part without a constraint.** symptom: training diverges with NaN after a few hundred steps when some $\operatorname{Re}\lambda$ crosses zero. fix: $\operatorname{Re}\lambda = -\exp(\cdot)$.
- **$\Delta$ uniform instead of log-uniform.** symptom: almost every channel gets $\Delta$ near $\Delta_{\max}$, so few channels have long memory. fix: sample $\log\Delta$ uniformly.
- **`step` without the skip.** symptom: recurrent and convolutional outputs differ by exactly $Du$. fix: add `D * u_t` in `step` too.

## going deeper

- Gu, Goel & Ré (2022), "Efficiently modeling long sequences with structured state spaces" (S4), *ICLR*. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Gu, Gupta, Goel & Ré (2022), "On the parameterization and initialization of diagonal state space models" (S4D), *NeurIPS*. the diagonal layer you built, its initializations and its ablations. [arXiv:2206.11893](https://arxiv.org/abs/2206.11893)
- Gu, Dao, Ermon, Rudra & Ré (2020), "HiPPO: recurrent memory with optimal polynomial projections", *NeurIPS*. [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)
- Gu, Johnson, Timalsina, Rudra & Ré (2023), "How to train your HiPPO: state space models with generalized orthogonal basis projections", *ICLR*. explains S4D-Lin and S4D-LegS kernels as bases. [arXiv:2206.12037](https://arxiv.org/abs/2206.12037)
- Gupta, Gu & Berant (2022), "Diagonal state spaces are as effective as structured state spaces", *NeurIPS*. [arXiv:2203.14343](https://arxiv.org/abs/2203.14343)
- Smith, Warrington & Linderman (2023), "Simplified state space layers for sequence modeling" (S5), *ICLR*. one multi-input multi-output SSM, computed with a parallel scan instead of a convolution (topic 08's tool). [arXiv:2208.04933](https://arxiv.org/abs/2208.04933)
- Tay et al. (2021), "Long Range Arena: a benchmark for efficient transformers", *ICLR*. [arXiv:2011.04006](https://arxiv.org/abs/2011.04006)
- Voelker, Kajić & Eliasmith (2019), "Legendre memory units: continuous-time representation in recurrent neural networks", *NeurIPS*. a Legendre-based recurrent memory from the same idea, independently.
- Jozefowicz, Zaremba & Sutskever (2015), "An empirical exploration of recurrent network architectures", *ICML*. the forget-gate bias initialization used for the lab's LSTM. [proceedings.mlr.press/v37/jozefowicz15](https://proceedings.mlr.press/v37/jozefowicz15.html)

## connections

- ← builds on: linear recurrences and state-space models (topics 05–06), eigenvalues and stability (ch 02), the FFT and convolution (ch 01, ch 16), LSTMs and vanishing gradients (ch 17), residual blocks with normalization (ch 15, ch 18).
- → used in: topic 08 makes $\Delta$, $B$ and $C$ functions of the input (selection), loses the convolution, and replaces it with a parallel scan; the block's project compares that selective model with a transformer, and uses this topic's layer as the time-invariant baseline.
