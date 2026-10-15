# variational circuits

## why this matters

today's quantum computers have tens to a few hundred noisy qubits and can run circuits only a few dozen gates deep before errors swamp the signal. that rules out the famous algorithms (Shor's factoring needs millions of error-corrected operations) and leaves one idea that fits the hardware: treat a short parameterized circuit as a **model**, and train its parameters with a classical optimizer. the circuit prepares a state that depends on the data and on the parameters, you measure an expectation value, and that number is your prediction. this is the **variational** recipe behind quantum classifiers, quantum neural networks, and the variational eigensolvers of quantum chemistry.

it looks like deep learning: angles are the parameters, running the circuit is the forward pass, and you need gradients. but you can't backpropagate through a quantum computer: intermediate states can't be read out without destroying them. instead there's the **parameter-shift rule**: the exact derivative of an expectation value with respect to a rotation angle is the difference of the same circuit run at two shifted angles. you'll derive it and implement it.

then the bad news, which is the most important result in the field. for random circuits of even modest size, gradients **vanish exponentially in the number of qubits** (McClean et al., 2018), not because of saturating activations as in old neural networks, but because random high-dimensional states all look the same to a measurement. a variance of $2^{-50}$ means no number of shots on any hardware will tell you which way is downhill. you'll measure this "barren plateau" on your own simulator and find the one setting in the lab where it disappears.

## warm-up: predict first

1. a gate $R_Y(\theta)$ appears once in a circuit. how does the expectation value $\langle Z\rangle$ at the output depend on $\theta$: polynomially, periodically, exponentially?
2. you estimate $\langle Z\rangle$ from 1,000 shots. roughly how large is the error?
3. to get the gradient of a circuit with 100 parameters, how many circuit runs do you need with finite differences, and with backpropagation on a classical network of the same size?
4. you draw the parameters of a 10-qubit circuit at random and compute $\partial\langle Z_0Z_1\rangle/\partial\theta_1$. what does its distribution over random draws look like?
5. a classifier encodes each feature once as a rotation angle. can it represent any function of the features, given enough layers?

## prerequisites check

- [ ] this notebook runs on your topic 30 simulator (→ topic 30). if a stage it needs is unfinished, it uses the reference simulator instead and says so in a box at the top; finishing topic 30 switches it back to yours (`MLFS_REFERENCE=qsim` forces the reference)
- [ ] you can derive a gradient with the chain rule and check it with finite differences (→ ch 02 · topic 07, ch 03)
- [ ] you know Adam and its bias correction (→ ch 06 · topic 04)
- [ ] you know the variance of a sample mean and of a binomial proportion (→ ch 05)

## core ideas

### idea 1: circuits as models

a **variational quantum model** has three parts:

1. a **data encoding** $S(x)$: gates whose angles are the features. we use **angle encoding**, $R_Y(x_q)$ on qubit $q$, so an $n$-feature input needs $n$ qubits.
2. a **trainable circuit** $W(\theta)$, the **ansatz**: rotations with free angles, interleaved with entangling gates.
3. an **observable** $O$, whose expectation is the output: $f(\theta, x) = \langle0|S(x)^\dagger W(\theta)^\dagger\,O\,W(\theta)S(x)|0\rangle$.

you'll build the **hardware-efficient** ansatz (named for being easy to run on devices, not for any theoretical virtue): each layer applies $R_Y$ then $R_Z$ to every qubit, then a ladder of CNOTs between neighbours. its parameters form an array $\theta$ of shape $(L, n, 2)$: layer, qubit, and which of the two rotations.

what functions can such a model represent? write each encoding rotation as $R_Y(x) = \cos\frac x2I - i\sin\frac x2Y$. the state after encoding one qubit has amplitudes in $\cos\frac x2$ and $\sin\frac x2$, and the expectation value is quadratic in them, so it's a combination of $1, \cos x, \sin x$ (using $\cos^2\frac x2 = \frac{1 + \cos x}2$ and so on). with every feature encoded once, $f$ is a **trigonometric polynomial of degree 1 in each feature**: a sum of products $r_a(x_1)r_b(x_2)\cdots$ with each $r \in \{1, \cos, \sin\}$, whatever the ansatz (Schuld, Sweke & Meyer, 2021). more layers change the coefficients, never the frequencies. to get higher frequencies you **re-upload** the data: repeat the encoding before every layer, so each feature passes through $L$ rotations and $f$ becomes a trig polynomial of degree $L$ in it (Pérez-Salinas et al., 2020). that's the quantum analogue of depth adding expressivity.

> **spec.** `ansatz_ops(theta)`: `theta` of shape $(L, n, 2)$ (else `ValueError`); returns the list of `(U, targets)` ops for your simulator: for each layer $l$, for each qubit $q = 0, \dots, n-1$, $R_Y(\theta_{l,q,0})$ then $R_Z(\theta_{l,q,1})$ on $q$; then CNOT$(q, q+1)$ for $q = 0, \dots, n-2$. `encoding_ops(x)`: $R_Y(x_q)$ on qubit $q$ for each feature. `model_ops(theta, x, *, reupload=False)`: the encoding followed by the whole ansatz; with `reupload=True`, the encoding before *every* layer (encoding, layer 0, encoding, layer 1, …). `len(x)` must equal $n$ (else `ValueError`).

> **▶ build: stage 1.** implement `ansatz_ops`, `encoding_ops` and `model_ops`.

### idea 2: outputs are expectation values, estimated from shots

the model's output is $f(\theta, x) = \langle\psi(\theta, x)|O|\psi(\theta, x)\rangle$, with $O$ a Pauli string (by default $Z_0$, i.e. `'ZI…I'`). on your simulator you can compute it exactly with `qsim.expval`. on hardware you can't: each shot measures the qubits and returns one bit string, and a Pauli string's eigenvalue on that outcome is the product of $\pm1$ over the qubits it touches (so $Z_0Z_1$ is $+1$ for `00` and `11`, $-1$ for `01` and `10`). to measure $X$ or $Y$ on a qubit, rotate that qubit into the $Z$ basis first: $H$ maps the $X$ eigenstates to $|0\rangle, |1\rangle$, and $HS^\dagger$ does the same for $Y$.

the estimate from $N$ shots is an average of $N$ independent $\pm1$ values with mean $\langle P\rangle$, so its variance is

$$\operatorname{Var}[\hat f] = \frac{1 - \langle P\rangle^2}{N}.$$

1,000 shots give an error of about $0.03$ (warm-up 2). halving it costs 4× the shots. every number a quantum model produces carries this noise, and anything you build on top, gradients included, inherits it.

> **spec.** `expectation(theta, x, observable=None, *, reupload=False)`: $f(\theta, x)$ computed exactly on your simulator, as a Python `float`; the observable is a Pauli string of length $n$, default `'Z' + 'I' * (n - 1)`. `estimate_pauli(state, pauli, shots, rng=None)`: rotate each qubit whose character is `'X'` (by $H$) or `'Y'` (by $HS^\dagger$, i.e. $S^\dagger$ first) into the $Z$ basis, draw `shots` outcomes with **one** call to `qsim.sample_counts`, and return the average of the eigenvalues $\prod_{q\in\text{support}}(-1)^{b_q}$. a string of the wrong length or with other characters raises `ValueError`.

> **▶ build: stage 2.** implement `expectation` and `estimate_pauli`.

### idea 3: the parameter-shift rule

how does $f$ depend on one angle $\theta$, with everything else fixed? the gate is $R_P(\theta) = \cos\frac\theta2I - i\sin\frac\theta2P$, and $f$ is quadratic in the gate's entries (the gate and its adjoint each appear once). products of $\cos\frac\theta2$ and $\sin\frac\theta2$ are combinations of $1, \cos\theta, \sin\theta$, so

$$f(\theta) = A + B\cos\theta + C\sin\theta$$

for constants $A, B, C$ that depend on everything else (warm-up 1). a sinusoid of known frequency is pinned down by a few samples, and its derivative is too:

$$f'(\theta) = -B\sin\theta + C\cos\theta = \frac{f(\theta + \frac\pi2) - f(\theta - \frac\pi2)}2.$$

(check: $\cos(\theta \pm \frac\pi2) = \mp\sin\theta$ and $\sin(\theta \pm \frac\pi2) = \pm\cos\theta$.) this is the **parameter-shift rule** (Mitarai et al., 2018; Schuld et al., 2019). it looks like a finite difference with step $\frac\pi2$, but it is **exact**: there's no truncation error, however large the shift. that matters because on hardware each $f$ is a shot estimate. a finite difference $(\hat f(\theta + h) - \hat f(\theta - h))/2h$ has noise of order $\frac{1}{h\sqrt N}$, which explodes as $h \to 0$, and truncation error of order $h^2$ if $h$ is large; the shift rule has noise of order $\frac1{\sqrt N}$ and no bias at all. you'll measure this in lab experiment 1.

the rule holds for any gate $e^{-i\theta G/2}$ whose generator $G$ has eigenvalues $\pm1$, which covers every rotation here. a parameter that appears in several gates (as with re-uploading, where $x$ appears $L$ times; or a shared weight) needs the product rule: shift each occurrence separately and add.

the cost: two circuits **per parameter**, $2P$ for a full gradient, each with its own shots, where backpropagation gets all $P$ derivatives for about one extra forward pass (warm-up 3).

> **spec.** `parameter_shift(f, theta)`: `f` maps an array shaped like `theta` to a float; returns the array of $\big(f(\theta + \frac\pi2e_i) - f(\theta - \frac\pi2e_i)\big)/2$ for every entry $i$, shaped like `theta`, with exactly $2\cdot$`theta.size` calls to `f` and `theta` unmodified.

> **🔮 predict:** with 1,000 shots per circuit, which finite-difference step is most accurate: $h = 1$, $0.1$, $0.01$ or $0.001$?

> **▶ build: stage 3.** implement `parameter_shift`.

### idea 4: training a classifier

for binary labels $y_i \in \{-1, +1\}$ the output $f(\theta, x) \in [-1, 1]$ is a natural score: predict $\operatorname{sign}f$, and train on the mean squared error

$$\mathcal L(\theta) = \frac1N\sum_i\big(f(\theta, x_i) - y_i\big)^2, \qquad \nabla_\theta\mathcal L = \frac1N\sum_i2\big(f(\theta, x_i) - y_i\big)\nabla_\theta f(\theta, x_i).$$

the chain rule with parameter-shift gradients. a neat implementation detail: since the shift rule is linear in $f$, you can apply it once to the scalar function $\theta \mapsto \sum_ir_i\,f(\theta, x_i)$ with the residual weights $r_i = 2(f_i - y_i)/N$ held fixed, instead of computing $N$ separate gradients. the circuit count is the same ($2P$ per sample), but the code is one line.

the optimizer is Adam (ch 06): angles live on a circle, gradients have wildly different scales across parameters, and Adam's per-parameter normalization copes well. a useful fact for testing: Adam's first bias-corrected step is $\hat m/\sqrt{\hat v} = g/|g|$, so every parameter moves by exactly the learning rate in the downhill direction (on paper 3).

initialize the angles uniformly in $[0, 2\pi)$, the standard choice, and the one that idea 5 shows to be dangerous at scale (smarter initializations, e.g. Grant et al., 2019, start near the identity).

> **spec.** `train_classifier(X, y, *, layers=2, steps=50, lr=0.1, reupload=False, rng=None)`: `X` $(N, n)$, `y` in $\{-1, +1\}$ (else `ValueError`). draw $\theta$ = `as_rng(rng).uniform(0, 2π, size=(layers, n, 2))` (the first and only draw), then take `steps` Adam steps ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, bias-corrected) on the MSE with parameter-shift gradients and the default observable $Z_0$. returns `(theta, losses)` where `losses` has `steps + 1` entries: the loss before each step, and the loss at the returned $\theta$. `classify(theta, X, *, reupload=False)`: labels $+1$ where $f \ge 0$, else $-1$.

> **🔮 predict:** two concentric circles, 40 training points. will a 2-qubit, 3-layer circuit classify them as well as an RBF SVM?

> **▶ build: stage 4.** implement `train_classifier` and `classify`.

### idea 5: barren plateaus

pick the parameters at random and look at one gradient component, $\partial_k f = \partial f/\partial\theta_k$. by symmetry its mean over random $\theta$ is zero. the question is its **variance**, because that's how far from zero a typical gradient is, and so how many shots you need to see it.

McClean et al. (2018) showed that if the random circuit is expressive enough to look like a random unitary (technically, if the parts before and after $\theta_k$ form **unitary 2-designs**, distributions that match the Haar measure in their first two moments), then

$$\operatorname{Var}_\theta[\partial_kf] \in O(2^{-n}).$$

the intuition is topic 30's lab experiment 3: a random state in $2^n$ dimensions is nearly maximally entangled, so every few-qubit observable has expectation close to its average over *all* states, with fluctuations shrinking like $2^{-n/2}$. the landscape is a plateau, flat to exponential precision almost everywhere, and a gradient estimated from shots is pure noise unless the shot count grows like $2^n$. this kills training at scale regardless of the optimizer: unlike a vanishing gradient in a deep network, you can't fix it by rescaling, since you'd rescale the shot noise too.

what drives it? two things, as Cerezo et al. (2021) made precise:

- **global observables.** if $O$ acts on all qubits (like $Z^{\otimes n}$, a "global cost" that compares the whole state to a target), gradients vanish exponentially even for shallow circuits.
- **depth.** with a **local** observable (like $Z_0Z_1$) and a circuit of depth $O(\log n)$, the variance decays at most polynomially: only gates inside the observable's backward **light cone** affect it, and in a shallow circuit that cone is small. as the depth grows the cone covers everything, the circuit approaches a 2-design, and the plateau returns.

the catch: a shallow circuit measured locally is close to something a classical computer can simulate (light cones are small), which undercuts the reason to use a quantum computer.

> **spec.** `gradient_samples(n, layers, n_samples, *, observable, rng=None)`: for each of `n_samples` samples, in order, draw $\theta$ = `gen.uniform(0, 2π, size=(layers, n, 2))` from `gen = as_rng(rng)` and compute $\partial\langle O\rangle/\partial\theta_{0,0,0}$ with the parameter-shift rule, for the model with no data (`x = zeros(n)`, so the encoding is the identity) and `observable` a Pauli string. returns an array $(n\_samples,)$.

> **▶ build: stage 5.** implement `gradient_samples`.

## worked example

**one qubit, exactly.** $n = 1$, one layer, $\theta = [[[a, b]]]$, feature $x$. the circuit is $R_Y(x)$, $R_Y(a)$, $R_Z(b)$, and $R_Y(x)R_Y(a) = R_Y(a + x)$, so the Bloch vector after the $R_Y$'s is $(\sin(a+x), 0, \cos(a+x))$ and $R_Z(b)$ rotates it about $z$:

$$\langle Z\rangle = \cos(a + x), \quad \langle X\rangle = \sin(a + x)\cos b, \quad \langle Y\rangle = \sin(a + x)\sin b.$$

**the shift rule on it.** for $f = \langle Z\rangle$ and the parameter $a$: $\frac12[\cos(a + x + \frac\pi2) - \cos(a + x - \frac\pi2)] = \frac12[-\sin(a + x) - \sin(a + x)] = -\sin(a + x)$, the exact derivative. with $a = 0.9$, $b = 0.4$, $x = 0.3$ and $f = \langle X\rangle$, the gradient with respect to $(a, b)$ is $(\cos1.2\cos0.4, -\sin1.2\sin0.4) = (0.3338, -0.3629)$.

**a classifier with $\theta = 0$.** $f(x) = \cos x$, so the inputs $0, \pi, 2, 1$ get labels $+1, -1, -1, +1$.

**Adam's first step.** whatever the gradient $g$ (where it isn't 0), $m_1 = 0.1g$, $v_1 = 0.001g^2$, the bias-corrected ratio is $g/|g|$, and each parameter moves by exactly $-\text{lr}\cdot\operatorname{sign}(g)$.

**one-qubit gradient variance.** with no data, $\langle Z\rangle = \cos\theta_{0,0,0}$, so each sample of `gradient_samples(1, 1, …, observable='Z')` is $-\sin\theta_{0,0,0}$ with $\theta_{0,0,0}\sim U(0, 2\pi)$: mean 0, variance $\mathbb E[\sin^2] = \frac12$.

your code must reproduce all of these.

## on paper

1. **the shift rule.** show that if $f(\theta) = A + B\cos\theta + C\sin\theta$, then for any shift $s$ with $\sin s \ne 0$, $f'(\theta) = \frac{f(\theta + s) - f(\theta - s)}{2\sin s}$. which $s$ minimizes the variance of the estimate when each $f$ has independent noise of variance $\sigma^2$?

<details><summary>solution</summary>

$f(\theta + s) - f(\theta - s) = B[\cos(\theta + s) - \cos(\theta - s)] + C[\sin(\theta + s) - \sin(\theta - s)] = -2B\sin\theta\sin s + 2C\cos\theta\sin s = 2\sin s\,f'(\theta)$. the estimate's variance is $2\sigma^2/(4\sin^2s) = \sigma^2/(2\sin^2s)$, smallest at $s = \pm\frac\pi2$: the standard rule is also the least noisy member of the family. as $s \to 0$ you recover the finite difference, with variance $\sigma^2/(2s^2)$.

</details>

2. **finite differences with shots.** with $N$ shots per circuit, so that each $\hat f$ has variance $\approx(1 - f^2)/N$, find the standard deviation of the central difference $(\hat f(\theta + h) - \hat f(\theta - h))/2h$ and its bias for a sinusoidal $f$. at $N = 1000$, which $h$ balances the two?

<details><summary>solution</summary>

the standard deviation is $\sqrt{2(1 - f^2)/N}/(2h) \approx \frac1{h\sqrt{2N}}$. for $f = A + B\cos\theta + C\sin\theta$ the central difference equals $f'(\theta)\frac{\sin h}h$ (from problem 1 with $s = h$), a bias of $f'(\theta)(\frac{\sin h}h - 1) \approx -f'h^2/6$ (about $0.16|f'|$ at $h = 1$). minimizing noise$^2$ + bias$^2$, $\frac{1}{2Nh^2} + \frac{f'^2h^4}{36}$, gives $h^* = (9/(Nf'^2))^{1/6}$, about $0.46\,|f'|^{-1/3}$ at $N = 1000$: $h^* \approx 1.1$ for the lab's $|f'| = 0.08$ (which is why $h = 1$ wins there), but $h^* \approx 0.46$ for $|f'| = 1$. the best $h$ depends on the unknown $f'$; the shift rule needs no tuning.

</details>

3. **Adam's first step.** with $m_0 = v_0 = 0$, show that the first bias-corrected Adam update is $-\text{lr}\cdot g/(|g| + \epsilon')$ elementwise, where $\epsilon'$ is tiny. what does it do to a parameter whose gradient is exactly zero?

<details><summary>solution</summary>

$m_1 = (1 - \beta_1)g$, $v_1 = (1 - \beta_2)g^2$, $\hat m_1 = m_1/(1 - \beta_1) = g$, $\hat v_1 = v_1/(1 - \beta_2) = g^2$, so the step is $-\text{lr}\,g/(\sqrt{g^2} + \epsilon) = -\text{lr}\,g/(|g| + \epsilon)$: $\pm\text{lr}$ for any $|g| \gg \epsilon$, and 0 for $g = 0$. in the ansatz, the last $R_Z$ on a qubit measured in $Z$ commutes with everything after it, so its gradient is exactly 0 and it never moves.

</details>

4. **degree-1 functions.** for one qubit with $R_Y(x)$ encoding followed by any unitary $W$ and observable $Z$, show that $f(x) = \alpha + \beta\cos x + \gamma\sin x$, and that $\alpha = 0$. (so a one-qubit model encoded once can't even shift its output: the bias must come from elsewhere.)

<details><summary>solution</summary>

the encoded state is $\rho(x) = \frac12(I + \sin x\,X + \cos x\,Z)$ (the Bloch vector of $R_Y(x)|0\rangle$). then $f = \operatorname{Tr}(W^\dagger ZW\rho(x)) = \frac12\operatorname{Tr}(M) + \frac12\sin x\operatorname{Tr}(MX) + \frac12\cos x\operatorname{Tr}(MZ)$ with $M = W^\dagger ZW$. $\operatorname{Tr}M = \operatorname{Tr}Z = 0$, so $\alpha = 0$. also $\beta^2 + \gamma^2 \le 1$, since $M$ is a unit-norm combination of Paulis. with $n$ qubits the same argument gives $f = \sum_{a,b,\dots}c_{ab\dots}r_a(x_1)r_b(x_2)\cdots$ with $c_{00\dots} = 0$ and $M = W^\dagger Z_0W$ squaring to $I$.

</details>

## experiment

`lab.py` investigates, with your code:

1. gradients from shots: finite differences at several step sizes against the parameter-shift rule.
2. a variational classifier on two concentric circles, with and without re-uploading, against your ch 09 SVM.
3. barren plateaus: the variance of a gradient component for 2 to 10 qubits, with global and local observables, shallow and deep circuits.

## check your understanding

1. why can't you backpropagate through a quantum circuit the way autograd does through a network?

<details><summary>answer</summary>

backprop stores the intermediate activations of the forward pass and reuses them. a quantum computer's intermediate states can't be copied (no-cloning) or read out without collapsing them, and a measurement gives one random bit string, not the state. so every derivative needs fresh circuit runs. (on a *simulator* you can backpropagate through the state vector, adjoint-style, at the cost of the exponential memory you'd have used anyway.)

</details>

2. (warm-up 3.) how many circuit runs does a parameter-shift gradient of a 100-parameter circuit need, with 1,000 shots each?

<details><summary>answer</summary>

200 circuits, so 200,000 shots, per data point per step. a training set of 100 points makes it 20 million shots per step. at a typical rate of thousands of shots per second on hardware, that's hours per step, which is why real-hardware experiments use few parameters, few points, or stochastic shortcuts.

</details>

3. the shift rule fails for a gate $e^{-i\theta G}$ where $G$ has eigenvalues $\{0, 1, 2\}$. why, and what would fix it?

<details><summary>answer</summary>

$f$ is then a trig polynomial with frequencies $0, 1, 2$ (the eigenvalue differences), so a two-point rule can't recover its derivative. fixes: decompose the gate into gates with $\pm1$ generators, or use a generalized shift rule with more evaluation points (one pair per distinct frequency), as in Wierichs et al. (2022).

</details>

4. (warm-up 5.) can a model that encodes each feature once represent $\cos(2x_1)$?

<details><summary>answer</summary>

no: with one encoding per feature, $f$ is degree 1 in each feature's $\cos, \sin$, whatever the ansatz and its depth. $\cos 2x_1$ needs frequency 2, which requires $x_1$ to be encoded at least twice (re-uploading), or encoded with a rescaled angle $2x_1$ (which changes the frequency instead of adding one).

</details>

5. in lab experiment 3, the local observable on a shallow circuit has a gradient variance that doesn't shrink with $n$. why is that not good news for quantum advantage?

<details><summary>answer</summary>

a local observable after a shallow circuit depends only on the gates in its backward light cone, a few qubits wide. you can simulate that small patch classically, whatever $n$ is, so the model is efficiently classically computable. trainability and classical hardness pull in opposite directions; finding models that have both is an open problem.

</details>

6. a colleague fixes a barren plateau by multiplying the loss by $2^n$. what happens?

<details><summary>answer</summary>

the true gradients get bigger, and so does the shot noise on the estimate of the loss, by the same factor. the signal-to-noise ratio, which is what determines how many shots you need, doesn't change. the plateau is a statement about how many measurements it takes to resolve the landscape, not about the scale of the numbers.

</details>

## common bugs

- **wrong ansatz order.** symptom: unitary tests fail by a lot. per layer it's $R_Y$ then $R_Z$ on each qubit (qubit by qubit), then the CNOT ladder; `theta[l, q, 0]` is the $R_Y$ angle.
- **shifting by $\pi$ instead of $\frac\pi2$, or forgetting the $\frac12$.** symptom: gradients are 0, or twice too big. the rule is $(f(\theta + \frac\pi2) - f(\theta - \frac\pi2))/2$.
- **mutating `theta` inside `parameter_shift`.** symptom: later entries of the gradient are evaluated at the wrong point. shift a copy, or undo the shift.
- **measuring $Y$ with $H$ alone.** symptom: biased $\langle Y\rangle$ estimates (you're measuring $X$ or $-X$). apply $S^\dagger$ before $H$.
- **adding the gradient instead of subtracting.** symptom: the loss climbs to 4 (every output has the wrong sign).
- **drawing $\theta$ in a different order from the spec.** symptom: `gradient_samples` gives a correct-looking distribution but fails the exact-reproduction check. one `uniform(…, size=(layers, n, 2))` call per sample.

## going deeper

- McClean, Boixo, Smelyanskiy, Babbush & Neven (2018), "Barren plateaus in quantum neural network training landscapes", *Nature Communications* 9, 4812. the exponentially vanishing gradients of lab experiment 3. [doi:10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
- Cerezo, Sone, Volkoff, Cincio & Coles (2021), "Cost function dependent barren plateaus in shallow parametrized quantum circuits", *Nature Communications* 12, 1791. global vs local costs. [doi:10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)
- Mitarai, Negoro, Kitagawa & Fujii (2018), "Quantum circuit learning", *Physical Review A* 98, 032309. variational circuits as models, and the shift rule. [arXiv:1803.00745](https://arxiv.org/abs/1803.00745)
- Schuld, Bergholm, Gogolin, Izaac & Killoran (2019), "Evaluating analytic gradients on quantum hardware", *Physical Review A* 99, 032331. the parameter-shift rule in general form. [arXiv:1811.11184](https://arxiv.org/abs/1811.11184)
- Schuld, Sweke & Meyer (2021), "Effect of data encoding on the expressive power of variational quantum machine learning models", *Physical Review A* 103, 032430. models as Fourier series. [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)
- Pérez-Salinas, Cervera-Lierta, Gil-Fuster & Latorre (2020), "Data re-uploading for a universal quantum classifier", *Quantum* 4, 226. [doi:10.22331/q-2020-02-06-226](https://doi.org/10.22331/q-2020-02-06-226)
- Grant, Wossnig, Ostaszewski & Benedetti (2019), "An initialization strategy for addressing barren plateaus in parametrized quantum circuits", *Quantum* 3, 214. [doi:10.22331/q-2019-12-09-214](https://doi.org/10.22331/q-2019-12-09-214)
- Wierichs, Izaac, Wang & Lin (2022), "General parameter-shift rules for quantum gradients", *Quantum* 6, 677. [doi:10.22331/q-2022-03-30-677](https://doi.org/10.22331/q-2022-03-30-677)
- Schuld & Petruccione (2021), *Machine Learning with Quantum Computers*, 2nd ed., Springer, chapters 4–5 (encoding and variational models). [doi:10.1007/978-3-030-83098-4](https://doi.org/10.1007/978-3-030-83098-4)

## connections

- ← builds on: your state-vector simulator (topic 30), gradient checking with finite differences (ch 02 · topic 07), Adam (ch 06 · topic 04), the variance of a mean (ch 05), vanishing gradients in deep networks (ch 15).
- → used in: topic 32 drops the training loop altogether: the encoding circuit becomes a feature map, and the overlaps of encoded states become a kernel for your SVM. the concentration that causes barren plateaus reappears there as **kernel concentration**.
