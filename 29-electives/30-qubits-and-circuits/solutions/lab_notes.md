# qubits & circuits · lab notes

numbers below are from the reference solution with the lab's default seeds; your simulator should reproduce them exactly if it draws random numbers in the same order (one `multinomial` per `sample_counts`, one `random()` per `measure_qubit`), and within the quoted standard errors otherwise. the whole lab runs in about 2 seconds.

## experiment 1: measuring an entangled pair

**what people predict.** most people get the $Z$ row right (Bob agrees with Alice). many expect that when Alice measures in the $X$ basis, Bob's qubit "becomes" $|+\rangle$ or $|-\rangle$ and his $Z$ results change somehow, and some expect Bob could detect which basis Alice chose.

**what happens.** measuring both qubits 4,000 times gives only `00` (1,983) and `11` (2,017). with Alice in the $Z$ basis, P(Bob = 1 | Alice = 0) = 0.000 and P(Bob = 1 | Alice = 1) = 1.000: perfect correlation. with Alice in the $X$ basis, the conditionals are 0.503 ± 0.011 and 0.483 ± 0.011: no correlation with Bob's $Z$ result at all. and Bob's overall fraction of 1s is 0.498 ± 0.008 in the first row and 0.493 ± 0.008 in the second: indistinguishable from $\frac12$ both times.

**why.** after Alice measures $Z$, Bob holds $|0\rangle$ or $|1\rangle$ with probability $\frac12$ each: $\rho_B = \frac12|0\rangle\langle0| + \frac12|1\rangle\langle1| = I/2$. after she measures $X$, Bob holds $|+\rangle$ or $|-\rangle$ with probability $\frac12$ each: $\rho_B = \frac12|+\rangle\langle+| + \frac12|-\rangle\langle-| = I/2$ again. the two ensembles are different, but they have the same density matrix, and every statistic Bob can measure is a function of $\rho_B$. so Alice's choice is invisible to Bob until she tells him her outcomes over a classical channel, at which point he can sort his results and see the correlation. this is the no-signalling theorem: entanglement gives correlations, not communication.

## experiment 2: the CHSH game

**what people predict.** "exactly 2" or "a bit less than 2, because of randomness" is common: quantum mechanics is "just probabilities", so why would it beat the best probabilistic strategy? people who have heard of Bell's theorem predict "more than 2" but rarely know by how much or for how wide a range of angles.

**what happens.** the best of the 16 deterministic answer sheets gives $|S| = 2$ (randomized strategies are mixtures of these, so they can't do better). the Bell pair at $\delta = 0$ gives $S = 2.861 \pm 0.031$ from 2,000 shots per setting (exact: $2\sqrt2 = 2.828$, one standard error away). the measured curve follows $2\sqrt2\cos\delta$ and stays above 2 for $|\delta| < 45°$ (exactly: $\cos\delta > 1/\sqrt2$); the points at $\pm40°$ are still clearly outside the grey band, those at $\pm50°$ inside.

**why.** for $|\Phi^+\rangle$, measuring along angles $a$ and $b$ in the $x$–$z$ plane means measuring $A = \cos a\,Z + \sin a\,X$ and $B = \cos b\,Z + \sin b\,X$, and $E(a, b) = \cos a\cos b\langle ZZ\rangle + \sin a\sin b\langle XX\rangle = \cos(a - b)$ (the cross terms $\langle ZX\rangle$, $\langle XZ\rangle$ are 0). with the angles $0, \pi/2$ and $\pm\pi/4$ all three positive terms are $\cos(\pi/4)$ and the negative one is $\cos(3\pi/4)$, giving $4/\sqrt2 = 2\sqrt2$. a classical answer sheet assigns fixed $\pm1$ values $a_0, a_1, b_0, b_1$, and $S = a_0(b_0 + b_1) + a_1(b_0 - b_1)$: one bracket is 0 and the other $\pm2$, so $|S| \le 2$ for each sheet and for any mixture. the violation rules out *local hidden variables*: the idea that each qubit carries answers to every question independent of what's asked of the other. it does not rule out quantum mechanics being a perfectly good probabilistic theory, and it doesn't allow signalling (experiment 1).

the win probability of the CHSH *game* (win when $a\oplus b$ matches the AND of the question bits) is $\frac12 + S/8$: 75% classically, $\cos^2(\pi/8) \approx 85.4\%$ with the Bell pair.

## experiment 3: how entangled is a random state?

**what people predict.** usually "around 1 bit" (since a Bell pair has 1 bit and the circuit is made of CNOTs) or "it keeps growing to 4".

**what happens.** the entropy climbs in steps of about 0.6–0.7 bits every other layer, and levels off after about 20 layers at **3.27 ± 0.05 bits** (mean ± s.d. over 20 circuits, 30 layers), right on Page's value $4 - 1/(2\ln2) = 3.28$ and well short of the maximum 4.

**why.** only layers whose CNOTs pair qubits $(3, 4)$ cross the cut between the halves, and those are every other layer. a single CNOT across a cut can raise the Schmidt rank by at most a factor of 2, so the entropy by at most 1 bit (in practice less, because the random rotations don't line up perfectly). once the circuit is deep enough, the state is effectively a random unit vector in $\mathbb C^{256}$, and Page showed that the average entanglement entropy of a random pure state of a $d_A\times d_B$ system ($d_A \le d_B$) is about $\log_2d_A - d_A/(2d_B\ln2)$: random states are almost, but not quite, maximally entangled. a consequence you'll meet in topic 31: in a nearly maximally entangled random state, every small subsystem is nearly maximally mixed, so $\langle Z_0\rangle \approx 0$ for almost every random circuit, and its gradient with respect to any one parameter is tiny too.

## experiment 4: the price of a qubit

**what people predict.** people know "exponential" but rarely put numbers on it. typical guesses for "largest state in 16 GB" range from 20 to 1,000 qubits.

**what happens.** one Hadamard takes about 0.01 ms up to $n \approx 10$ (Python and NumPy call overhead dominates), then doubles with every qubit, reaching about 6 ms at $n = 20$ (16 MB state; the exact times depend on your machine, the doubling doesn't). a complex128 state of 30 qubits needs exactly 16 GiB; 40 qubits, 16 TiB; 50 qubits, 16 PiB.

**why.** below ~1,000 amplitudes the fixed cost of a few NumPy calls dominates; above it, the work is proportional to $2^n$. full state-vector simulation tops out around 45–50 qubits even on the largest supercomputers. but circuits with limited entanglement (experiment 3: shallow circuits, or circuits whose CNOTs rarely cross a given cut) can be simulated far beyond that with tensor-network methods, whose cost grows with the entanglement rather than with $2^n$. "quantum advantage" claims, including the quantum-kernel claims of topic 32, live or die on whether the circuit is in the hard regime.
