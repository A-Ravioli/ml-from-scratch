# privacy & secure aggregation

## why this matters

federated learning keeps raw data on the device, and it's tempting to stop there and call it private. it isn't. a model update is a function of the data that produced it, and functions can be inverted: from a single client's gradient on a small batch, an attacker who knows the model can reconstruct recognizable training images (Zhu et al., 2019; Geiping et al., 2020), and from the final model alone, membership-inference attacks can tell whether a given record was in the training set (Shokri et al., 2017). so there are two separate leaks, and they need two separate tools.

**secure aggregation** (Bonawitz et al., 2017) stops the server from seeing any individual update: clients add random masks that cancel exactly when the server sums them, so the server learns the sum and nothing else, even if some clients drop out mid-round. **differential privacy** (Dwork et al., 2006) limits what the sum, and hence the model, reveals about any one user: clip each user's contribution, add calibrated Gaussian noise, and account for the privacy cost across rounds. McMahan et al. (2018) combined clipping, noise and user sampling into **DP-FedAvg**, and the combination of both tools is how production systems train keyboard models today (Xu et al., 2023).

neither is free: secure aggregation costs communication, DP costs accuracy, and buying accuracy back means averaging over more clients per round, which costs communication again. you'll build a secure-aggregation simulator with dropout recovery, DP-FedAvg and a privacy accountant, and the lab ends with the block's project: measuring that **privacy–accuracy–communication triangle**.

## warm-up: predict first

1. a client adds a uniformly random 32-bit integer $m$ to its encoded update $z$ (mod $2^{32}$) and sends $z + m$. what can the server learn about $z$ from the message alone?
2. why must the encoding be integers mod $2^{32}$ rather than floats with a large random number added?
3. clients 1 and 2 agree on a mask $m$; client 1 adds it, client 2 subtracts it. client 2 drops out before sending. what is left in the server's sum, and how could the server remove it without learning client 1's update?
4. DP-FedAvg clips each user's update to norm $S$ and adds Gaussian noise with standard deviation $\sigma S$ to the sum. 400 users per round instead of 25: for the same $\sigma$, how does the noise in the *average* change?
5. each round touches a random 1% of users. after 1,000 rounds, is the privacy cost closer to that of 10 rounds with everyone, or of 1,000 rounds with everyone?

## prerequisites check

- [ ] you've built `fedavg`, `client_update` and `partition_iid` (→ topic 27)
- [ ] you can work with modular arithmetic: $(a + b) \bmod M$, and modular inverses via `pow(a, -1, p)` (→ ch 00, ch 01)
- [ ] you know polynomial interpolation: $t$ points determine a unique polynomial of degree $t - 1$ (→ ch 01)
- [ ] you can compute $\mathbb E[e^{cZ}]$ for a Gaussian $Z$ (its moment-generating function) (→ ch 04)
- [ ] you've seen the definition of differential privacy, or at least KL and Rényi divergences (→ ch 05)

## core ideas

### idea 1: updates as integers mod $M$

masking only hides something if the mask is uniformly random over everything the message could be. over the real numbers there's no uniform distribution, and adding a large random float to a small one loses the small one to rounding. so secure aggregation works in a finite group: the integers mod $M$ (here $M = 2^{32}$), where adding an independent uniform mask makes the result exactly uniform (on paper 1) and the masks can cancel exactly.

so **fixed-point encode** each entry: multiply by a scale $s$ (here $2^{16}$), round, reduce mod $M$. negative numbers wrap around: $-0.25 \mapsto 2^{32} - 16{,}384$. sums of encodings decode correctly as long as the true sum stays in $(-M/(2s), M/(2s))$, here $\pm 32{,}768$: interpret residues $\ge M/2$ as negative, then divide by $s$. each encoding rounds by at most $0.5/s$, so the decoded sum of $n$ updates is off by at most $n/(2s)$. in DP-FedAvg the updates are clipped, so the overflow bound comes for free.

> **spec.** `encode(x, *, scale=2**16, modulus=2**32)`: `x` a NumPy array or torch tensor; returns `np.mod(np.round(x * scale), modulus)` as an `int64` array. `decode(z, *, scale=2**16, modulus=2**32)`: reduces mod `modulus`, maps entries `>= modulus // 2` to `entry - modulus`, divides by `scale`; returns `float64`.

> **▶ build: stage 1.** implement `encode` and `decode`.

### idea 2: pairwise masks that cancel

each pair of clients $i < j$ agrees on a shared random mask $m_{ij} \in \mathbb Z_M^d$. client $i$ adds it, client $j$ subtracts it:

$$y_i = z_i + \sum_{j > i}m_{ij} - \sum_{j < i}m_{ji} \pmod M.$$

every mask appears once with each sign, so $\sum_iy_i = \sum_iz_i \pmod M$ exactly, while each $y_i$ on its own is uniformly random (warm-up 1).

the pair agrees on $m_{ij}$ without ever sending it, using **Diffie–Hellman key agreement**: each client picks a secret $a_i$ and publishes $g^{a_i} \bmod p$; client $i$ computes $(g^{a_j})^{a_i}$ and client $j$ computes $(g^{a_i})^{a_j}$, the same number $g^{a_ia_j}$, which an eavesdropper who sees only $g^{a_i}$ and $g^{a_j}$ can't compute (that's the Diffie–Hellman assumption). the shared number seeds a pseudorandom generator that expands it into a $d$-dimensional mask, so a client sends one short public key instead of $d$ numbers per peer. here $p = 2^{61} - 1$ and $g = 3$, **a toy**: a 61-bit group is breakable in seconds. real systems use elliptic-curve Diffie–Hellman and a cryptographic PRG (AES in counter mode); NumPy's generator is fine for checking the arithmetic and nothing else.

> **spec.** `public_key(secret, *, prime=2**61 - 1, generator=3)` returns `pow(generator, secret, prime)`. `shared_secret(my_secret, their_public, *, prime=2**61 - 1)` returns `pow(their_public, my_secret, prime)`. `prg(seed, dim, *, modulus=2**32)` returns `np.random.default_rng(seed).integers(0, modulus, size=dim, dtype=np.int64)`. `mask(i, z, my_secret, publics, *, modulus=2**32, prime=2**61 - 1)`: `publics` maps every participating client id (including `i`, which is skipped) to its public key; returns $z + \sum_{j > i}\text{prg}(s_{ij}) - \sum_{j < i}\text{prg}(s_{ij}) \bmod M$ with $s_{ij}$ = `shared_secret(my_secret, publics[j])`, comparing ids, not positions.

> **▶ build: stage 2.** implement `public_key`, `shared_secret`, `prg` and `mask`.

### idea 3: surviving dropouts with secret sharing

phones drop out all the time. if client $d$ agreed masks with everyone and then never sends $y_d$, the survivors' sum contains $\pm m_{dj}$ for every survivor $j$, uniformly random garbage (warm-up 3). the server needs $d$'s masks to remove them, but it must not get them from $d$'s partners alone, or it could unmask a *living* client by claiming it dropped.

Bonawitz et al.'s answer is **Shamir secret sharing** (Shamir, 1979). before masking, each client splits its secret $a_i$ into $n$ shares, one per client, such that any $t$ shares reconstruct $a_i$ and any $t - 1$ reveal nothing about it: pick a random polynomial $f$ of degree $t - 1$ over $\mathbb Z_p$ with $f(0) = a_i$, and give client $k$ the share $(k, f(k))$. any $t$ points determine $f$, and Lagrange interpolation recovers $f(0)$:

$$f(0) = \sum_{k \in T}f(x_k)\prod_{l \in T, l \ne k}\frac{x_l}{x_l - x_k} \pmod p.$$

with $t - 1$ points, every possible secret is consistent with exactly one polynomial, so the shares say nothing (on paper 2). the protocol, simplified:

1. **keys.** every client sends its public key; the server relays them.
2. **shares.** every client sends one Shamir share of its secret to each other client (encrypted to that client, via the server).
3. **masked input.** every surviving client sends $y_i$.
4. **recovery.** the server announces who dropped; each survivor sends its shares of the dropped clients' secrets; with $t$ of them the server reconstructs $a_d$, recomputes $m_{dj}$ for every survivor $j$, and removes them.

if fewer than $t$ clients survive, the round fails. the real protocol adds one more layer, **double masking**: each client also adds a self-mask $b_i$ (also secret-shared), and in step 4 survivors reveal shares of $b_i$ for *survivors* and of $a_d$ for *dropped* clients, never both for the same client. without it, a client that was merely slow (declared dropped, then its $y_d$ arrives) would be fully unmasked. you'll skip double masking here and think about it in the check questions.

counting bytes: a surviving client uploads its masked vector ($4d$ bytes), its public key, $n - 1$ shares in step 2, and one share per dropped client in step 4. with 8-byte keys and shares that's $4d + 8(n + |D|)$ bytes: linear in $n$ per client, quadratic in total. for big cohorts and small models, the overhead dominates; Bell et al. (2020) cut it to $O(\log n)$ by masking only with a random sparse set of neighbours.

> **spec.** `shamir_share(secret, n, t, *, rng, prime=2**61 - 1)`: `ValueError` unless `1 <= t <= n`; coefficients $a_1..a_{t-1}$ from `rng.integers(0, prime, size=t - 1)`; returns `[(x, f(x)) for x in 1..n]` with Python-int arithmetic mod `prime`. `shamir_reconstruct(shares, *, prime=2**61 - 1)` returns $f(0)$ by Lagrange interpolation (modular inverses with `pow(den, -1, prime)`). `secure_aggregate(updates, *, threshold, dropped=(), scale=2**16, modulus=2**32, prime=2**61 - 1, seed=None, uploads=None, revealed=None)`: `updates` maps client ids to real vectors; simulate steps 1–4 for all ids (secrets from `np.random.default_rng(seed)`), where `dropped` clients share their secrets but never send a masked input; `ValueError` if fewer than `threshold` clients survive; return the decoded float64 sum of the survivors' updates. two optional dicts expose what the server saw: if `uploads` is given, store each survivor's masked vector as `uploads[i]`; if `revealed` is given, store for each dropped client `d` a dict `{holder_id: share}` of the shares the server used to rebuild `d`'s secret (all from survivors, at least `threshold` of them). rebuild it with your `shamir_reconstruct`. `secagg_upload_bytes(n_clients, dim, *, n_dropped=0, entry_bytes=4, key_bytes=8)` returns `entry_bytes * dim + key_bytes * (1 + (n_clients - 1) + n_dropped)`.

> **🔮 predict:** with 20% of 50 clients dropping out, how close do you expect the recovered sum to be to the true sum of the survivors' updates?

> **▶ build: stage 3.** implement `shamir_share`, `shamir_reconstruct`, `secure_aggregate` and `secagg_upload_bytes`.

### idea 4: differential privacy for a sum of updates

secure aggregation hides individual updates from the server, but the sum, and the model trained from it, still depends on every user. **differential privacy** bounds that dependence. a randomized mechanism $\mathcal M$ is $(\varepsilon, \delta)$-DP if for any two datasets $D, D'$ that differ in one unit and any set of outcomes $S$,

$$\Pr[\mathcal M(D) \in S] \le e^\varepsilon\Pr[\mathcal M(D') \in S] + \delta.$$

in federated learning the natural unit is a **user** (everything on one device), not one example: **user-level DP**, which is much stronger than the example-level DP of DP-SGD.

the recipe has two steps. **bound the sensitivity**: clip each user's update to norm $S$, $\tilde\Delta_k = \Delta_k\min(1, S/\|\Delta_k\|)$, so adding or removing one user changes the sum $\sum_k\tilde\Delta_k$ by a vector of norm at most $S$. **add Gaussian noise** calibrated to that sensitivity: release $\sum_k\tilde\Delta_k + \mathcal N(0, \sigma^2S^2I)$, where the **noise multiplier** $\sigma$ sets the privacy level. DP-FedAvg (McMahan et al., 2018) divides by the *expected* number of participants $qN$, not the actual count, so that the denominator doesn't depend on who was sampled, and adds the step to the model:

$$w^{t+1} = w^t + \eta_s\,\frac{1}{qN}\Big(\sum_{k \in S_t}\tilde\Delta_k + \mathcal N(0, \sigma^2S^2I)\Big).$$

the noise in the averaged update has standard deviation $\sigma S/(qN)$ per coordinate, while the signal, an average of clipped updates, has norm up to $S$. so for fixed $\sigma$, the noise-to-signal ratio falls like $1/(qN)$: more users per round drown the noise out (warm-up 4). that's the lever the project turns.

> **spec.** `clip_update(delta, clip)` returns `delta * min(1, clip / ||delta||)` (a zero vector stays zero). `dp_aggregate(deltas, *, clip, noise_multiplier, expected_clients, generator=None)`: `deltas` a non-empty list of 1-D tensors; returns $\big(\sum_k\text{clip}(\Delta_k) + \sigma S\cdot\xi\big)/\text{expected\_clients}$ with $\xi$ = `torch.randn(shape, generator=generator, dtype=...)`.

> **▶ build: stage 4.** implement `clip_update` and `dp_aggregate`.

### idea 5: accounting for many rounds

one noisy round costs a little privacy; training takes hundreds. the privacy cost of composing mechanisms is what an **accountant** tracks, and the bookkeeping matters enormously (lab, experiment 2). two facts make the numbers small.

**Rényi DP composes additively.** a mechanism is $(\alpha, \rho)$-RDP (Mironov, 2017) if the Rényi divergence of order $\alpha$ between its outputs on neighbouring datasets is at most $\rho$:

$$D_\alpha(P\,\|\,Q) = \frac{1}{\alpha - 1}\log\mathbb E_{x \sim Q}\Big[\Big(\frac{P(x)}{Q(x)}\Big)^\alpha\Big] \le \rho.$$

running $T$ mechanisms gives $(\alpha, \sum_t\rho_t)$-RDP, for every $\alpha$ at once. the Gaussian mechanism with sensitivity 1 and noise $\sigma$ is $(\alpha, \alpha/(2\sigma^2))$-RDP. at the end, convert to $(\varepsilon, \delta)$-DP with $\varepsilon = \rho + \log(1/\delta)/(\alpha - 1)$, minimized over the orders $\alpha$ you tracked (Mironov, 2017, Prop. 3).

**sampling amplifies privacy.** if each user joins a round independently with probability $q$ (Poisson sampling), a user is usually absent and the round reveals much less about them. the release is then a mixture: $\mathcal N(0, \sigma^2)$ without the user, $(1 - q)\mathcal N(0, \sigma^2) + q\mathcal N(1, \sigma^2)$ with them (in units of $S$, along the direction of their update). for integer $\alpha$ the Rényi divergence of this pair has an exact expression (Mironov, Talwar & Zhang, 2019), from expanding the $\alpha$-th power binomially (on paper 4):

$$\rho(\alpha) = \frac{1}{\alpha - 1}\log\sum_{k=0}^{\alpha}\binom{\alpha}{k}(1 - q)^{\alpha - k}q^k\exp\Big(\frac{k^2 - k}{2\sigma^2}\Big).$$

for small $q$ (and $\sigma$ not too small) this is close to $q^2\alpha/(2\sigma^2)$, the plain Gaussian's $\alpha/(2\sigma^2)$ times $q^2$: a factor $\sim q^2$ less privacy loss per round, not just $q$ (warm-up 5). sum over rounds, convert once. this is the same calculation as Abadi et al.'s (2016) moments accountant, and it reproduces their published example: $q = 0.01$, $\sigma = 4$, $10^4$ steps, $\delta = 10^{-5}$ gives $\varepsilon \approx 1.26$.

be careful about what this number is. it's an **upper bound** on the privacy loss under the assumptions (Poisson sampling, add/remove-one-user neighbours, the server adds the noise honestly). it isn't tight: better RDP-to-DP conversions (Balle et al., 2020) and numerical accountants that compose the full privacy-loss distribution (Gopi, Lee & Wutschitz, 2021) report noticeably smaller ε for the same mechanism. and real federated systems don't sample users by independent coin flips; production analyses have to account for how clients actually check in.

the terms $\exp(k^2/(2\sigma^2))$ overflow a float for large orders and small $\sigma$: use log-sum-exp.

> **spec.** `rdp_sampled_gaussian(q, sigma, orders)`: integer orders $\ge 2$; returns a NumPy array of $\rho(\alpha)$ by the formula above, computed in log space (`math.lgamma` for the binomials); $q = 0$ gives 0, $\sigma = 0$ (with $q > 0$) gives `inf`, and $q = 1$ gives $\alpha/(2\sigma^2)$. `rdp_to_dp(rdp, orders, delta)` returns `(epsilon, best_order)` minimizing $\rho_\alpha + \log(1/\delta)/(\alpha - 1)$. `compute_epsilon(q, sigma, steps, delta, orders=tuple(range(2, 257)))` composes `steps` rounds and returns ε.

> **▶ build: stage 5.** implement `rdp_sampled_gaussian`, `rdp_to_dp` and `compute_epsilon`.

### idea 6: DP-FedAvg, and the triangle

the full algorithm is topic 27's loop with three changes: Poisson sampling of users, clipped deltas, and the noisy aggregate. it reports its privacy spent so far after every round, and counts uploads as secure-aggregation messages (in a deployment, the clipped updates go through secure aggregation and the noise is added to the decoded sum; a further refinement, distributed DP, has the clients add the noise in pieces so the server need not be trusted to do it: Kairouz, Liu & Steinke, 2021).

now the triangle. **privacy** is set by $\sigma$, $q$ and the number of rounds $T$ through the accountant. **accuracy** depends on the noise in each averaged update, $\sigma S/(qN)$, on the clipping bias, and on $T$. **communication** is $T\cdot qN$ uploads, each $4d + 8(qN + |D|)$ bytes with secure aggregation. to hit a fixed ε with a larger cohort you must raise $\sigma$ (the per-round privacy cost grows with $q$), at most roughly in proportion to $q$, so the noise in the average, $\propto\sigma/q$, falls or at worst stays flat, while the average itself is taken over more users and gets cleaner: larger cohorts buy accuracy at fixed privacy with communication. a larger *population* $N$ shrinks the noise for free. every knob trades one corner against another, and the only way to know the exchange rate for your problem is to measure it.

> **spec.** `dp_fedavg(model, client_data, local_update, *, rounds, sampling_rate, clip, noise_multiplier, delta, server_lr=1.0, local_kwargs=None, eval_fn=None, eval_every=1, seed=None)`. one `rng = np.random.default_rng(seed)`; first seed a `torch.Generator` with `int(rng.integers(2**31))` for the noise. each round: `chosen = np.flatnonzero(rng.random(N) < sampling_rate)`; for each chosen client reset the model to the global weights, call `local_update(model, X_k, y_k, seed=int(rng.integers(2**31)), **local_kwargs)` (it returns `(flat weights, n_steps)`, like topic 27's `client_update`), and form $\Delta_k$. if nobody was chosen, use one zero delta (the round must still add noise). update the model with `server_lr * dp_aggregate(deltas, ..., expected_clients=sampling_rate * N)`. returns a dict of per-round lists: `round`, `clients` (the number sampled), `bytes_up` (cumulative; each chosen client uploads `secagg_upload_bytes(n_chosen, P)`), `epsilon` (`compute_epsilon(sampling_rate, noise_multiplier, r, delta)` after round r), and `eval_round`/`eval` as in topic 27 (round 0, every `eval_every` rounds, and the last).

> **🔮 predict:** at a fixed ε = 1 with 1,000 users, which costs more accuracy: going from 400 users per round to 25, or going from no privacy to ε = 1?

> **▶ build: stage 6.** implement `dp_fedavg`.

## worked example

**encoding.** $1.5 \mapsto 1.5\cdot2^{16} = 98{,}304$; $-0.25 \mapsto -16{,}384 \equiv 4{,}294{,}950{,}912 \pmod{2^{32}}$. adding the two encodings mod $2^{32}$ gives $81{,}920$, which decodes to $1.25$.

**masks.** clients 0 and 1 with shared mask $m$: $y_0 = z_0 + m$, $y_1 = z_1 - m$, and $y_0 + y_1 = z_0 + z_1 \pmod{2^{32}}$. with five clients (ids 3, 7, 10, 42, 99) the ten pairwise masks cancel the same way; the checks verify the sum bit for bit.

**Shamir.** with $p = 13$, $t = 3$ and $f(x) = 5 + 3x + 2x^2$: shares $(1, 10)$, $(2, 19 \bmod 13 = 6)$, $(3, 32 \bmod 13 = 6)$. the Lagrange weights at 0 are $\frac{2\cdot3}{(2-1)(3-1)} = 3$, $\frac{1\cdot3}{(1-2)(3-2)} = -3$ and $\frac{1\cdot2}{(1-3)(2-3)} = 1$, so $f(0) = 3\cdot10 - 3\cdot6 + 6 = 18 \equiv 5$.

**the DP aggregate.** deltas $(3, 4)$ and $(0.1, 0)$, clip $S = 1$, no noise, expected clients 2: the first is clipped to $(0.6, 0.8)$, the second is untouched, and the aggregate is $(0.7, 0.8)/2 = (0.35, 0.4)$. with $\sigma = 2$, $S = 0.5$ and 10 expected clients the noise in the aggregate has standard deviation $2\cdot0.5/10 = 0.1$ per coordinate.

**accounting.** $q = 1$, $\sigma = 2$: $\rho(\alpha) = \alpha/8$. converting RDP values $(1, 0.5)$ at orders $(2, 3)$ with $\delta = e^{-2}$: $\min(1 + 2/1,\ 0.5 + 2/2) = 1.5$, at order 3. and $q = 0.01$, $\sigma = 4$, $10^4$ rounds, $\delta = 10^{-5}$ gives $\varepsilon = 1.259$, Abadi et al.'s 1.26.

## on paper

1. **one-time pads in $\mathbb Z_M$.** show that if $m$ is uniform on $\mathbb Z_M$ and independent of $z$, then $z + m \bmod M$ is uniform whatever $z$ is. for two clients with masked inputs $y_0 = z_0 + m$, $y_1 = z_1 - m$, show that the pair $(y_0, y_1)$ has a distribution that depends on $(z_0, z_1)$ only through $z_0 + z_1$.

<details><summary>solution</summary>

$\Pr[z + m \equiv v] = \Pr[m \equiv v - z] = 1/M$ for every $v$. for the pair: $y_0$ is uniform (by the first part) and $y_1 = (z_0 + z_1) - y_0 \bmod M$ is a function of $y_0$ and the sum. so the joint distribution is "$y_0$ uniform, $y_1 = \text{sum} - y_0$", which depends only on the sum. the server learns the sum and nothing else.

</details>

2. **Shamir's privacy.** for a degree-$(t-1)$ polynomial over $\mathbb Z_p$ with random coefficients, show that any $t - 1$ shares are consistent with every possible secret, each by exactly one polynomial.

<details><summary>solution</summary>

fix $t - 1$ share points $(x_k, y_k)$ with distinct nonzero $x_k$, and a candidate secret $s$. adding the point $(0, s)$ gives $t$ points with distinct $x$'s, which determine exactly one polynomial of degree at most $t - 1$ (Lagrange interpolation works over any field). so each of the $p$ possible secrets corresponds to exactly one coefficient vector, and since the coefficients are uniform, the secret given the $t - 1$ shares is still uniform: they reveal nothing.

</details>

3. **sensitivity and noise.** with $\tilde\Delta_k$ clipped to norm $S$, show that adding or removing one user changes $\sum_k\tilde\Delta_k$ by at most $S$ in $\ell_2$. then with $m$ users sampled, each contributing roughly the same update $\bar u$ ($\|\bar u\| \le S$), compare the norms of the signal and of the noise in the DP-FedAvg aggregate in $d$ dimensions.

<details><summary>solution</summary>

the sums with and without user $u$ differ by exactly $\tilde\Delta_u$, whose norm is at most $S$. the aggregate is $\frac{m}{qN}\bar u + \frac{\sigma S}{qN}\xi$ with $\xi \sim \mathcal N(0, I_d)$, so with $m \approx qN$ the signal has norm $\|\bar u\| \le S$ and the noise has norm $\approx \sigma S\sqrt d/(qN)$. the ratio noise/signal is $\gtrsim \sigma\sqrt d/(qN)$: it grows with the model size and shrinks with the cohort. that's why small models and large cohorts are what make DP federated training work.

</details>

4. **the subsampled Gaussian's RDP.** let $\mu_0 = \mathcal N(0, \sigma^2)$ and $\mu = (1 - q)\mu_0 + q\,\mathcal N(1, \sigma^2)$. show $\mu(z)/\mu_0(z) = (1 - q) + q\exp\big(\frac{2z - 1}{2\sigma^2}\big)$, expand $\mathbb E_{\mu_0}[(\mu/\mu_0)^\alpha]$ with the binomial theorem, and use $\mathbb E[e^{cZ}] = e^{c^2\sigma^2/2}$ for $Z \sim \mu_0$ to get the formula in idea 5. check $q = 1$.

<details><summary>solution</summary>

$\frac{\mathcal N(z; 1, \sigma^2)}{\mathcal N(z; 0, \sigma^2)} = \exp\big(\frac{z^2 - (z - 1)^2}{2\sigma^2}\big) = \exp\big(\frac{2z - 1}{2\sigma^2}\big)$. then $\mathbb E[(\mu/\mu_0)^\alpha] = \sum_k\binom\alpha k(1 - q)^{\alpha - k}q^k\,\mathbb E\big[e^{k(2Z - 1)/(2\sigma^2)}\big]$, and with $c = k/\sigma^2$: $\mathbb E[e^{kZ/\sigma^2}]e^{-k/(2\sigma^2)} = e^{k^2/(2\sigma^2)}e^{-k/(2\sigma^2)} = e^{(k^2 - k)/(2\sigma^2)}$. at $q = 1$ only $k = \alpha$ survives: $\log e^{(\alpha^2 - \alpha)/(2\sigma^2)}/(\alpha - 1) = \alpha/(2\sigma^2)$. (Mironov et al., 2019, show this ordering of the two distributions is the worse one for integer $\alpha$.)

</details>

## experiment

the lab's experiment 1 runs your `secure_aggregate` with 5–50 clients and 20% dropouts at the project's model size, compares it with the plain sum, and tabulates the upload overhead for larger cohorts. experiment 2 compares your accountant with two cruder ways of counting privacy over 1,000 rounds. experiment 3, the project, trains DP-FedAvg on 1,000 simulated users at four privacy levels and three cohort sizes, calibrating the noise with your accountant, and maps out accuracy against ε and against megabytes uploaded.

## check your understanding

1. secure aggregation hides individual updates from the server. name an attack it does *not* prevent.

<details><summary>answer</summary>

anything that works from the aggregate or the model: membership inference on the final model, or a malicious server that samples one real client plus clients it controls and subtracts their known updates. that's why secure aggregation is paired with DP and minimum cohort sizes.

</details>

2. why does the server need $t$ shares to recover a dropped client's secret, and what goes wrong if $t$ is too small or too large?

<details><summary>answer</summary>

too small (say $t = 1$): any one client, or a server colluding with a few clients, can reconstruct anyone's secret and unmask them. too large (near $n$): a few dropouts beyond $n - t$ make the round fail. Bonawitz et al. choose $t$ above half the cohort to defend against a server colluding with a minority, and over-select clients so enough survive.

</details>

3. a client is slow, the server declares it dropped and reconstructs its secret, and then its masked input arrives. what can the server do, and how does double masking prevent it?

<details><summary>answer</summary>

with the secret, the server can compute every pairwise mask of that client, strip them from its masked input, and read its update in the clear. with double masking the input also carries a self-mask $b_i$ that is reconstructed only for survivors; a client declared dropped has its pairwise secret revealed but never its $b_i$, so its late message stays masked. honest survivors refuse to reveal both kinds of shares for the same client.

</details>

4. why clip each user's whole update rather than each example's gradient, as DP-SGD does?

<details><summary>answer</summary>

the unit of privacy is the user. clipping per example bounds each example's influence, but a user with 1,000 examples could still move the model a lot. clipping the user's full update bounds the user's influence, whatever they hold, which is what user-level DP needs.

</details>

5. why divide by the expected number of participants $qN$ rather than the number who actually showed up?

<details><summary>answer</summary>

the actual count depends on whether a given user was sampled, so dividing by it would make the released quantity's scale reveal information and break the sensitivity analysis: with a fixed denominator, one user changes the release by at most $S/(qN)$ regardless of who else participated. the price is a slightly noisier estimate (the effective step scales with $m/(qN)$).

</details>

6. in experiment 2 (σ = 1, q = 0.01), your accountant gives ε ≈ 1.3, 1.5, 1.6, 2.5, 7.5, 29 after $10^0, \dots, 10^5$ rounds, while adding up per-round ε's grows like $T$. why is the RDP curve flat at first, and how does it grow later?

<details><summary>answer</summary>

the final ε is $\min_\alpha T\rho(\alpha) + \log(1/\delta)/(\alpha - 1)$. the conversion term wants a large order, but the subsampled Gaussian's $\rho(\alpha)$ is tiny only up to a critical order (here around 10: $\rho(10) \approx 0.04$, $\rho(12) \approx 1$, $\rho(15) \approx 2.6$), where the $k = \alpha$ term with its $e^{\alpha^2/(2\sigma^2)}$ factor takes over and amplification stops helping. so for small $T$ the best order sits just below that cliff, $T\rho$ is negligible, and ε $\approx \log(1/\delta)/(\alpha^\star - 1)$: a plateau set by δ and the cliff, not by $T$. once $T\rho(\alpha)$ matters, the best order slides down and, in the region where $\rho(\alpha) \approx c\alpha$, minimizing $Tc\alpha + \log(1/\delta)/\alpha$ gives $\varepsilon \approx 2\sqrt{Tc\log(1/\delta)}$: the $\sqrt T$ regime, here roughly $10^3$–$10^5$ rounds (2.5 → 7.5 → 29, a factor of about 3–4 per decade). when the best order hits the smallest one tracked (2), growth becomes linear in $T$ again. composing in $(\varepsilon, \delta)$ form pays a per-round conversion $T$ times, so it is linear from the start.

</details>

7. a paper reports "ε = 1" for its federated model. what else do you need to know before believing it means anything?

<details><summary>answer</summary>

the unit (example or user), δ compared with $1/N$, the accountant and its assumptions (Poisson sampling?), who added the noise, and whether hyperparameter tuning on private data was counted. ε without these is a number, not a guarantee.

</details>

## common bugs

- **masks summed in the wrong group.** symptom: the masked sum is off by multiples of $2^{32}$, or by huge amounts. fix: reduce mod $M$ after every addition and subtraction, and keep everything in `int64` (values below $2^{32}$, so sums of a few never overflow).
- **signs by position instead of by id.** symptom: masks cancel when ids are `0..n-1` but not for ids like `{3, 7, 42}`. fix: compare `j > i` on the ids.
- **removing a dropped client's masks with the wrong sign.** symptom: the sum is right without dropouts and garbage with them. fix: survivor $j$ *added* $m_{jd}$ if $d > j$ and subtracted it if $d < j$; undo exactly that.
- **overflowing floats in the accountant.** symptom: `nan` or `inf` ε for large orders or small σ. fix: log-sum-exp over the $\alpha + 1$ terms, with `math.lgamma` for the binomial coefficients.
- **dividing by the number of sampled clients.** symptom: the fixed-denominator check fails; under DP the sensitivity analysis no longer holds. fix: divide by `sampling_rate * N`.
- **skipping empty rounds.** symptom: no noise is added when nobody is sampled, and the accountant's assumption (one noisy release per round) is violated. fix: aggregate a zero delta so the noise is still added.

## going deeper

- Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal & Seth (2017), *Practical Secure Aggregation for Privacy-Preserving Machine Learning*, CCS. the protocol, double masking, and its security proofs. [eprint 2017/281](https://eprint.iacr.org/2017/281)
- Bell, Bonawitz, Gascón, Lepoint & Raykova (2020), *Secure Single-Server Aggregation with (Poly)Logarithmic Overhead*, CCS. [eprint 2020/704](https://eprint.iacr.org/2020/704)
- Shamir (1979), *How to Share a Secret*, Communications of the ACM 22(11). [doi:10.1145/359168.359176](https://doi.org/10.1145/359168.359176)
- McMahan, Ramage, Talwar & Zhang (2018), *Learning Differentially Private Recurrent Language Models*, ICLR. DP-FedAvg. [arXiv:1710.06963](https://arxiv.org/abs/1710.06963)
- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang (2016), *Deep Learning with Differential Privacy*, CCS. DP-SGD and the moments accountant. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
- Mironov (2017), *Rényi Differential Privacy*, CSF; and Mironov, Talwar & Zhang (2019), *Rényi Differential Privacy of the Sampled Gaussian Mechanism*. [arXiv:1702.07476](https://arxiv.org/abs/1702.07476), [arXiv:1908.10530](https://arxiv.org/abs/1908.10530)
- Dwork & Roth (2014), *The Algorithmic Foundations of Differential Privacy*, chapters 2–3. [doi:10.1561/0400000042](https://doi.org/10.1561/0400000042)
- Kairouz, Liu & Steinke (2021), *The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation*, ICML. [arXiv:2102.06387](https://arxiv.org/abs/2102.06387)
- Zhu, Liu & Han (2019), *Deep Leakage from Gradients*, NeurIPS: why updates need protecting. [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)
- Xu, Zhang, Andrew, Choquette-Choo, Kairouz, McMahan, Rosenstock & Zhang (2023), *Federated Learning of Gboard Language Models with Differential Privacy*. the whole stack in production. [arXiv:2305.18465](https://arxiv.org/abs/2305.18465)

## connections

- ← builds on: your `fedavg`, `client_update` and partitions (topics 27–28); Rényi and KL divergences (ch 05); the Gaussian MGF (ch 04); modular arithmetic and interpolation (ch 00–01).
- → used in: chapter 28's privacy topic (DP-SGD, membership inference) applies the same accountant per example rather than per user; the ideas of clipping and noise calibration recur wherever a model is trained on sensitive data.
