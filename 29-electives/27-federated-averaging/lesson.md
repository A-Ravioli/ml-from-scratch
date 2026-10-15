# federated averaging

## why this matters

your phone's keyboard learns which word you'll type next. the training data (everything typed on a hundred million phones) is the most useful text corpus a keyboard team could want, and it can't be collected: it's private, it's huge, and uploading it would cost users bandwidth and battery. so the data stays where it is and the *model* travels instead. a server sends the current model to a few phones; each phone trains on its own data for a while and sends back only the change to the model; the server averages those changes and repeats. this is **federated learning** (McMahan et al., 2017), and it's how Google trains the next-word models in its Gboard keyboard (Hard et al., 2018).

moving the model instead of the data turns training into a different optimization problem. every message crosses a slow, metered uplink, so a "step" is no longer cheap: a phone might upload a few megabytes once a night, while it has spare compute to burn. the question that drives everything in this topic is **how to trade cheap local computation for expensive communication**. the answer McMahan et al. found, *federated averaging* (FedAvg), is almost embarrassingly simple: let each client run several epochs of ordinary SGD before averaging. in this topic you build a FedAvg simulator that counts every byte it sends, and in the lab you'll measure how many rounds local work saves, how much of the upload you can throw away, and how many clients a round really needs.

## warm-up: predict first

1. three clients hold 10, 10 and 80 examples. after one round, they report models $w_1, w_2, w_3$. what should the server's new model be?
2. if every client takes exactly one full-batch gradient step and the server averages, which centralized algorithm have you reproduced?
3. each client now runs 20 local epochs per round instead of 1. roughly how many fewer rounds do you expect to need to reach the same accuracy: 20×, a few times, or none?
4. you're allowed 2 bits per number instead of 32 when a client uploads its update. how much accuracy do you guess you'd lose?
5. you can ask 10 or 50 clients per round. 50 costs 5× the bandwidth per round. is it 5× faster?

## prerequisites check

- [ ] you can write a minibatch SGD loop in PyTorch and know what `zero_grad`, `backward` and an in-place parameter update do (→ ch 15 · topic 07)
- [ ] you know why SGD's minibatch gradient is an unbiased estimate of the full gradient, and how its variance shrinks with batch size (→ ch 06)
- [ ] you can compute the expected value and variance of a two-point random variable (→ ch 04)
- [ ] you've trained an MLP classifier on MNIST (→ ch 14, ch 15)

## core ideas

### idea 1: models as vectors, and the weighted average

the server and the clients exchange *models*, so the first thing you need is a way to treat a network as a single object you can send, add and average. a PyTorch module's parameters are a list of tensors of different shapes; flatten each one and concatenate them in `model.parameters()` order, and the whole model is one vector $w \in \mathbb R^P$. for an MLP 784 → 64 → 10, $P = 784\cdot64 + 64 + 64\cdot10 + 10 = 50{,}890$, and at 4 bytes per float32 one model is a 203,560-byte message.

the global objective is the average loss over *all* data, wherever it lives. if client $k$ (of $N$) holds $n_k$ examples with local average loss $F_k(w)$, and $n = \sum_k n_k$,

$$F(w) = \frac1n\sum_{\text{all examples } i}\ell_i(w) = \sum_{k=1}^N \frac{n_k}{n}F_k(w).$$

that weighting, $n_k/n$, is the one the server must use when it combines client results: a client with 80 examples speaks for 80 examples (warm-up 1). the server's aggregate of client vectors $w_1,\dots,w_m$ is

$$\bar w = \frac{\sum_k n_k w_k}{\sum_k n_k}.$$

the plain mean is wrong whenever clients have unequal amounts of data, which in practice is always: a power user's phone holds a hundred times more text than a new user's.

> **spec.** `get_weights(model)` returns all parameters flattened in `model.parameters()` order into one 1-D tensor, detached and copied (changing it must not change the model). `set_weights(model, weights)` copies a flat vector back into the existing parameters in place (under `torch.no_grad()`, keeping the same `Parameter` objects) and raises `ValueError` if the length doesn't match; it returns the model. `fedavg_aggregate(client_weights, num_samples)` takes a list of equal-length 1-D tensors and their sample counts and returns $\sum_k n_k w_k / \sum_k n_k$; it raises `ValueError` for an empty list or counts that sum to zero.

> **▶ build: stage 1.** implement `get_weights`, `set_weights` and `fedavg_aggregate`.

### idea 2: the client: plain local SGD

a client's job is the most familiar code in this course: start from the model it was sent, run SGD on its own data for $E$ epochs with batch size $B$ and learning rate $\eta$, and report the result. with $n_k$ examples it takes $u_k = E\lceil n_k/B\rceil$ local steps.

two special cases pin down what FedAvg is. with $B = \infty$ (one full batch) and $E = 1$, the client takes exactly one gradient step, $w_k = w - \eta\nabla F_k(w)$. then the server's weighted average is

$$\sum_k \frac{n_k}{n}\big(w - \eta\nabla F_k(w)\big) = w - \eta\sum_k\frac{n_k}{n}\nabla F_k(w) = w - \eta\nabla F(w),$$

one step of full-batch gradient descent on the global objective (warm-up 2). McMahan et al. call this baseline **FedSGD**: it's distributed gradient descent, one round per step, and it needs a lot of rounds. FedAvg is everything else: more epochs, smaller batches, more steps between averages. the averaged model is then *not* any step of centralized SGD, because each client's later steps use gradients at its own drifting iterate. whether that's a problem depends on how different the clients' data are, which is topic 28's subject; for IID clients the local trajectories stay close and averaging them works remarkably well.

notice what the client sends back. the server already knows $w$, so it only needs the **update** $\Delta_k = w_k - w$; that's equivalent for averaging ($\bar w = w + \sum_k\frac{n_k}{n}\Delta_k$ when the weights sum to one), but updates are small numbers centred near zero, which is what makes them compressible (idea 5) and what clipping acts on in topic 29.

> **spec.** `client_update(model, X, y, *, epochs=1, batch_size=10, lr=0.1, loss_fn=None, seed=None)`: trains `model` in place with plain SGD ($p \leftarrow p - \eta\,\nabla p$, no momentum) on the mean loss `loss_fn(model(xb), yb)` (default `F.cross_entropy`). each epoch shuffles with `np.random.default_rng(seed).permutation(n)` (one generator for the whole call) and walks through minibatches of `batch_size` in that order, the last one possibly smaller; `batch_size=None` means one full batch. returns `(get_weights(model), n_steps)`.

> **▶ build: stage 2.** implement `client_update`.

### idea 3: who holds what: IID and non-IID partitions

a simulator needs to decide which examples live on which client. the friendliest case is **IID**: shuffle the dataset and deal it out evenly, so every client's data looks like a small sample of the whole. real federations are nothing like that. a person who only texts in Portuguese, a hospital that only sees one demographic, a camera that only sees one street: each client's distribution is different, and many clients see only a few of the classes.

McMahan et al. used a deliberately brutal split to stress-test FedAvg, the **pathological non-IID** or **shards** partition: sort the examples by label, cut the sorted list into $N\cdot s$ equal shards, and give each client $s$ random shards. with MNIST, $N = 100$ clients and $s = 2$, each shard holds 300 examples of (almost always) one digit, so most clients see exactly two digits. topic 28 adds a tunable middle ground (Dirichlet label skew); here you build the two extremes.

> **spec.** `partition_iid(n, n_clients, *, seed=None)` returns a list of `n_clients` integer index arrays that together contain each of `range(n)` exactly once, in random order, with sizes differing by at most 1 (shuffle with `np.random.default_rng(seed)`, then `np.array_split`). `partition_shards(y, n_clients, shards_per_client=2, *, seed=None)`: sort the indices by label (`np.argsort(y, kind="stable")`), split them into `n_clients * shards_per_client` near-equal shards with `np.array_split`, deal a random permutation of the shards so client $k$ gets shards $ks, \dots, ks + s - 1$ of it, and return each client's concatenated indices.

> **🔮 predict:** with the shards split, every client trains a model that can only recognise two digits. what happens when the server averages fifty such "two-digit experts" with weights that start from the same point?

> **▶ build: stage 3.** implement `partition_iid` and `partition_shards`.

### idea 4: the FedAvg loop, and counting bytes

now the server. each round $t$:

1. sample $m = \max(1, \operatorname{round}(CN))$ of the $N$ clients uniformly without replacement ($C$ is the **client fraction**; real systems sample from whoever is plugged in, idle and on Wi-Fi);
2. send each the current global model $w^t$;
3. each runs `client_update` from $w^t$ and returns its update $\Delta_k$;
4. set $w^{t+1} = w^t + \sum_{k\in S_t}\frac{n_k}{\sum_{j\in S_t}n_j}\Delta_k$.

the knobs are $C$, $E$ and $B$, and they trade against each other in a way that's easiest to see with costs. per round, the server sends $m$ models down and receives $m$ updates up: $2mP\cdot 4$ bytes with float32. computation per round is $\sum_{k\in S_t}u_k$ local steps, spread over $m$ devices in parallel. a centralized trainer has the opposite cost profile, where steps are the cost and communication is free. counting both, explicitly, is the only way to compare federated algorithms honestly: "converges in fewer rounds" might mean "does 20× more work per round", and "converges in fewer steps" might mean "sends 50 clients' worth of traffic each round".

two things make the counting matter in practice. uplinks are slower than downlinks (often by 5–10× on mobile networks), so it's the upload you most want to shrink. and clients drop out: a phone that's unplugged mid-round never reports, so a round's cost is what you sent, not what you received. the simulator counts bytes per message; turning those into seconds is a deployment question.

> **spec.** `fedavg(model, client_data, *, rounds, client_fraction=0.1, local_epochs=1, batch_size=10, lr=0.1, local_update=None, compress=None, eval_fn=None, eval_every=1, seed=None)`. `client_data` is a list of `(X_k, y_k)`; `model` is trained in place and holds the final global model. use one `rng = np.random.default_rng(seed)`. each round: `rng.choice(N, size=m, replace=False)` picks the clients; for each, `set_weights(model, w_global)`, then call `local_update(model, X_k, y_k, epochs=local_epochs, batch_size=batch_size, lr=lr, seed=int(rng.integers(2**31)))` (`local_update` defaults to your `client_update`; it returns `(weights, n_steps)`), and form $\Delta_k$. if `compress` is given, call `delta, n_bytes = compress(delta, rng)` and count `n_bytes` uploaded; otherwise count the full message, `P * element_size` bytes. every selected client also downloads one full message. aggregate with `fedavg_aggregate` weighted by `len(y_k)`. returns a dict of per-round lists `round` (1, 2, …), `bytes_up`, `bytes_down` and `client_steps` (all **cumulative**), `clients` (the selected indices), and `eval_round`, `eval`: `eval_fn(model)` is called before the first round (round 0), after every `eval_every`-th round and after the last one.

> **🔮 predict:** with $E = 5$, $B = 10$ and 100 examples per client, how many local steps does each client take per round, and how many bytes does a round with 10 clients move, in total, for the 50,890-parameter MLP?

> **▶ build: stage 4.** implement `fedavg`.

### idea 5: compressing the upload

a 4-byte float per parameter is far more precision than an SGD update needs: the update is already noisy. the cheapest compression is **quantization**: map each entry onto one of $2^b$ evenly spaced levels between the update's min and max, send $b$ bits per entry plus the two endpoints.

rounding to the *nearest* level is biased. if most entries sit just below a level and all round down, the average update shrinks in a systematic direction, and averaging over clients doesn't fix it. **stochastic rounding** removes the bias: an entry at fractional position $x - \lfloor x\rfloor = p$ between two levels rounds up with probability $p$ and down with probability $1 - p$. then (on paper 2)

$$\mathbb E[\hat v_i] = v_i, \qquad \operatorname{Var}[\hat v_i] = p(1-p)\,\delta^2 \le \frac{\delta^2}{4}, \qquad \delta = \frac{\max v - \min v}{2^b - 1}.$$

the server averages $m$ independently quantized updates, so the quantization noise in the aggregate is unbiased and its variance falls like $1/m$, on top of the SGD noise that was there anyway. this is the idea behind QSGD (Alistarh et al., 2017) and the distributed mean estimation analysis of Suresh et al. (2017), and it's why surprisingly few bits suffice. the message is $\lceil Pb/8\rceil + 8$ bytes ($b$ bits per entry, plus min and max as two float32), about $32/b$ times smaller than the raw update.

> **spec.** `stochastic_quantize(v, bits, *, rng=None)`: `bits` an integer in 1..16 (else `ValueError`); levels $\text{lo} + j\delta$ for $j = 0,\dots,2^b-1$ with lo $=\min v$, $\delta$ as above. with $x = (v - \text{lo})/\delta$, round each entry up with probability $x - \lfloor x\rfloor$ using uniforms from `rng.random(v.shape)` (a NumPy generator). returns `(v_hat, n_bytes)` with `v_hat` a tensor of `v`'s shape and dtype and `n_bytes = ceil(numel * bits / 8) + 8`. a constant vector comes back unchanged. the signature fits `fedavg`'s compressor slot via `lambda d, rng: stochastic_quantize(d, 4, rng=rng)`.

> **▶ build: stage 5.** implement `stochastic_quantize`.

## worked example

**the weighted average.** two clients report $w_1 = (1, 2)$ with $n_1 = 1$ and $w_2 = (3, 6)$ with $n_2 = 3$. the aggregate is $\frac14(1, 2) + \frac34(3, 6) = (2.5, 5)$. the plain mean, $(2, 4)$, is what you get if you forget the weights.

**FedSGD is gradient descent.** three clients with 5, 12 and 30 examples each take one full-batch step with $\eta = 0.5$. the weighted average of $w - 0.5\nabla F_k(w)$ is $w - 0.5\sum_k\frac{n_k}{47}\nabla F_k(w)$, which is exactly $w - 0.5\nabla F(w)$ for the mean cross-entropy over all 47 examples. your `fedavg` must match centralized gradient descent to float precision in this setting; the checks build exactly this case.

**the cost of a round.** MLP 784 → 64 → 10: $P = 50{,}890$, one message $= 203{,}560$ bytes. with $C = 0.1$ of $N = 100$ clients, $m = 10$; each client with 100 examples, $E = 5$, $B = 10$ takes $5\cdot10 = 50$ local steps. one round moves $10\cdot203{,}560 \approx 2.04$ MB up and the same down, and does 500 local steps. after 25 rounds `bytes_up` reads 50,890,000.

**quantizing.** $v = (0, 0.1, 0.5, 0.77, 1, -0.3)$ with $b = 2$: lo $= -0.3$, hi $= 1$, $\delta = 1.3/3 \approx 0.433$, levels $(-0.3, 0.133, 0.567, 1)$. the entry $0.5$ sits at $x = 0.8/0.433 = 1.846$: it becomes $0.567$ with probability $0.846$ and $0.133$ with probability $0.154$, and indeed $0.846\cdot0.567 + 0.154\cdot0.133 = 0.5$. the endpoints $-0.3$ and $1$ come back exactly. the message is $\lceil 6\cdot2/8\rceil + 8 = 10$ bytes instead of 24.

## on paper

1. **FedSGD.** show that with $E = 1$, $B = \infty$ and all clients participating, one FedAvg round equals one step of gradient descent on $F(w) = \sum_k\frac{n_k}{n}F_k(w)$. then show what goes wrong with the plain mean $\frac1N\sum_k w_k$: which objective is it a gradient step on?

<details><summary>solution</summary>

$w_k = w - \eta\nabla F_k(w)$, so $\sum_k\frac{n_k}{n}w_k = w\sum_k\frac{n_k}{n} - \eta\sum_k\frac{n_k}{n}\nabla F_k(w) = w - \eta\nabla F(w)$, using $\sum_k n_k/n = 1$ and linearity of the gradient. the plain mean gives $w - \eta\nabla\big(\frac1N\sum_kF_k\big)(w)$: gradient descent on the *per-client* average loss, which weights an example on a small client more than one on a large client. that's a legitimate objective (it's "fair to clients" rather than "fair to examples") but it isn't the one you meant.

</details>

2. **stochastic rounding.** an entry sits at $x = j + p$ (in units of $\delta$, $0 \le p < 1$) and is rounded to $j + 1$ with probability $p$, else $j$. compute the mean and variance of the result, and show the variance is at most $\delta^2/4$. then bound the variance of the average of $m$ independently quantized updates.

<details><summary>solution</summary>

mean: $\delta[(j+1)p + j(1-p)] = \delta(j + p) = \delta x$, so after mapping back $\mathbb E[\hat v] = v$. variance: the result is $\delta j$ plus $\delta$ times a Bernoulli($p$), so $\operatorname{Var} = \delta^2p(1-p) \le \delta^2/4$ (maximized at $p = 1/2$). averaging $m$ independent quantizations with weights $\pi_k$ (summing to 1) gives variance $\sum_k\pi_k^2\operatorname{Var}_k \le \max_k\delta_k^2/4\cdot\sum_k\pi_k^2$, which is $\delta^2/(4m)$ for equal weights. summed over $P$ coordinates, the squared error of the aggregate is at most $P\delta^2/(4m)$.

</details>

3. **bytes vs rounds.** a phone's uplink manages 1 MB/s in good conditions, and a client is available for at most 5 minutes per round. for an MLP with $P = 50{,}890$ and 2-bit quantization, how long does an upload take? for a 100-million-parameter model at float32? what does this say about which models can be trained federatedly without compression?

<details><summary>solution</summary>

2-bit: $\lceil 50{,}890\cdot2/8\rceil + 8 = 12{,}731$ bytes, about 13 ms. float32 100M parameters: 400 MB, about 400 s, over the 5-minute window before any training happens. large models need compression, partial updates (train only adapters or a subset of layers), or both; this is why on-device federated training targets small models (keyboard language models of a few million parameters).

</details>

## experiment

the lab runs your `fedavg` on 10,000 MNIST images spread over 100 clients, IID and with the shards split. experiment 1 varies the local epochs $E \in \{1, 5, 20\}$ and plots accuracy against communication rounds and against total local steps. experiment 2 pushes uploads through your `stochastic_quantize` at 1–8 bits and plots accuracy against megabytes uploaded. experiment 3 varies the number of clients per round. record the rounds each setting needs to reach a target accuracy, and the seed-to-seed spread.

## check your understanding

1. why does the server need the sample counts $n_k$, and what is the privacy cost of sending them?

<details><summary>answer</summary>

they're the aggregation weights that make the average match the global objective. sending $n_k$ reveals how much data each client has, which can be sensitive (how much someone types, how many patients a clinic sees). systems often cap or bucket $n_k$, or use uniform weights; topic 29's DP-FedAvg uses fixed weights for exactly this reason.

</details>

2. FedAvg with $E = 1$, $B = 10$ and FedSGD both use each example once per round. why are they not the same algorithm?

<details><summary>answer</summary>

FedSGD computes one gradient per client at $w^t$ and takes one step. FedAvg with $B = 10$ takes $\lceil n_k/10\rceil$ sequential steps per client, each at the client's updated iterate, so the update is a composition of many small steps: roughly $\lceil n_k/10\rceil$ times the progress per round when the steps agree, which is most of FedAvg's advantage.

</details>

3. could the server just average the client *gradients* at the end of local training instead of the weights?

<details><summary>answer</summary>

the gradient at the final local iterate says almost nothing about the path the client took; you'd lose all the local progress. what's averaged must be the accumulated change $\Delta_k$ (equivalently, the final weights). you can view $-\Delta_k/\eta$ as a "pseudo-gradient", which is how server-side optimizers like FedAdam (Reddi et al., 2021) treat it.

</details>

4. in experiment 1 on IID data, why do the returns from more local epochs diminish?

<details><summary>answer</summary>

after enough local steps each client is close to the minimizer of its own empirical loss on 100 examples; more steps move it further toward overfitting that small sample rather than toward the global optimum. per round the gain saturates, and per local step it gets worse (more computation for the same progress).

</details>

5. why must the quantizer's randomness be independent across clients?

<details><summary>answer</summary>

the variance of the averaged quantization noise falls like $1/m$ only if the clients' rounding errors are independent. if every client used the same uniforms, errors in similar updates would be correlated and wouldn't average out. (an extreme case: identical updates with identical uniforms give identical errors, and averaging does nothing.)

</details>

6. a round selects 10 clients but 3 drop out before uploading. what should the server do with the other 7 updates, and what has the round cost?

<details><summary>answer</summary>

aggregate the 7 that arrived, weighted by their own $n_k$ (the weights renormalize over the reporting clients); that's still an unbiased-ish step if dropout is unrelated to the data. the round cost 10 downloads and 7 uploads. production systems over-select clients (say 130% of the target) so that enough report before a deadline (Bonawitz et al., 2019). dropout correlated with the data (slow phones are cheap phones are certain demographics) biases the model, which is a fairness problem, not just an efficiency one.

</details>

7. why is quantizing $\Delta_k$ much better than quantizing $w_k$?

<details><summary>answer</summary>

the quantization step is proportional to the range of what you quantize. weights span a range set by initialization and training (order 1), while an update spans the range of one round's change (often 10–100× smaller), so the same number of bits gives a far finer grid. and the error in $\hat w_k$ would be applied in full every round, while the error in $\hat\Delta_k$ only perturbs one step.

</details>

## common bugs

- **the plain mean instead of the weighted one.** symptom: the FedSGD check fails only when clients have different sizes; on real data, small clients pull the model around. fix: weight by $n_k$ (and normalize by the sum over the *selected* clients).
- **not resetting the model before each client.** symptom: client 2 starts where client 1 stopped, so the "round" is really sequential training on 10 clients, with an aggregate that double-counts early clients. fix: `set_weights(model, w_global)` before every `local_update` call, from a copy of $w^t$ taken at the start of the round.
- **`get_weights` returning a view.** symptom: the server's snapshot of $w^t$ changes while a client trains, so every $\Delta_k$ comes out zero and the model never moves. fix: `torch.cat` already copies; if you return a single parameter's `.view(-1)`, `.clone()` it.
- **cumulative vs per-round bytes.** symptom: the bytes-vs-accuracy plot is a vertical line. fix: the history stores running totals.
- **nearest-level rounding in the quantizer.** symptom: the unbiasedness check fails, and with few bits training stalls even though each update "looks right". fix: round up with probability equal to the fractional part.
- **dividing by zero for a constant update.** symptom: NaNs after a round where some client didn't move (e.g. zero learning rate or an empty dataset). fix: return a constant vector unchanged.

## going deeper

- McMahan, Moore, Ramage, Hampson & Agüera y Arcas (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS. the FedAvg paper: the shards split, the $C$/$E$/$B$ experiments you reproduce. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- Kairouz, McMahan et al. (2021), *Advances and Open Problems in Federated Learning*, Foundations and Trends in ML. the field's survey; sections 1 and 3 cover the setting and communication. [arXiv:1912.04977](https://arxiv.org/abs/1912.04977)
- Konečný, McMahan, Yu, Richtárik, Suresh & Bacon (2016), *Federated Learning: Strategies for Improving Communication Efficiency*. structured and sketched updates, random rotations before quantization. [arXiv:1610.05492](https://arxiv.org/abs/1610.05492)
- Alistarh, Grubic, Li, Tomioka & Vojnovic (2017), *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding*, NeurIPS. [arXiv:1610.02132](https://arxiv.org/abs/1610.02132)
- Suresh, Yu, Kumar & McMahan (2017), *Distributed Mean Estimation with Limited Communication*, ICML. the variance analysis of stochastic quantization, and why rotating first helps. [arXiv:1611.00429](https://arxiv.org/abs/1611.00429)
- Stich (2019), *Local SGD Converges Fast and Communicates Little*, ICLR; and Woodworth et al. (2020), *Is Local SGD Better than Minibatch SGD?*, ICML: when local steps provably help on IID data. [arXiv:1805.09767](https://arxiv.org/abs/1805.09767), [arXiv:2002.07839](https://arxiv.org/abs/2002.07839)
- Hard et al. (2018), *Federated Learning for Mobile Keyboard Prediction*. FedAvg in production on Gboard. [arXiv:1811.03604](https://arxiv.org/abs/1811.03604)
- Bonawitz et al. (2019), *Towards Federated Learning at Scale: System Design*, MLSys. how a production system selects clients, handles dropouts and paces rounds. [arXiv:1902.01046](https://arxiv.org/abs/1902.01046)

## connections

- ← builds on: minibatch SGD (ch 06), PyTorch training loops and `mlfs.torchnn.MLP` (ch 15 · topic 07), unbiased estimators and their variance (ch 04, ch 05).
- → used in: topic 28 plugs FedProx into your `fedavg` through `local_update` and studies non-IID data with your partitions; topic 29 replaces the server's average with secure aggregation and a differentially private aggregate, and counts the extra bytes.
