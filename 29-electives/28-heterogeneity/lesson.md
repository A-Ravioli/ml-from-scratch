# heterogeneity

## why this matters

in topic 27 you split MNIST evenly at random and FedAvg worked beautifully: more local work, fewer rounds, and accuracy close to centralized training. real federations never look like that. a keyboard user in Lisbon types Portuguese, one in Osaka types Japanese; one hospital sees mostly children, another mostly the elderly; a phone that photographs only cats has a label distribution of one class. the clients' data are **non-IID**, and that breaks the argument that made FedAvg look so good: when every client's loss is a sample of the same function, many local steps on it approximate many steps on the global loss; when each client has *its own* loss, many local steps walk each client toward *its own* optimum, and the server averages points that were never aiming at the same place.

this topic makes that failure precise and then looks at three responses. **FedProx** (Li et al., 2020) adds a leash to each client. **SCAFFOLD** (Karimireddy et al., 2020) estimates each client's bias and subtracts it. **personalization** gives up on one model for everyone and adapts the global model to each client. you'll build tunable label skew, compute FedAvg's error exactly on quadratics (where it has a closed form you'll derive), implement SCAFFOLD and FedProx, and in the lab measure how much skew FedAvg tolerates on MNIST, whether FedProx's leash helps, and who benefits from personalization.

## warm-up: predict first

1. ten clients each hold only one digit class of MNIST. each trains for 50 SGD steps from the global model and reports back. what does the average of their models predict?
2. two clients: client 1's loss is $\tfrac12w^2$, client 2's is $\tfrac32(w-4)^2$. the global loss (the average) is minimized at $w = 3$. if each client runs gradient descent to convergence every round and the server averages, where does FedAvg end up?
3. same clients, but each takes exactly one gradient step per round. where does FedAvg end up?
4. FedProx adds $\tfrac\mu2\|w - w^t\|^2$ to each client's loss, pulling it toward the round's starting model. can a large enough $\mu$ make FedAvg's endpoint exactly right?
5. a client with 80 images of only two digits trains a fresh model on just its own data. on test images *from that client's own distribution*, does it beat a federated model trained on all 10,000 images?

## prerequisites check

- [ ] you've built `fedavg`, `client_update` and `partition_shards` (→ topic 27)
- [ ] you can solve a linear system and know when $I - T$ is invertible, e.g. when $T$'s eigenvalues are inside the unit circle (→ ch 02)
- [ ] you know gradient descent on a quadratic: the iteration $w \leftarrow w - \eta A(w - c)$ converges for $0 < \eta < 2/\lambda_{\max}(A)$ (→ ch 06)
- [ ] you know the Dirichlet distribution as a distribution over probability vectors (→ ch 04, ch 12)
- [ ] you've met control variates as a variance-reduction trick (→ ch 05, ch 23 · topic 03's baselines)

## core ideas

### idea 1: kinds of heterogeneity, and a dial for label skew

clients can differ in many ways (Kairouz et al., 2021, §3.1): **label skew** (different class proportions, the hospitals and the cat phone), **feature skew** (same labels, different inputs: handwriting styles, camera sensors), **concept shift** (same input, different label: "football" in Boston and in Barcelona), and **quantity skew** (a power user holds 100× more data). label skew is the easiest to simulate and the most studied, so it's the one you'll build.

topic 27's shards split is the extreme. the standard tunable alternative (Hsu, Qi & Brown, 2019) uses a Dirichlet distribution *per class*: for each class $c$, draw a vector of client proportions $\pi_c \sim \operatorname{Dir}(\alpha\mathbf 1_N)$ (one entry per client, $N$ clients) and deal that class's examples to the clients in those proportions. as $\alpha \to \infty$, $\pi_c \to (1/N, \dots, 1/N)$ and the split becomes IID; at $\alpha = 1$, $\pi_c$ is uniform on the simplex and clients have visibly different but broad label mixes; as $\alpha \to 0$, each class lands almost entirely on one client. one side effect: client sizes vary too (a client that drew small proportions everywhere ends up nearly empty), so implementations resample until every client has at least a minimum number of examples.

to put a number on the skew, compare each client's label distribution $p_k$ with the global one $\bar p = \sum_k\frac{n_k}{n}p_k$ using the **total variation distance** $\operatorname{TV}(p, q) = \tfrac12\sum_c|p_c - q_c|$ (the largest difference in probability the two assign to any set of classes), and average over clients, weighted by size:

$$H = \sum_k\frac{n_k}{n}\operatorname{TV}(p_k, \bar p) \in [0, 1].$$

$H \approx 0$ for IID; with 10 balanced classes, one label per client gives $H = 0.9$.

> **spec.** `partition_dirichlet(y, n_clients, alpha, *, seed=None, min_size=1, max_tries=100)`: one `rng = np.random.default_rng(seed)`. for each class in `np.unique(y)` (sorted), shuffle that class's indices with `rng.permutation`, draw `rng.dirichlet(alpha * ones(n_clients))`, and split the shuffled indices at `(cumsum(p)[:-1] * n_c).astype(int)`, client $k$ taking the $k$-th piece. if any client ends up with fewer than `min_size` examples, start over; after `max_tries` failed attempts raise `RuntimeError`. returns a list of integer index arrays. `label_distribution(parts, y, n_classes)` returns the `(n_clients, n_classes)` array of normalized label histograms (an empty client's row is all zeros). `heterogeneity(parts, y, n_classes)` returns $H$ as a float.

> **▶ build: stage 1.** implement `partition_dirichlet`, `label_distribution` and `heterogeneity`.

### idea 2: client drift, exactly

to see what goes wrong, take the simplest possible clients: client $k$'s loss is a quadratic $f_k(w) = \tfrac12(w - c_k)^\top A_k(w - c_k)$ with its own optimum $c_k$ and curvature $A_k$ (positive definite). the global objective $F = \sum_kp_kf_k$ is minimized where $\sum_kp_kA_k(w - c_k) = 0$:

$$w^{\text{opt}} = \Big(\sum_kp_kA_k\Big)^{-1}\sum_kp_kA_kc_k,$$

a curvature-weighted average of the local optima. now run FedAvg with all clients, each taking $K$ steps of gradient descent with step size $\eta$ from the global model $w^t$. one local step is the affine map $w \mapsto Gw + \eta A_kc_k$ with $G = I - \eta A_k$, so after $K$ steps from $w^t$

$$w_k = T_kw^t + s_k, \qquad T_k = G^K,\quad s_k = \eta\Big(\sum_{j<K}G^j\Big)A_kc_k.$$

the server's average is again affine, $w^{t+1} = \bar Tw^t + \bar s$ with $\bar T = \sum_kp_kT_k$, $\bar s = \sum_kp_ks_k$, and if it converges it converges to the **fixed point**

$$w^\star = (I - \bar T)^{-1}\bar s.$$

that's an exact formula for where FedAvg ends up, and in general $w^\star \ne w^{\text{opt}}$. this gap is **client drift**. three special cases tell you where it comes from (on paper 1 and 2):

- $K = 1$: $w^\star = w^{\text{opt}}$. one local step is FedSGD, gradient descent on $F$ (warm-up 3).
- $K \to \infty$: each client reaches its own optimum, $w_k \to c_k$, so $w^\star \to \sum_kp_kc_k$, the plain average of the local optima, which ignores curvature (warm-up 2: $(0 + 4)/2 = 2$, not 3).
- identical curvatures $A_k = A$: $w^\star = w^{\text{opt}}$ for every $K$. drift needs clients that disagree about *both* where the optimum is and how sharply the loss curves around it.

in between, the error grows with $K$ and approaches the distance between $w^{\text{opt}}$ and the average of the $c_k$ (it's the $K \to \infty$ limit, not a ceiling: with two clients in one dimension the error rises to it monotonically, but with more clients or dimensions it can overshoot at intermediate $K$ before settling). with stochastic gradients and neural networks there's no closed form, but the mechanism is the same: local steps follow each client's own gradient field, which near the end of local training points at the client's own optimum.

the same algebra covers FedProx (idea 4): with the proximal term, a local step is $w \leftarrow w - \eta\big(A_k(w - c_k) + \mu(w - w^t)\big)$, so $G = I - \eta(A_k + \mu I)$ and the anchor adds $\eta\mu\big(\sum_{j<K}G^j\big)w^t$ to $T_k$. one function handles both.

> **spec.** all three take `As` of shape `(n_clients, d, d)`, `cs` of shape `(n_clients, d)` and optional `weights` (normalized to $p_k$; default uniform). `global_optimum(As, cs, weights=None)` returns $w^{\text{opt}}$. `quadratic_fedavg(As, cs, w0, *, rounds, local_steps, lr, mu=0.0, weights=None)` runs full-participation FedAvg (FedProx if `mu > 0`) with `local_steps` steps of exact gradient descent per client, and returns the `(rounds + 1, d)` array of global iterates starting with `w0`. `quadratic_fixed_point(As, cs, *, local_steps, lr, mu=0.0, weights=None)` returns $w^\star = (I - \bar T)^{-1}\bar s$ computed from the formulas above.

> **🔮 predict:** in the 1-D example of warm-up 2 with $\eta = 0.1$, is FedAvg's fixed point with $K = 2$ closer to 3 (the optimum) or to 2 (the average of local optima)? (the worked example computes it.)

> **▶ build: stage 2.** implement `global_optimum`, `quadratic_fedavg` and `quadratic_fixed_point`.

### idea 3: SCAFFOLD: subtract the bias

client $k$'s local steps go wrong because it follows $\nabla f_k$ instead of $\nabla F$. if the client knew the difference $\nabla F(y) - \nabla f_k(y)$ it could add it to every step and follow the global gradient while only touching its own data. it can't compute that, but it can *estimate* it from the previous round. that's SCAFFOLD (Karimireddy et al., 2020), short for stochastic controlled averaging: each client keeps a **control variate** $c_k$, an estimate of its own gradient, and the server keeps $c = \sum_kp_kc_k$, an estimate of the global gradient. local steps use the corrected direction

$$y \leftarrow y - \eta\big(\nabla f_k(y) - c_k + c\big).$$

after $K$ steps from $x$ (the global model), the client updates its control variate from what it just observed, its average gradient over the trajectory ("option II" in the paper):

$$c_k^+ = c_k - c + \frac{x - y_K}{K\eta},$$

and the server moves both the model and the global control variate by the weighted average change: $x \leftarrow x + \sum_kp_k(y_k - x)$, $c \leftarrow c + \sum_kp_k(c_k^+ - c_k)$. this is the same idea as the baseline in policy gradients or SVRG's anchor gradient: subtract an estimate of the part you know is coming, so what's left is smaller. the correction costs communication: each client uploads $c_k^+ - c_k$ alongside its model update, doubling the upload.

with control variates starting at zero, SCAFFOLD's first round is FedAvg's. after that, the corrections remove the drift: at SCAFFOLD's fixed point $c_k = \nabla f_k(x^\star)$ and $c = \nabla F(x^\star) = 0$ (on paper 3), so it converges to the true optimum for any $K$, in the quadratic case exactly. with partial participation and stochastic gradients its guarantees are statements about convergence rates rather than exactness, and in deep-learning benchmarks its advantage over FedAvg is real but less dramatic than on quadratics (Li et al., 2021).

> **spec.** `quadratic_scaffold(As, cs, w0, *, rounds, local_steps, lr, weights=None)`: full participation, server learning rate 1, all control variates initialized to zero, the updates above; returns the `(rounds + 1, d)` array of global iterates starting with `w0`.

> **▶ build: stage 3.** implement `quadratic_scaffold`.

### idea 4: FedProx: a leash on each client

FedProx (Li et al., 2020) asks each client to minimize its loss plus a proximal term that keeps it near the model it was sent:

$$\min_w\ f_k(w) + \frac\mu2\|w - w^t\|^2.$$

the gradient of the extra term is $\mu(w - w^t)$: zero at the start of the round and growing as the client wanders. the motivation in the paper is twofold. statistically, it limits how far a client can drift. systemically, it makes partial work safe: in a real deployment some devices finish 20 epochs and some only 2, and plain FedAvg either drops the slow ones (biasing the model toward fast devices) or averages wildly different amounts of progress; with the proximal term, every client solves a well-posed problem that a few steps can approximately solve, and the paper's convergence analysis allows each client a different amount of inexactness.

as idea 2 showed, the proximal term changes FedAvg's fixed point but doesn't remove the drift for any finite $\mu$: each client is pulled toward $w^t$, not toward $w^{\text{opt}}$. in the $K \to \infty$ limit each client returns $\operatorname{prox}_{f_k/\mu}(w^t)$, and the fixed point is a curvature-reweighted average that approaches $w^{\text{opt}}$ only as $\mu \to \infty$, where clients barely move (on paper 4). FedProx trades drift for progress per round.

in code, FedProx is your `client_update` with one more term in the loss. the global weights are whatever the model holds when the function is called (`fedavg` resets the model to $w^t$ before every client), so the function needs nothing extra to plug into your topic 27 simulator: `fedavg(..., local_update=functools.partial(fedprox_client_update, mu=0.1))`.

> **spec.** `fedprox_client_update(model, X, y, *, mu=0.01, epochs=1, batch_size=10, lr=0.1, loss_fn=None, seed=None)`: exactly topic 27's `client_update` (plain SGD, `np.random.default_rng(seed).permutation` each epoch, `batch_size=None` for one full batch, default loss `F.cross_entropy`) except that each minibatch loss gets $+\frac\mu2\sum_p\|p - p^{(0)}\|^2$, with $p^{(0)}$ a detached copy of each parameter taken at entry. returns `(flat weights, n_steps)`.

> **▶ build: stage 4.** implement `fedprox_client_update`.

### idea 5: personalization: one model each

all of the above assumes the goal is one model that's good on the union of the data. but a keyboard user doesn't care how the model does on everyone else's typing; they care how it does on theirs. with skewed clients those are different targets, and a single global model has to compromise between them. **personalization** gives each client its own model, while still using the federation to learn what's shared.

the simplest version works surprisingly well (Wang et al., 2019): train a global model with FedAvg, then let each client **fine-tune** it for a few SGD steps on its own data before using it. variants keep a per-client output head and federate only the body (Arivazhagan et al., 2019), or fine-tune only the head, which is cheaper and less prone to overfitting a small local dataset. at the other extreme, a client can ignore the federation and train **locally only**. comparing the four on each client's own test data shows when federation helps a client and when it's in the way.

evaluating personalization needs per-client test data from the client's own distribution, and reports the **mean per-client accuracy**, which can be very different from the global model's accuracy on a pooled test set.

> **spec.** `fine_tune(model, X, y, *, steps=10, lr=0.05, batch_size=10, head_only=False, loss_fn=None, seed=None)`: returns a `copy.deepcopy` of `model` after exactly `steps` plain-SGD steps on (X, y); the input model is untouched. minibatches walk through `rng.permutation(n)` (`rng = np.random.default_rng(seed)`), drawing a new permutation whenever fewer than `batch_size` unused indices remain, so every step uses exactly `min(batch_size, n)` examples (`batch_size=None`: full batch). with `head_only=True` only the parameters of the last `nn.Linear` in `model.modules()` are updated. `per_client_accuracy(models, client_data)` takes one model (used for every client) or a list with one model per client, and returns a NumPy array of each model's accuracy on its client's `(X_k, y_k)` (models in eval mode, no gradients).

> **🔮 predict:** on IID clients, does fine-tuning a good global model for 20 steps on a client's 80 examples raise or lower its accuracy on that client's test data?

> **▶ build: stage 5.** implement `fine_tune` and `per_client_accuracy`.

## worked example

**heterogeneity.** labels $y = (0, 0, 1, 1)$; client 0 holds example 0, client 1 holds examples 1–3. label distributions: $p_0 = (1, 0)$, $p_1 = (1/3, 2/3)$; global $\bar p = (1/2, 1/2)$. TVs: $\tfrac12(0.5 + 0.5) = 0.5$ and $\tfrac12(1/6 + 1/6) = 1/6$. weighted: $\tfrac14\cdot0.5 + \tfrac34\cdot\tfrac16 = 0.25$.

**drift in one dimension.** $f_1 = \tfrac12w^2$ ($a_1 = 1$, $c_1 = 0$), $f_2 = \tfrac32(w - 4)^2$ ($a_2 = 3$, $c_2 = 4$), equal weights, $\eta = 0.1$.

- optimum: $w^{\text{opt}} = (1\cdot0 + 3\cdot4)/(1 + 3) = 3$.
- $K = 2$: $G_1 = 0.9$, $G_2 = 0.7$, so $T_1 = 0.81$, $T_2 = 0.49$; $s_1 = 0$, $s_2 = 0.1(1 + 0.7)\cdot3\cdot4 = 2.04$. then $\bar T = 0.65$, $\bar s = 1.02$ and $w^\star = 1.02/0.35 = 2.914$. two local steps already cost 0.086.
- $K \to \infty$: $T_k \to 0$ and $s_k \to c_k$, so $w^\star \to (0 + 4)/2 = 2$.
- FedProx with $\mu = 1$, $K \to \infty$: client $k$ returns $\frac{a_kc_k + \mu w}{a_k + \mu}$, so $w^\star$ solves $w = \tfrac12\big(\tfrac{w}{2} + \tfrac{12 + w}{4}\big)$, giving $w^\star = 1.5/0.625 = 2.4$: better than 2, still not 3.
- SCAFFOLD converges to 3 for every $K$.

your code must reproduce the first three numbers and SCAFFOLD's limit.

## on paper

1. **the fixed point.** derive $T_k$ and $s_k$ for $K$ local GD steps on $f_k$ (with $\mu = 0$), show $w^\star = (I - \bar T)^{-1}\bar s$, and check that $K = 1$ gives $w^{\text{opt}}$.

<details><summary>solution</summary>

one step: $w \mapsto w - \eta A_k(w - c_k) = Gw + \eta A_kc_k$. by induction, $K$ steps from $w^t$ give $G^Kw^t + \sum_{j<K}G^j\eta A_kc_k$. averaging: $w^{t+1} = \bar Tw^t + \bar s$, and a fixed point satisfies $w = \bar Tw + \bar s$, i.e. $(I - \bar T)w = \bar s$. ($I - \bar T$ is invertible when $\eta < 1/\lambda_{\max}(A_k)$ for all $k$: then every $T_k$ is symmetric with eigenvalues in $[0, 1)$, and so is their average.) for $K = 1$: $I - \bar T = \eta\sum_kp_kA_k$ and $\bar s = \eta\sum_kp_kA_kc_k$, so $w^\star = w^{\text{opt}}$.

</details>

2. **the two limits.** show that as $K \to \infty$, $w^\star \to \sum_kp_kc_k$, and that if all $A_k = A$, $w^\star = w^{\text{opt}}$ for every $K$.

<details><summary>solution</summary>

for $\mu = 0$, $\sum_{j<K}G^j\eta A_k = (I - G^K)$ (a geometric series, using $\eta A_k = I - G$), so $s_k = (I - T_k)c_k$. as $K \to \infty$, $T_k \to 0$, $\bar T \to 0$ and $\bar s \to \sum_kp_kc_k$. if $A_k = A$, all $T_k = T$ and $\bar s = (I - T)\sum_kp_kc_k$, so $w^\star = \sum_kp_kc_k$, which is also $w^{\text{opt}} = A^{-1}A\sum_kp_kc_k$.

</details>

3. **SCAFFOLD's fixed point.** assume SCAFFOLD (full participation, option II) is at a fixed point: $x$, $c$ and every $c_k$ unchanged by a round. show that $c_k = \nabla f_k(x)$ and $\nabla F(x) = 0$. (assume $\eta$ is small enough that each local step is a contraction.)

<details><summary>solution</summary>

$c_k^+ = c_k$ means $(x - y_k)/(K\eta) = c$ for every $k$. $x$ unchanged means $\sum_kp_k(y_k - x) = 0$, so $c = 0$ and therefore $y_k = x$: $K$ corrected steps from $x$ return to $x$. the corrected step $y \mapsto y - \eta(\nabla f_k(y) - c_k)$ is a contraction, so the only point it returns to after $K$ steps is its own fixed point: $\nabla f_k(x) = c_k$. finally $\nabla F(x) = \sum_kp_k\nabla f_k(x) = \sum_kp_kc_k = c = 0$ (the server keeps $c = \sum_kp_kc_k$ because both start at zero and move by the same weighted increments).

</details>

4. **FedProx in the limit.** with $K \to \infty$, client $k$ minimizes $f_k(w) + \frac\mu2\|w - w^t\|^2$ exactly. in one dimension ($f_k = \frac{a_k}{2}(w - c_k)^2$), find the fixed point and show it tends to $w^{\text{opt}}$ as $\mu \to \infty$ and to the average of the $c_k$ as $\mu \to 0$.

<details><summary>solution</summary>

the client returns $w_k = \frac{a_kc_k + \mu w^t}{a_k + \mu}$. at the fixed point $w = \sum_kp_kw_k$, so $w\sum_kp_k\frac{a_k}{a_k + \mu} = \sum_kp_k\frac{a_kc_k}{a_k + \mu}$: a weighted average of the $c_k$ with weights $\propto p_k\frac{a_k}{a_k + \mu}$. as $\mu \to \infty$ the weights become $\propto p_ka_k$, giving $w^{\text{opt}}$; as $\mu \to 0$ they become $p_k$, giving $\sum_kp_kc_k$. but the clients' progress per round, $w_k - w^t = \frac{a_k}{a_k + \mu}(c_k - w^t)$, shrinks like $1/\mu$: exactness costs speed.

</details>

## experiment

the lab runs your topic 27 `fedavg` on five splits of MNIST with increasing heterogeneity, from IID to one label per client, and plots accuracy against your heterogeneity score. experiment 2 computes FedAvg's, FedProx's and SCAFFOLD's limits on a random 10-client quadratic problem as the number of local steps grows (with a slider for $\mu$). experiment 3 plugs your FedProx into `fedavg` on the hardest MNIST split. experiment 4 compares the global model, fine-tuned models and local-only models on each client's own test data.

## check your understanding

1. why does the Dirichlet split draw one proportion vector per *class* rather than one label distribution per *client*?

<details><summary>answer</summary>

drawing per class makes every example land on exactly one client with no replacement, and client sizes come out of the draw (a known side effect). drawing a label distribution per client and then sampling examples would need sampling with replacement or careful bookkeeping to avoid using an example twice. both produce label skew; the per-class version (Hsu et al., 2019) became the convention, so results across papers are comparable.

</details>

2. FedAvg's drift on quadratics vanishes with $K = 1$. why not just use $K = 1$?

<details><summary>answer</summary>

$K = 1$ is FedSGD: one gradient step per round, so it needs as many rounds as gradient descent needs steps. the whole point of local steps was to cut rounds. drift is the price of that saving; the algorithms in this topic try to keep the saving and lose the price.

</details>

3. in experiment 2, FedAvg's error grows with $K$ and then levels off. what is the level, and is it a hard ceiling?

<details><summary>answer</summary>

the distance between $w^{\text{opt}}$ and $\sum_kp_kc_k$: the $K \to \infty$ limit, where every client reaches its own optimum. it is not a hard ceiling. for two clients in one dimension the fixed point slides monotonically from $w^{\text{opt}}$ to the average, so the limit is also the maximum. with more clients or more dimensions, clients converge at different rates along different directions, and at intermediate $K$ the fixed point can land further from the optimum than its limit. in random problems this happens often but usually by a fraction of a percent; it can be large when the average of the local optima happens to sit close to $w^{\text{opt}}$.

</details>

4. SCAFFOLD needs each client to keep state ($c_k$) across rounds. why is that a problem in cross-device federated learning?

<details><summary>answer</summary>

with millions of devices and small cohorts, a given phone participates rarely (perhaps a few times in all of training), so its $c_k$ is stale or uninitialized when it's next sampled. SCAFFOLD fits cross-silo settings (a few dozen hospitals that participate every round) much better than cross-device ones. the doubled upload is the other cost.

</details>

5. why might FedProx improve *stability* on real deployments even if it doesn't improve the final accuracy?

<details><summary>answer</summary>

with heterogeneous systems, some clients do far more local work than others, and without a proximal term the clients that did the most work dominate the average and can throw the model far from $w^t$. the proximal term caps each client's displacement relative to its progress, which smooths the rounds. Li et al. (2020) show the largest gains in exactly that regime, with many stragglers doing partial work.

</details>

6. a client's fine-tuned model beats the global model on the client's own test data. is that a reason to stop federating?

<details><summary>answer</summary>

no: the fine-tuned model started from the global model, which encodes everything learned from other clients. a local-only model would have to learn the shared structure from the client's data alone. the lab's experiment 4 shows when that works (very skewed clients with an easy local task) and when it fails badly (broad label mixes).

</details>

7. why report the mean per-client accuracy, and why is it not the same as accuracy on the pooled test set?

<details><summary>answer</summary>

it measures what a typical *user* experiences, with every client counted once. the pooled test set weights clients by how much data they contribute and uses the global label mix, which may not resemble any client. a model can be good on the pooled set and bad for many individual clients (or the reverse), so both numbers matter, and the per-client distribution (not just its mean) is what fairness questions are about.

</details>

## common bugs

- **the proximal term anchored to the wrong point.** symptom: FedProx pulls the model toward zero (anchored at nothing, like weight decay) or not at all (anchored to the current iterate, so the term's gradient is always zero). fix: snapshot the weights at the *start* of `fedprox_client_update` and keep that copy fixed.
- **one Dirichlet draw per client instead of per class.** symptom: with tiny $\alpha$, classes are spread across many clients; the "each class concentrates on one client" check fails. fix: loop over classes, one `rng.dirichlet` draw each.
- **an unbounded resampling loop.** symptom: `partition_dirichlet` hangs when `min_size` is impossible. fix: cap the attempts and raise.
- **SCAFFOLD with the server's $c$ updated by the plain mean.** symptom: SCAFFOLD converges to the wrong point when client weights differ. fix: the server's model and control-variate updates must use the same weights $p_k$, so that $c = \sum_kp_kc_k$ holds every round.
- **forgetting the $\eta\mu S$ term in FedProx's $T_k$.** symptom: the closed-form fixed point and the simulated iterates disagree when $\mu > 0$. fix: the anchor $w^t$ enters every local step, so it contributes to the linear part of the map.
- **fine-tuning the global model in place.** symptom: client 2's fine-tuned model starts from client 1's fine-tuned model. fix: `copy.deepcopy` first.

## going deeper

- Hsu, Qi & Brown (2019), *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*. the Dirichlet split and server momentum. [arXiv:1909.06335](https://arxiv.org/abs/1909.06335)
- Zhao, Li, Lai, Suda, Civin & Chandra (2018), *Federated Learning with Non-IID Data*. measured how FedAvg's accuracy degrades with label skew and traced it to weight divergence. [arXiv:1806.00582](https://arxiv.org/abs/1806.00582)
- Li, Sahu, Zaheer, Sanjabi, Talwalkar & Smith (2020), *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)
- Karimireddy, Kale, Mohri, Reddi, Stich & Suresh (2020), *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*, ICML. client drift and control variates. [arXiv:1910.06378](https://arxiv.org/abs/1910.06378)
- Pathak & Wainwright (2020), *FedSplit: An Algorithmic Framework for Fast Federated Optimization*, NeurIPS. shows FedAvg's fixed points need not be optima, even with exact local computation. [arXiv:2005.05238](https://arxiv.org/abs/2005.05238)
- Li, Diao, Chen & He (2021), *Federated Learning on Non-IID Data Silos: An Experimental Study*. a careful benchmark of FedAvg, FedProx, SCAFFOLD and FedNova across kinds of skew. [arXiv:2102.02079](https://arxiv.org/abs/2102.02079)
- Wang, Mathews, Kiddon, Eichner, Beaufays & Ramage (2019), *Federated Evaluation of On-device Personalization*; Arivazhagan, Aggarwal, Singh & Choudhary (2019), *Federated Learning with Personalization Layers*. [arXiv:1910.10252](https://arxiv.org/abs/1910.10252), [arXiv:1912.00818](https://arxiv.org/abs/1912.00818)
- Fallah, Mokhtari & Ozdaglar (2020), *Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach*, NeurIPS. personalization as meta-learning (the MAML elective, topic 09). [arXiv:2002.07948](https://arxiv.org/abs/2002.07948)

## connections

- ← builds on: your `fedavg`, `client_update` and `partition_shards` (topic 27); gradient descent on quadratics and its convergence (ch 06); fixed points of affine maps (ch 02); control variates (ch 05, ch 23).
- → used in: topic 29 reuses these partitions for DP-FedAvg and its privacy–accuracy–communication project; personalization by fine-tuning connects to the meta-learning elective (topics 09–11), where "a good starting point for fast adaptation" becomes the training objective itself.
