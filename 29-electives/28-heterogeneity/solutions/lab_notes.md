# heterogeneity · lab notes

numbers are from the reference solution on the cached MNIST data, one CPU thread (the lab takes 70–115 s depending on machine load). MNIST results use two seeds; the per-seed values are in each table. "accuracy, rounds 26–30" is the mean of the last five evaluations, which smooths out the round-to-round wobble of the skewed splits.

## experiment 1: how much skew does FedAvg tolerate?

**what people usually predict:** a steady decline with heterogeneity, already clearly visible at α = 1 (every client's label mix looks very different), with one label per client "not working at all", near 10%.

**what happens:**

| split | heterogeneity | accuracy, rounds 26–30 (per seed) | mean round-to-round change, last 10 rounds |
|---|---|---|---|
| IID (Dirichlet α = 100) | 0.04 | 0.885 (0.885, 0.885) | 0.003 |
| Dirichlet α = 1 | 0.35 | 0.880 (0.884, 0.876) | 0.004 |
| Dirichlet α = 0.1 | 0.71 | 0.813 (0.806, 0.821) | 0.029 |
| shards, 2 labels per client | 0.80 | 0.761 (0.759, 0.762) | 0.040 |
| shards, 1 label per client | 0.89 | 0.483 (0.492, 0.474) | 0.086 |

the decline is anything but steady. α = 1 costs half a point, inside the seed spread. the damage starts around heterogeneity 0.7 (α = 0.1: −7 points, with rounds wobbling by 3 points), grows at two labels per client (−12 points), and then falls off a cliff at one label per client: 48% averaged over rounds 26–30, with the test accuracy jumping by almost 9 points from one round to the next. that curve hasn't converged: it is still climbing at round 30 (its last evaluation is about 0.57), just slowly and erratically, so "far behind after 30 rounds" is the accurate summary, not "stuck at 48%".

**why:** at α = 1 every client still sees most classes, so its local optimum is a reasonable 10-class classifier, and the local trajectories stay compatible: averaging works as on IID data. with one label per client, 50 local steps on a single digit do something specific and destructive: the fastest way to lower the loss is to raise that class's output bias and its last-layer row until the model predicts that digit for every input. each client's update is mostly "make everything class $c_k$", and the 10 sampled clients cover only some of the classes, so each round's average favours whichever digits were sampled and penalizes the rest. the global model swings between rounds (the wobble) and makes progress only slowly, as the sampled classes rotate. this is client drift in its neural-network form: the local optima ("always say $c_k$") are far from each other and from the global optimum, and many local steps reach them.

## experiment 2: client drift, exactly

**what people usually predict:** FedAvg's error grows without bound in $K$, or stays small; FedProx with a large μ fixes it; SCAFFOLD "helps".

**what happens:** (10 random 5-dimensional quadratic clients, lr 0.1, μ = 1)

| K | FedAvg | FedProx μ = 1 | SCAFFOLD (400 rounds) |
|---|---|---|---|
| 1 | 1.3e-15 | 7.9e-16 | 8.0e-16 |
| 2 | 0.026 | 0.028 | 4e-16 |
| 5 | 0.096 | 0.094 | 5e-16 |
| 10 | 0.186 | 0.160 | 5e-16 |
| 20 | 0.300 | 0.206 | 2e-16 |
| 50 | 0.434 | 0.218 | 4e-16 |
| 100 | 0.475 | 0.218 | 6e-11 |
| 200 | 0.479 | 0.218 | 3e-6 |

the average of the local optima is 0.479 from the global optimum.

FedAvg's error is exactly zero at $K = 1$, grows with $K$, and saturates at 0.479, the distance of the plain average of the $c_k$ from the optimum. FedProx with μ = 1 is no better for small $K$ (slightly worse at $K = 2$) and levels off at 0.218, less than half of FedAvg's error but far from zero; larger μ lowers the plateau further (move the slider) but never reaches zero. SCAFFOLD's limit is the optimum to machine precision; at $K = 100$ and $200$ it simply hasn't finished converging after 400 rounds (run it longer and the residual keeps shrinking).

**why:** with $K \to \infty$ each FedAvg client lands on its own optimum, and the server averages the optima, ignoring that some clients' losses curve far more sharply than others (the true optimum weights client $k$ by $A_k$). FedProx's clients land on $\operatorname{prox}_{f_k/\mu}(w^t)$, a compromise between $c_k$ and the anchor, which reweights the clients by $A_k(A_k + \mu I)^{-1}$: closer to the right weighting, exact only as $\mu \to \infty$, where clients stop moving (on paper 4). the anchor is the round's starting point, not the optimum, so the leash limits the drift without pointing it in the right direction. SCAFFOLD's correction does point it in the right direction: at its fixed point each $c_k$ equals $\nabla f_k$ at the optimum and $c = 0$, so every client's corrected local step is a step on the global objective (on paper 3).

## experiment 3: does FedProx fix the broken case?

**what people usually predict:** a few points of improvement, the more the larger μ, since FedProx was designed for non-IID data.

**what happens:** (one label per client, E = 5, two seeds)

| μ | accuracy, rounds 26–30 (per seed) |
|---|---|
| 0 (FedAvg) | 0.483 (0.492, 0.474) |
| 0.1 | 0.468 (0.478, 0.458) |
| 1 | 0.393 (0.402, 0.384) |

FedProx doesn't help: μ = 0.1 is within noise of FedAvg (about 1.5 points lower), and μ = 1 is 9 points *worse*. the curves in the plot are almost on top of each other for the first 12 rounds and separate only as μ = 1 falls behind.

**why:** the proximal term doesn't change what each client is trying to do ("predict my digit"); it only limits how far it gets per round. on this problem the drift is not a small bias on top of useful progress, as in experiment 2: almost all of a client's progress *is* the drift. shrinking it shrinks the useful part too, and the round's average still favours the sampled classes, just more slowly. careful benchmarks find the same: FedProx rarely beats a well-tuned FedAvg on accuracy under label skew (Li et al., 2021, *Federated Learning on Non-IID Data Silos*). its design target was systems heterogeneity, clients doing very different amounts of local work, where the leash keeps the over-achievers from dominating; this lab gives every client the same $E$. a correction that targets the bias (SCAFFOLD, or server-side momentum, Hsu et al., 2019) or sharing a small amount of global data (Zhao et al., 2018) does better here.

## experiment 4: one model for everyone, or one each?

**what people usually predict:** the global model wins on IID data and fine-tuning wins on skewed data; local-only models, trained on 80 images, lose everywhere.

**what happens:** (mean per-client accuracy ± standard error over the 100 clients' local test sets, about 20 images each)

| split | global model, pooled test set | global | fine-tuned (all) | fine-tuned (head) | local only |
|---|---|---|---|---|---|
| shards, 2 labels per client | 0.719 | 0.773 ± 0.017 | 0.978 ± 0.004 | 0.974 ± 0.004 | 0.976 ± 0.004 |
| Dirichlet α = 1 | 0.878 | 0.907 ± 0.006 | 0.915 ± 0.006 | 0.918 ± 0.006 | 0.739 ± 0.013 |
| IID | 0.877 | 0.900 ± 0.007 | 0.888 ± 0.007 | 0.896 ± 0.007 | 0.677 ± 0.010 |

on the 2-label split, a model trained **locally only** on 80 images is as good as the fine-tuned federated model, about 98%, and both crush the global model's 77%. at α = 1 fine-tuning adds about a point (within two standard errors) and local-only training collapses to 74%; on IID data fine-tuning slightly *hurts* (−0.4 to −1.2 points) and local-only is worst of all.

**why:** a client with two digits faces a 2-class problem, which 80 images solve easily; the global model has to solve the 10-class problem and spends capacity on digits this client never sees, and its biases still reflect the last round's sampled classes. so on very skewed clients, *what* the model is asked to do matters more than how much data it saw, and the federation adds little for that client's own test distribution (though the local model is useless on any other digit). at α = 1 every client faces a near-10-class problem: the local model can't learn it from ~100 images, and the global model, already close to the client's mix, has little to gain from fine-tuning. on IID data the global model is already the best estimate for every client, so 20 noisy steps on 80 examples just add variance. head-only fine-tuning is the safer choice: it gets nearly all of the gain where there is one and loses less where there isn't.

the pooled-test-set accuracy (0.719 on the 2-label split) and the mean per-client accuracy (0.773) differ because each client's test set contains only its own two digits. a deployment should report both, and the spread across clients, not only the mean.
