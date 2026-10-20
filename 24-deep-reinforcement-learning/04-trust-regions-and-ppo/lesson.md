# trust regions & PPO

## why this matters

A2C takes one gradient step per batch and throws the batch away. that's wasteful (the batch cost thousands of environment steps), but the obvious fix, taking many steps on the same batch, breaks the method: the policy gradient is only valid *at* the policy that collected the data, and after a few steps you're optimizing an estimate for a policy you no longer have. push hard enough and the policy collapses: one bad update makes it act badly, so the next batch is bad, so the next update is worse, and on-policy methods have no replay buffer to recover from.

**trust-region methods** make the big step safe by limiting how far the policy may move. TRPO (Schulman et al., 2015) does it properly: maximize a surrogate objective subject to a bound on the KL divergence between old and new policy, which needs second-order machinery (the Fisher information, conjugate gradients, a line search). **PPO** (Schulman et al., 2017) gets most of the benefit with one line: clip the importance ratio in the objective so that there's no incentive to move it beyond $1 \pm \varepsilon$, then just run several epochs of minibatch SGD. PPO became the default policy-gradient method, the "RL" in RLHF (ch 27), and a lesson in how much an algorithm's results depend on implementation details that never make it into the paper's pseudocode. in this topic you build both, and in the lab you'll find out that the clip doesn't actually keep the policy inside the trust region, and which of the details matter on CartPole.

## warm-up: predict first

1. you collected a batch with policy $\pi_{\text{old}}$. how can you evaluate (an estimate of) how good a *different* policy $\pi_\theta$ would be, using the same batch?
2. an action had advantage $+2$ and its probability ratio $\pi_\theta/\pi_{\text{old}}$ is already 1.5. with $\varepsilon = 0.2$, does PPO's loss still push the ratio up?
3. the same action, but the advantage is $-2$. does the loss push the ratio down?
4. after 10 epochs of PPO on one batch, what's the largest ratio $\pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$ you'd expect to find in the batch: about 1.2, about 2, or much more?
5. why might normalizing the advantages in each minibatch to mean 0, standard deviation 1 help, given that it changes the gradient's scale?

## prerequisites check

- [ ] you know importance sampling and why ratios of probabilities appear in off-policy estimators (→ ch 23 · topic 04)
- [ ] you know KL divergence and its second-order expansion (→ ch 05)
- [ ] you know the conjugate gradient method, or at least SPD linear systems (→ ch 06, ch 02)
- [ ] you've built `compute_gae`, `ActorCritic` and `RolloutCollector` (→ topic 03)

## core ideas

### idea 1: the clipped surrogate

write $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)$, the **probability ratio** of an action in the batch under the new and the old policy. by importance sampling, $\mathbb E_{\pi_{\text{old}}}[r_t(\theta)\hat A_t]$ estimates how much better than $\pi_{\text{old}}$ the policy $\pi_\theta$ would be, *if the state distribution didn't change* (warm-up 1; Kakade & Langford, 2002). that's the **surrogate objective** $L(\theta) = \hat{\mathbb E}_t[r_t(\theta)\hat A_t]$. at $\theta = \theta_{\text{old}}$ its gradient is exactly the policy gradient (on paper 1), so one step on it is A2C. but the surrogate is only trustworthy near $\theta_{\text{old}}$: the ignored change in the state distribution grows with the distance between the policies, and maximizing $L$ without limit drives ratios to extremes.

PPO's **clipped surrogate** removes the incentive to go far:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon)\hat A_t\big)\Big].$$

for a good action ($\hat A > 0$) the objective grows with $r$ until $r = 1 + \varepsilon$, then goes flat: no gradient pushes the ratio further up (warm-up 2). for a bad action ($\hat A < 0$) it's flat below $1 - \varepsilon$ (warm-up 3). the min makes it a *pessimistic* bound: if the ratio has moved in the direction that makes the objective *worse* (a good action that became less likely), the unclipped term is the smaller one and its gradient still applies, so mistakes can always be undone. you minimize $-L^{\text{CLIP}}$.

two diagnostics come free with every minibatch. the **clip fraction**, the share of samples with $|r - 1| > \varepsilon$, says how hard you're pushing against the clip. **approximate KL**: the KL divergence from the old policy to the new one, estimated from the batch as $\hat{\mathbb E}[(r - 1) - \log r]$, an unbiased and always-nonnegative estimator (on paper 4) that's far less noisy than the naive $\hat{\mathbb E}[-\log r]$. in RLHF these are the numbers people watch.

> **spec.** `ppo_clip_loss(log_probs, old_log_probs, advantages, *, clip_eps=0.2)`: tensors `(B,)`; `old_log_probs` and `advantages` are constants (detach them). returns `(loss, info)` with `loss` $= -L^{\text{CLIP}}$ (the mean over the batch) and `info` a dict of Python floats: `clip_frac` (fraction with $|r - 1| > \varepsilon$) and `approx_kl` (mean of $(r - 1) - \log r$).

> **🔮 predict:** five samples with ratios $(1.5, 0.5, 0.5, 1.5, 1.1)$ and advantages $(2, 1, -1, -2, 1)$, $\varepsilon = 0.2$. which samples get zero gradient? (the worked example has the answer.)

> **▶ build: stage 1.** implement `ppo_clip_loss`.

### idea 2: is the critic any good?

GAE's advantages are only as good as the critic, and a broken critic is one of the commonest silent failures in actor–critic code. the standard health check is the **explained variance** of the returns by the critic's values:

$$\text{EV} = 1 - \frac{\operatorname{Var}[\hat V - V_\phi]}{\operatorname{Var}[\hat V]}.$$

1 means the critic predicts the returns perfectly, 0 means it does no better than predicting their mean, and negative means it's worse than that. a healthy run climbs from around 0 towards 0.8–0.95; an EV stuck near 0 (or negative) means the advantages are mostly noise. it's cheap: use the rollout's values (the "old" values) and GAE's returns.

> **spec.** `explained_variance(values, returns)` accepts tensors or arrays, flattens them, and returns $1 - \operatorname{Var}[\text{returns} - \text{values}]/\operatorname{Var}[\text{returns}]$ as a Python float, or `nan` if the returns have zero variance.

> **▶ build: stage 2.** implement `explained_variance`.

### idea 3: the trust region, done properly

TRPO asks for the best surrogate inside a KL ball:

$$\max_\theta\ L(\theta)\quad\text{subject to}\quad\bar D_{\text{KL}}(\pi_{\text{old}}\,\|\,\pi_\theta) \le \delta,$$

with $\bar D_{\text{KL}}$ the KL averaged over the batch's states. near $\theta_{\text{old}}$, $L(\theta) \approx g^\top\Delta$ (with $g$ the policy gradient, $\Delta = \theta - \theta_{\text{old}}$) and the KL is quadratic, $\bar D_{\text{KL}} \approx \tfrac12\Delta^\top F\Delta$, where $F$ is its Hessian at $\theta_{\text{old}}$, which is also the **Fisher information** $\mathbb E[\nabla\log\pi\,\nabla\log\pi^\top]$ (on paper 3). maximizing a linear function in an ellipsoid has a closed form:

$$\Delta^* = \sqrt{\frac{2\delta}{g^\top F^{-1}g}}\;F^{-1}g,$$

the **natural gradient** direction $F^{-1}g$ (Amari, 1998; Kakade, 2001), scaled so the quadratic KL estimate equals $\delta$. the natural gradient measures step size in "how much the policy's behaviour changes", not in raw parameter distance, so it's invariant to how you parameterize the network.

for a network with $10^5$ parameters you can't form $F$ ($10^{10}$ entries), let alone invert it. two tricks make it feasible. **Fisher-vector products**: $Fv$ is the gradient of $(\nabla\bar D_{\text{KL}})\cdot v$, two backward passes, no matrix. **conjugate gradient**: CG solves $Fx = g$ using only products $v \mapsto Fv$, and 10 iterations get close enough. TRPO then line-searches along $\Delta^*$, shrinking it until the *actual* KL is below $\delta$ and the surrogate improves. you'll build the step (without the line search); PPO's clip is a first-order stand-in for all of it.

> **spec.** `conjugate_gradient(Avp, b, *, n_iters=10, tol=1e-10)`: CG from $x_0 = 0$ for a symmetric positive-definite $A$ given as a function `Avp(v) -> A v`; at most `n_iters` iterations (one `Avp` call each), stopping early once the squared residual norm is at most `tol`. returns $x$. `fisher_vector_product(dist_fn, params, v, *, damping=0.0)`: `dist_fn()` returns the current policy's `Categorical` over a batch of states; compute the mean $\text{KL}(\pi_{\text{old}}\,\|\,\pi_\theta)$ with $\pi_{\text{old}}$ the same distribution with detached logits, its gradient with respect to `params` (with `create_graph=True`), and the gradient of that flattened gradient dotted with `v` (a flat tensor). returns $(F + \text{damping}\cdot I)v$, flat and detached. `natural_gradient_step(g, fvp, max_kl, *, cg_iters=10)` returns $\Delta^*$ above, with $F^{-1}g$ from `conjugate_gradient(fvp, g, n_iters=cg_iters)`.

> **▶ build: stage 3.** implement `conjugate_gradient`, `fisher_vector_product` and `natural_gradient_step`.

### idea 4: the PPO update

PPO's update does what A2C couldn't: it reuses each batch for several **epochs**, each a pass over the batch in shuffled **minibatches**, with the loss

$$\mathcal L = -L^{\text{CLIP}} + c_v\,\hat{\mathbb E}\big[(V_\phi(s) - \hat V)^2\big] - c_e\,\hat{\mathbb E}\big[\mathcal H[\pi_\theta(\cdot\mid s)]\big].$$

the old log-probabilities are the ones stored at collection time; they stay fixed for the whole update (recomputing them each epoch would make every ratio 1 and turn PPO back into repeated A2C steps). three of the "details" are in the update itself (Huang et al., 2022):

- **advantage normalization**: standardize the minibatch's advantages to mean 0, std 1. the scale of returns changes by orders of magnitude during training (and across tasks); normalizing makes the policy step size independent of it (warm-up 5).
- **global gradient clipping** to norm 0.5, as in A2C.
- **early stopping on KL** (optional): after each epoch, stop if the mean approximate KL exceeds a target. the clip alone doesn't bound the KL (lab, experiment 1), so many implementations add this backstop.

> **spec.** `ppo_update(model, optimizer, batch, *, n_epochs=10, minibatch_size=64, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, norm_adv=True, target_kl=None, seed=None)`: `model(obs) -> (dist, value)`; `batch` is a dict of flat tensors `obs`, `actions`, `log_probs` (old), `values` (old), `advantages`, `returns`. for each epoch, shuffle with `np.random.default_rng(seed).permutation` and take minibatch steps on the loss above (the value term a plain MSE), with the minibatch's advantages standardized as `(A - A.mean()) / (A.std() + 1e-8)` if `norm_adv`, and the gradient norm clipped to `max_grad_norm` unless it's `None`. if `target_kl` is set and an epoch's mean `approx_kl` exceeds it, stop after that epoch. returns a dict with the mean over all steps of `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_frac` (Python floats), `n_updates` (the number of optimizer steps, an int), and `explained_variance` of the batch's old values and returns.

> **▶ build: stage 4.** implement `ppo_update`.

### idea 5: the PPO agent

the full agent is topic 03's loop with `ppo_update` in place of the single A2C step: collect $N \times T$ steps with the `RolloutCollector`, compute GAE, flatten, update, repeat. two more details live here: the learning rate is **annealed** linearly to 0 over training (late updates are small, so a good policy isn't knocked off its peak), and Adam's $\epsilon$ is set to $10^{-5}$ rather than PyTorch's default $10^{-8}$ (it slightly damps updates of parameters with tiny gradient variance). none of these are in the original paper's algorithm box, and Engstrom et al. (2020) showed that much of PPO's advantage over TRPO in benchmarks came from them.

> **spec.** `train_ppo(make_env, total_steps, *, n_envs=8, n_steps=128, gamma=0.99, lam=0.95, lr=3e-4, anneal_lr=True, n_epochs=10, minibatch_size=256, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, norm_adv=True, target_kl=None, hidden_sizes=(64, 64), continuous=False, seed=None)`: like `train_a2c`, with `max(1, total_steps // (n_envs * n_steps))` updates. torch is seeded from `seed` before building the `ActorCritic`; Adam with `eps=1e-5`; the collector and each `ppo_update` get seeds drawn from `np.random.default_rng(seed)`. before update $k$ (0-based) of $K$, set the learning rate to `lr * (1 - k / K)` if `anneal_lr`. returns `(model, history)` with the collector's `episode_returns` and `episode_ends`, and per-update lists `lr`, `approx_kl`, `clip_frac`, `explained_variance`, `entropy`.

> **🔮 predict:** which of these matters most for PPO on CartPole: advantage normalization, gradient clipping, learning-rate annealing, the number of epochs, or the clip itself?

> **▶ build: stage 5.** implement `train_ppo`.

## worked example

**the clipped loss.** ratios $r = (1.5, 0.5, 0.5, 1.5, 1.1)$, advantages $(2, 1, -1, -2, 1)$, $\varepsilon = 0.2$:

| sample | $rA$ | $\operatorname{clip}(r)A$ | min | gradient through $r$? |
|---|---|---|---|---|
| 1 | 3 | $1.2\cdot2 = 2.4$ | 2.4 | no: a good action already 50% more likely |
| 2 | 0.5 | $0.8\cdot1 = 0.8$ | 0.5 | yes: a good action that became less likely |
| 3 | −0.5 | $0.8\cdot(-1) = -0.8$ | −0.8 | no: a bad action already much less likely |
| 4 | −3 | $1.2\cdot(-2) = -2.4$ | −3 | yes: a bad action that became more likely |
| 5 | 1.1 | 1.1 | 1.1 | yes: inside the clip range |

the loss is $-(2.4 + 0.5 - 0.8 - 3 + 1.1)/5 = -0.04$. its gradient with respect to the log-probabilities is $-r_iA_i/5$ where the unclipped term is active and 0 elsewhere: $(0, -0.1, 0, 0.6, -0.22)$. the clip fraction is $4/5$ (it counts ratios outside the range, whether or not the clip was the active term).

**a natural-gradient step.** with $F = \begin{pmatrix}2 & 0.5\\0.5 & 1\end{pmatrix}$, $g = (1, -1)$ and $\delta = 0.01$: $F^{-1}g = \tfrac{1}{1.75}(1.5, -2.5) = (0.857, -1.429)$, $g^\top F^{-1}g = 2.286$, so $\Delta^* = \sqrt{0.02/2.286}\,(0.857, -1.429) = (0.0802, -0.1336)$. note it's not parallel to $g$: the natural gradient moves further in the direction where the policy is less sensitive (the second parameter, with the smaller Fisher entry).

your code must reproduce these numbers.

## on paper

1. **the surrogate's gradient at the old policy.** show that $\nabla_\theta\hat{\mathbb E}[r_t(\theta)\hat A_t]$ at $\theta = \theta_{\text{old}}$ equals the policy gradient $\hat{\mathbb E}[\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)]$, and that the same holds for $L^{\text{CLIP}}$.

<details><summary>solution</summary>

$\nabla_\theta r_t = r_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$, and $r_t = 1$ at $\theta_{\text{old}}$. for $L^{\text{CLIP}}$: at $r = 1$ the clip is inactive (for $\varepsilon > 0$), both terms of the min are equal with equal gradients, so the gradient is the same. one PPO step on a fresh batch, with plain SGD, is one A2C step (the checks test exactly this).

</details>

2. **the clip's piecewise shape.** for $\hat A > 0$, sketch $\min(r\hat A, \operatorname{clip}(r, 1 - \varepsilon, 1 + \varepsilon)\hat A)$ as a function of $r$; do the same for $\hat A < 0$. where is the gradient with respect to $r$ zero?

<details><summary>solution</summary>

$\hat A > 0$: equal to $r\hat A$ for $r \le 1 + \varepsilon$ (slope $\hat A$), then constant $(1 + \varepsilon)\hat A$: zero gradient for $r > 1 + \varepsilon$. (for $r < 1 - \varepsilon$ the unclipped term $r\hat A$ is the smaller one, so the slope stays $\hat A$.) $\hat A < 0$: constant $(1 - \varepsilon)\hat A$ for $r < 1 - \varepsilon$ (zero gradient), $r\hat A$ for $r \ge 1 - \varepsilon$. the objective is a lower bound on the unclipped surrogate, equal to it near $r = 1$.

</details>

3. **KL is locally quadratic, with the Fisher as Hessian.** show that $D_{\text{KL}}(\pi_{\theta_0}\,\|\,\pi_\theta)$ has zero value and zero gradient at $\theta = \theta_0$, and Hessian $\mathbb E_{a\sim\pi_{\theta_0}}[\nabla\log\pi_{\theta_0}(a)\nabla\log\pi_{\theta_0}(a)^\top]$.

<details><summary>solution</summary>

$D(\theta) = \sum_a\pi_0(a)[\log\pi_0(a) - \log\pi_\theta(a)]$. $\nabla D = -\sum_a\pi_0\nabla\log\pi_\theta$, which at $\theta_0$ is $-\sum_a\nabla\pi_0 = 0$. $\nabla^2D = -\sum_a\pi_0\nabla^2\log\pi_\theta$; at $\theta_0$, using $\nabla^2\log\pi = \nabla^2\pi/\pi - \nabla\log\pi\nabla\log\pi^\top$ and $\sum_a\nabla^2\pi_0 = 0$, it equals $\sum_a\pi_0\nabla\log\pi_0\nabla\log\pi_0^\top = F$. so $D \approx \tfrac12\Delta^\top F\Delta$, which is the constraint TRPO linearizes.

</details>

4. **the approximate-KL estimator.** with $r = \pi_\theta(a)/\pi_{\text{old}}(a)$ and $a \sim \pi_{\text{old}}$, show $\mathbb E[(r - 1) - \log r] = D_{\text{KL}}(\pi_{\text{old}}\,\|\,\pi_\theta)$, and that each term is nonnegative.

<details><summary>solution</summary>

$\mathbb E_{\text{old}}[r] = \sum_a\pi_\theta(a) = 1$, so $\mathbb E[r - 1] = 0$ and $\mathbb E[-\log r] = \mathbb E_{\text{old}}[\log\pi_{\text{old}} - \log\pi_\theta] = D_{\text{KL}}$. nonnegative because $\log r \le r - 1$ for all $r > 0$. adding the zero-mean control variate $r - 1$ to the naive estimator $-\log r$ removes most of its variance (Schulman, 2020, "Approximating KL divergence").

</details>

## experiment

`lab.py` investigates, with your code on your `CartPole`:

1. does the clip keep the new policy close to the old one? one batch, many epochs, with and without clipping: the largest ratio and the true KL after each update.
2. implementation details that matter: PPO with one detail removed at a time, over several seeds.

## check your understanding

1. why does PPO need a critic, while REINFORCE could do without?

<details><summary>answer</summary>

it doesn't strictly need one (you could use Monte Carlo returns), but reusing a batch for many epochs magnifies the variance of the advantages: every noisy weight is applied ten times. GAE with a critic gives low-variance advantages, which is what makes many epochs on one batch productive.

</details>

2. (the predict callout in idea 1.) which samples get zero gradient?

<details><summary>answer</summary>

samples 1 and 3: a good action whose ratio is already above $1 + \varepsilon$, and a bad action whose ratio is already below $1 - \varepsilon$. samples 2 and 4 moved the wrong way, so the pessimistic min keeps their (unclipped) gradient to correct them.

</details>

3. why is the clip fraction a useful thing to log?

<details><summary>answer</summary>

it tells you how hard the optimizer is pushing against the trust region. near 0: steps are tiny, you could use a larger learning rate or more epochs. above about 0.3: most samples have left the region, the update relies on the clip as a brake, and the KL is probably large; lower the learning rate or the number of epochs.

</details>

4. TRPO's step has a closed form; why does TRPO still need a line search?

<details><summary>answer</summary>

the closed form uses a linear model of the surrogate and a quadratic model of the KL. the real KL can be larger than its quadratic approximation, and the real surrogate can decrease along a step that the linear model says improves it. the backtracking line search checks both on the batch and shrinks the step until they hold.

</details>

5. what goes wrong if you compute the old log-probabilities by a forward pass at the start of *each epoch*?

<details><summary>answer</summary>

every epoch starts with ratios of exactly 1, so the clip never engages; each epoch is A2C-style steps from wherever the policy has drifted to, and nothing limits the drift over epochs. the update loses its trust region entirely.

</details>

6. explained variance is 0.9 but returns aren't improving. what might be going on?

<details><summary>answer</summary>

the critic is fine; look at the actor. common causes: entropy has collapsed (the policy is deterministic and can't explore), the learning rate has annealed to near zero, advantages are being normalized over a degenerate minibatch, or the clip fraction is near zero because the steps are tiny. EV only certifies the critic.

</details>

## common bugs

- **recomputing old log-probabilities each epoch.** symptom: clip fraction always 0, approx_kl tiny at the start of each epoch, unstable learning.
- **advantages with gradients.** symptom: the loss pushes the critic around through the policy term. detach (GAE under `no_grad` already does).
- **normalizing advantages over a minibatch of size 1.** symptom: `nan` (std of one number is undefined). skip normalization for tiny minibatches.
- **the ratio from probabilities.** `probs / old_probs` underflows; use `exp(log_probs - old_log_probs)`.
- **annealing to exactly 0 too early.** symptom: training stalls at whatever the policy reached by the middle of the run. anneal over the *planned* number of updates.
- **mixing up `min` and `max`.** symptom: the clipped term becomes optimistic and updates overshoot. the min is pessimistic: it keeps the unclipped term when that's *worse*.

## going deeper

- Schulman, Wolski, Dhariwal, Radford & Klimov (2017), "Proximal policy optimization algorithms". [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman, Levine, Moritz, Jordan & Abbeel (2015), "Trust region policy optimization", *ICML*. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- Kakade & Langford (2002), "Approximately optimal approximate reinforcement learning", *ICML*. the surrogate and its improvement bound.
- Kakade (2001), "A natural policy gradient", *NeurIPS* 14. the natural gradient for policies.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry (2020), "Implementation matters in deep policy gradients: a case study on PPO and TRPO", *ICLR*. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Andrychowicz et al. (2021), "What matters in on-policy reinforcement learning? A large-scale empirical study", *ICLR*. [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
- Huang, Dossa, Raffin, Kanervisto & Wang (2022), "The 37 implementation details of proximal policy optimization", *ICLR blog track*. [iclr-blog-track.github.io](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## connections

- ← builds on: importance sampling (ch 23 · topic 04), KL divergence (ch 05), conjugate gradients and SPD systems (ch 02, ch 06), GAE, `ActorCritic` and `RolloutCollector` (topic 03).
- → used in: the capstone's PPO agent; `ppo_clip_loss` and `compute_gae` are the optimizer of RLHF (ch 27 · topic 04), where the "actions" are tokens and a KL penalty to the reference model joins the clip.
