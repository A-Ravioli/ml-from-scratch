import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical

    from mlfs import checks, lesson
    from mlfs.deeprl import ActorCritic, RolloutCollector, compute_gae  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # trust regions & PPO · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `ppo_clip_loss`: the clipped surrogate objective, with its diagnostics → `mlfs.deeprl` |
    | 2 | `explained_variance`: how much of the returns' variance the critic explains → `mlfs.deeprl` |
    | 3 | `conjugate_gradient`, `fisher_vector_product`, `natural_gradient_step`: TRPO's trust-region step |
    | 4 | `ppo_update`: several epochs of minibatch updates on one batch |
    | 5 | `train_ppo`: the full PPO agent |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def ppo_clip_loss(log_probs, old_log_probs, advantages, *, clip_eps=0.2):
    """stage 1: -mean(min(r A, clip(r, 1 - eps, 1 + eps) A)) with r = exp(log_probs - old); (loss, info)."""
    log_ratio = log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    adv = advantages.detach()
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    loss = -surrogate.mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()
    return loss, {"clip_frac": clip_frac, "approx_kl": approx_kl}


@app.cell(hide_code=True)
def _():
    checks.run(1, ppo_clip_loss=ppo_clip_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def explained_variance(values, returns):
    """stage 2: 1 - Var(returns - values) / Var(returns), a Python float (nan if Var(returns) = 0)."""
    v = np.asarray(values.detach() if torch.is_tensor(values) else values, dtype=np.float64).ravel()
    r = np.asarray(returns.detach() if torch.is_tensor(returns) else returns, dtype=np.float64).ravel()
    var_r = r.var()
    return float("nan") if var_r == 0 else float(1.0 - (r - v).var() / var_r)


@app.cell(hide_code=True)
def _():
    checks.run(2, explained_variance=explained_variance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def conjugate_gradient(Avp, b, *, n_iters=10, tol=1e-10):
    """stage 3: approximately solve A x = b for symmetric positive-definite A, given only v -> A v; returns x."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)
    for _ in range(n_iters):
        if rr <= tol:
            break
        Ap = Avp(p)
        alpha = rr / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = torch.dot(r, r)
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


@app.function
def fisher_vector_product(dist_fn, params, v, *, damping=0.0):
    """stage 3: (F + damping I) v, F the Hessian of mean KL(pi_old || pi_theta) at theta = theta_old; v flat."""
    dist = dist_fn()
    old = Categorical(logits=dist.logits.detach())
    kl = torch.distributions.kl_divergence(old, dist).mean()
    grads = torch.autograd.grad(kl, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    hv = torch.autograd.grad(torch.dot(flat, v), params)
    return torch.cat([h.reshape(-1) for h in hv]).detach() + damping * v


@app.function
def natural_gradient_step(g, fvp, max_kl, *, cg_iters=10):
    """stage 3: the step along F^-1 g whose quadratic KL estimate 0.5 s^T F s equals max_kl."""
    x = conjugate_gradient(fvp, g, n_iters=cg_iters)
    shs = torch.dot(x, fvp(x))
    return torch.sqrt(2 * max_kl / shs) * x


@app.cell(hide_code=True)
def _():
    checks.run(3, conjugate_gradient=conjugate_gradient, fisher_vector_product=fisher_vector_product,
               natural_gradient_step=natural_gradient_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ppo_update(model, optimizer, batch, *, n_epochs=10, minibatch_size=64, clip_eps=0.2, vf_coef=0.5,
               ent_coef=0.0, max_grad_norm=0.5, norm_adv=True, target_kl=None, seed=None):
    """stage 4: n_epochs passes of shuffled minibatch steps on the PPO loss; returns a dict of diagnostics."""
    rng = np.random.default_rng(seed)
    n = len(batch["advantages"])
    infos = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    n_updates = 0
    for _ in range(n_epochs):
        epoch_kl = []
        perm = rng.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = torch.as_tensor(perm[start:start + minibatch_size])
            dist, values = model(batch["obs"][idx])
            log_probs = dist.log_prob(batch["actions"][idx])
            adv = batch["advantages"][idx]
            if norm_adv and len(idx) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss, info = ppo_clip_loss(log_probs, batch["log_probs"][idx], adv, clip_eps=clip_eps)
            value_loss = ((values - batch["returns"][idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            n_updates += 1
            for k, val in [("policy_loss", pg_loss.item()), ("value_loss", value_loss.item()),
                           ("entropy", entropy.item()), ("approx_kl", info["approx_kl"]),
                           ("clip_frac", info["clip_frac"])]:
                infos[k].append(val)
            epoch_kl.append(info["approx_kl"])
        if target_kl is not None and np.mean(epoch_kl) > target_kl:
            break
    stats = {k: float(np.mean(v)) for k, v in infos.items()}
    stats["n_updates"] = n_updates
    stats["explained_variance"] = explained_variance(batch["values"], batch["returns"])
    return stats


@app.cell(hide_code=True)
def _():
    checks.run(4, ppo_update=ppo_update)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_ppo(make_env, total_steps, *, n_envs=8, n_steps=128, gamma=0.99, lam=0.95, lr=3e-4, anneal_lr=True,
              n_epochs=10, minibatch_size=256, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
              norm_adv=True, target_kl=None, hidden_sizes=(64, 64), continuous=False, seed=None):
    """stage 5: PPO with GAE on n_envs copies of make_env(); returns (model, history)."""
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        envs = [make_env() for _ in range(n_envs)]
        obs_dim = int(np.prod(envs[0].observation_space.shape))
        act_dim = int(np.prod(envs[0].action_space.shape)) if continuous else envs[0].action_space.n
        model = ActorCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes, continuous=continuous)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        collector = RolloutCollector(envs, model, gamma=gamma, seed=int(rng.integers(2**31)))
        n_updates = max(1, total_steps // (n_envs * n_steps))
        history = {"lr": [], "approx_kl": [], "clip_frac": [], "explained_variance": [], "entropy": []}
        for k in range(n_updates):
            lr_now = lr * (1 - k / n_updates) if anneal_lr else lr
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            b = collector.collect(n_steps)
            adv, ret = compute_gae(b["rewards"], b["values"], b["dones"], b["last_value"], gamma=gamma, lam=lam)
            flat = {"obs": b["obs"].reshape(-1, obs_dim),
                    "actions": b["actions"].reshape((-1, act_dim) if continuous else (-1,)),
                    "log_probs": b["log_probs"].reshape(-1), "values": b["values"].reshape(-1),
                    "advantages": adv.reshape(-1), "returns": ret.reshape(-1)}
            stats = ppo_update(model, optimizer, flat, n_epochs=n_epochs, minibatch_size=minibatch_size,
                               clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef, max_grad_norm=max_grad_norm,
                               norm_adv=norm_adv, target_kl=target_kl, seed=int(rng.integers(2**31)))
            history["lr"].append(lr_now)
            for key in ("approx_kl", "clip_frac", "explained_variance", "entropy"):
                history[key].append(stats[key])
        history["episode_returns"] = collector.episode_returns
        history["episode_ends"] = collector.episode_ends
    return model, history


@app.cell(hide_code=True)
def _():
    checks.run(5, train_ppo=train_ppo)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `ppo_clip_loss` and `explained_variance` are part of
    `mlfs.deeprl` (chapter 27's RLHF uses your `ppo_clip_loss` and `compute_gae`). open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
