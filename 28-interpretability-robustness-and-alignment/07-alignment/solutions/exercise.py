import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.llm import accuracy_ci, paired_accuracy_diff  # yours, from ch 27 · 06
    from mlfs.rl import policy_evaluation, value_iteration  # yours, from ch 23 · 02-03
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # alignment · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. nothing in this topic becomes part of `mlfs`: it's a
    workbench for studying what goes wrong when you optimise a proxy.

    | stage | you build |
    |---|---|
    | 1 | `kl_regularized_policy`, `policy_stats`: optimising a proxy reward with a KL leash, in closed form |
    | 2 | `best_of_n_distribution`: exactly what best-of-n sampling does to a distribution |
    | 3 | `corridor_mdp`, `gaming_gap`, `rollout_states`: specification gaming in a tiny MDP |
    | 4 | `eval_suite`, `compare_agents`: a small behavioural eval with intervals |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def kl_regularized_policy(ref_probs, rewards, beta):
    """stage 1: pi(i) proportional to ref_probs(i) exp(rewards(i) / beta), the maximiser of E[r] - beta KL(pi || ref)."""
    p = np.asarray(ref_probs, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64)
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    logits = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) + r / beta, -np.inf)
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


@app.function
def policy_stats(policy, ref_probs, proxy, true):
    """stage 1: {'kl': KL(policy || ref) in nats, 'proxy': E_policy[proxy], 'true': E_policy[true]}."""
    pi = np.asarray(policy, dtype=np.float64)
    ref = np.asarray(ref_probs, dtype=np.float64)
    mask = pi > 0
    kl = float(np.sum(pi[mask] * (np.log(pi[mask]) - np.log(ref[mask]))))
    return {"kl": kl, "proxy": float(pi @ np.asarray(proxy, dtype=np.float64)),
            "true": float(pi @ np.asarray(true, dtype=np.float64))}


@app.cell(hide_code=True)
def _():
    checks.run(1, kl_regularized_policy=kl_regularized_policy, policy_stats=policy_stats)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def best_of_n_distribution(ref_probs, scores, n):
    """stage 2: P(best-of-n picks item i) for n samples from ref_probs, ties split in proportion to ref_probs."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    p = np.asarray(ref_probs, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(p)
    levels = np.unique(s)
    below = 0.0
    for v in levels:
        group = s == v
        mass = p[group].sum()
        p_max_here = (below + mass) ** n - below ** n  # P(the best sample's score is exactly v)
        if mass > 0:
            out[group] = p_max_here * p[group] / mass
        below += mass
    return out


@app.cell(hide_code=True)
def _():
    checks.run(2, best_of_n_distribution=best_of_n_distribution)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def corridor_mdp(length, bonus_pos, bonus, *, goal_reward=10.0):
    """stage 3: (P (S, 2, S), R_proxy (S, 2), R_true (S, 2)) for a corridor 0..length-1, actions left/right, goal at the end."""
    if length < 3 or not 0 < bonus_pos < length - 1:
        raise ValueError(f"need length >= 3 and 0 < bonus_pos < length - 1, got {length}, {bonus_pos}")
    S, goal = length, length - 1
    P = np.zeros((S, 2, S))
    R_true = np.zeros((S, 2))
    R_proxy = np.zeros((S, 2))
    for s in range(S):
        for a in range(2):
            if s == goal:  # absorbing, no reward
                P[s, a, s] = 1.0
                continue
            nxt = max(s - 1, 0) if a == 0 else s + 1
            P[s, a, nxt] = 1.0
            if nxt == goal:
                R_true[s, a] = goal_reward
            R_proxy[s, a] = R_true[s, a] + (bonus if nxt == bonus_pos and s != bonus_pos else 0.0)
    return P, R_proxy, R_true


@app.function
def gaming_gap(P, R_proxy, R_true, gamma=0.95, start=0):
    """stage 3: the proxy-optimal policy, its TRUE value at start, the true-optimal value at start, and the regret."""
    pi_proxy, _ = value_iteration(P, R_proxy, gamma)
    _, V_true = value_iteration(P, R_true, gamma)
    v = float(policy_evaluation(P, R_true, pi_proxy, gamma)[start])
    return {"policy": pi_proxy, "true_value": v, "optimal_value": float(V_true[start]),
            "regret": float(V_true[start]) - v}


@app.function
def rollout_states(P, policy, start, horizon):
    """stage 3: the states visited by a deterministic policy in a deterministic MDP: [start, s1, ..., s_horizon]."""
    states = [int(start)]
    for _ in range(horizon):
        s = states[-1]
        a = int(np.argmax(policy[s]))
        states.append(int(np.argmax(P[s, a])))
    return states


@app.cell(hide_code=True)
def _():
    checks.run(3, corridor_mdp=corridor_mdp, gaming_gap=gaming_gap, rollout_states=rollout_states)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def eval_suite(agent, cases, *, n_boot=2000, rng=None):
    """stage 4: {category: (pass rate, lo, hi)} plus 'overall', over cases (category, check); check(agent) -> bool;
    one shared generator, used for the categories in sorted order, then 'overall'."""
    if not cases:
        raise ValueError("an eval needs at least one case")
    rng = as_rng(rng)
    results = [(cat, bool(check(agent))) for cat, check in cases]
    out = {}
    for cat in sorted({c for c, _ in results}):
        out[cat] = accuracy_ci(np.array([ok for c, ok in results if c == cat], dtype=float), n_boot=n_boot, rng=rng)
    out["overall"] = accuracy_ci(np.array([ok for _, ok in results], dtype=float), n_boot=n_boot, rng=rng)
    return out


@app.function
def compare_agents(agent_a, agent_b, cases, *, n_boot=2000, rng=None):
    """stage 4: paired pass-rate difference a - b over the same cases, with a bootstrap interval: (diff, lo, hi)."""
    a = np.array([bool(check(agent_a)) for _, check in cases], dtype=float)
    b = np.array([bool(check(agent_b)) for _, check in cases], dtype=float)
    return paired_accuracy_diff(a, b, n_boot=n_boot, rng=rng)


@app.cell(hide_code=True)
def _():
    checks.run(4, eval_suite=eval_suite, compare_agents=compare_agents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then the chapter's
    capstone: train a small transformer on modular addition, watch it grok, and use this
    chapter's tools to read the algorithm it found.
    """)
    return


if __name__ == "__main__":
    app.run()
