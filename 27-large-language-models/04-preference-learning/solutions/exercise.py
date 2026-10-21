import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # preference learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `token_logprobs`, `sequence_logprobs`: how likely a model finds a response → `mlfs.llm` |
    | 2 | `RewardModel`, `bradley_terry_loss`: a scalar score learned from pairwise preferences → `mlfs.llm` |
    | 3 | `dpo_loss`: direct preference optimization, the reward model folded into the policy → `mlfs.llm` |
    | 4 | `dpo_step`: one DPO update against a frozen reference model |
    | 5 | `kl_penalized_rewards`, `best_of_n_kl`: the KL-regularized reward of RLHF, and what best-of-n costs in KL |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def token_logprobs(model, input_ids, targets):
    """stage 1: (B, T) log p(targets[t] | input_ids[:t+1]) under the causal LM, 0.0 where targets == -100."""
    logits, _ = model(input_ids)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = targets != -100
    picked = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return picked * mask


@app.function
def sequence_logprobs(model, input_ids, targets):
    """stage 1: (B,) sum of log p(targets) over positions where targets != -100."""
    return token_logprobs(model, input_ids, targets).sum(dim=-1)


@app.cell(hide_code=True)
def _():
    checks.run(1, token_logprobs=token_logprobs, sequence_logprobs=sequence_logprobs)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class RewardModel(nn.Module):
    """a causal LM backbone plus a linear value head read at the last real token: (B,) scalar rewards."""

    def __init__(self, backbone, d_model):
        """stage 2: keep backbone (has hidden_states(idx)) and value_head = nn.Linear(d_model, 1)."""
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask=None):
        """stage 2: value_head(hidden state at the last True of attention_mask (B, T) bool; default the last position)."""
        h = self.backbone.hidden_states(input_ids)
        B, T, _ = h.shape
        if attention_mask is None:
            last = torch.full((B,), T - 1, dtype=torch.long, device=h.device)
        else:
            last = T - 1 - attention_mask.flip(-1).long().argmax(dim=-1)
        return self.value_head(h[torch.arange(B, device=h.device), last]).squeeze(-1)


@app.function
def bradley_terry_loss(chosen_rewards, rejected_rewards):
    """stage 2: -mean(logsigmoid(chosen - rejected))."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


@app.cell(hide_code=True)
def _():
    checks.run(2, RewardModel=RewardModel, bradley_terry_loss=bradley_terry_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, *, beta=0.1):
    """stage 3: -mean(logsigmoid(beta ((pi_c - ref_c) - (pi_r - ref_r)))); returns (loss, chosen_rewards, rejected_rewards)."""
    chosen = policy_chosen_logps - ref_chosen_logps.detach()
    rejected = policy_rejected_logps - ref_rejected_logps.detach()
    loss = -F.logsigmoid(beta * (chosen - rejected)).mean()
    return loss, (beta * chosen).detach(), (beta * rejected).detach()


@app.cell(hide_code=True)
def _():
    checks.run(3, dpo_loss=dpo_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def dpo_step(policy, reference, batch, optimizer, *, beta=0.1):
    """stage 4: one DPO update; reference in eval mode without gradients, policy in train mode; returns float metrics."""
    reference.eval()
    with torch.no_grad():
        ref_c = sequence_logprobs(reference, batch["chosen_ids"], batch["chosen_targets"])
        ref_r = sequence_logprobs(reference, batch["rejected_ids"], batch["rejected_targets"])
    policy.train()
    pol_c = sequence_logprobs(policy, batch["chosen_ids"], batch["chosen_targets"])
    pol_r = sequence_logprobs(policy, batch["rejected_ids"], batch["rejected_targets"])
    loss, r_c, r_r = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "accuracy": (r_c > r_r).float().mean().item(), "margin": (r_c - r_r).mean().item(),
            "chosen_reward": r_c.mean().item(), "rejected_reward": r_r.mean().item()}


@app.cell(hide_code=True)
def _():
    checks.run(4, dpo_step=dpo_step, sequence_logprobs=sequence_logprobs, dpo_loss=dpo_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def kl_penalized_rewards(scores, policy_logps, ref_logps, mask, *, beta=0.1):
    """stage 5: (B, T) per-token rewards: -beta (logp_policy - logp_ref) on mask, plus scores at each row's last True; 0 off mask."""
    r = -beta * (policy_logps - ref_logps) * mask
    T = mask.shape[1]
    last = T - 1 - mask.flip(-1).long().argmax(dim=-1)
    r = r.clone()
    r[torch.arange(len(scores), device=r.device), last] += scores
    return r


@app.function
def best_of_n_kl(n):
    """stage 5: KL(best-of-n || base) = log n - (n - 1) / n, in nats."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.log(n) - (n - 1) / n


@app.cell(hide_code=True)
def _():
    checks.run(5, kl_penalized_rewards=kl_penalized_rewards, best_of_n_kl=best_of_n_kl)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import sequence_logprobs, RewardModel, bradley_terry_loss, dpo_loss` gives
    you the core of preference learning; the capstone aligns its assistant with DPO, and
    chapter 28 comes back to reward models when it studies reward hacking.
    """)
    return


if __name__ == "__main__":
    app.run()
