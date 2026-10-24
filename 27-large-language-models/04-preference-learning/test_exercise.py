"""Checks for 27 · 04 preference learning.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. a tiny causal LM written here
(embeddings, a causal running mean, a head) stands in for your Llama. log-probabilities against a manual
gather; the reward model for which position it reads (padding), its head and its training; the Bradley-Terry
and DPO losses against their formulas (Rafailov et al., 2023), for beta, signs and the frozen reference;
the DPO step for what it updates, train/eval modes and the optimizer; the KL-penalized RLHF rewards and the
best-of-n KL against hand computations.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

V = 7


class _TinyLM(nn.Module):
    """causal LM protocol: hidden state at t = running mean of embeddings 0..t (then optional dropout)."""

    def __init__(self, d=8, dropout=0.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(V, d)
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d, V)

    def hidden_states(self, idx):
        e = self.emb(idx)
        h = e.cumsum(1) / torch.arange(1, idx.shape[1] + 1, device=idx.device)[None, :, None]
        return self.drop(torch.tanh(h))

    def forward(self, idx, targets=None, *, cache=None):
        logits = self.lm_head(self.hidden_states(idx))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
        return logits, loss


def _batch(seed=0, B=4, T=6):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    tg = torch.randint(0, V, (B, T), generator=g)
    tg[:, :2] = -100
    tg[0, -1] = -100
    return ids, tg


# ---------------------------------------------------------------- stage 1: log-probabilities


def test_stage1_logprobs(nb):
    m = _TinyLM()
    ids, tg = _batch()
    lp = F.log_softmax(m(ids)[0], -1)
    want_tok = torch.zeros(tg.shape)
    for b in range(tg.shape[0]):
        for t in range(tg.shape[1]):
            if tg[b, t] != -100:
                want_tok[b, t] = lp[b, t, tg[b, t]]
    tok = nb.token_logprobs(m, ids, tg)
    assert tok.shape == tg.shape and torch.allclose(tok, want_tok, atol=1e-5), (
        "token_logprobs[b, t] = log softmax(logits[b, t])[targets[b, t]] (targets are pre-shifted), and 0.0 where the "
        "target is -100")
    seq = nb.sequence_logprobs(m, ids, tg)
    assert seq.shape == (4,) and torch.allclose(seq, want_tok.sum(1), atol=1e-5), (
        "sequence_logprobs sums the log-probabilities of the non-ignored targets of each row")
    assert (seq <= 0).all()
    seq.sum().backward()
    assert m.lm_head.weight.grad is not None, "the log-probabilities are differentiable (DPO trains through them)"
    ids2 = ids.clone()
    ids2[1:] = 0
    assert torch.allclose(nb.sequence_logprobs(m, ids2, tg)[0], seq[0].detach(), atol=1e-5), "rows are independent"


# ---------------------------------------------------------------- stage 2: reward model


def test_stage2_reads_last_real_token(nb):
    torch.manual_seed(0)
    bb = _TinyLM()
    rm = nb.RewardModel(bb, 8)
    assert rm.backbone is bb and isinstance(rm.value_head, nn.Linear) and rm.value_head.weight.shape == (1, 8), (
        "keep the backbone and a value_head = nn.Linear(d_model, 1)")
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 4, 4, 4, 5]])
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    got = rm(ids, mask)
    h = bb.hidden_states(ids)
    want = torch.stack([rm.value_head(h[0, 2]), rm.value_head(h[1, 4])]).squeeze(-1)
    assert got.shape == (2,) and torch.allclose(got, want, atol=1e-6), (
        "the reward is the value head applied to the hidden state at the LAST True position of attention_mask "
        "(position 2 for the right-padded first row, 4 for the second)")
    assert torch.allclose(rm(ids[:, :3], mask[:, :3])[0], got[0], atol=1e-6), "padding after the last token doesn't matter"
    nomask = rm(ids)
    assert torch.allclose(nomask, rm.value_head(h[:, -1]).squeeze(-1), atol=1e-6), "no mask: the last position"
    gappy = torch.tensor([[True, False, True, False, False]])
    assert torch.allclose(rm(ids[:1], gappy), rm.value_head(h[0, 2]).squeeze(-1), atol=1e-6), (
        "the last True, not the number of Trues minus one")
    got.sum().backward()
    assert bb.emb.weight.grad is not None, "the backbone trains with the head"


def test_stage2_bradley_terry(nb):
    c, r = torch.tensor([2.0, 0.0, -1.0]), torch.tensor([0.0, 0.0, 1.0])
    got = nb.bradley_terry_loss(c, r)
    want = -(F.logsigmoid(torch.tensor(2.0)) + F.logsigmoid(torch.tensor(0.0)) + F.logsigmoid(torch.tensor(-2.0))) / 3
    assert torch.allclose(got, want), f"-mean(logsigmoid(chosen - rejected)) = {want.item():.4f}; got {got.item():.4f}"
    assert torch.allclose(nb.bradley_terry_loss(torch.zeros(5), torch.zeros(5)), torch.tensor(math.log(2))), (
        "equal rewards: P(chosen preferred) = 1/2, loss log 2")
    assert torch.allclose(nb.bradley_terry_loss(c + 3, r + 3), got), "only differences matter"


def test_stage2_learns_preferences(nb):
    # hidden rule: the response with more 3s is preferred
    rng = np.random.default_rng(0)
    seqs = torch.tensor(rng.integers(0, V, (400, 6)))
    score = (seqs == 3).sum(1)
    pairs = [(i, j) for i, j in rng.integers(0, 400, (600, 2)) if score[i] != score[j]]
    ch = torch.tensor([i if score[i] > score[j] else j for i, j in pairs])
    rj = torch.tensor([j if score[i] > score[j] else i for i, j in pairs])
    rm = nb.RewardModel(_TinyLM(seed=1), 8)
    opt = torch.optim.Adam(rm.parameters(), lr=0.03)
    for _ in range(150):
        loss = nb.bradley_terry_loss(rm(seqs[ch]), rm(seqs[rj]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (rm(seqs[ch]) > rm(seqs[rj])).float().mean().item()
    assert acc > 0.9, f"a reward model trained with your loss should rank 90% of pairs by the hidden rule; got {acc:.2f}"


# ---------------------------------------------------------------- stage 3: DPO loss


def test_stage3_formula(nb):
    pc, pr = torch.tensor([-5.0, -3.0, -8.0]), torch.tensor([-6.0, -2.0, -8.0])
    rc, rr = torch.tensor([-5.5, -3.0, -7.0]), torch.tensor([-5.5, -3.0, -9.0])
    loss, cr, rj = nb.dpo_loss(pc, pr, rc, rr, beta=0.5)
    margin = 0.5 * ((pc - rc) - (pr - rr))
    want = -F.logsigmoid(margin).mean()
    assert torch.allclose(loss, want), (
        f"DPO (Rafailov et al. 2023): -mean(logsigmoid(beta ((pi_c - ref_c) - (pi_r - ref_r)))) = {want.item():.4f}; "
        f"got {loss.item():.4f}")
    assert torch.allclose(cr, 0.5 * (pc - rc)) and torch.allclose(rj, 0.5 * (pr - rr)), (
        "chosen_rewards = beta (pi_c - ref_c), rejected_rewards = beta (pi_r - ref_r): DPO's implicit rewards")
    l01, _, _ = nb.dpo_loss(pc, pr, rc, rr)
    assert torch.allclose(l01, -F.logsigmoid(0.2 * margin).mean()), "default beta is 0.1"
    same, _, _ = nb.dpo_loss(rc, rr, rc, rr, beta=3.0)
    assert torch.allclose(same, torch.tensor(math.log(2))), "policy = reference: margin 0, loss log 2"


def test_stage3_gradients(nb):
    pc = torch.tensor([-4.0, -2.0], requires_grad=True)
    pr = torch.tensor([-3.0, -2.5], requires_grad=True)
    rc = torch.tensor([-4.0, -2.0], requires_grad=True)
    rr = torch.tensor([-3.0, -2.5], requires_grad=True)
    loss, cr, rj = nb.dpo_loss(pc, pr, rc, rr, beta=0.2)
    assert not cr.requires_grad and not rj.requires_grad, "the returned rewards are detached (they're for logging)"
    loss.backward()
    assert (pc.grad < 0).all() and (pr.grad > 0).all(), (
        "descending the loss raises the policy's log-prob of the chosen response and lowers the rejected one's")
    assert torch.allclose(pc.grad, torch.full((2,), -0.2 * 0.5 / 2)), "at margin 0: dL/dpi_c = -beta * sigmoid(0) / B"
    assert rc.grad is None or torch.all(rc.grad == 0), "the reference log-probs are constants: no gradient flows into them"


# ---------------------------------------------------------------- stage 4: DPO step


def _pref_batch():
    chosen = torch.tensor([[1, 2, 3, 3], [4, 3, 3, 0]])
    rejected = torch.tensor([[1, 2, 5, 5], [4, 6, 6, 0]])
    ct = torch.tensor([[-100, 3, 3, 6], [-100, 3, 0, -100]])
    rt = torch.tensor([[-100, 5, 5, 6], [-100, 6, 0, -100]])
    return {"chosen_ids": chosen, "chosen_targets": ct, "rejected_ids": rejected, "rejected_targets": rt}


def test_stage4_what_moves(nb):
    policy, ref = _TinyLM(seed=2), _TinyLM(seed=4, dropout=0.5)
    grad_modes = []
    ref.register_forward_hook(lambda mod, inp, out: grad_modes.append(torch.is_grad_enabled()))
    ref.train()
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    pol_before = {k: v.clone() for k, v in policy.state_dict().items()}
    batch = _pref_batch()
    ref.eval()
    with torch.no_grad():
        rc = nb.sequence_logprobs(ref, batch["chosen_ids"], batch["chosen_targets"])
        rr = nb.sequence_logprobs(ref, batch["rejected_ids"], batch["rejected_targets"])
        pc = nb.sequence_logprobs(policy, batch["chosen_ids"], batch["chosen_targets"])
        pr = nb.sequence_logprobs(policy, batch["rejected_ids"], batch["rejected_targets"])
    want = nb.dpo_loss(pc, pr, rc, rr, beta=0.3)[0].item()
    ref.train()
    policy.eval()  # dpo_step must switch it to train mode itself
    grad_modes.clear()
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    out = nb.dpo_step(policy, ref, batch, opt, beta=0.3)
    assert grad_modes and not any(grad_modes), "run the reference under torch.no_grad(): it's never trained"
    assert isinstance(out, dict) and all(isinstance(out[k], float) for k in ("loss", "accuracy", "margin")), (
        "dpo_step returns a dict of plain floats with at least 'loss', 'accuracy' and 'margin'")
    assert abs(out["loss"] - want) < 1e-5, (
        f"the loss must use the reference in eval mode (no dropout) and beta=0.3: want {want:.5f}, got {out['loss']:.5f}")
    assert not ref.training and policy.training, "after the step the reference is in eval mode, the policy in train mode"
    assert all(torch.equal(before[k], v) for k, v in ref.state_dict().items()), "the reference never changes"
    assert all(p.grad is None for p in ref.parameters()), "no gradients are computed for the reference"
    assert any(not torch.equal(pol_before[k], v) for k, v in policy.state_dict().items()), "the optimizer steps the policy"
    frozen = _TinyLM(seed=2)
    snap = {k: v.clone() for k, v in frozen.state_dict().items()}
    nb.dpo_step(frozen, _TinyLM(seed=2), batch, torch.optim.SGD(frozen.parameters(), lr=0.0))
    assert all(torch.equal(snap[k], v) for k, v in frozen.state_dict().items()), "use the optimizer you're given"
    stale = _TinyLM(seed=2)
    snap = {n: p.detach().clone() for n, p in stale.named_parameters()}
    for p in stale.parameters():
        p.grad = torch.full_like(p, 100.0)  # leftovers from some earlier backward pass
    nb.dpo_step(stale, _TinyLM(seed=4), batch, torch.optim.SGD(stale.parameters(), lr=1e-3))
    moved = max((p.detach() - snap[n]).abs().max().item() for n, p in stale.named_parameters())
    assert moved < 0.05, (
        f"a parameter moved by {moved:.3f}: stale gradients were added to this step's. zero the gradients before backward")


def test_stage4_learns(nb):
    policy, ref = _TinyLM(seed=3), _TinyLM(seed=3)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    batch = _pref_batch()
    hist = [nb.dpo_step(policy, ref, batch, opt, beta=0.5) for _ in range(30)]
    assert abs(hist[0]["loss"] - math.log(2)) < 1e-5, "first step: policy = reference, loss log 2"
    assert hist[0]["accuracy"] == 0.0, (
        "accuracy counts pairs whose chosen implicit reward is STRICTLY higher; at step 0 every pair ties at 0")
    assert hist[-1]["loss"] < 0.2 and hist[-1]["accuracy"] == 1.0 and hist[-1]["margin"] > 2, (
        f"30 Adam steps on one batch should drive the DPO loss below 0.2 with every chosen response ahead; got "
        f"{hist[-1]}")
    swapped = {"chosen_ids": batch["rejected_ids"], "chosen_targets": batch["rejected_targets"],
               "rejected_ids": batch["chosen_ids"], "rejected_targets": batch["chosen_targets"]}
    back = nb.dpo_step(policy, ref, swapped, torch.optim.SGD(policy.parameters(), lr=0.0), beta=0.5)
    assert back["margin"] < -2 and back["accuracy"] == 0.0, (
        f"with chosen and rejected swapped the margin (chosen - rejected implicit reward) is negative; got {back}")


# ---------------------------------------------------------------- stage 5: RLHF rewards and best-of-n


def test_stage5_kl_rewards(nb):
    scores = torch.tensor([1.0, -2.0])
    pol = torch.tensor([[-3.0, -1.0, -2.0, -0.5], [-1.0, -1.0, -1.0, 0.0]])
    ref = torch.tensor([[0.0, -1.5, -1.0, -2.5], [0.0, -2.0, -1.0, 0.0]])
    mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    got = nb.kl_penalized_rewards(scores, pol, ref, mask, beta=0.5)
    want = torch.tensor([[0.0, -0.25, 0.5 + 1.0, 0.0], [0.0, -0.5, 0.0, 0.0 - 2.0]])
    assert got.shape == (2, 4) and torch.allclose(got, want), (
        f"per-token reward -beta (logp - ref_logp) on response tokens, plus the score at each row's last response "
        f"token, 0 elsewhere: want {want.tolist()}, got {got.tolist()}")
    d = nb.kl_penalized_rewards(scores, pol, ref, mask)
    assert torch.allclose(d[0, 1], torch.tensor(-0.05)), "default beta 0.1"
    assert torch.allclose(got.sum(1), scores - 0.5 * ((pol - ref) * mask).sum(1)), (
        "each row sums to score - beta * (sequence log-ratio): the KL-regularized RLHF objective")


def test_stage5_best_of_n_kl(nb):
    assert nb.best_of_n_kl(1) == 0.0, "best of 1 is the base policy: KL 0"
    assert math.isclose(nb.best_of_n_kl(2), math.log(2) - 0.5), "n=2: log 2 - 1/2 = 0.193"
    assert math.isclose(nb.best_of_n_kl(64), math.log(64) - 63 / 64)
    vals = [nb.best_of_n_kl(n) for n in (1, 2, 4, 8, 16)]
    assert vals == sorted(vals), "more samples, more KL"
    with pytest.raises(ValueError):
        nb.best_of_n_kl(0)
