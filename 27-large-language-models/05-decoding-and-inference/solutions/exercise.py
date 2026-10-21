import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # decoding & inference · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `apply_repetition_penalty`, `min_p_filter`: two more ways to shape the next-token distribution |
    | 2 | `speculative_accept`: the rejection step that makes a draft model's guesses exact |
    | 3 | `speculative_decode`: draft k tokens, verify them in one target pass → `mlfs.llm` |
    | 4 | `constrained_decode`: sampling only from tokens a grammar allows |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def apply_repetition_penalty(logits, prev_tokens, penalty):
    """stage 1: logits (B, V); for tokens in prev_tokens (B, L) divide positive logits by penalty, multiply negative ones."""
    if penalty <= 0:
        raise ValueError(f"penalty must be > 0, got {penalty}")
    out = logits.clone()
    seen = torch.zeros_like(out, dtype=torch.bool).scatter(-1, prev_tokens, True)
    out[seen] = torch.where(out[seen] > 0, out[seen] / penalty, out[seen] * penalty)
    return out


@app.function
def min_p_filter(logits, min_p):
    """stage 1: set logits to -inf where softmax prob < min_p * (max prob of that row); logits (B, V)."""
    if not 0.0 <= min_p <= 1.0:
        raise ValueError(f"min_p must be in [0, 1], got {min_p}")
    probs = F.softmax(logits.float(), dim=-1)
    keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(~keep, float("-inf"))


@app.cell(hide_code=True)
def _():
    checks.run(1, apply_repetition_penalty=apply_repetition_penalty, min_p_filter=min_p_filter)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def speculative_accept(p, q, draft_tokens, *, generator=None):
    """stage 2: p (k+1, V) target probs, q (k, V) draft probs, draft_tokens (k,) -> (n_accepted, next_token int)."""
    k = len(draft_tokens)
    dev = None if generator is None else generator.device  # draw where the generator lives (e.g. CPU, with a GPU model)

    def draw(probs):
        return int(torch.multinomial(probs if dev is None else probs.to(dev), 1, generator=generator))

    for i in range(k):
        x = int(draft_tokens[i])
        u = torch.rand((), generator=generator, device=dev).item()
        if u * q[i, x] < p[i, x]:  # accept with probability min(1, p/q)
            continue
        residual = (p[i] - q[i]).clamp(min=0)
        residual = residual / residual.sum() if residual.sum() > 0 else p[i]
        return i, draw(residual)
    return k, draw(p[k])


@app.cell(hide_code=True)
def _():
    checks.run(2, speculative_accept=speculative_accept)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
@torch.no_grad()
def speculative_decode(target, draft, idx, max_new_tokens, *, k=4, temperature=1.0, generator=None):
    """stage 3: batch 1; draft proposes k tokens, target verifies in one forward; exact; (tokens (1, T + max_new_tokens), acceptance_rate)."""
    if idx.shape[0] != 1:
        raise ValueError("speculative_decode works on one sequence (batch size 1)")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    was = (target.training, draft.training)
    target.eval()
    draft.eval()

    def probs(model, seq):
        bs = getattr(model, "block_size", None)
        logits, _ = model(seq if bs is None else seq[:, -bs:])
        return logits[0].float()

    seq = idx
    proposed = accepted = 0
    try:
        while seq.shape[1] - idx.shape[1] < max_new_tokens:
            n_draft = min(k, max_new_tokens - (seq.shape[1] - idx.shape[1]))
            drafted, qs = seq, []
            for _ in range(n_draft):
                logits = probs(draft, drafted)[-1]
                if temperature == 0:
                    tok = logits.argmax()
                    qs.append(F.one_hot(tok, logits.shape[-1]).float())
                else:
                    q = F.softmax(logits / temperature, dim=-1)
                    q_g = q if generator is None else q.to(generator.device)  # sample where the generator lives
                    tok = torch.multinomial(q_g, 1, generator=generator)[0]
                    qs.append(q)
                drafted = torch.cat([drafted, tok.view(1, 1).to(seq.device)], dim=1)
            t_logits = probs(target, drafted)[-(n_draft + 1):]  # the target's distribution at every drafted position
            guesses = drafted[0, seq.shape[1]:]
            if temperature == 0:
                best = t_logits.argmax(-1)
                n_ok = 0
                while n_ok < n_draft and int(best[n_ok]) == int(guesses[n_ok]):
                    n_ok += 1
                nxt = int(best[n_ok])
            else:
                p = F.softmax(t_logits / temperature, dim=-1)
                n_ok, nxt = speculative_accept(p, torch.stack(qs), guesses, generator=generator)
            proposed += n_draft
            accepted += n_ok
            new = torch.cat([guesses[:n_ok], torch.tensor([nxt], device=seq.device)])
            seq = torch.cat([seq, new.view(1, -1)], dim=1)
    finally:
        target.train(was[0])
        draft.train(was[1])
    return seq[:, :idx.shape[1] + max_new_tokens], accepted / max(proposed, 1)


@app.cell(hide_code=True)
def _():
    checks.run(3, speculative_decode=speculative_decode, speculative_accept=speculative_accept)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
@torch.no_grad()
def constrained_decode(model, idx, max_new_tokens, allowed, *, temperature=1.0, generator=None):
    """stage 4: batch 1; at each step sample only from tokens where allowed(generated: list[int]) (bool (V,)) is True; stop early if none."""
    if idx.shape[0] != 1:
        raise ValueError("constrained_decode works on one sequence (batch size 1)")
    was = model.training
    model.eval()
    seq, generated = idx, []
    try:
        for _ in range(max_new_tokens):
            mask = torch.as_tensor(allowed(list(generated)), dtype=torch.bool)
            if not mask.any():
                break
            bs = getattr(model, "block_size", None)
            logits = model(seq if bs is None else seq[:, -bs:])[0][0, -1].float()
            logits = logits.masked_fill(~mask.to(logits.device), float("-inf"))
            if temperature == 0:
                tok = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, -1)
                probs = probs if generator is None else probs.to(generator.device)  # sample where the generator lives
                tok = int(torch.multinomial(probs, 1, generator=generator))
            generated.append(tok)
            seq = torch.cat([seq, torch.tensor([[tok]], device=seq.device)], dim=1)
    finally:
        model.train(was)
    return seq


@app.cell(hide_code=True)
def _():
    checks.run(4, constrained_decode=constrained_decode)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import speculative_decode` generates from any pair of causal LMs with the
    same vocabulary, exactly as the big one would, only faster when the small one guesses well.
    """)
    return


if __name__ == "__main__":
    app.run()
