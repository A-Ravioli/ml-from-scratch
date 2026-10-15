import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # seq2seq & attention · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build an **encoder–decoder** that
    reads one sequence and writes another, with **Bahdanau attention** so the decoder can look
    back at any input position, and **beam search** to decode from it. these complete
    **`mlfs.seq`**; chapter 27 reuses your `beam_search` for language models.

    | stage | you build |
    |---|---|
    | 1 | `BahdanauAttention`: additive scores, masked softmax, a weighted sum of the keys |
    | 2 | `Seq2Seq`: GRU encoder, GRU decoder with teacher forcing, with or without attention |
    | 3 | `greedy_decode`: generate an output one token at a time |
    | 4 | `beam_search`: keep the `beam_size` best partial outputs instead of one |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class BahdanauAttention(nn.Module):
    """additive attention: score_j = v . tanh(W_q q + W_k k_j); weights = masked softmax; context = sum_j w_j k_j."""

    def __init__(self, query_dim, key_dim, hidden_dim):
        """stage 1: `W_q` = Linear(query_dim, hidden_dim, bias=False), `W_k` = Linear(key_dim, hidden_dim), `v` = Linear(hidden_dim, 1, bias=False)."""
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        """stage 1: query (B, Dq), keys (B, T, Dk), mask (B, T) bool (True = may attend) -> (context (B, Dk), weights (B, T))."""
        scores = self.v(torch.tanh(self.W_q(query).unsqueeze(1) + self.W_k(keys))).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


@app.cell(hide_code=True)
def _():
    checks.run(1, BahdanauAttention=BahdanauAttention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Seq2Seq(nn.Module):
    """GRU encoder -> GRU decoder (teacher forcing), optionally with Bahdanau attention over the encoder outputs."""

    def __init__(self, src_vocab_size, tgt_vocab_size, *, embed_dim=64, hidden_dim=128, attention=True):
        """stage 2: src_embed, tgt_embed, encoder (nn.GRU), decoder (nn.GRUCell), attn (or None), out (nn.Linear)."""
        super().__init__()
        self.hidden_dim, self.use_attention = hidden_dim, attention
        ctx = hidden_dim if attention else 0
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRUCell(embed_dim + ctx, hidden_dim)
        self.attn = BahdanauAttention(hidden_dim, hidden_dim, hidden_dim) if attention else None
        self.out = nn.Linear(hidden_dim + ctx, tgt_vocab_size)
        self.last_attention = None

    def encode(self, src, src_mask=None):
        """stage 2: (keys (B, S, H) = encoder outputs, initial decoder state (B, H) = the state at each last valid position, mask)."""
        if src_mask is None:
            src_mask = torch.ones(src.shape, dtype=torch.bool, device=src.device)
        keys, _ = self.encoder(self.src_embed(src))
        last = src_mask.long().sum(1) - 1  # index of each sequence's last real token
        state = keys[torch.arange(src.shape[0], device=src.device), last]
        return keys, state, src_mask

    def decode_step(self, y_prev, state, keys, src_mask):
        """stage 2: one decoder step from y_prev (B,) -> (logits (B, V), new state (B, H), attention weights (B, S) or None)."""
        emb = self.tgt_embed(y_prev)
        if self.attn is None:
            state = self.decoder(emb, state)
            return self.out(state), state, None
        context, weights = self.attn(state, keys, src_mask)  # Bahdanau: attend with the previous state
        state = self.decoder(torch.cat([emb, context], dim=-1), state)
        return self.out(torch.cat([state, context], dim=-1)), state, weights

    def forward(self, src, tgt_in, *, src_mask=None):
        """stage 2: teacher forcing: logits (B, T, V) for every position of tgt_in; sets last_attention (B, T, S) or None."""
        keys, state, src_mask = self.encode(src, src_mask)
        logits, attn = [], []
        for t in range(tgt_in.shape[1]):
            step_logits, state, w = self.decode_step(tgt_in[:, t], state, keys, src_mask)
            logits.append(step_logits)
            attn.append(w)
        self.last_attention = torch.stack(attn, dim=1).detach() if self.attn is not None else None
        return torch.stack(logits, dim=1)


@app.cell(hide_code=True)
def _():
    checks.run(2, Seq2Seq=Seq2Seq, BahdanauAttention=BahdanauAttention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def greedy_decode(model, src, bos_id, eos_id, *, max_len=50, src_mask=None):
    """stage 3: for each source row, [bos] + the argmax token at every step, stopping after eos or max_len tokens."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            keys, state, mask = model.encode(src, src_mask)
            B = src.shape[0]
            y = torch.full((B,), bos_id, dtype=torch.long, device=src.device)
            out = [[bos_id] for _ in range(B)]
            done = [False] * B
            for _ in range(max_len):
                logits, state, _ = model.decode_step(y, state, keys, mask)
                y = logits.argmax(-1)
                for b in range(B):
                    if not done[b]:
                        out[b].append(int(y[b]))
                        done[b] = int(y[b]) == eos_id
                if all(done):
                    break
    finally:
        model.train(was_training)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(3, greedy_decode=greedy_decode, Seq2Seq=Seq2Seq)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def beam_search(step_fn, bos_id, eos_id, *, beam_size=5, max_len=50, length_penalty=1.0):
    """stage 4: [(tokens incl. bos/eos, score)] best first; score = sum log p / (len(tokens) - 1) ** length_penalty."""
    alive, finished = [([bos_id], 0.0)], []
    for _ in range(max_len):
        if not alive or len(finished) >= beam_size:
            break
        lp = step_fn([seq for seq, _ in alive])
        lp = lp.detach().cpu().double().numpy() if torch.is_tensor(lp) else np.asarray(lp, dtype=np.float64)
        cand = (np.array([s for _, s in alive])[:, None] + lp).ravel()
        width = beam_size - len(finished)  # finished hypotheses keep their slots: the beam shrinks
        new_alive = []
        for flat in np.argsort(-cand, kind="stable")[:width]:  # ties: earlier beam, then smaller token id
            if not np.isfinite(cand[flat]):
                break
            i, tok = divmod(int(flat), lp.shape[1])
            hyp = (alive[i][0] + [tok], float(cand[flat]))
            (finished if tok == eos_id else new_alive).append(hyp)
        alive = new_alive
    results = [(seq, s / max(len(seq) - 1, 1) ** length_penalty) for seq, s in finished + alive]
    results.sort(key=lambda r: -r[1])
    return results[:beam_size]


@app.cell(hide_code=True)
def _():
    checks.run(4, beam_search=beam_search)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.seq import Seq2Seq, BahdanauAttention, beam_search` gives you these tools.
    chapter 18 replaces the recurrence with attention everywhere, and chapter 27 decodes
    language models with your `beam_search`.
    """)
    return


if __name__ == "__main__":
    app.run()
