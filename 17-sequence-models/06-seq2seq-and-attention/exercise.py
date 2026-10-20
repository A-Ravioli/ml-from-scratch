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
    # seq2seq & attention · build

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
        raise NotImplementedError("stage1: implement BahdanauAttention.__init__")

    def forward(self, query, keys, mask=None):
        """stage 1: query (B, Dq), keys (B, T, Dk), mask (B, T) bool (True = may attend) -> (context (B, Dk), weights (B, T))."""
        raise NotImplementedError("stage1: implement BahdanauAttention.forward")


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
        raise NotImplementedError("stage2: implement Seq2Seq.__init__")

    def encode(self, src, src_mask=None):
        """stage 2: (keys (B, S, H) = encoder outputs, initial decoder state (B, H) = the state at each last valid position, mask)."""
        raise NotImplementedError("stage2: implement Seq2Seq.encode")

    def decode_step(self, y_prev, state, keys, src_mask):
        """stage 2: one decoder step from y_prev (B,) -> (logits (B, V), new state (B, H), attention weights (B, S) or None)."""
        raise NotImplementedError("stage2: implement Seq2Seq.decode_step")

    def forward(self, src, tgt_in, *, src_mask=None):
        """stage 2: teacher forcing: logits (B, T, V) for every position of tgt_in; sets last_attention (B, T, S) or None."""
        raise NotImplementedError("stage2: implement Seq2Seq.forward")


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
    raise NotImplementedError("stage3: implement greedy_decode")


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
    raise NotImplementedError("stage4: implement beam_search")


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
