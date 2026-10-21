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
    from mlfs.torchnn import RMSNorm  # yours, from ch 15 · 07
    from mlfs.transformer import RotaryEmbedding, repeat_kv, sample_next_token  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # modern LLM architecture · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `SwiGLU`: the gated feed-forward layer of Llama, PaLM and most LLMs since → `mlfs.llm` |
    | 2 | `GroupedQueryAttention`: RoPE attention whose keys and values are shared across groups of heads → `mlfs.llm` |
    | 3 | `MoE`: a mixture-of-experts layer with top-k routing and a load-balancing loss → `mlfs.llm` |
    | 4 | `LlamaBlock`: pre-norm RMSNorm, attention, SwiGLU or MoE → `mlfs.llm` |
    | 5 | `Llama`: the whole model, compatible with your chapter 18 KV cache and `generate_cached` → `mlfs.llm` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class SwiGLU(nn.Module):
    """the gated MLP: w_down(silu(w_gate x) * w_up x), no biases."""

    def __init__(self, d_model, d_ff=None):
        """stage 1: w_gate, w_up (d_model -> d_ff) and w_down (d_ff -> d_model); d_ff default int(8 d / 3) rounded up to a multiple of 8."""
        raise NotImplementedError("stage1: implement SwiGLU.__init__")

    def forward(self, x):
        """stage 1: (..., d_model) -> (..., d_model)."""
        raise NotImplementedError("stage1: implement SwiGLU.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, SwiGLU=SwiGLU)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GroupedQueryAttention(nn.Module):
    """causal self-attention with rotary positions and n_kv_heads key/value heads shared by groups of query heads."""

    def __init__(self, d_model, n_heads, n_kv_heads, *, rope_base=10000.0, max_seq_len=2048, dropout=0.0):
        """stage 2: q_proj, k_proj, v_proj, o_proj (no biases), rope = RotaryEmbedding(head_dim), save_weights = False; ValueError if heads don't divide."""
        raise NotImplementedError("stage2: implement GroupedQueryAttention.__init__")

    def forward(self, x, *, cache=None):
        """stage 2: x (B, T, d) -> (B, T, d); with a LayerKVCache, rotate at offset cache.seq_len and append the n_kv_heads k, v; save_weights stores last_weights."""
        raise NotImplementedError("stage2: implement GroupedQueryAttention.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, GroupedQueryAttention=GroupedQueryAttention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MoE(nn.Module):
    """a sparse mixture of SwiGLU experts: each token goes to its top_k experts by router probability."""

    def __init__(self, d_model, d_ff, n_experts, *, top_k=2):
        """stage 3: router (d_model -> n_experts, no bias) and experts (ModuleList of SwiGLU(d_model, d_ff)); ValueError if top_k > n_experts."""
        raise NotImplementedError("stage3: implement MoE.__init__")

    def forward(self, x):
        """stage 3: x (B, T, d) -> (out (B, T, d), aux_loss); each expert runs only on its tokens; Switch load-balancing aux."""
        raise NotImplementedError("stage3: implement MoE.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, MoE=MoE, SwiGLU=SwiGLU)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class LlamaBlock(nn.Module):
    """pre-norm block: x + attn(norm1(x)), then h + mlp(norm2(h)), with RMSNorm and SwiGLU (or MoE)."""

    def __init__(self, d_model, n_heads, n_kv_heads, *, d_ff=None, n_experts=0, top_k=2, max_seq_len=2048):
        """stage 4: norm1, attn (GroupedQueryAttention), norm2, mlp (SwiGLU, or MoE when n_experts > 0); aux_loss = 0.0."""
        raise NotImplementedError("stage4: implement LlamaBlock.__init__")

    def forward(self, x, *, cache=None):
        """stage 4: (B, T, d) -> (B, T, d); stores the MoE aux loss in self.aux_loss (0.0 for a dense block)."""
        raise NotImplementedError("stage4: implement LlamaBlock.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, LlamaBlock=LlamaBlock, MoE=MoE, SwiGLU=SwiGLU, GroupedQueryAttention=GroupedQueryAttention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class Llama(nn.Module):
    """a Llama-style decoder-only LM: token embedding, LlamaBlocks, RMSNorm, (tied) head; no position embedding (RoPE)."""

    def __init__(self, vocab_size, *, block_size=256, n_layers=4, n_heads=4, n_kv_heads=2, d_model=128, d_ff=None,
                 n_experts=0, top_k=2, aux_loss_coef=0.01, tie_weights=True):
        """stage 5: tok_emb, blocks, norm, lm_head (no bias); N(0, 0.02) init, residual outputs scaled by 1/sqrt(2 n_layers)."""
        raise NotImplementedError("stage5: implement Llama.__init__")

    def _stream(self, idx, cache=None):
        T = idx.shape[1]
        offset = cache.seq_len if cache is not None else 0
        if offset + T > self.block_size:
            raise ValueError(f"sequence of {offset + T} tokens is longer than block_size={self.block_size}")
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=None if cache is None else cache[i])
        return self.norm(x)

    def hidden_states(self, idx):
        """stage 5: idx (B, T) -> (B, T, d_model), the final-norm output that lm_head reads."""
        raise NotImplementedError("stage5: implement Llama.hidden_states")

    def forward(self, idx, targets=None, *, cache=None):
        """stage 5: (logits (B, T, V), loss or None); loss = CE (ignore -100) + aux_loss_coef * sum of blocks' aux_loss."""
        raise NotImplementedError("stage5: implement Llama.forward")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None):
        """stage 5: append max_new_tokens tokens sampled with sample_next_token; context cropped to block_size; no cache."""
        raise NotImplementedError("stage5: implement Llama.generate")


@app.cell(hide_code=True)
def _():
    checks.run(5, Llama=Llama, LlamaBlock=LlamaBlock, MoE=MoE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import Llama` is the model the rest of the chapter fine-tunes, aligns,
    decodes and evaluates, and chapter 28 opens up to look inside.
    """)
    return


if __name__ == "__main__":
    app.run()
