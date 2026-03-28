import numpy as np

from exercise import MultiHeadAttention, PositionalEncoding, Transformer


def test_positional_encoding_changes_values():
    x = np.zeros((2, 4, 8))
    pe = PositionalEncoding(8, max_len=10)
    out = pe.forward(x)
    assert out.shape == x.shape
    assert not np.allclose(out, x)


def test_multi_head_attention_shape():
    np.random.seed(0)
    attn = MultiHeadAttention(8, 2)
    x = np.random.randn(2, 4, 8)
    out = attn.forward(x, x, x)
    assert out.shape == (2, 4, 8)
    assert np.isfinite(out).all()


def test_transformer_forward_shape():
    np.random.seed(0)
    model = Transformer(32, 40, d_model=8, num_heads=2, d_ff=16, num_layers=2, max_len=10)
    src = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    tgt = np.array([[1, 2, 3, 0], [2, 3, 4, 0]])
    logits = model.forward(src, tgt)
    assert logits.shape == (2, 4, 40)
    assert np.isfinite(logits).all()
