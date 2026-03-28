import numpy as np

from exercise import EfficientTransformerBlock, LinearAttention, PerformerAttention


def test_linear_attention_shape():
    np.random.seed(0)
    attn = LinearAttention(d_model=8, num_heads=2)
    x = np.random.randn(2, 5, 8)
    out = attn.forward(x)
    assert out.shape == x.shape


def test_performer_feature_map_is_finite():
    np.random.seed(0)
    attn = PerformerAttention(d_model=8, num_heads=2, num_features=16)
    x = np.random.randn(2, 5, 2, 4)
    mapped = attn.feature_map(x)
    assert mapped.shape[-1] == 16
    assert np.isfinite(mapped).all()


def test_efficient_transformer_block_shape():
    np.random.seed(0)
    block = EfficientTransformerBlock(d_model=8, num_heads=2, d_ff=16, attention_type="linear")
    x = np.random.randn(2, 5, 8)
    out = block.forward(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()
