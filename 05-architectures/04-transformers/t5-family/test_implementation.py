import numpy as np

from exercise import RelativePositionBias, T5Attention, T5Model


def test_relative_position_bias_shape():
    bias = RelativePositionBias(num_heads=2, num_buckets=8, max_distance=16)
    out = bias.forward(query_length=4, key_length=4)
    assert out.shape == (1, 2, 4, 4)


def test_t5_attention_shape():
    np.random.seed(0)
    attn = T5Attention(d_model=8, num_heads=2)
    x = np.random.randn(2, 4, 8)
    out = attn.forward(x, x, x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_t5_model_forward():
    np.random.seed(0)
    model = T5Model(vocab_size=48, d_model=8, num_heads=2, d_ff=16, num_layers=2)
    encoder_ids = np.array([[1, 2, 3, 4]])
    decoder_ids = np.array([[0, 1, 2, 3]])
    logits = model.forward(encoder_ids, decoder_ids)
    assert logits.shape == (1, 4, 48)
