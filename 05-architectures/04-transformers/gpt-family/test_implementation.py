import numpy as np

from exercise import CausalMultiHeadAttention, GPTModel, GPTTrainer


def test_causal_attention_shape():
    np.random.seed(0)
    attn = CausalMultiHeadAttention(8, 2)
    x = np.random.randn(2, 5, 8)
    out = attn.forward(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_gpt_forward_and_generate():
    np.random.seed(0)
    model = GPTModel(vocab_size=32, d_model=8, num_heads=2, d_ff=16, num_layers=2, max_len=12)
    input_ids = np.array([[1, 2, 3, 4]])
    logits = model.forward(input_ids)
    generated = model.generate(input_ids, max_new_tokens=2, temperature=1.0)
    assert logits.shape == (1, 4, 32)
    assert generated.shape == (1, 6)


def test_gpt_trainer_loss():
    np.random.seed(0)
    model = GPTModel(vocab_size=24, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=10)
    trainer = GPTTrainer(model)
    tokens = np.array([[1, 2, 3, 4, 5]])
    loss = trainer.compute_loss(tokens, tokens)
    assert np.isfinite(loss)
    assert loss > 0.0
