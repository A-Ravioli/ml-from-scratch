import numpy as np

from exercise import BertEmbeddings, BertLayer, BertPreTrainingModel


def test_bert_embeddings_shape():
    np.random.seed(0)
    embeddings = BertEmbeddings(vocab_size=32, d_model=8, max_len=10)
    input_ids = np.array([[1, 2, 3, 4]])
    token_types = np.zeros_like(input_ids)
    out = embeddings.forward(input_ids, token_types)
    assert out.shape == (1, 4, 8)


def test_bert_layer_shape():
    np.random.seed(0)
    layer = BertLayer(d_model=8, num_heads=2, d_ff=16)
    x = np.random.randn(2, 4, 8)
    out = layer.forward(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_bert_pretraining_heads():
    np.random.seed(0)
    model = BertPreTrainingModel(vocab_size=40, d_model=8, num_heads=2, d_ff=16, num_layers=2, max_len=10)
    input_ids = np.array([[1, 2, 3, 4]])
    mlm_logits, nsp_logits = model.forward(input_ids)
    assert mlm_logits.shape == (1, 4, 40)
    assert nsp_logits.shape == (1, 2)
