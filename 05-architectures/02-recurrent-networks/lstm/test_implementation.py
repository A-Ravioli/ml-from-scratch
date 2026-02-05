import numpy as np


def test_lstm_cell_shapes():
    from exercise import LSTMCell

    np.random.seed(0)
    cell = LSTMCell(input_size=5, hidden_size=7)
    x = np.random.randn(3, 5)
    h0 = np.random.randn(3, 7)
    c0 = np.random.randn(3, 7)
    h1, c1 = cell.forward(x, h0, c0)
    assert h1.shape == (3, 7)
    assert c1.shape == (3, 7)


def test_lstm_unidirectional_shapes():
    from exercise import LSTM

    np.random.seed(0)
    lstm = LSTM(input_size=4, hidden_size=6, num_layers=2, bidirectional=False)
    x = np.random.randn(2, 5, 4)
    y, (h, c) = lstm.forward(x)
    assert y.shape == (2, 5, 6)
    assert h.shape == (2, 2, 6)
    assert c.shape == (2, 2, 6)


def test_lstm_bidirectional_shapes():
    from exercise import LSTM

    np.random.seed(0)
    lstm = LSTM(input_size=3, hidden_size=5, num_layers=1, bidirectional=True)
    x = np.random.randn(2, 4, 3)
    y, (h, c) = lstm.forward(x)
    assert y.shape == (2, 4, 10)
    assert h.shape == (1, 2, 10)
    assert c.shape == (1, 2, 10)


def test_language_model_forward_and_generate():
    from exercise import LSTMLanguageModel

    np.random.seed(0)
    lm = LSTMLanguageModel(vocab_size=50, embed_dim=8, hidden_size=16, num_layers=1)
    input_ids = np.random.randint(0, 50, size=(2, 6))
    logits = lm.forward(input_ids)
    assert logits.shape == (2, 6, 50)

    np.random.seed(123)
    generated = lm.generate(start_token=1, length=12, temperature=1.0)
    assert len(generated) == 12
    assert all(0 <= t < 50 for t in generated)

