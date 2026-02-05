"""
LSTM reference solution.

Completed clone of `exercise.py`'s public API.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class LSTMCell:
    """Single LSTM cell with forget, input, and output gates."""

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        self.W = np.random.randn(4, self.hidden_size, self.input_size + self.hidden_size) * 0.1
        self.b = np.zeros((4, self.hidden_size), dtype=float)
        self.b[0] = np.ones(self.hidden_size, dtype=float)

    def forward(self, x: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = x.shape[0]
        combined = np.concatenate([h_prev, x], axis=1)
        gates = np.dot(combined, self.W.reshape(4 * self.hidden_size, -1).T) + self.b.flatten()
        gates = gates.reshape(batch_size, 4, self.hidden_size)

        forget_gate = self.sigmoid(gates[:, 0, :])
        input_gate = self.sigmoid(gates[:, 1, :])
        output_gate = self.sigmoid(gates[:, 2, :])
        candidate = np.tanh(gates[:, 3, :])

        C_new = forget_gate * C_prev + input_gate * candidate
        h_new = output_gate * np.tanh(C_new)
        return h_new, C_new

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class LSTM:
    """Multi-layer LSTM network."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        self.forward_layers: List[LSTMCell] = []
        layer_input_size = self.input_size
        for _ in range(self.num_layers):
            self.forward_layers.append(LSTMCell(layer_input_size, self.hidden_size))
            layer_input_size = self.hidden_size

        if self.bidirectional:
            self.backward_layers: List[LSTMCell] = []
            layer_input_size = self.input_size
            for _ in range(self.num_layers):
                self.backward_layers.append(LSTMCell(layer_input_size, self.hidden_size))
                layer_input_size = self.hidden_size

    def forward(
        self, x: np.ndarray, initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_size, seq_len, _ = x.shape
        if initial_state is None:
            h_0 = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
            C_0 = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            h_0, C_0 = initial_state

        forward_outputs, (h_forward, C_forward) = self._forward_direction(x, h_0, C_0)
        if not self.bidirectional:
            return forward_outputs, (h_forward, C_forward)

        backward_outputs, (h_backward, C_backward) = self._backward_direction(x, h_0, C_0)
        outputs = np.concatenate([forward_outputs, backward_outputs], axis=2)
        h_final = np.concatenate([h_forward, h_backward], axis=2)
        C_final = np.concatenate([C_forward, C_backward], axis=2)
        return outputs, (h_final, C_final)

    def _forward_direction(self, x: np.ndarray, h_0: np.ndarray, C_0: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        _, seq_len, _ = x.shape
        outputs = []
        h_states = [h_0[i] for i in range(self.num_layers)]
        C_states = [C_0[i] for i in range(self.num_layers)]
        for t in range(seq_len):
            layer_input = x[:, t, :]
            for layer_idx in range(self.num_layers):
                h_new, C_new = self.forward_layers[layer_idx].forward(layer_input, h_states[layer_idx], C_states[layer_idx])
                h_states[layer_idx] = h_new
                C_states[layer_idx] = C_new
                layer_input = h_new
            outputs.append(h_states[-1])
        outputs = np.stack(outputs, axis=1)
        h_final = np.stack(h_states, axis=0)
        C_final = np.stack(C_states, axis=0)
        return outputs, (h_final, C_final)

    def _backward_direction(self, x: np.ndarray, h_0: np.ndarray, C_0: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        _, seq_len, _ = x.shape
        outputs = []
        h_states = [h_0[i] for i in range(self.num_layers)]
        C_states = [C_0[i] for i in range(self.num_layers)]
        for t in range(seq_len - 1, -1, -1):
            layer_input = x[:, t, :]
            for layer_idx in range(self.num_layers):
                h_new, C_new = self.backward_layers[layer_idx].forward(layer_input, h_states[layer_idx], C_states[layer_idx])
                h_states[layer_idx] = h_new
                C_states[layer_idx] = C_new
                layer_input = h_new
            outputs.append(h_states[-1])
        outputs.reverse()
        outputs = np.stack(outputs, axis=1)
        h_final = np.stack(h_states, axis=0)
        C_final = np.stack(C_states, axis=0)
        return outputs, (h_final, C_final)


class LSTMLanguageModel:
    """LSTM-based language model for sequence prediction."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 2):
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.embedding = np.random.randn(self.vocab_size, self.embed_dim) * 0.1
        self.lstm = LSTM(self.embed_dim, int(hidden_size), int(num_layers))
        self.output_projection = np.random.randn(int(hidden_size), self.vocab_size) * 0.1
        self.output_bias = np.zeros(self.vocab_size, dtype=float)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        embeddings = self.embedding[input_ids]
        lstm_output, _ = self.lstm.forward(embeddings)
        logits = np.dot(lstm_output, self.output_projection) + self.output_bias
        return logits

    def generate(self, start_token: int, length: int, temperature: float = 1.0) -> List[int]:
        generated = [int(start_token)]
        batch_size = 1
        h_state = np.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), dtype=float)
        C_state = np.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), dtype=float)
        for _ in range(int(length) - 1):
            current_input = np.array([[generated[-1]]], dtype=int)
            embeddings = self.embedding[current_input]
            lstm_output, (h_state, C_state) = self.lstm.forward(embeddings, (h_state, C_state))
            logits = np.dot(lstm_output[:, -1, :], self.output_projection) + self.output_bias
            logits = logits / float(temperature)
            probs = self.softmax(logits)
            next_token = int(np.random.choice(self.vocab_size, p=probs[0]))
            generated.append(next_token)
        return generated

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GradientAnalyzer:
    """Analyze gradient flow in LSTM vs vanilla RNN (toy simulation)."""

    def __init__(self, hidden_size: int):
        self.hidden_size = int(hidden_size)

    def analyze_lstm_gradients(self, lstm_cell: LSTMCell, sequence_length: int) -> Dict[str, List[float]]:
        batch_size = 1
        input_size = lstm_cell.input_size
        h = np.random.randn(batch_size, self.hidden_size) * 0.1
        C = np.random.randn(batch_size, self.hidden_size) * 0.1
        for _ in range(int(sequence_length)):
            x = np.random.randn(batch_size, input_size) * 0.1
            h, C = lstm_cell.forward(x, h, C)

        gradient_magnitudes = {"cell_state": [], "hidden_state": [], "combined": []}
        grad_h = np.random.randn(batch_size, self.hidden_size)
        grad_C = np.random.randn(batch_size, self.hidden_size)
        for _ in range(int(sequence_length) - 1, -1, -1):
            cell_mag = float(np.mean(np.abs(grad_C)))
            hidden_mag = float(np.mean(np.abs(grad_h)))
            gradient_magnitudes["cell_state"].append(cell_mag)
            gradient_magnitudes["hidden_state"].append(hidden_mag)
            gradient_magnitudes["combined"].append(cell_mag + hidden_mag)
            grad_h = grad_h * 0.9
            grad_C = grad_C * 0.98
        for k in gradient_magnitudes:
            gradient_magnitudes[k].reverse()
        return gradient_magnitudes


# Exercise helpers are kept for parity with the curriculum.
def exercise_1_lstm_cell():
    lstm_cell = LSTMCell(10, 20)
    x = np.random.randn(3, 10)
    h_prev = np.random.randn(3, 20)
    C_prev = np.random.randn(3, 20)
    h_new, C_new = lstm_cell.forward(x, h_prev, C_prev)
    assert h_new.shape == (3, 20)
    assert C_new.shape == (3, 20)
    return h_new, C_new


def exercise_2_multilayer_lstm():
    lstm = LSTM(50, 64, num_layers=3)
    x = np.random.randn(2, 20, 50)
    outputs, (h_final, C_final) = lstm.forward(x)
    assert outputs.shape == (2, 20, 64)
    assert h_final.shape == (3, 2, 64)
    assert C_final.shape == (3, 2, 64)
    return outputs, (h_final, C_final)


def exercise_3_bidirectional_lstm():
    lstm = LSTM(20, 32, num_layers=2, bidirectional=True)
    x = np.random.randn(3, 10, 20)
    outputs, (h_final, C_final) = lstm.forward(x)
    assert outputs.shape == (3, 10, 64)
    assert h_final.shape == (2, 3, 64)
    return outputs, (h_final, C_final)


def exercise_4_language_model():
    lm = LSTMLanguageModel(100, 16, 32, num_layers=1)
    input_ids = np.random.randint(0, 100, (2, 5))
    logits = lm.forward(input_ids)
    assert logits.shape == (2, 5, 100)
    generated = lm.generate(start_token=1, length=10)
    assert len(generated) == 10
    return logits, generated


def exercise_5_gradient_analysis():
    lstm_cell = LSTMCell(20, 50)
    analyzer = GradientAnalyzer(50)
    gradients = analyzer.analyze_lstm_gradients(lstm_cell, 50)
    return gradients


def exercise_6_performance_comparison():
    class VanillaRNN:
        def __init__(self, input_size: int, hidden_size: int):
            self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
            self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
            self.b_h = np.zeros(hidden_size)

        def forward(self, x: np.ndarray) -> np.ndarray:
            batch_size, seq_len, _ = x.shape
            hidden_size = self.W_hh.shape[0]
            outputs = []
            h = np.zeros((batch_size, hidden_size))
            for t in range(seq_len):
                h = np.tanh(np.dot(h, self.W_hh) + np.dot(x[:, t], self.W_xh.T) + self.b_h)
                outputs.append(h)
            return np.stack(outputs, axis=1)

    input_size = 20
    hidden_size = 64
    seq_len = 50
    batch_size = 4
    lstm = LSTM(input_size, hidden_size)
    rnn = VanillaRNN(input_size, hidden_size)
    x = np.random.randn(batch_size, seq_len, input_size)
    start = time.time()
    lstm_outputs, _ = lstm.forward(x)
    lstm_time = time.time() - start
    start = time.time()
    rnn_outputs = rnn.forward(x)
    rnn_time = time.time() - start
    return {"lstm_time": lstm_time, "rnn_time": rnn_time, "ratio": lstm_time / max(1e-9, rnn_time)}


if __name__ == "__main__":
    exercise_1_lstm_cell()

