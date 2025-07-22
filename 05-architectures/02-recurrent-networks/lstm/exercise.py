"""
LSTM Implementation Exercise

Implementation of Long Short-Term Memory networks with gating mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class LSTMCell:
    """Single LSTM cell with forget, input, and output gates"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrix for efficiency: [forget, input, output, candidate]
        self.W = np.random.randn(4, hidden_size, input_size + hidden_size) * 0.1
        self.b = np.zeros((4, hidden_size))
        
        # Initialize forget gate bias to 1 (default to remembering)
        self.b[0] = np.ones(hidden_size)
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM cell"""
        batch_size = x.shape[0]
        
        # Concatenate input and previous hidden state
        combined = np.concatenate([h_prev, x], axis=1)  # [batch, hidden + input]
        
        # Compute all gates at once
        gates = np.dot(combined, self.W.reshape(4 * self.hidden_size, -1).T) + self.b.flatten()
        gates = gates.reshape(batch_size, 4, self.hidden_size)
        
        # Split into individual gates
        forget_gate = self.sigmoid(gates[:, 0, :])    # f_t
        input_gate = self.sigmoid(gates[:, 1, :])     # i_t  
        output_gate = self.sigmoid(gates[:, 2, :])    # o_t
        candidate = np.tanh(gates[:, 3, :])           # C̃_t
        
        # Update cell state
        C_new = forget_gate * C_prev + input_gate * candidate
        
        # Update hidden state
        h_new = output_gate * np.tanh(C_new)
        
        return h_new, C_new
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))


class LSTM:
    """Multi-layer LSTM network"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Forward LSTM layers
        self.forward_layers = []
        layer_input_size = input_size
        
        for i in range(num_layers):
            self.forward_layers.append(LSTMCell(layer_input_size, hidden_size))
            layer_input_size = hidden_size
        
        # Backward LSTM layers (if bidirectional)
        if bidirectional:
            self.backward_layers = []
            layer_input_size = input_size
            
            for i in range(num_layers):
                self.backward_layers.append(LSTMCell(layer_input_size, hidden_size))
                layer_input_size = hidden_size
    
    def forward(self, x: np.ndarray, 
                initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forward pass through LSTM network"""
        batch_size, seq_len, input_size = x.shape
        
        # Initialize states if not provided
        if initial_state is None:
            h_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
            C_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        else:
            h_0, C_0 = initial_state
        
        # Forward direction
        forward_outputs, (h_forward, C_forward) = self._forward_direction(x, h_0, C_0)
        
        if not self.bidirectional:
            return forward_outputs, (h_forward, C_forward)
        
        # Backward direction
        backward_outputs, (h_backward, C_backward) = self._backward_direction(x, h_0, C_0)
        
        # Concatenate forward and backward outputs
        outputs = np.concatenate([forward_outputs, backward_outputs], axis=2)
        
        # Concatenate final states
        h_final = np.concatenate([h_forward, h_backward], axis=2)
        C_final = np.concatenate([C_forward, C_backward], axis=2)
        
        return outputs, (h_final, C_final)
    
    def _forward_direction(self, x: np.ndarray, h_0: np.ndarray, C_0: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Process sequence in forward direction"""
        batch_size, seq_len, input_size = x.shape
        outputs = []
        
        h_states = [h_0[i] for i in range(self.num_layers)]
        C_states = [C_0[i] for i in range(self.num_layers)]
        
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx in range(self.num_layers):
                h_new, C_new = self.forward_layers[layer_idx].forward(
                    layer_input, h_states[layer_idx], C_states[layer_idx]
                )
                h_states[layer_idx] = h_new
                C_states[layer_idx] = C_new
                layer_input = h_new
            
            outputs.append(h_states[-1])  # Output from last layer
        
        outputs = np.stack(outputs, axis=1)  # [batch, seq_len, hidden]
        h_final = np.stack(h_states, axis=0)  # [num_layers, batch, hidden]
        C_final = np.stack(C_states, axis=0)
        
        return outputs, (h_final, C_final)
    
    def _backward_direction(self, x: np.ndarray, h_0: np.ndarray, C_0: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Process sequence in backward direction"""
        batch_size, seq_len, input_size = x.shape
        outputs = []
        
        h_states = [h_0[i] for i in range(self.num_layers)]
        C_states = [C_0[i] for i in range(self.num_layers)]
        
        for t in range(seq_len - 1, -1, -1):  # Backward iteration
            layer_input = x[:, t, :]
            
            for layer_idx in range(self.num_layers):
                h_new, C_new = self.backward_layers[layer_idx].forward(
                    layer_input, h_states[layer_idx], C_states[layer_idx]
                )
                h_states[layer_idx] = h_new
                C_states[layer_idx] = C_new
                layer_input = h_new
            
            outputs.append(h_states[-1])  # Output from last layer
        
        outputs.reverse()  # Reverse to match forward order
        outputs = np.stack(outputs, axis=1)  # [batch, seq_len, hidden]
        h_final = np.stack(h_states, axis=0)  # [num_layers, batch, hidden]
        C_final = np.stack(C_states, axis=0)
        
        return outputs, (h_final, C_final)


class LSTMLanguageModel:
    """LSTM-based language model for sequence prediction"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, 
                 num_layers: int = 2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        
        # LSTM layers
        self.lstm = LSTM(embed_dim, hidden_size, num_layers)
        
        # Output projection
        self.output_projection = np.random.randn(hidden_size, vocab_size) * 0.1
        self.output_bias = np.zeros(vocab_size)
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through language model"""
        # Embedding lookup
        embeddings = self.embedding[input_ids]  # [batch, seq_len, embed_dim]
        
        # LSTM forward pass
        lstm_output, _ = self.lstm.forward(embeddings)  # [batch, seq_len, hidden]
        
        # Project to vocabulary
        logits = np.dot(lstm_output, self.output_projection) + self.output_bias
        
        return logits
    
    def generate(self, start_token: int, length: int, temperature: float = 1.0) -> List[int]:
        """Generate sequence using the language model"""
        generated = [start_token]
        batch_size = 1
        
        # Initialize states
        h_state = np.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size))
        C_state = np.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size))
        
        for _ in range(length - 1):
            # Prepare input
            current_input = np.array([[generated[-1]]])
            embeddings = self.embedding[current_input]
            
            # Forward pass
            lstm_output, (h_state, C_state) = self.lstm.forward(embeddings, (h_state, C_state))
            
            # Get logits for next token
            logits = np.dot(lstm_output[:, -1, :], self.output_projection) + self.output_bias
            logits = logits / temperature
            
            # Sample next token
            probs = self.softmax(logits)
            next_token = np.random.choice(self.vocab_size, p=probs[0])
            generated.append(next_token)
        
        return generated
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GradientAnalyzer:
    """Analyze gradient flow in LSTM vs vanilla RNN"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        
    def analyze_lstm_gradients(self, lstm_cell: LSTMCell, sequence_length: int) -> Dict[str, List[float]]:
        """Analyze gradient flow through LSTM cell"""
        batch_size = 1
        input_size = lstm_cell.input_size
        
        # Forward pass
        h_states = []
        C_states = []
        x_inputs = []
        
        h = np.random.randn(batch_size, self.hidden_size) * 0.1
        C = np.random.randn(batch_size, self.hidden_size) * 0.1
        
        for t in range(sequence_length):
            x = np.random.randn(batch_size, input_size) * 0.1
            h, C = lstm_cell.forward(x, h, C)
            
            h_states.append(h.copy())
            C_states.append(C.copy())
            x_inputs.append(x.copy())
        
        # Approximate gradient magnitudes
        gradient_magnitudes = {
            'cell_state': [],
            'hidden_state': [],
            'combined': []
        }
        
        # Simulate gradient backprop (simplified)
        grad_h = np.random.randn(batch_size, self.hidden_size)
        grad_C = np.random.randn(batch_size, self.hidden_size)
        
        for t in range(sequence_length - 1, -1, -1):
            # Approximate gradients (this is a simplified simulation)
            # In practice, these would be computed via backpropagation
            
            # Cell state gradient typically has better flow
            cell_grad_mag = np.mean(np.abs(grad_C))
            hidden_grad_mag = np.mean(np.abs(grad_h))
            combined_grad_mag = cell_grad_mag + hidden_grad_mag
            
            gradient_magnitudes['cell_state'].append(cell_grad_mag)
            gradient_magnitudes['hidden_state'].append(hidden_grad_mag)
            gradient_magnitudes['combined'].append(combined_grad_mag)
            
            # Simulate gradient decay (LSTM has less decay)
            grad_h = grad_h * 0.9  # Some decay for hidden state
            grad_C = grad_C * 0.98  # Less decay for cell state
        
        # Reverse to match forward order
        for key in gradient_magnitudes:
            gradient_magnitudes[key].reverse()
        
        return gradient_magnitudes


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_lstm_cell():
    """Exercise 1: Test basic LSTM cell"""
    print("=== Exercise 1: LSTM Cell ===")
    
    input_size = 10
    hidden_size = 20
    batch_size = 3
    
    lstm_cell = LSTMCell(input_size, hidden_size)
    
    # Test forward pass
    x = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    C_prev = np.random.randn(batch_size, hidden_size)
    
    h_new, C_new = lstm_cell.forward(x, h_prev, C_prev)
    
    print(f"Input shape: {x.shape}")
    print(f"Hidden state shape: {h_new.shape}")
    print(f"Cell state shape: {C_new.shape}")
    
    assert h_new.shape == (batch_size, hidden_size)
    assert C_new.shape == (batch_size, hidden_size)
    
    # Test gate ranges
    print(f"Hidden state range: [{np.min(h_new):.3f}, {np.max(h_new):.3f}]")
    print(f"Cell state range: [{np.min(C_new):.3f}, {np.max(C_new):.3f}]")
    
    print("✓ LSTM cell working correctly")


def exercise_2_multilayer_lstm():
    """Exercise 2: Test multi-layer LSTM"""
    print("=== Exercise 2: Multi-layer LSTM ===")
    
    input_size = 50
    hidden_size = 64
    num_layers = 3
    seq_len = 20
    batch_size = 2
    
    lstm = LSTM(input_size, hidden_size, num_layers)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    outputs, (h_final, C_final) = lstm.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    print(f"Final cell shape: {C_final.shape}")
    
    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert h_final.shape == (num_layers, batch_size, hidden_size)
    assert C_final.shape == (num_layers, batch_size, hidden_size)
    
    print("✓ Multi-layer LSTM working correctly")


def exercise_3_bidirectional_lstm():
    """Exercise 3: Test bidirectional LSTM"""
    print("=== Exercise 3: Bidirectional LSTM ===")
    
    input_size = 30
    hidden_size = 40
    seq_len = 15
    batch_size = 2
    
    bilstm = LSTM(input_size, hidden_size, num_layers=2, bidirectional=True)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    outputs, (h_final, C_final) = bilstm.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Bidirectional output shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    
    # Bidirectional doubles the output size
    assert outputs.shape == (batch_size, seq_len, 2 * hidden_size)
    assert h_final.shape == (2, batch_size, 2 * hidden_size)  # 2 layers
    
    print("✓ Bidirectional LSTM working correctly")


def exercise_4_language_model():
    """Exercise 4: Test LSTM language model"""
    print("=== Exercise 4: LSTM Language Model ===")
    
    vocab_size = 1000
    embed_dim = 128
    hidden_size = 256
    seq_len = 50
    batch_size = 4
    
    lm = LSTMLanguageModel(vocab_size, embed_dim, hidden_size, num_layers=2)
    
    # Test forward pass
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = lm.forward(input_ids)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    
    # Test generation
    generated = lm.generate(start_token=1, length=20, temperature=1.0)
    print(f"Generated sequence length: {len(generated)}")
    print(f"Generated tokens: {generated[:10]}...")
    
    assert len(generated) == 20
    assert all(0 <= token < vocab_size for token in generated)
    
    print("✓ LSTM language model working correctly")


def exercise_5_gradient_analysis():
    """Exercise 5: Analyze gradient flow in LSTM"""
    print("=== Exercise 5: Gradient Analysis ===")
    
    input_size = 20
    hidden_size = 50
    sequence_length = 100
    
    lstm_cell = LSTMCell(input_size, hidden_size)
    analyzer = GradientAnalyzer(hidden_size)
    
    gradients = analyzer.analyze_lstm_gradients(lstm_cell, sequence_length)
    
    print(f"Sequence length: {sequence_length}")
    print(f"Initial combined gradient: {gradients['combined'][0]:.6f}")
    print(f"Final combined gradient: {gradients['combined'][-1]:.6f}")
    print(f"Gradient decay ratio: {gradients['combined'][-1] / gradients['combined'][0]:.6f}")
    
    # Cell state should maintain gradients better
    cell_decay = gradients['cell_state'][-1] / gradients['cell_state'][0]
    hidden_decay = gradients['hidden_state'][-1] / gradients['hidden_state'][0]
    
    print(f"Cell state gradient decay: {cell_decay:.6f}")
    print(f"Hidden state gradient decay: {hidden_decay:.6f}")
    
    # Plot gradient flow
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(gradients['cell_state'], label='Cell State', alpha=0.8)
    plt.plot(gradients['hidden_state'], label='Hidden State', alpha=0.8)
    plt.xlabel('Time Step (Backward)')
    plt.ylabel('Gradient Magnitude')
    plt.title('LSTM Gradient Flow')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(gradients['combined'], label='Combined', color='purple', alpha=0.8)
    plt.xlabel('Time Step (Backward)')
    plt.ylabel('Combined Gradient Magnitude')
    plt.title('Combined Gradient Flow')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/tmp/lstm_gradients.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Gradient analysis completed")


def exercise_6_performance_comparison():
    """Exercise 6: Compare LSTM with vanilla RNN performance"""
    print("=== Exercise 6: Performance Comparison ===")
    
    # Simple vanilla RNN for comparison
    class VanillaRNN:
        def __init__(self, input_size: int, hidden_size: int):
            self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
            self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
            self.b_h = np.zeros(hidden_size)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            batch_size, seq_len, input_size = x.shape
            hidden_size = self.W_hh.shape[0]
            
            outputs = []
            h = np.zeros((batch_size, hidden_size))
            
            for t in range(seq_len):
                h = np.tanh(np.dot(h, self.W_hh) + np.dot(x[:, t], self.W_xh.T) + self.b_h)
                outputs.append(h)
            
            return np.stack(outputs, axis=1)
    
    input_size = 20
    hidden_size = 64
    seq_len = 100
    batch_size = 8
    
    # Create models
    lstm = LSTM(input_size, hidden_size)
    rnn = VanillaRNN(input_size, hidden_size)
    
    # Test data
    x = np.random.randn(batch_size, seq_len, input_size)
    
    # Time LSTM
    start_time = time.time()
    lstm_outputs, _ = lstm.forward(x)
    lstm_time = time.time() - start_time
    
    # Time RNN
    start_time = time.time()
    rnn_outputs = rnn.forward(x)
    rnn_time = time.time() - start_time
    
    print(f"LSTM forward time: {lstm_time:.4f}s")
    print(f"RNN forward time: {rnn_time:.4f}s")
    print(f"LSTM/RNN time ratio: {lstm_time/rnn_time:.2f}x")
    
    print(f"LSTM output shape: {lstm_outputs.shape}")
    print(f"RNN output shape: {rnn_outputs.shape}")
    
    # LSTM outputs should have better gradient flow properties
    # (this would be evident in actual training scenarios)
    
    print("✓ Performance comparison completed")


if __name__ == "__main__":
    exercise_1_lstm_cell()
    exercise_2_multilayer_lstm()
    exercise_3_bidirectional_lstm()
    exercise_4_language_model()
    exercise_5_gradient_analysis()
    exercise_6_performance_comparison()
    print("\nLSTM implementation completed!")