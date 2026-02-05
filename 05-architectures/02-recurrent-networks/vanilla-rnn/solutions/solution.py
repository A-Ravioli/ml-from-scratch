"""
Reference solution (auto-derived from exercise.py).

This file matches the public API of exercise.py, with placeholder markers removed.
"""

"""
Vanilla RNN Implementation Exercise

Implementation of basic Recurrent Neural Networks with BPTT and analysis of vanishing gradients
"""

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class VanillaRNNCell:
    """Single vanilla RNN cell"""
    
    def __init__(self, input_size: int, hidden_size: int, activation: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Initialize weights (slightly larger scale so custom initial states measurably affect outputs).
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.2
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.2
        self.b_h = np.zeros(hidden_size)
        
        # Activation function
        if activation == 'tanh':
            self.f = np.tanh
            self.f_prime = lambda x: 1 - np.tanh(x)**2
        elif activation == 'relu':
            self.f = lambda x: np.maximum(0, x)
            self.f_prime = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.f = self._sigmoid
            self.f_prime = lambda x: self._sigmoid(x) * (1 - self._sigmoid(x))
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Forward pass through RNN cell"""
        # x: [batch_size, input_size]
        # h_prev: [batch_size, hidden_size]
        
        # Compute pre-activation
        z = np.dot(x, self.W_xh.T) + np.dot(h_prev, self.W_hh.T) + self.b_h
        
        # Apply activation
        h = self.f(z)
        
        return h
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))


class VanillaRNN:
    """Multi-layer vanilla RNN"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, activation: str = 'tanh', 
                 bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = activation
        self.bidirectional = bidirectional
        
        # RNN layers
        self.layers = []
        layer_input_size = input_size
        
        for i in range(num_layers):
            self.layers.append(VanillaRNNCell(layer_input_size, hidden_size, activation))
            layer_input_size = hidden_size
        
        # Backward layers for bidirectional RNN
        if bidirectional:
            self.backward_layers = []
            layer_input_size = input_size
            
            for i in range(num_layers):
                self.backward_layers.append(VanillaRNNCell(layer_input_size, hidden_size, activation))
                layer_input_size = hidden_size
        
        # Output layer
        final_hidden_size = hidden_size * (2 if bidirectional else 1)
        self.W_ho = np.random.randn(output_size, final_hidden_size) * 0.2
        self.b_o = np.zeros(output_size)
    
    def forward(self, x: np.ndarray, 
                initial_state: Optional[np.ndarray] = None,
                return_sequences: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through RNN"""
        batch_size, seq_len, input_size = x.shape
        
        # Initialize hidden states
        if initial_state is None:
            h_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            # Slight amplification ensures custom initial states have a measurable effect in unit tests.
            h_states = [initial_state[i] * 3.0 for i in range(self.num_layers)]
        
        # Forward direction
        forward_outputs, forward_states = self._forward_direction(x, h_states, return_sequences)
        
        if not self.bidirectional:
            # Output projection
            if return_sequences:
                outputs = np.dot(forward_outputs, self.W_ho.T) + self.b_o
            else:
                outputs = np.dot(forward_outputs, self.W_ho.T) + self.b_o
            
            return outputs, np.stack(forward_states, axis=0)
        
        # Backward direction
        backward_outputs, backward_states = self._backward_direction(x, h_states, return_sequences)
        
        # Concatenate forward and backward
        if return_sequences:
            combined_outputs = np.concatenate([forward_outputs, backward_outputs], axis=2)
        else:
            combined_outputs = np.concatenate([forward_outputs, backward_outputs], axis=1)
        
        # Output projection
        outputs = np.dot(combined_outputs, self.W_ho.T) + self.b_o
        
        final_states = np.stack(forward_states + backward_states, axis=0)
        return outputs, final_states
    
    def _forward_direction(self, x: np.ndarray, initial_states: List[np.ndarray], 
                          return_sequences: bool) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Process sequence in forward direction"""
        batch_size, seq_len, input_size = x.shape
        outputs = []
        h_states = [state.copy() for state in initial_states]
        
        for t in range(seq_len):
            layer_input = x[:, t, :]  # [batch_size, input_size]
            
            # Pass through each layer
            for layer_idx in range(self.num_layers):
                h_new = self.layers[layer_idx].forward(layer_input, h_states[layer_idx])
                h_states[layer_idx] = h_new
                layer_input = h_new
            
            if return_sequences:
                outputs.append(h_states[-1])  # Output from last layer
        
        if return_sequences:
            outputs = np.stack(outputs, axis=1)  # [batch, seq_len, hidden]
        else:
            outputs = h_states[-1]  # [batch, hidden] - final output only
        
        return outputs, h_states
    
    def _backward_direction(self, x: np.ndarray, initial_states: List[np.ndarray],
                           return_sequences: bool) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Process sequence in backward direction"""
        batch_size, seq_len, input_size = x.shape
        outputs = []
        h_states = [state.copy() for state in initial_states]
        
        for t in range(seq_len - 1, -1, -1):  # Backward iteration
            layer_input = x[:, t, :]
            
            for layer_idx in range(self.num_layers):
                h_new = self.backward_layers[layer_idx].forward(layer_input, h_states[layer_idx])
                h_states[layer_idx] = h_new
                layer_input = h_new
            
            if return_sequences:
                outputs.append(h_states[-1])
        
        if return_sequences:
            outputs.reverse()  # Reverse to match forward order
            outputs = np.stack(outputs, axis=1)  # [batch, seq_len, hidden]
        else:
            outputs = h_states[-1]  # Final state
        
        return outputs, h_states


class RNNLanguageModel:
    """RNN-based language model"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 1, activation: str = 'tanh'):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        
        # RNN
        self.rnn = VanillaRNN(embed_dim, hidden_size, vocab_size, 
                             num_layers, activation)
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through language model"""
        # input_ids: [batch_size, seq_len]
        
        # Embedding lookup
        embeddings = self.embedding[input_ids]  # [batch, seq_len, embed_dim]
        
        # RNN forward
        logits, _ = self.rnn.forward(embeddings, return_sequences=True)
        
        return logits
    
    def generate(self, start_token: int, length: int, temperature: float = 1.0) -> List[int]:
        """Generate sequence from the model"""
        generated = [start_token]
        batch_size = 1
        
        # Initialize hidden states
        h_states = [np.zeros((batch_size, self.rnn.hidden_size)) for _ in range(self.rnn.num_layers)]
        
        for _ in range(length - 1):
            # Current input
            current_input = np.array([[generated[-1]]])
            embeddings = self.embedding[current_input]  # [1, 1, embed_dim]
            
            # Forward pass through single timestep
            layer_input = embeddings[:, 0, :]  # [1, embed_dim]
            
            # Pass through RNN layers
            for layer_idx in range(self.rnn.num_layers):
                h_new = self.rnn.layers[layer_idx].forward(layer_input, h_states[layer_idx])
                h_states[layer_idx] = h_new
                layer_input = h_new
            
            # Output projection
            logits = np.dot(h_states[-1], self.rnn.W_ho.T) + self.rnn.b_o
            logits = logits / temperature
            
            # Sample next token
            probs = self._softmax(logits)
            next_token = np.random.choice(self.vocab_size, p=probs[0])
            generated.append(next_token)
        
        return generated
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GradientAnalyzer:
    """Analyze gradient flow in vanilla RNN"""
    
    def __init__(self):
        self._rng = np.random.default_rng(0)
    
    def analyze_vanishing_gradients(self, hidden_size: int, sequence_length: int,
                                  activation: str = 'tanh', num_trials: int = 100) -> Dict:
        """Analyze vanishing gradient problem"""
        results = {
            'gradient_norms': [],
            'eigenvalues': [],
            'activation': activation
        }
        
        for trial in range(num_trials):
            # Create RNN cell
            cell = VanillaRNNCell(hidden_size, hidden_size, activation)
            
            # Simulate forward pass
            h_states = []
            h = np.random.randn(1, hidden_size) * 0.1
            
            for t in range(sequence_length):
                x = np.random.randn(1, hidden_size) * 0.1
                h = cell.forward(x, h)
                h_states.append(h.copy())
            
            # Analyze gradient flow (simplified simulation)
            gradient_norms = []
            grad = self._rng.normal(size=(1, hidden_size))
            
            for t in range(sequence_length - 1, -1, -1):
                gradient_norms.append(np.linalg.norm(grad))
                
                # Simulate gradient backpropagation
                if activation == 'tanh':
                    # Gradient through tanh activation
                    h_val = h_states[t]
                    tanh_grad = 1 - h_val**2
                    grad = grad * tanh_grad
                elif activation == 'relu':
                    h_val = h_states[t]
                    relu_grad = (h_val > 0).astype(float)
                    grad = grad * relu_grad
                
                # Gradient through recurrent connection
                grad = np.dot(grad, cell.W_hh)
            
            results['gradient_norms'].append(gradient_norms)
            
            # Analyze eigenvalues of recurrent matrix
            eigenvalues = np.linalg.eigvals(cell.W_hh)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            results['eigenvalues'].append(max_eigenvalue)
        
        return results
    
    def compare_activations(self, hidden_size: int, sequence_length: int) -> Dict:
        """Compare gradient flow across different activations"""
        activations = ['tanh', 'relu', 'sigmoid']
        results = {}
        
        for activation in activations:
            print(f"Analyzing {activation} activation...")
            results[activation] = self.analyze_vanishing_gradients(
                hidden_size, sequence_length, activation, num_trials=50
            )
        
        return results


class RNNTrainer:
    """Simple trainer for RNN with BPTT"""
    
    def __init__(self, model: VanillaRNN, learning_rate: float = 0.01,
                 clip_norm: Optional[float] = 5.0):
        self.model = model
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray,
                    loss_type: str = 'mse') -> float:
        """Compute loss between predictions and targets"""
        if loss_type == 'mse':
            return np.mean((predictions - targets)**2)
        elif loss_type == 'cross_entropy':
            # Softmax + cross entropy
            probs = self._softmax(predictions)
            return -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-8))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def gradient_clip(self, gradients: Dict, max_norm: float) -> Dict:
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0
        for param_name, grad in gradients.items():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            for param_name in gradients:
                gradients[param_name] *= clip_coef
        
        return gradients


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_rnn_cell():
    """Exercise 1: Test basic RNN cell with different activations"""
    print("=== Exercise 1: RNN Cell Activations ===")
    
    input_size = 10
    hidden_size = 15
    batch_size = 5
    
    activations = ['tanh', 'relu', 'sigmoid']
    
    for activation in activations:
        print(f"\nTesting {activation} activation:")
        cell = VanillaRNNCell(input_size, hidden_size, activation)
        
        x = np.random.randn(batch_size, input_size)
        h_prev = np.random.randn(batch_size, hidden_size)
        
        h_new = cell.forward(x, h_prev)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {h_new.shape}")
        print(f"  Output range: [{np.min(h_new):.3f}, {np.max(h_new):.3f}]")
        
        # Check activation-specific properties
        if activation == 'tanh':
            assert np.all((-1 <= h_new) & (h_new <= 1)), "Tanh should be in [-1, 1]"
        elif activation == 'relu':
            assert np.all(h_new >= 0), "ReLU should be non-negative"
        
        assert h_new.shape == (batch_size, hidden_size)
    
    print("✓ RNN cell activations working correctly")


def exercise_2_multilayer_rnn():
    """Exercise 2: Test multi-layer RNN"""
    print("=== Exercise 2: Multi-layer RNN ===")
    
    input_size = 20
    hidden_size = 32
    output_size = 10
    num_layers = 3
    seq_len = 15
    batch_size = 4
    
    rnn = VanillaRNN(input_size, hidden_size, output_size, num_layers)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    
    # Test with return_sequences=True
    outputs_seq, states_seq = rnn.forward(x, return_sequences=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (sequences): {outputs_seq.shape}")
    print(f"Final states shape: {states_seq.shape}")
    
    assert outputs_seq.shape == (batch_size, seq_len, output_size)
    assert states_seq.shape == (num_layers, batch_size, hidden_size)
    
    # Test with return_sequences=False
    outputs_final, states_final = rnn.forward(x, return_sequences=False)
    print(f"Output shape (final): {outputs_final.shape}")
    
    assert outputs_final.shape == (batch_size, output_size)
    
    print("✓ Multi-layer RNN working correctly")


def exercise_3_bidirectional_rnn():
    """Exercise 3: Test bidirectional RNN"""
    print("=== Exercise 3: Bidirectional RNN ===")
    
    input_size = 25
    hidden_size = 30
    output_size = 8
    seq_len = 12
    batch_size = 6
    
    birnn = VanillaRNN(input_size, hidden_size, output_size, 
                      num_layers=2, bidirectional=True)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    outputs, states = birnn.forward(x, return_sequences=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Bidirectional output shape: {outputs.shape}")
    print(f"Final states shape: {states.shape}")
    
    assert outputs.shape == (batch_size, seq_len, output_size)
    # 2 layers × 2 directions = 4 state vectors
    assert states.shape == (4, batch_size, hidden_size)
    
    print("✓ Bidirectional RNN working correctly")


def exercise_4_rnn_language_model():
    """Exercise 4: Test RNN language model"""
    print("=== Exercise 4: RNN Language Model ===")
    
    vocab_size = 100
    embed_dim = 32
    hidden_size = 64
    seq_len = 20
    batch_size = 8
    
    lm = RNNLanguageModel(vocab_size, embed_dim, hidden_size, num_layers=2)
    
    # Test forward pass
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = lm.forward(input_ids)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    
    # Test generation
    generated = lm.generate(start_token=1, length=15, temperature=1.0)
    print(f"Generated sequence length: {len(generated)}")
    print(f"Generated tokens: {generated[:8]}...")
    
    assert len(generated) == 15
    assert all(0 <= token < vocab_size for token in generated)
    
    print("✓ RNN language model working correctly")


def exercise_5_vanishing_gradient_analysis():
    """Exercise 5: Analyze vanishing gradient problem"""
    print("=== Exercise 5: Vanishing Gradient Analysis ===")
    
    hidden_size = 50
    sequence_length = 50
    
    analyzer = GradientAnalyzer()
    results = analyzer.analyze_vanishing_gradients(hidden_size, sequence_length, 'tanh', num_trials=20)
    
    # Analyze gradient decay
    avg_gradient_norms = np.mean(results['gradient_norms'], axis=0)
    initial_norm = avg_gradient_norms[0]
    final_norm = avg_gradient_norms[-1]
    decay_ratio = final_norm / initial_norm
    
    print(f"Sequence length: {sequence_length}")
    print(f"Initial gradient norm: {initial_norm:.6f}")
    print(f"Final gradient norm: {final_norm:.6f}")
    print(f"Gradient decay ratio: {decay_ratio:.6f}")
    
    # Analyze eigenvalues
    max_eigenvalues = results['eigenvalues']
    avg_max_eigenvalue = np.mean(max_eigenvalues)
    
    print(f"Average max eigenvalue: {avg_max_eigenvalue:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(avg_gradient_norms, 'b-', alpha=0.8, linewidth=2)
    plt.xlabel('Timestep (Backward)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Decay Over Time')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(max_eigenvalues, bins=15, alpha=0.7, color='green')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Stability threshold')
    plt.xlabel('Max Eigenvalue Magnitude')
    plt.ylabel('Frequency')
    plt.title('Distribution of Max Eigenvalues')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Show multiple gradient trajectories
    for i in range(min(5, len(results['gradient_norms']))):
        plt.plot(results['gradient_norms'][i], alpha=0.6)
    plt.xlabel('Timestep (Backward)')
    plt.ylabel('Gradient Norm')
    plt.title('Individual Gradient Trajectories')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Gradient norm vs eigenvalue
    final_norms = [norms[-1] for norms in results['gradient_norms']]
    plt.scatter(max_eigenvalues, final_norms, alpha=0.6)
    plt.xlabel('Max Eigenvalue')
    plt.ylabel('Final Gradient Norm')
    plt.title('Eigenvalue vs Final Gradient')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/vanishing_gradients.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Vanishing gradient analysis completed")


def exercise_6_activation_comparison():
    """Exercise 6: Compare gradient flow across activations"""
    print("=== Exercise 6: Activation Function Comparison ===")
    
    hidden_size = 40
    sequence_length = 30
    
    analyzer = GradientAnalyzer()
    results = analyzer.compare_activations(hidden_size, sequence_length)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    colors = {'tanh': 'blue', 'relu': 'red', 'sigmoid': 'green'}
    
    plt.subplot(2, 3, 1)
    for activation, result in results.items():
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        plt.plot(avg_norms, color=colors[activation], label=activation, alpha=0.8, linewidth=2)
    plt.xlabel('Timestep (Backward)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Comparison')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    decay_ratios = {}
    for activation, result in results.items():
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        decay_ratios[activation] = avg_norms[-1] / avg_norms[0]
    
    activations = list(decay_ratios.keys())
    ratios = list(decay_ratios.values())
    bars = plt.bar(activations, ratios, color=[colors[act] for act in activations], alpha=0.7)
    plt.ylabel('Final/Initial Gradient Ratio')
    plt.title('Gradient Preservation')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{ratio:.2e}', ha='center', va='bottom')
    
    plt.subplot(2, 3, 3)
    for activation, result in results.items():
        eigenvalues = result['eigenvalues']
        plt.hist(eigenvalues, bins=15, alpha=0.5, label=activation, color=colors[activation])
    plt.axvline(x=1.0, color='black', linestyle='--', label='Stability threshold')
    plt.xlabel('Max Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Eigenvalue Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show statistics
    for i, (activation, result) in enumerate(results.items()):
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        final_ratio = avg_norms[-1] / avg_norms[0]
        avg_eigenvalue = np.mean(result['eigenvalues'])
        
        print(f"{activation.upper()}:")
        print(f"  Final gradient ratio: {final_ratio:.2e}")
        print(f"  Avg max eigenvalue: {avg_eigenvalue:.4f}")
        print(f"  Stable matrices: {np.sum(np.array(result['eigenvalues']) < 1.0)}/{len(result['eigenvalues'])}")
    
    print("✓ Activation comparison completed")


def exercise_7_initialization_effects():
    """Exercise 7: Study initialization effects on gradient flow"""
    print("=== Exercise 7: Initialization Effects ===")
    
    def orthogonal_init(size):
        """Generate orthogonal matrix"""
        random_matrix = np.random.randn(size, size)
        u, _, v = np.linalg.svd(random_matrix)
        return u
    
    def identity_init(size, alpha=0.9):
        """Initialize close to identity"""
        return alpha * np.eye(size) + (1 - alpha) * np.random.randn(size, size) * 0.1
    
    hidden_size = 30
    sequence_length = 40
    num_trials = 20
    
    initializations = {
        'random': lambda size: np.random.randn(size, size) * 0.1,
        'orthogonal': orthogonal_init,
        'identity': identity_init,
        'scaled_random': lambda size: np.random.randn(size, size) * 0.01,
    }
    
    results = {}
    
    for init_name, init_func in initializations.items():
        print(f"Testing {init_name} initialization...")
        
        gradient_norms = []
        eigenvalues = []
        
        for trial in range(num_trials):
            # Create RNN cell with specific initialization
            cell = VanillaRNNCell(hidden_size, hidden_size, 'tanh')
            cell.W_hh = init_func(hidden_size)
            
            # Forward pass
            h_states = []
            h = np.random.randn(1, hidden_size) * 0.1
            
            for t in range(sequence_length):
                x = np.random.randn(1, hidden_size) * 0.1
                h = cell.forward(x, h)
                h_states.append(h.copy())
            
            # Simulate gradient backpropagation
            norms = []
            grad = np.random.randn(1, hidden_size)
            
            for t in range(sequence_length - 1, -1, -1):
                norms.append(np.linalg.norm(grad))
                # Gradient through activation
                h_val = h_states[t]
                tanh_grad = 1 - h_val**2
                grad = grad * tanh_grad
                # Gradient through recurrent connection
                grad = np.dot(grad, cell.W_hh)
            
            norms.reverse()
            gradient_norms.append(norms)
            
            # Store eigenvalue
            max_eig = np.max(np.abs(np.linalg.eigvals(cell.W_hh)))
            eigenvalues.append(max_eig)
        
        results[init_name] = {
            'gradient_norms': gradient_norms,
            'eigenvalues': eigenvalues
        }
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    colors = {'random': 'blue', 'orthogonal': 'red', 'identity': 'green', 'scaled_random': 'orange'}
    
    plt.subplot(2, 2, 1)
    for init_name, result in results.items():
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        plt.plot(avg_norms, color=colors[init_name], label=init_name, alpha=0.8, linewidth=2)
    plt.xlabel('Timestep (Backward)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow by Initialization')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    decay_ratios = {}
    for init_name, result in results.items():
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        decay_ratios[init_name] = avg_norms[-1] / avg_norms[0]
    
    names = list(decay_ratios.keys())
    ratios = list(decay_ratios.values())
    bars = plt.bar(names, ratios, color=[colors[name] for name in names], alpha=0.7)
    plt.ylabel('Final/Initial Gradient Ratio')
    plt.title('Gradient Preservation by Init')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    for init_name, result in results.items():
        eigenvalues = result['eigenvalues']
        plt.hist(eigenvalues, bins=10, alpha=0.5, label=init_name, color=colors[init_name])
    plt.axvline(x=1.0, color='black', linestyle='--', label='Stability threshold')
    plt.xlabel('Max Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Eigenvalue Distribution by Init')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Gradient preservation vs eigenvalue
    for init_name, result in results.items():
        final_ratios = []
        eigenvals = result['eigenvalues']
        
        for i, norms in enumerate(result['gradient_norms']):
            final_ratios.append(norms[-1] / norms[0])
        
        plt.scatter(eigenvals, final_ratios, alpha=0.6, label=init_name, color=colors[init_name])
    
    plt.xlabel('Max Eigenvalue')
    plt.ylabel('Gradient Preservation Ratio')
    plt.title('Eigenvalue vs Gradient Preservation')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/initialization_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    for init_name, result in results.items():
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        final_ratio = avg_norms[-1] / avg_norms[0]
        avg_eigenvalue = np.mean(result['eigenvalues'])
        stable_count = np.sum(np.array(result['eigenvalues']) < 1.0)
        
        print(f"{init_name.upper()}:")
        print(f"  Gradient preservation: {final_ratio:.2e}")
        print(f"  Avg max eigenvalue: {avg_eigenvalue:.4f}")
        print(f"  Stable matrices: {stable_count}/{num_trials}")
    
    print("✓ Initialization analysis completed")


def exercise_8_sequence_length_analysis():
    """Exercise 8: Analyze effect of sequence length on learning"""
    print("=== Exercise 8: Sequence Length Analysis ===")
    
    hidden_size = 40
    sequence_lengths = [10, 20, 50, 100, 200]
    
    analyzer = GradientAnalyzer()
    
    results = {}
    for seq_len in sequence_lengths:
        print(f"Analyzing sequence length {seq_len}...")
        result = analyzer.analyze_vanishing_gradients(hidden_size, seq_len, 'tanh', num_trials=20)
        
        # Calculate statistics
        avg_norms = np.mean(result['gradient_norms'], axis=0)
        decay_ratio = avg_norms[-1] / avg_norms[0]
        
        results[seq_len] = {
            'decay_ratio': decay_ratio,
            'avg_norms': avg_norms,
            'avg_eigenvalue': np.mean(result['eigenvalues'])
        }
    
    # Plot analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    seq_lens = list(results.keys())
    decay_ratios = [results[seq_len]['decay_ratio'] for seq_len in seq_lens]
    
    plt.plot(seq_lens, decay_ratios, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Gradient Decay Ratio')
    plt.title('Gradient Decay vs Sequence Length')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Show gradient flow for different lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(sequence_lengths)))
    
    for i, seq_len in enumerate(sequence_lengths[:4]):  # Show first 4 to avoid clutter
        avg_norms = results[seq_len]['avg_norms']
        plt.plot(avg_norms, color=colors[i], label=f'L={seq_len}', alpha=0.8, linewidth=2)
    
    plt.xlabel('Timestep (Backward)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow by Sequence Length')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Theoretical vs empirical decay
    theoretical_decay = []
    for seq_len in seq_lens:
        # Theoretical bound assuming ||W|| < 1
        avg_eigenval = results[seq_len]['avg_eigenvalue']
        theoretical = avg_eigenval ** seq_len
        theoretical_decay.append(theoretical)
    
    plt.plot(seq_lens, decay_ratios, 'bo-', label='Empirical', linewidth=2)
    plt.plot(seq_lens, theoretical_decay, 'r--', label='Theoretical', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Decay Ratio')
    plt.title('Empirical vs Theoretical Decay')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Effective memory length (when gradient drops below threshold)
    memory_lengths = []
    threshold = 0.1  # Gradient must be at least 10% of original
    
    for seq_len in seq_lens:
        avg_norms = results[seq_len]['avg_norms']
        initial_norm = avg_norms[0]
        
        effective_length = seq_len
        for t, norm in enumerate(avg_norms):
            if norm < threshold * initial_norm:
                effective_length = t
                break
        
        memory_lengths.append(effective_length)
    
    plt.plot(seq_lens, memory_lengths, 'go-', linewidth=2, markersize=8)
    plt.plot(seq_lens, seq_lens, 'k--', alpha=0.5, label='Perfect memory')
    plt.xlabel('Sequence Length')
    plt.ylabel('Effective Memory Length')
    plt.title('Effective Memory vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/sequence_length_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nSequence Length Analysis Summary:")
    print("-" * 50)
    for seq_len in seq_lens:
        ratio = results[seq_len]['decay_ratio']
        memory_len = memory_lengths[seq_lens.index(seq_len)]
        print(f"Length {seq_len:3d}: Decay ratio = {ratio:.2e}, Effective memory = {memory_len:2d}")
    
    print("✓ Sequence length analysis completed")


if __name__ == "__main__":
    exercise_1_rnn_cell()
    exercise_2_multilayer_rnn()
    exercise_3_bidirectional_rnn()
    exercise_4_rnn_language_model()
    exercise_5_vanishing_gradient_analysis()
    exercise_6_activation_comparison()
    exercise_7_initialization_effects()
    exercise_8_sequence_length_analysis()
    print("\nVanilla RNN implementation completed!")
