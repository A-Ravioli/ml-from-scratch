"""
GRU Implementation Exercise

Implementation of Gated Recurrent Units with reset and update gates
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class GRUCell:
    """Single GRU cell with reset and update gates"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrix for efficiency: [reset, update, candidate]
        self.W = np.random.randn(3, hidden_size, input_size + hidden_size) * 0.1
        self.b = np.zeros((3, hidden_size))
        
        # Initialize update gate bias to 1 (bias toward memory)
        self.b[1] = np.ones(hidden_size)
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Forward pass through GRU cell"""
        batch_size = x.shape[0]
        
        # Concatenate input and previous hidden state
        combined = np.concatenate([h_prev, x], axis=1)  # [batch, hidden + input]
        
        # Compute reset and update gates
        gates = np.dot(combined, self.W[:2].reshape(2 * self.hidden_size, -1).T) + self.b[:2].flatten()
        gates = gates.reshape(batch_size, 2, self.hidden_size)
        
        reset_gate = self.sigmoid(gates[:, 0, :])    # r_t
        update_gate = self.sigmoid(gates[:, 1, :])   # z_t
        
        # Compute candidate hidden state with reset gate
        reset_hidden = reset_gate * h_prev
        candidate_input = np.concatenate([reset_hidden, x], axis=1)
        candidate = np.tanh(np.dot(candidate_input, self.W[2].T) + self.b[2])  # h̃_t
        
        # Update hidden state: linear interpolation
        h_new = (1 - update_gate) * h_prev + update_gate * candidate
        
        return h_new
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))


class GRU:
    """Multi-layer GRU network"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Forward GRU layers
        self.forward_layers = []
        layer_input_size = input_size
        
        for i in range(num_layers):
            self.forward_layers.append(GRUCell(layer_input_size, hidden_size))
            layer_input_size = hidden_size
        
        # Backward GRU layers (if bidirectional)
        if bidirectional:
            self.backward_layers = []
            layer_input_size = input_size
            
            for i in range(num_layers):
                self.backward_layers.append(GRUCell(layer_input_size, hidden_size))
                layer_input_size = hidden_size
    
    def forward(self, x: np.ndarray, 
                initial_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through GRU network"""
        batch_size, seq_len, input_size = x.shape
        
        # Initialize states if not provided
        if initial_state is None:
            h_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        else:
            h_0 = initial_state
        
        # Forward direction
        forward_outputs, h_forward = self._forward_direction(x, h_0)
        
        if not self.bidirectional:
            return forward_outputs, h_forward
        
        # Backward direction
        backward_outputs, h_backward = self._backward_direction(x, h_0)
        
        # Concatenate forward and backward outputs
        outputs = np.concatenate([forward_outputs, backward_outputs], axis=2)
        
        # Concatenate final states
        h_final = np.concatenate([h_forward, h_backward], axis=2)
        
        return outputs, h_final
    
    def _forward_direction(self, x: np.ndarray, h_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process sequence in forward direction"""
        batch_size, seq_len, input_size = x.shape
        outputs = []
        
        h_states = [h_0[i] for i in range(self.num_layers)]
        
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx in range(self.num_layers):
                h_new = self.forward_layers[layer_idx].forward(layer_input, h_states[layer_idx])
                h_states[layer_idx] = h_new
                layer_input = h_new
            
            outputs.append(h_states[-1])  # Output from last layer
        
        outputs = np.stack(outputs, axis=1)  # [batch, seq_len, hidden]
        h_final = np.stack(h_states, axis=0)  # [num_layers, batch, hidden]
        
        return outputs, h_final
    
    def _backward_direction(self, x: np.ndarray, h_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process sequence in backward direction"""
        batch_size, seq_len, input_size = x.shape
        outputs = []
        
        h_states = [h_0[i] for i in range(self.num_layers)]
        
        for t in range(seq_len - 1, -1, -1):  # Backward iteration
            layer_input = x[:, t, :]
            
            for layer_idx in range(self.num_layers):
                h_new = self.backward_layers[layer_idx].forward(layer_input, h_states[layer_idx])
                h_states[layer_idx] = h_new
                layer_input = h_new
            
            outputs.append(h_states[-1])  # Output from last layer
        
        outputs.reverse()  # Reverse to match forward order
        outputs = np.stack(outputs, axis=1)  # [batch, seq_len, hidden]
        h_final = np.stack(h_states, axis=0)  # [num_layers, batch, hidden]
        
        return outputs, h_final


class GRULanguageModel:
    """GRU-based language model for sequence prediction"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, 
                 num_layers: int = 2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        
        # GRU layers
        self.gru = GRU(embed_dim, hidden_size, num_layers)
        
        # Output projection
        self.output_projection = np.random.randn(hidden_size, vocab_size) * 0.1
        self.output_bias = np.zeros(vocab_size)
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through language model"""
        # Embedding lookup
        embeddings = self.embedding[input_ids]  # [batch, seq_len, embed_dim]
        
        # GRU forward pass
        gru_output, _ = self.gru.forward(embeddings)  # [batch, seq_len, hidden]
        
        # Project to vocabulary
        logits = np.dot(gru_output, self.output_projection) + self.output_bias
        
        return logits
    
    def generate(self, start_token: int, length: int, temperature: float = 1.0) -> List[int]:
        """Generate sequence using the language model"""
        generated = [start_token]
        batch_size = 1
        
        # Initialize state
        h_state = np.zeros((self.gru.num_layers, batch_size, self.gru.hidden_size))
        
        for _ in range(length - 1):
            # Prepare input
            current_input = np.array([[generated[-1]]])
            embeddings = self.embedding[current_input]
            
            # Forward pass
            gru_output, h_state = self.gru.forward(embeddings, h_state)
            
            # Get logits for next token
            logits = np.dot(gru_output[:, -1, :], self.output_projection) + self.output_bias
            logits = logits / temperature
            
            # Sample next token
            probs = self.softmax(logits)
            next_token = np.random.choice(self.vocab_size, p=probs[0])
            generated.append(next_token)
        
        return generated
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GRUClassifier:
    """GRU-based sequence classifier"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 1, bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # GRU network
        self.gru = GRU(input_size, hidden_size, num_layers, bidirectional)
        
        # Classification head
        output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = np.random.randn(output_size, num_classes) * 0.1
        self.bias = np.zeros(num_classes)
        
        # Dropout simulation (not implemented for simplicity)
        self.dropout_rate = 0.1
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through classifier"""
        # GRU forward pass
        gru_output, final_states = self.gru.forward(x)  # [batch, seq_len, hidden]
        
        # Use final hidden state for classification
        # In a real implementation, you might use attention pooling instead
        if mask is not None:
            # Use last non-masked position
            seq_lengths = np.sum(mask, axis=1)  # [batch]
            batch_indices = np.arange(x.shape[0])
            last_outputs = gru_output[batch_indices, seq_lengths - 1]  # [batch, hidden]
        else:
            # Use last timestep
            last_outputs = gru_output[:, -1, :]  # [batch, hidden]
        
        # Classification
        logits = np.dot(last_outputs, self.classifier) + self.bias
        
        return logits
    
    def predict(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions"""
        logits = self.forward(x, mask)
        return np.argmax(logits, axis=1)


class GRUPerformanceAnalyzer:
    """Analyze GRU performance vs LSTM and vanilla RNN"""
    
    def __init__(self):
        pass
    
    def benchmark_models(self, input_size: int, hidden_size: int, seq_len: int, 
                        batch_size: int, num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """Benchmark different RNN architectures"""
        
        # Create models
        gru = GRU(input_size, hidden_size)
        
        # Simple LSTM for comparison
        class SimpleLSTM:
            def __init__(self, input_size: int, hidden_size: int):
                # Simplified LSTM (4 gates)
                self.W = np.random.randn(4, hidden_size, input_size + hidden_size) * 0.1
                self.b = np.zeros((4, hidden_size))
                self.b[0] = np.ones(hidden_size)  # forget gate bias
            
            def forward(self, x: np.ndarray) -> np.ndarray:
                batch_size, seq_len, input_size = x.shape
                hidden_size = self.W.shape[1]
                
                outputs = []
                h = np.zeros((batch_size, hidden_size))
                C = np.zeros((batch_size, hidden_size))
                
                for t in range(seq_len):
                    combined = np.concatenate([h, x[:, t]], axis=1)
                    gates = np.dot(combined, self.W.reshape(-1, self.W.shape[-1]).T) + self.b.flatten()
                    gates = gates.reshape(batch_size, 4, hidden_size)
                    
                    f_t = self._sigmoid(gates[:, 0, :])
                    i_t = self._sigmoid(gates[:, 1, :])
                    o_t = self._sigmoid(gates[:, 2, :])
                    C_tilde = np.tanh(gates[:, 3, :])
                    
                    C = f_t * C + i_t * C_tilde
                    h = o_t * np.tanh(C)
                    outputs.append(h)
                
                return np.stack(outputs, axis=1)
            
            def _sigmoid(self, x):
                return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        
        # Simple vanilla RNN
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
        
        lstm = SimpleLSTM(input_size, hidden_size)
        rnn = VanillaRNN(input_size, hidden_size)
        
        # Test data
        x = np.random.randn(batch_size, seq_len, input_size)
        
        results = {}
        
        # Benchmark each model
        for name, model in [("GRU", gru), ("LSTM", lstm), ("RNN", rnn)]:
            times = []
            memory_usage = []
            
            for _ in range(num_runs):
                # Time forward pass
                start_time = time.time()
                if name == "GRU":
                    output, _ = model.forward(x)
                else:
                    output = model.forward(x)
                forward_time = time.time() - start_time
                
                times.append(forward_time)
                
                # Estimate memory usage (rough approximation)
                if name == "GRU":
                    # GRU: 3 weight matrices + hidden states
                    param_memory = 3 * hidden_size * (input_size + hidden_size)
                    state_memory = batch_size * seq_len * hidden_size
                elif name == "LSTM":
                    # LSTM: 4 weight matrices + hidden + cell states  
                    param_memory = 4 * hidden_size * (input_size + hidden_size)
                    state_memory = 2 * batch_size * seq_len * hidden_size
                else:  # RNN
                    # RNN: 2 weight matrices + hidden states
                    param_memory = hidden_size * (input_size + hidden_size)
                    state_memory = batch_size * seq_len * hidden_size
                
                total_memory = param_memory + state_memory
                memory_usage.append(total_memory)
            
            results[name] = {
                "avg_time": np.mean(times),
                "std_time": np.std(times),
                "memory_usage": np.mean(memory_usage),
                "params": param_memory
            }
        
        return results
    
    def analyze_gradient_flow(self, hidden_size: int, seq_len: int) -> Dict[str, List[float]]:
        """Analyze gradient flow characteristics"""
        gru_cell = GRUCell(hidden_size, hidden_size)
        
        # Simulate forward pass
        batch_size = 1
        h_states = []
        gate_states = []
        
        h = np.random.randn(batch_size, hidden_size) * 0.1
        
        for t in range(seq_len):
            x = np.random.randn(batch_size, hidden_size) * 0.1
            
            # Extract gate values for analysis
            combined = np.concatenate([h, x], axis=1)
            gates = np.dot(combined, gru_cell.W[:2].reshape(2 * hidden_size, -1).T) + gru_cell.b[:2].flatten()
            gates = gates.reshape(batch_size, 2, hidden_size)
            
            reset_gate = gru_cell.sigmoid(gates[:, 0, :])
            update_gate = gru_cell.sigmoid(gates[:, 1, :])
            
            h = gru_cell.forward(x, h)
            
            h_states.append(h.copy())
            gate_states.append({
                'reset': reset_gate.copy(),
                'update': update_gate.copy()
            })
        
        # Analyze gradient flow (simplified simulation)
        gradient_flow = {
            'magnitude': [],
            'reset_gate_avg': [],
            'update_gate_avg': []
        }
        
        # Simulate backward pass
        grad_h = np.random.randn(batch_size, hidden_size)
        
        for t in range(seq_len - 1, -1, -1):
            # Gradient magnitude
            grad_magnitude = np.mean(np.abs(grad_h))
            gradient_flow['magnitude'].append(grad_magnitude)
            
            # Gate statistics
            reset_avg = np.mean(gate_states[t]['reset'])
            update_avg = np.mean(gate_states[t]['update'])
            
            gradient_flow['reset_gate_avg'].append(reset_avg)
            gradient_flow['update_gate_avg'].append(update_avg)
            
            # Simulate gradient flow through GRU
            # When update_gate is close to 0, gradient flows directly through (1-z_t) path
            update_gate_t = gate_states[t]['update'][0]
            grad_h = grad_h * (1 - update_gate_t) + grad_h * update_gate_t * 0.9  # Some decay
        
        # Reverse to match forward order
        for key in gradient_flow:
            gradient_flow[key].reverse()
        
        return gradient_flow


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_gru_cell():
    """Exercise 1: Test basic GRU cell"""
    print("=== Exercise 1: GRU Cell ===")
    
    input_size = 10
    hidden_size = 20
    batch_size = 3
    
    gru_cell = GRUCell(input_size, hidden_size)
    
    # Test forward pass
    x = np.random.randn(batch_size, input_size)
    h_prev = np.random.randn(batch_size, hidden_size)
    
    h_new = gru_cell.forward(x, h_prev)
    
    print(f"Input shape: {x.shape}")
    print(f"Previous hidden shape: {h_prev.shape}")
    print(f"New hidden shape: {h_new.shape}")
    
    assert h_new.shape == (batch_size, hidden_size)
    
    # Test gate ranges
    print(f"Hidden state range: [{np.min(h_new):.3f}, {np.max(h_new):.3f}]")
    
    # Test that gates are working (different inputs should give different outputs)
    x2 = np.random.randn(batch_size, input_size) * 10
    h_new2 = gru_cell.forward(x2, h_prev)
    
    assert not np.allclose(h_new, h_new2), "GRU should respond to different inputs"
    
    print("✓ GRU cell working correctly")


def exercise_2_multilayer_gru():
    """Exercise 2: Test multi-layer GRU"""
    print("=== Exercise 2: Multi-layer GRU ===")
    
    input_size = 50
    hidden_size = 64
    num_layers = 3
    seq_len = 20
    batch_size = 2
    
    gru = GRU(input_size, hidden_size, num_layers)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    outputs, h_final = gru.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    
    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert h_final.shape == (num_layers, batch_size, hidden_size)
    
    print("✓ Multi-layer GRU working correctly")


def exercise_3_bidirectional_gru():
    """Exercise 3: Test bidirectional GRU"""
    print("=== Exercise 3: Bidirectional GRU ===")
    
    input_size = 30
    hidden_size = 40
    seq_len = 15
    batch_size = 2
    
    bigru = GRU(input_size, hidden_size, num_layers=2, bidirectional=True)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    outputs, h_final = bigru.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Bidirectional output shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    
    # Bidirectional doubles the output size
    assert outputs.shape == (batch_size, seq_len, 2 * hidden_size)
    assert h_final.shape == (2, batch_size, 2 * hidden_size)  # 2 layers
    
    print("✓ Bidirectional GRU working correctly")


def exercise_4_gru_language_model():
    """Exercise 4: Test GRU language model"""
    print("=== Exercise 4: GRU Language Model ===")
    
    vocab_size = 1000
    embed_dim = 128
    hidden_size = 256
    seq_len = 50
    batch_size = 4
    
    lm = GRULanguageModel(vocab_size, embed_dim, hidden_size, num_layers=2)
    
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
    
    print("✓ GRU language model working correctly")


def exercise_5_gru_classifier():
    """Exercise 5: Test GRU sequence classifier"""
    print("=== Exercise 5: GRU Sequence Classifier ===")
    
    input_size = 100
    hidden_size = 128
    num_classes = 5
    seq_len = 30
    batch_size = 8
    
    # Test unidirectional classifier
    classifier = GRUClassifier(input_size, hidden_size, num_classes, num_layers=2)
    
    x = np.random.randn(batch_size, seq_len, input_size)
    logits = classifier.forward(x)
    predictions = classifier.predict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    assert logits.shape == (batch_size, num_classes)
    assert predictions.shape == (batch_size,)
    assert np.all((0 <= predictions) & (predictions < num_classes))
    
    # Test bidirectional classifier
    bi_classifier = GRUClassifier(input_size, hidden_size, num_classes, 
                                 num_layers=1, bidirectional=True)
    
    bi_logits = bi_classifier.forward(x)
    bi_predictions = bi_classifier.predict(x)
    
    print(f"Bidirectional logits shape: {bi_logits.shape}")
    print(f"Bidirectional predictions shape: {bi_predictions.shape}")
    
    assert bi_logits.shape == (batch_size, num_classes)
    
    print("✓ GRU classifier working correctly")


def exercise_6_performance_analysis():
    """Exercise 6: Analyze GRU performance vs other RNNs"""
    print("=== Exercise 6: Performance Analysis ===")
    
    input_size = 50
    hidden_size = 128
    seq_len = 100
    batch_size = 16
    
    analyzer = GRUPerformanceAnalyzer()
    
    print("Benchmarking models...")
    results = analyzer.benchmark_models(input_size, hidden_size, seq_len, batch_size, num_runs=5)
    
    print("\nPerformance Results:")
    print("-" * 60)
    print(f"{'Model':<8} {'Time(ms)':<10} {'Memory(MB)':<12} {'Params':<10}")
    print("-" * 60)
    
    for model_name, stats in results.items():
        avg_time_ms = stats['avg_time'] * 1000
        memory_mb = stats['memory_usage'] / (1024 * 1024)  # Convert to MB
        params_k = stats['params'] / 1000  # Convert to K
        
        print(f"{model_name:<8} {avg_time_ms:<10.2f} {memory_mb:<12.2f} {params_k:<10.1f}K")
    
    # Calculate efficiency metrics
    gru_time = results['GRU']['avg_time']
    lstm_time = results['LSTM']['avg_time']
    rnn_time = results['RNN']['avg_time']
    
    print(f"\nEfficiency Ratios:")
    print(f"GRU vs LSTM speed: {lstm_time/gru_time:.2f}x faster")
    print(f"GRU vs RNN speed: {gru_time/rnn_time:.2f}x slower")
    
    gru_memory = results['GRU']['memory_usage']
    lstm_memory = results['LSTM']['memory_usage']
    
    print(f"GRU vs LSTM memory: {lstm_memory/gru_memory:.2f}x less memory")
    
    print("✓ Performance analysis completed")


def exercise_7_gradient_flow_analysis():
    """Exercise 7: Analyze gradient flow in GRU"""
    print("=== Exercise 7: Gradient Flow Analysis ===")
    
    hidden_size = 64
    seq_len = 50
    
    analyzer = GRUPerformanceAnalyzer()
    gradient_info = analyzer.analyze_gradient_flow(hidden_size, seq_len)
    
    print(f"Sequence length: {seq_len}")
    print(f"Initial gradient magnitude: {gradient_info['magnitude'][0]:.6f}")
    print(f"Final gradient magnitude: {gradient_info['magnitude'][-1]:.6f}")
    print(f"Gradient preservation ratio: {gradient_info['magnitude'][-1]/gradient_info['magnitude'][0]:.4f}")
    
    # Analyze gate behavior
    avg_reset = np.mean(gradient_info['reset_gate_avg'])
    avg_update = np.mean(gradient_info['update_gate_avg'])
    
    print(f"Average reset gate activation: {avg_reset:.4f}")
    print(f"Average update gate activation: {avg_update:.4f}")
    
    # Plot gradient flow and gate behavior
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(gradient_info['magnitude'], label='Gradient Magnitude', color='blue', alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('Gradient Magnitude')
    plt.title('GRU Gradient Flow')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(gradient_info['reset_gate_avg'], label='Reset Gate', color='red', alpha=0.8)
    plt.plot(gradient_info['update_gate_avg'], label='Update Gate', color='green', alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('Gate Activation')
    plt.title('Gate Activation Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Correlation between gates
    reset_vals = gradient_info['reset_gate_avg']
    update_vals = gradient_info['update_gate_avg']
    correlation = np.corrcoef(reset_vals, update_vals)[0, 1]
    
    plt.scatter(reset_vals, update_vals, alpha=0.6, c=range(len(reset_vals)), cmap='viridis')
    plt.xlabel('Reset Gate Activation')
    plt.ylabel('Update Gate Activation')
    plt.title(f'Gate Correlation (ρ={correlation:.3f})')
    plt.colorbar(label='Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Gradient magnitude vs update gate
    plt.scatter(gradient_info['update_gate_avg'], gradient_info['magnitude'], alpha=0.6)
    plt.xlabel('Update Gate Activation')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow vs Update Gate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/gru_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Gradient flow analysis completed")


def exercise_8_gru_vs_lstm_comparison():
    """Exercise 8: Direct comparison of GRU vs LSTM characteristics"""
    print("=== Exercise 8: GRU vs LSTM Comparison ===")
    
    input_size = 100
    hidden_size = 256
    seq_len = 200
    batch_size = 32
    
    # Create both models
    gru = GRU(input_size, hidden_size, num_layers=2)
    
    # Count parameters
    def count_gru_params():
        # 3 weight matrices per layer: reset, update, candidate
        # Each matrix: (input_size + hidden_size) x hidden_size
        # Plus biases: 3 x hidden_size per layer
        layer1_params = 3 * (input_size + hidden_size) * hidden_size + 3 * hidden_size
        layer2_params = 3 * (hidden_size + hidden_size) * hidden_size + 3 * hidden_size
        return layer1_params + layer2_params
    
    def count_lstm_params():
        # 4 weight matrices per layer: forget, input, output, candidate
        layer1_params = 4 * (input_size + hidden_size) * hidden_size + 4 * hidden_size
        layer2_params = 4 * (hidden_size + hidden_size) * hidden_size + 4 * hidden_size
        return layer1_params + layer2_params
    
    gru_params = count_gru_params()
    lstm_params = count_lstm_params()
    
    print(f"Model Comparison:")
    print(f"GRU Parameters: {gru_params:,}")
    print(f"LSTM Parameters: {lstm_params:,}")
    print(f"Parameter Reduction: {(1 - gru_params/lstm_params)*100:.1f}%")
    
    # Memory usage comparison
    x = np.random.randn(batch_size, seq_len, input_size)
    
    # Time forward passes
    start_time = time.time()
    gru_output, gru_states = gru.forward(x)
    gru_time = time.time() - start_time
    
    print(f"\nPerformance Metrics:")
    print(f"GRU Forward Time: {gru_time:.4f}s")
    print(f"GRU Output Shape: {gru_output.shape}")
    print(f"GRU State Memory: {gru_states.nbytes / 1024:.1f} KB")
    
    # Memory efficiency
    gru_state_memory = gru_states.nbytes
    lstm_estimated_memory = gru_state_memory * 2  # LSTM has both h and C states
    
    print(f"GRU State Memory: {gru_state_memory / 1024:.1f} KB")
    print(f"LSTM Estimated Memory: {lstm_estimated_memory / 1024:.1f} KB")
    print(f"Memory Reduction: {(1 - gru_state_memory/lstm_estimated_memory)*100:.1f}%")
    
    # Analyze expressiveness
    print(f"\nArchitecture Analysis:")
    print(f"GRU Gates: 2 (Reset, Update)")
    print(f"LSTM Gates: 3 (Forget, Input, Output)")
    print(f"GRU States: 1 (Hidden)")
    print(f"LSTM States: 2 (Hidden, Cell)")
    
    print("✓ GRU vs LSTM comparison completed")


if __name__ == "__main__":
    exercise_1_gru_cell()
    exercise_2_multilayer_gru()
    exercise_3_bidirectional_gru()
    exercise_4_gru_language_model()
    exercise_5_gru_classifier()
    exercise_6_performance_analysis()
    exercise_7_gradient_flow_analysis()
    exercise_8_gru_vs_lstm_comparison()
    print("\nGRU implementation completed!")