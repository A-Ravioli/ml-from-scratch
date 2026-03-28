"""
GPT Family Implementation Exercise

Implementation of GPT-style decoder-only Transformers for autoregressive language modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class CausalMultiHeadAttention:
    """Multi-head attention with causal masking for autoregressive models"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Combined QKV projection for efficiency
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V in one pass
        qkv = np.dot(x, self.W_qkv)
        Q, K, V = np.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention with causal mask
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(causal_mask == 0, -np.inf, scores)
        
        attn_weights = self.softmax(scores)
        attn_output = np.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MLP:
    """Position-wise MLP with GELU activation (as used in GPT)"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.W1) + self.b1
        h = self.gelu(h)
        return np.dot(h, self.W2) + self.b2
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    """Layer normalization (pre-norm as in modern Transformers)"""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class TransformerBlock:
    """Single Transformer decoder block with pre-norm"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pre-norm architecture
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x


class GPTModel:
    """GPT-style decoder-only Transformer"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12, 
                 d_ff: int = 3072, num_layers: int = 12, max_len: int = 1024):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Learned position embeddings (as in GPT-2)
        self.position_embedding = np.random.randn(max_len, d_model) * 0.01
        
        # Transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, d_ff))
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(d_model)
        self.lm_head = self.token_embedding.T  # Weight tying
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        
        # Token + position embeddings
        token_emb = self.token_embedding[input_ids]
        pos_emb = self.position_embedding[:seq_len]
        x = token_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Final layer norm and projection to vocab
        x = self.ln_f.forward(x)
        logits = np.dot(x, self.lm_head)
        
        return logits
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50, 
                 temperature: float = 1.0, top_k: Optional[int] = None) -> np.ndarray:
        """Generate text using the model"""
        generated = input_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < np.partition(next_token_logits, -top_k, axis=-1)[..., -top_k, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = self.softmax(next_token_logits)
            next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])
            
            # Append to sequence
            generated = np.concatenate([generated, next_token], axis=1)
        
        return generated
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GPTTrainer:
    """Simple trainer for GPT models"""
    
    def __init__(self, model: GPTModel, learning_rate: float = 3e-4):
        self.model = model
        self.learning_rate = learning_rate
    
    def compute_loss(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss for next-token prediction"""
        logits = self.model.forward(input_ids)
        
        # Shift targets for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = targets[:, 1:]
        
        # Compute cross-entropy loss
        batch_size, seq_len, vocab_size = shift_logits.shape
        shift_logits_flat = shift_logits.reshape(-1, vocab_size)
        shift_labels_flat = shift_labels.flatten()
        
        # Softmax and cross-entropy
        probs = self.softmax(shift_logits_flat)
        loss = -np.mean(np.log(probs[np.arange(len(shift_labels_flat)), shift_labels_flat] + 1e-10))
        
        return loss
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_causal_attention():
    """Exercise 1: Test causal attention mechanism"""
    print("=== Exercise 1: Causal Attention ===")
    
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    causal_attn = CausalMultiHeadAttention(d_model, num_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = causal_attn.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Causal attention output shape: {output.shape}")
    assert output.shape == x.shape
    
    print("✓ Causal attention working correctly")


def exercise_2_gpt_model():
    """Exercise 2: Test complete GPT model"""
    print("=== Exercise 2: GPT Model ===")
    
    vocab_size = 1000
    seq_len = 20
    batch_size = 2
    
    model = GPTModel(vocab_size, d_model=512, num_layers=6, num_heads=8)
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = model.forward(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    
    print("✓ GPT model working correctly")


def exercise_3_text_generation():
    """Exercise 3: Test text generation"""
    print("=== Exercise 3: Text Generation ===")
    
    vocab_size = 100
    model = GPTModel(vocab_size, d_model=256, num_layers=3, num_heads=4)
    
    # Start with a simple prompt
    prompt = np.array([[1, 2, 3]])  # Some token IDs
    
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    
    print(f"Prompt: {prompt[0]}")
    print(f"Generated: {generated[0]}")
    assert generated.shape[1] == prompt.shape[1] + 10
    
    print("✓ Text generation working correctly")


def exercise_4_scaling_analysis():
    """Exercise 4: Analyze scaling properties"""
    print("=== Exercise 4: Scaling Analysis ===")
    
    vocab_size = 1000
    seq_len = 100
    
    model_configs = [
        {"d_model": 256, "num_layers": 6, "num_heads": 4, "name": "Small"},
        {"d_model": 512, "num_layers": 8, "num_heads": 8, "name": "Medium"},
        {"d_model": 768, "num_layers": 12, "num_heads": 12, "name": "Large"}
    ]
    
    for config in model_configs:
        model = GPTModel(vocab_size, **{k: v for k, v in config.items() if k != "name"})
        
        # Count parameters
        total_params = 0
        total_params += model.token_embedding.size
        total_params += model.position_embedding.size
        
        # Approximate block parameters
        block_params = (
            config["d_model"] * 3 * config["d_model"] +  # QKV projection
            config["d_model"] * config["d_model"] +       # Output projection
            config["d_model"] * 4 * config["d_model"] * 2 +  # MLP
            config["d_model"] * 4                         # Layer norms
        )
        total_params += block_params * config["num_layers"]
        total_params += config["d_model"]  # Final layer norm
        
        print(f"{config['name']}: ~{total_params/1e6:.1f}M parameters")
    
    print("✓ Scaling analysis completed")


def exercise_5_training_simulation():
    """Exercise 5: Simulate training process"""
    print("=== Exercise 5: Training Simulation ===")
    
    vocab_size = 500
    model = GPTModel(vocab_size, d_model=256, num_layers=4, num_heads=4)
    trainer = GPTTrainer(model)
    
    # Create dummy training data
    batch_size = 4
    seq_len = 50
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # Compute initial loss
    initial_loss = trainer.compute_loss(input_ids, input_ids)
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Expected random loss: {np.log(vocab_size):.4f}")
    
    # Verify loss is reasonable for random initialization
    assert initial_loss > np.log(vocab_size) * 0.8
    assert initial_loss < np.log(vocab_size) * 1.5
    
    print("✓ Training simulation working correctly")


if __name__ == "__main__":
    exercise_1_causal_attention()
    exercise_2_gpt_model()
    exercise_3_text_generation()
    exercise_4_scaling_analysis()
    exercise_5_training_simulation()
    print("\nGPT Family implementation completed!")