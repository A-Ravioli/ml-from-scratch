"""
BERT Family Implementation Exercise

Implementation of BERT-style encoder-only Transformers for bidirectional language understanding
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class MultiHeadSelfAttention:
    """Multi-head self-attention for BERT (bidirectional)"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value projections
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, x: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + (attention_mask * -1e9)
        
        attn_weights = self.softmax(scores)
        attn_output = np.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class BertMLP:
    """BERT MLP with GELU activation"""
    
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
    """Layer normalization (post-norm as in original BERT)"""
    
    def __init__(self, d_model: int, eps: float = 1e-12):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class BertLayer:
    """Single BERT transformer layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.attention_norm = LayerNorm(d_model)
        self.mlp = BertMLP(d_model, d_ff)
        self.mlp_norm = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Self-attention with residual connection and layer norm
        attn_output = self.attention.forward(x, attention_mask)
        x = self.attention_norm.forward(x + attn_output)
        
        # MLP with residual connection and layer norm
        mlp_output = self.mlp.forward(x)
        x = self.mlp_norm.forward(x + mlp_output)
        
        return x


class BertEmbeddings:
    """BERT embeddings: token + position + segment"""
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, 
                 num_segments: int = 2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Learned position embeddings
        self.position_embedding = np.random.randn(max_len, d_model) * 0.02
        
        # Segment embeddings
        self.segment_embedding = np.random.randn(num_segments, d_model) * 0.02
        
        # Layer norm and dropout
        self.layer_norm = LayerNorm(d_model)
    
    def forward(self, input_ids: np.ndarray, 
                token_type_ids: Optional[np.ndarray] = None,
                position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding[input_ids]
        
        # Position embeddings
        if position_ids is None:
            position_ids = np.arange(seq_len)
        pos_emb = self.position_embedding[position_ids]
        
        # Segment embeddings
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        segment_emb = self.segment_embedding[token_type_ids]
        
        # Combine embeddings
        embeddings = token_emb + pos_emb + segment_emb
        
        return self.layer_norm.forward(embeddings)


class BertModel:
    """BERT encoder model"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 d_ff: int = 3072, num_layers: int = 12, max_len: int = 512):
        self.embeddings = BertEmbeddings(vocab_size, d_model, max_len)
        
        # Transformer layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(BertLayer(d_model, num_heads, d_ff))
    
    def forward(self, input_ids: np.ndarray, 
                attention_mask: Optional[np.ndarray] = None,
                token_type_ids: Optional[np.ndarray] = None) -> np.ndarray:
        # Embeddings
        x = self.embeddings.forward(input_ids, token_type_ids)
        
        # Transform layers
        for layer in self.layers:
            x = layer.forward(x, attention_mask)
        
        return x


class MaskedLanguageModel:
    """MLM head for BERT pre-training"""
    
    def __init__(self, d_model: int, vocab_size: int):
        self.transform = BertMLP(d_model, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.decoder = np.random.randn(d_model, vocab_size) * 0.02
        self.bias = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        hidden_states = self.transform.forward(hidden_states)
        hidden_states = self.layer_norm.forward(hidden_states)
        return np.dot(hidden_states, self.decoder) + self.bias


class NextSentencePrediction:
    """NSP head for BERT pre-training"""
    
    def __init__(self, d_model: int):
        self.classifier = np.random.randn(d_model, 2) * 0.02
        self.bias = np.zeros(2)
    
    def forward(self, cls_hidden_state: np.ndarray) -> np.ndarray:
        return np.dot(cls_hidden_state, self.classifier) + self.bias


class BertPreTrainingModel:
    """Complete BERT model with MLM and NSP heads"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 d_ff: int = 3072, num_layers: int = 12, max_len: int = 512):
        self.bert = BertModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.mlm_head = MaskedLanguageModel(d_model, vocab_size)
        self.nsp_head = NextSentencePrediction(d_model)
    
    def forward(self, input_ids: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                token_type_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Get hidden states from BERT
        hidden_states = self.bert.forward(input_ids, attention_mask, token_type_ids)
        
        # MLM predictions for all tokens
        mlm_logits = self.mlm_head.forward(hidden_states)
        
        # NSP prediction from [CLS] token (first token)
        cls_hidden = hidden_states[:, 0, :]  # [batch_size, d_model]
        nsp_logits = self.nsp_head.forward(cls_hidden)
        
        return mlm_logits, nsp_logits


class BertTrainer:
    """Simple BERT trainer for MLM and NSP"""
    
    def __init__(self, model: BertPreTrainingModel, learning_rate: float = 1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.mask_token_id = 103  # [MASK] token ID
    
    def create_masked_lm_data(self, input_ids: np.ndarray, 
                             mask_prob: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """Create masked language modeling data"""
        masked_input = input_ids.copy()
        labels = np.full_like(input_ids, -100)  # -100 = ignore in loss
        
        batch_size, seq_len = input_ids.shape
        
        for i in range(batch_size):
            for j in range(seq_len):
                if np.random.random() < mask_prob:
                    labels[i, j] = input_ids[i, j]
                    
                    # 80% replace with [MASK], 10% random token, 10% unchanged
                    rand = np.random.random()
                    if rand < 0.8:
                        masked_input[i, j] = self.mask_token_id
                    elif rand < 0.9:
                        masked_input[i, j] = np.random.randint(0, 1000)  # Random token
        
        return masked_input, labels
    
    def compute_mlm_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute masked language modeling loss"""
        # Only compute loss for masked tokens (labels != -100)
        mask = labels != -100
        
        if not np.any(mask):
            return 0.0
        
        logits_masked = logits[mask]
        labels_masked = labels[mask]
        
        # Softmax and cross-entropy
        probs = self.softmax(logits_masked)
        loss = -np.mean(np.log(probs[np.arange(len(labels_masked)), labels_masked] + 1e-10))
        
        return loss
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_bert_embeddings():
    """Exercise 1: Test BERT embeddings"""
    print("=== Exercise 1: BERT Embeddings ===")
    
    vocab_size = 30522  # BERT vocab size
    d_model = 768
    seq_len = 128
    batch_size = 2
    
    embeddings = BertEmbeddings(vocab_size, d_model)
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    token_type_ids = np.random.randint(0, 2, (batch_size, seq_len))
    
    output = embeddings.forward(input_ids, token_type_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Embeddings output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    print("✓ BERT embeddings working correctly")


def exercise_2_bert_layer():
    """Exercise 2: Test BERT transformer layer"""
    print("=== Exercise 2: BERT Layer ===")
    
    d_model = 768
    num_heads = 12
    d_ff = 3072
    seq_len = 128
    batch_size = 2
    
    layer = BertLayer(d_model, num_heads, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Create attention mask (0 = attend, 1 = don't attend)
    attention_mask = np.zeros((batch_size, 1, 1, seq_len))
    attention_mask[:, :, :, seq_len//2:] = 1  # Mask second half
    
    output = layer.forward(x, attention_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Layer output shape: {output.shape}")
    assert output.shape == x.shape
    
    print("✓ BERT layer working correctly")


def exercise_3_complete_bert():
    """Exercise 3: Test complete BERT model"""
    print("=== Exercise 3: Complete BERT ===")
    
    vocab_size = 1000
    seq_len = 64
    batch_size = 2
    
    model = BertModel(vocab_size, d_model=256, num_layers=6, num_heads=8)
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    output = model.forward(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"BERT output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 256)
    
    print("✓ Complete BERT working correctly")


def exercise_4_mlm_pretraining():
    """Exercise 4: Test MLM pre-training"""
    print("=== Exercise 4: MLM Pre-training ===")
    
    vocab_size = 1000
    model = BertPreTrainingModel(vocab_size, d_model=256, num_layers=4, num_heads=4)
    trainer = BertTrainer(model)
    
    batch_size = 4
    seq_len = 32
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create masked data
    masked_input, labels = trainer.create_masked_lm_data(input_ids)
    
    # Forward pass
    mlm_logits, nsp_logits = model.forward(masked_input)
    
    print(f"Original input shape: {input_ids.shape}")
    print(f"Masked input shape: {masked_input.shape}")
    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"NSP logits shape: {nsp_logits.shape}")
    
    # Compute MLM loss
    mlm_loss = trainer.compute_mlm_loss(mlm_logits, labels)
    print(f"MLM loss: {mlm_loss:.4f}")
    
    assert mlm_logits.shape == (batch_size, seq_len, vocab_size)
    assert nsp_logits.shape == (batch_size, 2)
    
    print("✓ MLM pre-training working correctly")


def exercise_5_attention_analysis():
    """Exercise 5: Analyze BERT attention patterns"""
    print("=== Exercise 5: Attention Analysis ===")
    
    vocab_size = 100
    d_model = 256
    num_heads = 4
    
    attention = MultiHeadSelfAttention(d_model, num_heads)
    
    # Create a simple sequence
    seq_len = 10
    batch_size = 1
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Modify attention to return attention weights
    batch_size, seq_len, d_model = x.shape
    Q = np.dot(x, attention.W_q).reshape(batch_size, seq_len, num_heads, attention.d_k).transpose(0, 2, 1, 3)
    K = np.dot(x, attention.W_k).reshape(batch_size, seq_len, num_heads, attention.d_k).transpose(0, 2, 1, 3)
    
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(attention.d_k)
    attn_weights = attention.softmax(scores)
    
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Sum of attention weights per position: {np.sum(attn_weights[0, 0, 0, :]):.4f}")
    
    # Check that attention weights sum to 1
    assert np.allclose(np.sum(attn_weights, axis=-1), 1.0)
    
    print("✓ Attention analysis completed")


if __name__ == "__main__":
    exercise_1_bert_embeddings()
    exercise_2_bert_layer()
    exercise_3_complete_bert()
    exercise_4_mlm_pretraining()
    exercise_5_attention_analysis()
    print("\nBERT Family implementation completed!")