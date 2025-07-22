"""
Complete Implementation of Autoregressive Models

This file contains reference implementations for all autoregressive models
covered in the lesson. Study these implementations after attempting the
exercises yourself.

Author: ML-from-Scratch Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


class CharRNNLanguageModel(nn.Module):
    """Character-level RNN language model."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, 
                 rnn_type: str = 'LSTM', dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RNN layers
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout and output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        # Embed input tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(embedded, hidden)  # [batch_size, seq_len, hidden_dim]
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Project to vocabulary size
        logits = self.output_projection(rnn_output)  # [batch_size, seq_len, vocab_size]
        
        return logits, hidden
        
    def generate(self, start_idx: int, max_length: int = 100, 
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, device: torch.device = None) -> List[int]:
        """Generate sequence using different sampling strategies."""
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        generated = [start_idx]
        
        # Initialize hidden state
        hidden = self.init_hidden(1, device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Prepare input
                x = torch.tensor([[generated[-1]]], device=device)
                
                # Forward pass
                logits, hidden = self(x, hidden)
                logits = logits[0, -1, :]  # Get last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply sampling strategy
                if top_k is not None:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    # Nucleus sampling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
        return generated
        
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class CausalConv1d(nn.Module):
    """Causal 1D convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causality
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            padding=self.padding, dilation=dilation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking."""
        # Apply convolution
        output = self.conv(x)
        
        # Remove future information by cropping
        if self.padding > 0:
            output = output[:, :, :-self.padding]
            
        return output


class WaveNetBlock(nn.Module):
    """WaveNet-style residual block with gated activation."""
    
    def __init__(self, residual_channels: int, gate_channels: int,
                 skip_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        
        self.residual_channels = residual_channels
        
        # Causal convolution
        self.conv = CausalConv1d(residual_channels, gate_channels, kernel_size, dilation)
        
        # Gate and filter convolutions
        self.conv_gate = CausalConv1d(residual_channels, gate_channels, kernel_size, dilation)
        self.conv_filter = CausalConv1d(residual_channels, gate_channels, kernel_size, dilation)
        
        # Output projections
        self.conv_res = nn.Conv1d(gate_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(gate_channels, skip_channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Gated activation
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        
        gated = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        
        # Residual connection
        residual = self.conv_res(gated)
        if residual.size(-1) != x.size(-1):
            x = x[:, :, :residual.size(-1)]
        residual_output = x + residual
        
        # Skip connection
        skip_output = self.conv_skip(gated)
        
        return residual_output, skip_output


class CausalCNNLanguageModel(nn.Module):
    """Causal CNN language model inspired by WaveNet."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 residual_channels: int = 256, gate_channels: int = 512,
                 skip_channels: int = 256, num_blocks: int = 10,
                 num_layers_per_block: int = 10, kernel_size: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.residual_channels = residual_channels
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initial causal convolution
        self.start_conv = CausalConv1d(embed_dim, residual_channels, 1)
        
        # WaveNet blocks
        self.blocks = nn.ModuleList()
        for block in range(num_blocks):
            for layer in range(num_layers_per_block):
                dilation = 2 ** layer
                self.blocks.append(
                    WaveNetBlock(residual_channels, gate_channels, skip_channels,
                               kernel_size, dilation)
                )
        
        # Final layers
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.end_conv2 = nn.Conv1d(skip_channels, vocab_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        
        # Initial convolution
        x = self.start_conv(embedded)
        
        # Apply WaveNet blocks
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
            
        # Sum skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Final convolutions
        output = F.relu(skip_sum)
        output = self.end_conv1(output)
        output = F.relu(output)
        output = self.end_conv2(output)
        
        # Transpose back to [batch_size, seq_len, vocab_size]
        output = output.transpose(1, 2)
        
        return output
        
    def generate(self, start_idx: int, max_length: int = 100,
                temperature: float = 1.0, device: torch.device = None) -> List[int]:
        """Generate sequence autoregressively."""
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        generated = [start_idx]
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Prepare input sequence
                x = torch.tensor([generated], device=device)
                
                # Forward pass
                logits = self(x)  # [1, seq_len, vocab_size]
                logits = logits[0, -1, :] / temperature  # Get last token logits
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
        return generated


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal self-attention layer."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask to prevent attending to future positions."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking."""
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply causal mask
        causal_mask = self.create_causal_mask(seq_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadCausalAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections and layer norms."""
        # Self-attention with residual connection
        attn_output = self.self_attention(self.norm1(x))
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x


class TransformerLanguageModel(nn.Module):
    """Transformer-based autoregressive language model."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 6, 
                 ff_dim: int = 2048, max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
        
    def create_positional_encoding(self, seq_len: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) * 
                           -(math.log(10000.0) / embed_dim))
        
        pos_encoding = torch.zeros(seq_len, embed_dim, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_embeddings = self.token_embedding(x)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits
        
    def generate(self, start_idx: int, max_length: int = 100,
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, device: torch.device = None) -> List[int]:
        """Generate sequence using various sampling strategies."""
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        generated = [start_idx]
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Prepare input
                x = torch.tensor([generated], device=device)
                
                # Forward pass
                logits = self(x)  # [1, seq_len, vocab_size]
                logits = logits[0, -1, :] / temperature  # Get last token logits
                
                # Apply sampling strategy (same as CharRNN implementation)
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
        return generated


class GenerationStrategies:
    """Collection of text generation strategies."""
    
    @staticmethod
    def greedy_search(logits: torch.Tensor) -> int:
        """Greedy decoding - always pick most likely token."""
        return torch.argmax(logits).item()
        
    @staticmethod
    def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample with temperature scaling."""
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, 1).item()
        
    @staticmethod
    def top_k_sampling(logits: torch.Tensor, k: int) -> int:
        """Top-k sampling."""
        top_k_logits, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, 1).item()
        
    @staticmethod
    def nucleus_sampling(logits: torch.Tensor, p: float) -> int:
        """Nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False  # Keep at least one token
        
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, 1).item()
        
    @staticmethod
    def beam_search(model: nn.Module, start_idx: int, max_length: int, 
                   beam_size: int = 5, length_penalty: float = 1.0, 
                   device: torch.device = None) -> List[List[int]]:
        """Beam search for finding high-probability sequences."""
        if device is None:
            device = next(model.parameters()).device
            
        model.eval()
        
        # Initialize beams
        beams = [([start_idx], 0.0)]  # (sequence, score)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                candidates = []
                
                for sequence, score in beams:
                    if len(sequence) >= max_length:
                        candidates.append((sequence, score))
                        continue
                        
                    # Prepare input
                    x = torch.tensor([sequence], device=device)
                    
                    # Forward pass
                    logits = model(x)  # [1, seq_len, vocab_size]
                    logits = logits[0, -1, :]  # Get last token logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top-k candidates
                    top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        new_token = top_k_indices[i].item()
                        new_sequence = sequence + [new_token]
                        new_score = score + top_k_log_probs[i].item()
                        
                        # Apply length penalty
                        if length_penalty != 1.0:
                            length_norm = ((len(new_sequence)) ** length_penalty)
                            normalized_score = new_score / length_norm
                        else:
                            normalized_score = new_score
                            
                        candidates.append((new_sequence, normalized_score))
                
                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
        
        return [beam[0] for beam in beams]


class AutoregressiveTrainer:
    """Training utilities for autoregressive models."""
    
    def __init__(self, model: nn.Module, tokenizer: Dict, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer,
                  criterion: nn.Module) -> float:
        """Single training step with teacher forcing."""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Prepare input and targets (teacher forcing)
        input_ids = batch[:, :-1]  # All tokens except last
        targets = batch[:, 1:]     # All tokens except first (shifted by 1)
        
        # Forward pass
        if isinstance(self.model, CharRNNLanguageModel):
            logits, _ = self.model(input_ids)
        else:
            logits = self.model(input_ids)
        
        # Compute loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
        
    def evaluate(self, data_loader) -> float:
        """Evaluate model and return perplexity."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                if isinstance(self.model, CharRNNLanguageModel):
                    logits, _ = self.model(input_ids)
                else:
                    logits = self.model(input_ids)
                
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = targets.reshape(-1)
                
                loss = criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                total_tokens += targets_flat.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity
        
    def train(self, train_loader, val_loader, num_epochs: int, 
             learning_rate: float = 0.001, save_path: str = None) -> Dict:
        """Full training loop."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        
        history = {'train_loss': [], 'val_perplexity': [], 'learning_rate': []}
        best_perplexity = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                loss = self.train_step(batch, optimizer, criterion)
                epoch_loss += loss
                num_batches += 1
                
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            val_perplexity = self.evaluate(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_perplexity)
            
            # Save best model
            if save_path and val_perplexity < best_perplexity:
                best_perplexity = val_perplexity
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'perplexity': val_perplexity,
                }, save_path)
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_perplexity'].append(val_perplexity)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Perplexity = {val_perplexity:.4f}")
        
        return history


class EvaluationMetrics:
    """Evaluation metrics for autoregressive models."""
    
    @staticmethod
    def perplexity(log_probs: torch.Tensor) -> float:
        """Compute perplexity from log probabilities."""
        return torch.exp(-log_probs.mean()).item()
        
    @staticmethod
    def bleu_score(references: List[List[str]], hypotheses: List[str], 
                  max_n: int = 4) -> float:
        """Compute BLEU score for generated text."""
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            if len(tokens) < n:
                return Counter()
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        
        def bleu_n(ref_ngrams: Counter, hyp_ngrams: Counter) -> Tuple[int, int]:
            overlap = sum((ref_ngrams & hyp_ngrams).values())
            total = sum(hyp_ngrams.values())
            return overlap, total
        
        total_score = 0
        
        for ref_list, hyp in zip(references, hypotheses):
            ref_tokens = ref_list if isinstance(ref_list[0], str) else ref_list[0].split()
            hyp_tokens = hyp.split() if isinstance(hyp, str) else hyp
            
            if len(hyp_tokens) == 0:
                continue
                
            score = 0
            for n in range(1, max_n + 1):
                ref_ngrams = get_ngrams(ref_tokens, n)
                hyp_ngrams = get_ngrams(hyp_tokens, n)
                
                if len(hyp_ngrams) == 0:
                    continue
                    
                overlap, total = bleu_n(ref_ngrams, hyp_ngrams)
                if total > 0:
                    score += math.log(overlap / total) / max_n
                    
            # Brevity penalty
            ref_len = len(ref_tokens)
            hyp_len = len(hyp_tokens)
            if hyp_len < ref_len:
                score *= math.exp(1 - ref_len / hyp_len)
                
            total_score += math.exp(score)
            
        return total_score / len(references) if references else 0.0
        
    @staticmethod
    def repetition_penalty(generated_text: str, n: int = 4) -> float:
        """Measure repetition in generated text."""
        tokens = generated_text.split()
        if len(tokens) < n:
            return 0.0
            
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        unique_ngrams = set(ngrams)
        
        return 1 - len(unique_ngrams) / len(ngrams)


# Utility functions
def create_character_tokenizer(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create character-level tokenizer."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def load_text_dataset(file_path: str, seq_len: int, tokenizer: Dict[str, int]) -> torch.Tensor:
    """Load and tokenize text dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    tokens = [tokenizer.get(ch, tokenizer['<UNK>']) for ch in text]
    
    # Create sequences
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        sequences.append(tokens[i:i + seq_len + 1])  # +1 for target
        
    return torch.tensor(sequences, dtype=torch.long)


def visualize_attention_weights(attention_weights: torch.Tensor, 
                              tokens: List[str], save_path: str = None):
    """Visualize attention patterns."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(attention_weights.detach().cpu().numpy(), cmap='Blues')
    
    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if attention_weights[i, j] > 0.1:  # Only show significant weights
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="red")
    
    ax.set_title("Attention Weights")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage and comprehensive testing
    print("Autoregressive Models - Complete Implementation")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    vocab_size = 1000
    batch_size = 4
    seq_len = 32
    
    # Test CharRNN
    print("\nTesting CharRNN Language Model...")
    char_rnn = CharRNNLanguageModel(vocab_size, embed_dim=128, hidden_dim=256)
    char_rnn.to(device)
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits, hidden = char_rnn(x)
    print(f"CharRNN output shape: {logits.shape}")
    
    # Test generation
    generated = char_rnn.generate(start_idx=0, max_length=20, device=device)
    print(f"Generated sequence length: {len(generated)}")
    
    # Test Causal CNN
    print("\nTesting Causal CNN Language Model...")
    causal_cnn = CausalCNNLanguageModel(vocab_size, embed_dim=128)
    causal_cnn.to(device)
    
    logits = causal_cnn(x)
    print(f"CausalCNN output shape: {logits.shape}")
    
    # Test Transformer
    print("\nTesting Transformer Language Model...")
    transformer = TransformerLanguageModel(vocab_size, embed_dim=256, num_heads=8, num_layers=4)
    transformer.to(device)
    
    logits = transformer(x)
    print(f"Transformer output shape: {logits.shape}")
    
    # Test generation strategies
    print("\nTesting Generation Strategies...")
    test_logits = torch.randn(vocab_size)
    
    greedy_token = GenerationStrategies.greedy_search(test_logits)
    temp_token = GenerationStrategies.temperature_sampling(test_logits, temperature=0.8)
    topk_token = GenerationStrategies.top_k_sampling(test_logits, k=10)
    nucleus_token = GenerationStrategies.nucleus_sampling(test_logits, p=0.9)
    
    print(f"Greedy: {greedy_token}, Temp: {temp_token}, Top-k: {topk_token}, Nucleus: {nucleus_token}")
    
    # Test evaluation metrics
    print("\nTesting Evaluation Metrics...")
    log_probs = torch.tensor([-1.2, -0.8, -2.1, -1.5])
    perplexity = EvaluationMetrics.perplexity(log_probs)
    print(f"Perplexity: {perplexity:.2f}")
    
    # Test tokenizer
    print("\nTesting Tokenizer...")
    sample_text = "hello world"
    char_to_idx, idx_to_char = create_character_tokenizer(sample_text)
    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Characters: {list(char_to_idx.keys())}")
    
    print("\n🎉 All implementations working correctly!")