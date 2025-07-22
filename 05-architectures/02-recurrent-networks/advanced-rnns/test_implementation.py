"""
Test Suite for Advanced RNN Architectures

Comprehensive tests for Neural Turing Machines, Differentiable Neural Computers,
and memory-augmented networks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from exercise import (
    ContentAddressing, LocationAddressing, NeuralTuringMachine,
    DifferentiableNeuralComputer, MemoryNetwork, DifferentiableStack,
    AdaptiveComputationTime, BenchmarkTasks, MemoryState
)


class TestContentAddressing:
    """Test content-based addressing mechanism"""
    
    def test_initialization(self):
        """Test content addressing initialization"""
        content_addr = ContentAddressing(memory_size=16, key_size=8)
        assert content_addr.memory_size == 16
        assert content_addr.key_size == 8
    
    def test_forward_shape(self):
        """Test content addressing forward pass shapes"""
        content_addr = ContentAddressing(memory_size=16, key_size=8)
        
        batch_size = 4
        key = torch.randn(batch_size, 8)
        beta = torch.ones(batch_size, 1)
        memory = torch.randn(batch_size, 16, 8)
        
        weights = content_addr(key, beta, memory)
        
        assert weights.shape == (batch_size, 16)
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    def test_similarity_computation(self):
        """Test cosine similarity computation"""
        content_addr = ContentAddressing(memory_size=4, key_size=3)
        
        # Create identical key and memory slot
        key = torch.tensor([[1.0, 0.0, 0.0]])
        memory = torch.tensor([[[1.0, 0.0, 0.0],  # Identical
                              [0.0, 1.0, 0.0],   # Orthogonal
                              [0.5, 0.5, 0.0],   # Similar
                              [-1.0, 0.0, 0.0]]]) # Opposite
        beta = torch.ones(1, 1)
        
        weights = content_addr(key, beta, memory)
        
        # Highest weight should be for identical vector
        assert torch.argmax(weights[0]) == 0
    
    def test_beta_scaling(self):
        """Test beta parameter scaling effect"""
        content_addr = ContentAddressing(memory_size=3, key_size=2)
        
        key = torch.ones(1, 2)
        memory = torch.tensor([[[1.0, 1.0], [1.0, 0.9], [0.0, 0.0]]])
        
        # Low beta - more uniform weights
        beta_low = torch.tensor([[0.1]])
        weights_low = content_addr(key, beta_low, memory)
        
        # High beta - more concentrated weights
        beta_high = torch.tensor([[10.0]])
        weights_high = content_addr(key, beta_high, memory)
        
        # High beta should be more concentrated (higher max weight)
        assert torch.max(weights_high) > torch.max(weights_low)


class TestLocationAddressing:
    """Test location-based addressing mechanism"""
    
    def test_initialization(self):
        """Test location addressing initialization"""
        loc_addr = LocationAddressing(memory_size=8, shift_range=2)
        assert loc_addr.memory_size == 8
        assert loc_addr.shift_range == 2
    
    def test_circular_convolution(self):
        """Test circular convolution for shifting"""
        loc_addr = LocationAddressing(memory_size=5, shift_range=1)
        
        # Single peak at position 2
        weights = torch.zeros(1, 5)
        weights[0, 2] = 1.0
        
        # Shift right by 1 position
        shift = torch.zeros(1, 3)  # [-1, 0, +1]
        shift[0, 2] = 1.0  # +1 shift
        
        shifted = loc_addr._circular_convolution(weights, shift)
        
        # Peak should move from position 2 to position 3
        assert torch.argmax(shifted[0]) == 3
    
    def test_sharpening(self):
        """Test attention sharpening"""
        loc_addr = LocationAddressing(memory_size=4, shift_range=1)
        
        # Uniform weights
        weights = torch.ones(1, 4) / 4
        
        # Apply strong sharpening
        gamma = torch.tensor([[10.0]])
        sharpened = loc_addr._sharpen(weights, gamma)
        
        # Should still sum to 1
        assert torch.allclose(sharpened.sum(), torch.tensor(1.0))
        
        # Should be more concentrated than original
        assert torch.max(sharpened) > torch.max(weights)
    
    def test_full_addressing(self):
        """Test complete location addressing pipeline"""
        loc_addr = LocationAddressing(memory_size=6, shift_range=1)
        
        content_weights = torch.ones(1, 6) / 6
        prev_weights = torch.zeros(1, 6)
        prev_weights[0, 0] = 1.0  # Peak at position 0
        
        gate = torch.tensor([[0.5]])  # Interpolate 50-50
        shift = torch.zeros(1, 3)
        shift[0, 1] = 1.0  # No shift
        gamma = torch.tensor([[1.0]])  # No sharpening
        
        final_weights = loc_addr(content_weights, prev_weights, gate, shift, gamma)
        
        assert final_weights.shape == (1, 6)
        assert torch.allclose(final_weights.sum(), torch.tensor(1.0))


class TestNeuralTuringMachine:
    """Test Neural Turing Machine implementation"""
    
    @pytest.fixture
    def ntm_model(self):
        """Create NTM model for testing"""
        return NeuralTuringMachine(
            input_size=4, output_size=4, hidden_size=16,
            memory_size=8, memory_dim=6, num_heads=1
        )
    
    def test_initialization(self, ntm_model):
        """Test NTM initialization"""
        assert ntm_model.input_size == 4
        assert ntm_model.memory_size == 8
        assert ntm_model.memory_dim == 6
    
    def test_memory_state_initialization(self, ntm_model):
        """Test memory state initialization"""
        batch_size = 2
        device = torch.device('cpu')
        
        initial_state = ntm_model._initialize_state(batch_size, device)
        
        assert initial_state.memory.shape == (batch_size, 8, 6)
        assert initial_state.read_weights.shape[0] == batch_size
        assert initial_state.write_weights.shape[0] == batch_size
    
    def test_memory_read(self, ntm_model):
        """Test memory reading operation"""
        batch_size = 2
        memory = torch.randn(batch_size, 8, 6)
        read_weights = torch.zeros(batch_size, 1, 8)
        read_weights[:, 0, 0] = 1.0  # Read from first location
        
        read_vectors = ntm_model._read_memory(memory, read_weights)
        
        assert read_vectors.shape == (batch_size, 6)  # 1 head * 6 dims
        # Should match first memory location
        assert torch.allclose(read_vectors, memory[:, 0, :])
    
    def test_memory_write(self, ntm_model):
        """Test memory writing operation"""
        batch_size = 2
        memory = torch.zeros(batch_size, 8, 6)
        
        write_weights = torch.zeros(batch_size, 8)
        write_weights[:, 0] = 1.0  # Write to first location
        
        erase_vector = torch.zeros(batch_size, 6)
        add_vector = torch.ones(batch_size, 6)
        
        new_memory = ntm_model._write_memory(memory, write_weights, erase_vector, add_vector)
        
        # First location should be all ones
        assert torch.allclose(new_memory[:, 0, :], torch.ones(batch_size, 6))
        # Other locations should remain zero
        assert torch.allclose(new_memory[:, 1:, :], torch.zeros(batch_size, 7, 6))
    
    def test_forward_pass(self, ntm_model):
        """Test complete forward pass"""
        batch_size, seq_len = 2, 5
        input_seq = torch.randn(batch_size, seq_len, 4)
        
        output_seq, final_state = ntm_model(input_seq)
        
        assert output_seq.shape == (batch_size, seq_len, 4)
        assert isinstance(final_state, MemoryState)


class TestDifferentiableNeuralComputer:
    """Test Differentiable Neural Computer implementation"""
    
    @pytest.fixture
    def dnc_model(self):
        """Create DNC model for testing"""
        return DifferentiableNeuralComputer(
            input_size=6, output_size=6, hidden_size=20,
            memory_size=16, memory_dim=8, num_read_heads=2
        )
    
    def test_initialization(self, dnc_model):
        """Test DNC initialization"""
        assert dnc_model.memory_size == 16
        assert dnc_model.num_read_heads == 2
    
    def test_allocation_weights(self, dnc_model):
        """Test memory allocation mechanism"""
        batch_size = 3
        
        # All memory free
        usage_free = torch.zeros(batch_size, 16)
        allocation_free = dnc_model._compute_allocation_weights(usage_free)
        
        # Should allocate to first location
        assert torch.allclose(allocation_free[:, 0], torch.ones(batch_size))
        assert torch.allclose(allocation_free[:, 1:].sum(dim=1), torch.zeros(batch_size))
        
        # Half memory used
        usage_half = torch.ones(batch_size, 16) * 0.5
        allocation_half = dnc_model._compute_allocation_weights(usage_half)
        
        # Should be more uniform allocation
        assert allocation_half.std(dim=1).mean() > 0
    
    def test_usage_update(self, dnc_model):
        """Test usage vector update"""
        batch_size = 2
        usage = torch.zeros(batch_size, 16)
        
        write_weights = torch.zeros(batch_size, 16)
        write_weights[:, 0] = 1.0  # Write to first location
        
        read_weights = torch.zeros(batch_size, 2, 16)
        read_weights[:, 0, 1] = 1.0  # Read from second location
        
        free_gates = torch.ones(batch_size, 2)  # Free all reads
        
        new_usage = dnc_model._update_usage(usage, write_weights, read_weights, free_gates)
        
        # First location should have high usage (written to)
        assert new_usage[:, 0].mean() > 0
    
    def test_temporal_links(self, dnc_model):
        """Test temporal link matrix update"""
        batch_size = 2
        memory_size = 16
        
        link_matrix = torch.zeros(batch_size, memory_size, memory_size)
        precedence = torch.zeros(batch_size, memory_size)
        
        write_weights = torch.zeros(batch_size, memory_size)
        write_weights[:, 0] = 1.0  # Write to location 0
        
        new_link, new_precedence = dnc_model._update_temporal_links(
            link_matrix, precedence, write_weights
        )
        
        assert new_link.shape == (batch_size, memory_size, memory_size)
        assert new_precedence.shape == (batch_size, memory_size)
        # Location 0 should have precedence
        assert new_precedence[:, 0].mean() > 0


class TestMemoryNetwork:
    """Test Memory Network implementation"""
    
    @pytest.fixture
    def memory_net(self):
        """Create Memory Network for testing"""
        return MemoryNetwork(vocab_size=100, embed_dim=32, 
                           memory_size=10, num_hops=2)
    
    def test_initialization(self, memory_net):
        """Test Memory Network initialization"""
        assert memory_net.vocab_size == 100
        assert memory_net.num_hops == 2
    
    def test_input_encoding(self, memory_net):
        """Test input fact encoding"""
        batch_size = 4
        num_facts = 6
        seq_len = 5
        
        inputs = torch.randint(0, 100, (batch_size, num_facts, seq_len))
        encoded = memory_net._encode_input(inputs)
        
        assert encoded.shape == (batch_size, num_facts, 32)
    
    def test_attention_hop(self, memory_net):
        """Test single attention hop"""
        batch_size = 4
        query = torch.randn(batch_size, 32)
        memories = torch.randn(batch_size, 10, 32)
        
        updated_query = memory_net._attention_hop(query, memories, hop_idx=0)
        
        assert updated_query.shape == (batch_size, 32)
    
    def test_forward_pass(self, memory_net):
        """Test complete forward pass"""
        batch_size = 3
        facts = torch.randint(0, 100, (batch_size, 8, 4))
        question = torch.randint(0, 100, (batch_size, 3))
        
        answer_logits = memory_net(facts, question)
        
        assert answer_logits.shape == (batch_size, 100)


class TestDifferentiableStack:
    """Test Differentiable Stack implementation"""
    
    @pytest.fixture
    def stack(self):
        """Create differentiable stack for testing"""
        return DifferentiableStack(batch_size=2, stack_depth=5, element_dim=4)
    
    def test_initialization(self, stack):
        """Test stack initialization"""
        assert stack.batch_size == 2
        assert stack.stack_depth == 5
        assert stack.element_dim == 4
    
    def test_push_operation(self, stack):
        """Test push operation"""
        values = torch.ones(2, 4)
        strengths = torch.ones(2) * 0.8
        
        prev_stack = torch.zeros(2, 5, 4)
        prev_strengths = torch.zeros(2, 5)
        
        new_stack, new_strengths = stack.push(values, strengths, prev_stack, prev_strengths)
        
        assert new_stack.shape == (2, 5, 4)
        assert new_strengths.shape == (2, 5)
        
        # Top of stack should have highest strength
        assert new_strengths[:, 0].mean() > new_strengths[:, 1:].mean()
    
    def test_pop_operation(self, stack):
        """Test pop operation"""
        # Initialize stack with some content
        prev_stack = torch.randn(2, 5, 4)
        prev_strengths = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0],
                                      [0.7, 0.3, 0.0, 0.0, 0.0]])
        
        pop_strengths = torch.ones(2) * 0.5
        
        popped_values, new_stack, new_strengths = stack.pop(
            pop_strengths, prev_stack, prev_strengths
        )
        
        assert popped_values.shape == (2, 4)
        assert new_stack.shape == (2, 5, 4)
        assert new_strengths.shape == (2, 5)
    
    def test_read_operation(self, stack):
        """Test read operation"""
        test_stack = torch.randn(2, 5, 4)
        strengths = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.6, 0.4, 0.0, 0.0, 0.0]])
        
        read_values = stack.read(test_stack, strengths)
        
        assert read_values.shape == (2, 4)
        
        # First batch should read exactly top element
        assert torch.allclose(read_values[0], test_stack[0, 0])


class TestAdaptiveComputationTime:
    """Test Adaptive Computation Time mechanism"""
    
    @pytest.fixture
    def act_model(self):
        """Create ACT model for testing"""
        return AdaptiveComputationTime(hidden_size=16, max_steps=3, ponder_cost=0.01)
    
    def test_initialization(self, act_model):
        """Test ACT initialization"""
        assert act_model.hidden_size == 16
        assert act_model.max_steps == 3
        assert act_model.ponder_cost == 0.01
    
    def test_halting_probability(self, act_model):
        """Test halting probability computation"""
        state = torch.randn(4, 16)
        halt_prob = act_model._compute_halting_probability(state)
        
        assert halt_prob.shape == (4,)
        assert torch.all(halt_prob >= 0) and torch.all(halt_prob <= 1)
    
    def test_state_update(self, act_model):
        """Test state update mechanism"""
        state = torch.randn(4, 16)
        new_state = act_model._update_state(state)
        
        assert new_state.shape == (4, 16)
    
    def test_forward_pass(self, act_model):
        """Test complete ACT forward pass"""
        initial_state = torch.randn(6, 16)
        final_state, ponder_cost = act_model(initial_state)
        
        assert final_state.shape == (6, 16)
        assert ponder_cost.shape == (6,)
        assert torch.all(ponder_cost >= 0)


class TestBenchmarkTasks:
    """Test benchmark task generation"""
    
    def test_copy_task(self):
        """Test copy task generation"""
        batch_size, seq_len, vocab_size = 4, 8, 10
        input_seq, target_seq = BenchmarkTasks.copy_task(batch_size, seq_len, vocab_size)
        
        assert input_seq.shape[0] == batch_size
        assert target_seq.shape[0] == batch_size
        assert input_seq.shape[1] == target_seq.shape[1]
    
    def test_associative_recall(self):
        """Test associative recall task"""
        batch_size, num_items, item_dim = 3, 5, 6
        input_seq, target = BenchmarkTasks.associative_recall(batch_size, num_items, item_dim)
        
        assert input_seq.shape[0] == batch_size
        assert target.shape[0] == batch_size
    
    def test_priority_sort(self):
        """Test priority sort task"""
        batch_size, seq_len = 5, 7
        input_seq, sorted_seq = BenchmarkTasks.priority_sort(batch_size, seq_len)
        
        assert input_seq.shape == sorted_seq.shape
        assert input_seq.shape[0] == batch_size


class TestIntegration:
    """Integration tests for complete systems"""
    
    def test_ntm_copy_task(self):
        """Test NTM on copy task"""
        ntm = NeuralTuringMachine(
            input_size=8, output_size=8, hidden_size=32,
            memory_size=16, memory_dim=8, num_heads=1
        )
        
        # Generate copy task
        batch_size, seq_len = 2, 6
        input_seq, target_seq = BenchmarkTasks.copy_task(batch_size, seq_len, 8)
        
        # Forward pass (no training, just shape check)
        output_seq, final_state = ntm(input_seq)
        
        assert output_seq.shape == target_seq.shape
        assert final_state.memory.shape == (batch_size, 16, 8)
    
    def test_dnc_associative_recall(self):
        """Test DNC on associative recall task"""
        dnc = DifferentiableNeuralComputer(
            input_size=10, output_size=10, hidden_size=40,
            memory_size=20, memory_dim=10, num_read_heads=2
        )
        
        # Generate associative recall task
        batch_size, num_items, item_dim = 3, 4, 5
        input_seq, target = BenchmarkTasks.associative_recall(batch_size, num_items, item_dim)
        
        # Forward pass
        output_seq, final_state = dnc(input_seq)
        
        assert output_seq.shape[0] == batch_size
        assert hasattr(final_state, 'link_matrix')
    
    def test_memory_network_qa(self):
        """Test Memory Network on simple QA"""
        memnet = MemoryNetwork(vocab_size=50, embed_dim=16, 
                              memory_size=8, num_hops=2)
        
        # Create simple QA scenario
        batch_size = 2
        facts = torch.randint(0, 50, (batch_size, 5, 3))
        question = torch.randint(0, 50, (batch_size, 2))
        
        answer_logits = memnet(facts, question)
        
        assert answer_logits.shape == (batch_size, 50)
        assert not torch.isnan(answer_logits).any()


def test_memory_capacity():
    """Test memory capacity of different architectures"""
    
    def test_architecture_capacity(model, task_generator, max_length=20):
        """Test how sequence length affects performance"""
        capacities = []
        
        for length in range(2, max_length, 2):
            try:
                # Generate task of given length
                input_seq, target_seq = task_generator(length)
                
                # Forward pass
                output_seq, _ = model(input_seq)
                
                # Simple success metric (no NaN outputs)
                success = not torch.isnan(output_seq).any()
                capacities.append((length, success))
                
            except Exception as e:
                capacities.append((length, False))
                break
        
        return capacities
    
    # Test NTM capacity
    ntm = NeuralTuringMachine(
        input_size=4, output_size=4, hidden_size=16,
        memory_size=32, memory_dim=8, num_heads=1
    )
    
    def copy_task_gen(length):
        return BenchmarkTasks.copy_task(1, length, 4)
    
    ntm_capacities = test_architecture_capacity(ntm, copy_task_gen)
    print(f"NTM capacity test: {ntm_capacities}")
    
    assert len(ntm_capacities) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])