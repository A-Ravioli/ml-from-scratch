"""
Test Suite for Graph Convolutional Networks

Comprehensive tests for spectral, spatial, and attention-based graph convolutions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import scipy.sparse as sp
from typing import Dict, List, Tuple

from exercise import (
    SpectralGraphConv, ChebyshevGraphConv, SpatialGraphConv,
    GraphSAGEConv, GraphAttentionConv, ResidualGCNLayer,
    GraphNeuralNetwork, GraphSAINT, GraphDataLoader
)


class TestSpectralGraphConv:
    """Test spectral graph convolution implementation"""
    
    @pytest.fixture
    def graph_data(self):
        """Create synthetic graph for testing"""
        num_nodes = 10
        num_features = 8
        
        # Create simple path graph
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes - 1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
        
        # Compute Laplacian
        degree = torch.diag(adj.sum(dim=1))
        laplacian = degree - adj
        
        x = torch.randn(num_nodes, num_features)
        
        return x, adj, laplacian
    
    def test_initialization(self):
        """Test spectral GCN initialization"""
        conv = SpectralGraphConv(16, 32)
        assert conv.in_features == 16
        assert conv.out_features == 32
    
    def test_eigendecomposition(self, graph_data):
        """Test eigendecomposition computation"""
        x, adj, laplacian = graph_data
        conv = SpectralGraphConv(8, 16)
        
        eigenvals, eigenvecs = conv._compute_eigendecomposition(laplacian)
        
        assert eigenvals.shape == (10,)
        assert eigenvecs.shape == (10, 10)
        
        # Check if eigendecomposition is correct
        reconstructed = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
        assert torch.allclose(reconstructed, laplacian, atol=1e-4)
    
    def test_forward_pass(self, graph_data):
        """Test forward pass"""
        x, adj, laplacian = graph_data
        conv = SpectralGraphConv(8, 16)
        
        output = conv(x, laplacian)
        
        assert output.shape == (10, 16)
        assert not torch.isnan(output).any()
    
    def test_spectral_filter(self, graph_data):
        """Test spectral filter application"""
        x, adj, laplacian = graph_data
        conv = SpectralGraphConv(8, 8)
        
        eigenvals = torch.tensor([0., 0.5, 1.0, 1.5, 2.0])
        filtered = conv._spectral_filter(eigenvals)
        
        assert filtered.shape == eigenvals.shape
        assert not torch.isnan(filtered).any()


class TestChebyshevGraphConv:
    """Test Chebyshev polynomial graph convolution"""
    
    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing"""
        num_nodes = 8
        num_features = 6
        
        # Create cycle graph
        adj = torch.zeros(num_nodes, num_features)
        for i in range(num_nodes):
            adj[i, (i+1) % num_nodes] = 1
            adj[(i+1) % num_nodes, i] = 1
        
        degree = torch.diag(adj.sum(dim=1))
        laplacian = degree - adj
        
        x = torch.randn(num_nodes, num_features)
        
        return x, adj, laplacian
    
    def test_initialization(self):
        """Test Chebyshev GCN initialization"""
        conv = ChebyshevGraphConv(10, 20, K=3)
        assert conv.in_features == 10
        assert conv.out_features == 20
        assert conv.K == 3
    
    def test_chebyshev_polynomial(self):
        """Test Chebyshev polynomial computation"""
        conv = ChebyshevGraphConv(5, 5, K=3)
        
        x = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0])
        
        # Test T_0(x) = 1
        t0 = conv._chebyshev_polynomial(x, 0)
        assert torch.allclose(t0, torch.ones_like(x))
        
        # Test T_1(x) = x
        t1 = conv._chebyshev_polynomial(x, 1)
        assert torch.allclose(t1, x)
        
        # Test T_2(x) = 2x^2 - 1
        t2 = conv._chebyshev_polynomial(x, 2)
        expected_t2 = 2 * x**2 - 1
        assert torch.allclose(t2, expected_t2)
    
    def test_laplacian_normalization(self, graph_data):
        """Test Laplacian normalization for Chebyshev"""
        x, adj, laplacian = graph_data
        conv = ChebyshevGraphConv(6, 12, K=2)
        
        normalized_L = conv._normalize_laplacian(laplacian)
        
        assert normalized_L.shape == laplacian.shape
        
        # Check eigenvalues are in [-1, 1]
        eigenvals = torch.linalg.eigvals(normalized_L).real
        assert torch.all(eigenvals >= -1.1)  # Allow small numerical error
        assert torch.all(eigenvals <= 1.1)
    
    def test_forward_pass(self, graph_data):
        """Test Chebyshev convolution forward pass"""
        x, adj, laplacian = graph_data
        conv = ChebyshevGraphConv(6, 12, K=3)
        
        output = conv(x, laplacian)
        
        assert output.shape == (8, 12)
        assert not torch.isnan(output).any()


class TestSpatialGraphConv:
    """Test spatial graph convolution (Kipf & Welling GCN)"""
    
    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing"""
        num_nodes = 12
        num_features = 10
        
        # Create random sparse adjacency matrix
        adj = torch.rand(num_nodes, num_nodes) < 0.3
        adj = adj.float()
        adj = (adj + adj.T) / 2  # Make symmetric
        torch.fill_diagonal_(adj, 0)  # Remove self-loops
        
        x = torch.randn(num_nodes, num_features)
        
        return x, adj
    
    def test_initialization(self):
        """Test spatial GCN initialization"""
        conv = SpatialGraphConv(15, 30, add_self_loops=True)
        assert conv.in_features == 15
        assert conv.out_features == 30
        assert conv.add_self_loops == True
    
    def test_adjacency_normalization(self, graph_data):
        """Test adjacency matrix normalization"""
        x, adj = graph_data
        conv = SpatialGraphConv(10, 20)
        
        norm_adj = conv._normalize_adjacency(adj)
        
        assert norm_adj.shape == adj.shape
        assert not torch.isnan(norm_adj).any()
        
        # Check symmetry is preserved
        assert torch.allclose(norm_adj, norm_adj.T, atol=1e-5)
    
    def test_self_loops(self, graph_data):
        """Test self-loop addition"""
        x, adj = graph_data
        conv = SpatialGraphConv(10, 20, add_self_loops=True)
        
        norm_adj = conv._normalize_adjacency(adj)
        
        # Self-loops should be present (diagonal > 0)
        assert torch.all(torch.diag(norm_adj) > 0)
    
    def test_forward_pass(self, graph_data):
        """Test spatial GCN forward pass"""
        x, adj = graph_data
        conv = SpatialGraphConv(10, 20)
        
        output = conv(x, adj)
        
        assert output.shape == (12, 20)
        assert not torch.isnan(output).any()
    
    def test_message_passing(self, graph_data):
        """Test that message passing works correctly"""
        x, adj = graph_data
        conv = SpatialGraphConv(10, 10, use_bias=False)
        
        # Set weights to identity for easier testing
        with torch.no_grad():
            conv.weight.copy_(torch.eye(10))
        
        output = conv(x, adj)
        
        # Output should be different from input due to neighborhood aggregation
        assert not torch.allclose(output, x)


class TestGraphSAGEConv:
    """Test GraphSAGE convolution layer"""
    
    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing"""
        num_nodes = 15
        num_features = 8
        
        adj = torch.rand(num_nodes, num_nodes) < 0.2
        adj = adj.float()
        adj = (adj + adj.T) / 2
        torch.fill_diagonal_(adj, 0)
        
        x = torch.randn(num_nodes, num_features)
        
        return x, adj
    
    def test_initialization(self):
        """Test GraphSAGE initialization"""
        conv = GraphSAGEConv(12, 24, aggregation='mean')
        assert conv.in_features == 12
        assert conv.out_features == 24
        assert conv.aggregation == 'mean'
    
    def test_mean_aggregation(self, graph_data):
        """Test mean aggregation of neighbors"""
        x, adj = graph_data
        conv = GraphSAGEConv(8, 16, aggregation='mean')
        
        neighbor_features = conv._aggregate_neighbors(x, adj)
        
        assert neighbor_features.shape == x.shape
        assert not torch.isnan(neighbor_features).any()
    
    def test_max_aggregation(self, graph_data):
        """Test max aggregation of neighbors"""
        x, adj = graph_data
        conv = GraphSAGEConv(8, 16, aggregation='max')
        
        neighbor_features = conv._aggregate_neighbors(x, adj)
        
        assert neighbor_features.shape == x.shape
        assert not torch.isnan(neighbor_features).any()
    
    def test_forward_pass(self, graph_data):
        """Test GraphSAGE forward pass"""
        x, adj = graph_data
        conv = GraphSAGEConv(8, 16, aggregation='mean')
        
        output = conv(x, adj)
        
        assert output.shape == (15, 16)
        assert not torch.isnan(output).any()
    
    def test_concatenation(self, graph_data):
        """Test that self and neighbor features are concatenated"""
        x, adj = graph_data
        conv = GraphSAGEConv(8, 8)
        
        # Mock the aggregation to return zeros for easier testing
        def mock_aggregate(x_in, adj_in):
            return torch.zeros_like(x_in)
        
        conv._aggregate_neighbors = mock_aggregate
        
        output = conv(x, adj)
        
        # With zero neighbor features, output should depend only on self features
        assert output.shape == (15, 8)


class TestGraphAttentionConv:
    """Test Graph Attention Network convolution"""
    
    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing"""
        num_nodes = 10
        num_features = 6
        
        adj = torch.tensor([
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        ]).float()
        
        x = torch.randn(num_nodes, num_features)
        
        return x, adj
    
    def test_initialization(self):
        """Test GAT initialization"""
        conv = GraphAttentionConv(8, 16, num_heads=4, dropout=0.1)
        assert conv.in_features == 8
        assert conv.out_features == 16
        assert conv.num_heads == 4
        assert conv.dropout == 0.1
    
    def test_attention_weights_computation(self, graph_data):
        """Test attention weight computation"""
        x, adj = graph_data
        conv = GraphAttentionConv(6, 12, num_heads=2)
        
        attention_weights = conv._compute_attention_weights(x, adj)
        
        assert attention_weights.shape == (2, 10, 10)  # [num_heads, N, N]
        
        # Attention weights should sum to 1 for each node's neighbors
        # (after masking with adjacency)
        for head in range(2):
            for node in range(10):
                neighbors = adj[node].nonzero().flatten()
                if len(neighbors) > 0:
                    neighbor_weights = attention_weights[head, node, neighbors]
                    assert torch.allclose(neighbor_weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_multi_head_attention(self, graph_data):
        """Test multi-head attention mechanism"""
        x, adj = graph_data
        conv = GraphAttentionConv(6, 8, num_heads=3, concat=True)
        
        output = conv(x, adj)
        
        # With concatenation, output size should be num_heads * out_features
        assert output.shape == (10, 3 * 8)
        assert not torch.isnan(output).any()
    
    def test_attention_averaging(self, graph_data):
        """Test attention head averaging"""
        x, adj = graph_data
        conv = GraphAttentionConv(6, 8, num_heads=3, concat=False)
        
        output = conv(x, adj)
        
        # With averaging, output size should be out_features
        assert output.shape == (10, 8)
        assert not torch.isnan(output).any()
    
    def test_attention_masking(self, graph_data):
        """Test that attention is properly masked by adjacency"""
        x, adj = graph_data
        conv = GraphAttentionConv(6, 8, num_heads=1)
        
        attention_weights = conv._compute_attention_weights(x, adj)
        
        # Attention weights should be zero where adjacency is zero
        masked_weights = attention_weights[0] * (1 - adj)  # Should be all zeros
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))


class TestResidualGCNLayer:
    """Test residual GCN layer"""
    
    def test_initialization(self):
        """Test residual layer initialization"""
        layer = ResidualGCNLayer(16, conv_type='spatial', dropout=0.1)
        assert layer.features == 16
        assert layer.dropout == 0.1
    
    def test_residual_connection(self):
        """Test residual connection"""
        layer = ResidualGCNLayer(12, conv_type='spatial')
        
        x = torch.randn(8, 12)
        adj = torch.rand(8, 8) < 0.3
        adj = adj.float()
        
        output = layer(x, adj)
        
        assert output.shape == x.shape
        # Output should be different from input due to transformation + residual
        assert not torch.allclose(output, x)


class TestGraphNeuralNetwork:
    """Test complete GNN architecture"""
    
    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing"""
        num_nodes = 20
        num_features = 10
        num_classes = 4
        
        adj = torch.rand(num_nodes, num_nodes) < 0.15
        adj = adj.float()
        adj = (adj + adj.T) / 2
        torch.fill_diagonal_(adj, 0)
        
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        return x, adj, y
    
    def test_initialization(self):
        """Test GNN initialization"""
        gnn = GraphNeuralNetwork(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=8,
            num_classes=5,
            conv_type='spatial'
        )
        
        assert len(gnn.layers) > 0
    
    def test_forward_pass(self, graph_data):
        """Test GNN forward pass"""
        x, adj, y = graph_data
        
        gnn = GraphNeuralNetwork(
            input_dim=10,
            hidden_dims=[16, 8],
            output_dim=4,
            num_classes=4,
            conv_type='spatial'
        )
        
        logits = gnn(x, adj)
        
        assert logits.shape == (20, 4)
        assert not torch.isnan(logits).any()
    
    def test_different_conv_types(self, graph_data):
        """Test GNN with different convolution types"""
        x, adj, y = graph_data
        
        for conv_type in ['spatial', 'chebyshev']:
            gnn = GraphNeuralNetwork(
                input_dim=10,
                hidden_dims=[12],
                output_dim=6,
                num_classes=4,
                conv_type=conv_type
            )
            
            logits = gnn(x, adj)
            assert logits.shape == (20, 4)


class TestGraphSAINT:
    """Test GraphSAINT sampling methods"""
    
    @pytest.fixture
    def graph_data(self):
        """Create graph data for testing"""
        num_nodes = 50
        num_features = 8
        
        # Create random graph
        edge_prob = 0.1
        edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if torch.rand(1) < edge_prob:
                    edges.append([i, j])
                    edges.append([j, i])
        
        edge_index = torch.tensor(edges).t().contiguous() if edges else torch.empty(2, 0).long()
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, 3, (num_nodes,))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def test_initialization(self):
        """Test GraphSAINT initialization"""
        sampler = GraphSAINT(sampling_method='node', sample_coverage=10)
        assert sampler.sampling_method == 'node'
        assert sampler.sample_coverage == 10
    
    def test_node_sampling(self, graph_data):
        """Test node sampling"""
        sampler = GraphSAINT(sampling_method='node')
        
        sampled_data = sampler.sample_subgraph(graph_data, batch_size=20)
        
        assert isinstance(sampled_data, Data)
        assert sampled_data.x.size(0) <= 20  # At most 20 nodes
        assert sampled_data.x.size(1) == 8   # Same feature dimension
    
    def test_edge_sampling(self, graph_data):
        """Test edge sampling"""
        sampler = GraphSAINT(sampling_method='edge')
        
        num_edges = graph_data.edge_index.size(1) // 2  # Divide by 2 for undirected
        sample_size = min(10, num_edges)
        
        if sample_size > 0:
            sampled_data = sampler.sample_subgraph(graph_data, batch_size=sample_size)
            
            assert isinstance(sampled_data, Data)
            assert sampled_data.edge_index.size(1) <= 2 * sample_size  # At most sample_size edges
    
    def test_random_walk_sampling(self, graph_data):
        """Test random walk sampling"""
        sampler = GraphSAINT(sampling_method='walk')
        
        sampled_data = sampler.sample_subgraph(graph_data, batch_size=15)
        
        assert isinstance(sampled_data, Data)
        # Should contain some subset of original nodes
        assert sampled_data.x.size(0) <= graph_data.x.size(0)


class TestGraphDataLoader:
    """Test graph data loading and preprocessing"""
    
    def test_initialization(self):
        """Test data loader initialization"""
        loader = GraphDataLoader('cora', split='train')
        assert loader.dataset_name == 'cora'
        assert loader.split == 'train'
    
    def test_feature_normalization(self):
        """Test feature normalization"""
        loader = GraphDataLoader('cora')
        
        # Test with random features
        features = torch.randn(10, 5)
        normalized = loader._normalize_features(features)
        
        assert normalized.shape == features.shape
        assert not torch.isnan(normalized).any()
        
        # Check that rows are normalized (L2 norm = 1)
        row_norms = torch.norm(normalized, dim=1)
        assert torch.allclose(row_norms, torch.ones(10), atol=1e-5)


class TestIntegrationAndPerformance:
    """Integration tests for complete graph learning pipeline"""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline"""
        # Create synthetic dataset
        num_nodes = 30
        num_features = 8
        num_classes = 3
        
        adj = torch.rand(num_nodes, num_nodes) < 0.2
        adj = adj.float()
        adj = (adj + adj.T) / 2
        torch.fill_diagonal_(adj, 0)
        
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # Create train/val/test masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:15] = True
        val_mask[15:22] = True
        test_mask[22:] = True
        
        data = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        # Initialize model
        model = GraphNeuralNetwork(
            input_dim=num_features,
            hidden_dims=[16],
            output_dim=8,
            num_classes=num_classes,
            conv_type='spatial'
        )
        
        # Test forward pass
        logits = model(data.x, adj)
        
        assert logits.shape == (num_nodes, num_classes)
        assert not torch.isnan(logits).any()
        
        # Test loss computation
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits[train_mask], y[train_mask])
        
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through GNN layers"""
        num_nodes = 15
        adj = torch.eye(num_nodes)  # Simple identity adjacency
        x = torch.randn(num_nodes, 5, requires_grad=True)
        y = torch.randint(0, 2, (num_nodes,))
        
        model = GraphNeuralNetwork(
            input_dim=5,
            hidden_dims=[8],
            output_dim=4,
            num_classes=2,
            conv_type='spatial'
        )
        
        logits = model(x, adj)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        
        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_memory_efficiency(self):
        """Test memory efficiency of different architectures"""
        num_nodes = 100
        adj = torch.rand(num_nodes, num_nodes) < 0.05
        adj = adj.float()
        x = torch.randn(num_nodes, 16)
        
        models = {
            'spatial': SpatialGraphConv(16, 32),
            'chebyshev': ChebyshevGraphConv(16, 32, K=2),
            'sage': GraphSAGEConv(16, 32),
            'gat': GraphAttentionConv(16, 32, num_heads=2)
        }
        
        for name, model in models.items():
            try:
                output = model(x, adj)
                assert output.shape == (num_nodes, 32)
                print(f"{name} model: Memory test passed")
            except Exception as e:
                pytest.fail(f"{name} model failed memory test: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])