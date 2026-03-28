import torch

from exercise import ChebyshevGraphConv, GraphNeuralNetwork, SpatialGraphConv, SpectralGraphConv


def _toy_graph():
    adj = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    degree = torch.diag(adj.sum(dim=1))
    laplacian = degree - adj
    x = torch.randn(4, 3)
    return x, adj, laplacian


def test_spectral_graph_conv_shape():
    x, _, laplacian = _toy_graph()
    conv = SpectralGraphConv(3, 5)
    eigenvalues, eigenvectors = conv._compute_eigendecomposition(laplacian)
    out = conv(x, laplacian)
    assert eigenvalues.shape == (4,)
    assert eigenvectors.shape == (4, 4)
    assert out.shape == (4, 5)


def test_chebyshev_polynomial_and_forward():
    x, _, laplacian = _toy_graph()
    conv = ChebyshevGraphConv(3, 4, K=3)
    poly = conv._chebyshev_polynomial(torch.tensor([0.0, 0.5, 1.0]), 2)
    out = conv(x, laplacian)
    assert poly.shape == (3,)
    assert out.shape == (4, 4)


def test_spatial_gnn_forward():
    x, adj, _ = _toy_graph()
    model = GraphNeuralNetwork(3, 6, 2, num_layers=2)
    logits = model(x, adj)
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()
