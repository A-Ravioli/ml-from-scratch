import torch

from exercise import DynamicRouting, PrimaryCapsuleLayer, SquashFunction


def test_squash_function_bounds():
    squash = SquashFunction()
    x = torch.randn(4, 6)
    out = squash(x)
    lengths = torch.norm(out, dim=-1)
    assert out.shape == x.shape
    assert torch.all(lengths < 1.0)


def test_primary_capsule_layer_shape():
    layer = PrimaryCapsuleLayer(in_channels=4, out_channels=3, capsule_dim=5, kernel_size=3, stride=1)
    x = torch.randn(2, 4, 6, 6)
    out = layer(x)
    assert out.shape[0] == 2
    assert out.shape[-1] == 5


def test_dynamic_routing_shape():
    routing = DynamicRouting(8, 4, 5, 6, num_iterations=3)
    x = torch.randn(2, 8, 5)
    out = routing(x)
    assert out.shape == (2, 4, 6)
    assert torch.isfinite(out).all()
