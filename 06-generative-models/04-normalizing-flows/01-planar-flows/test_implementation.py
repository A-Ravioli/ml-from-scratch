import torch

from exercise import NormalizingFlow, PlanarFlow


def test_planar_flow_forward():
    flow = PlanarFlow(dim=3)
    z = torch.randn(4, 3)
    z_next, log_det = flow(z)
    assert z_next.shape == z.shape
    assert log_det.shape == (4,)


def test_normalizing_flow_stack():
    model = NormalizingFlow(dim=3, n_flows=2)
    z = torch.randn(5, 3)
    z_k, total_log_det = model(z)
    assert z_k.shape == z.shape
    assert total_log_det.shape == (5,)
