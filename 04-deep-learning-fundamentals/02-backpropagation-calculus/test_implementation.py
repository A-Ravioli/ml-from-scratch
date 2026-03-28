import numpy as np

from exercise import AutogradLayer, AutogradNetwork, ComputationNode, add, matmul, mse_loss, multiply, numerical_gradient, relu


def test_add_and_multiply_backward():
    a = ComputationNode(np.array([1.0, 2.0]))
    b = ComputationNode(np.array([3.0, 4.0]))
    summed = add(a, b)
    product = multiply(summed, summed)
    product.backward(np.ones_like(product.value))
    assert np.all(a.grad != 0)
    assert np.all(b.grad != 0)


def test_matmul_and_relu_shapes():
    x = ComputationNode(np.ones((2, 3)))
    w = ComputationNode(np.ones((3, 4)))
    out = relu(matmul(x, w))
    assert out.value.shape == (2, 4)


def test_autograd_layer_and_network():
    layer = AutogradLayer(3, 2)
    x = ComputationNode(np.ones((4, 3)))
    out = layer.forward(x)
    assert out.value.shape == (4, 2)

    network = AutogradNetwork([3, 4, 2])
    net_out = network.forward(x)
    assert net_out.value.shape == (4, 2)


def test_mse_and_numerical_gradient():
    pred = ComputationNode(np.array([1.0, 2.0, 3.0]))
    target = ComputationNode(np.array([1.0, 1.0, 1.0]))
    loss = mse_loss(pred, target)
    assert loss.value.shape == ()

    grad = numerical_gradient(lambda z: np.sum(z ** 2), np.array([1.0, -2.0]))
    assert np.allclose(grad, np.array([2.0, -4.0]), atol=1e-4)
