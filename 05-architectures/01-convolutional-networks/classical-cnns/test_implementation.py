import numpy as np


def test_conv_layer_forward_shape():
    from exercise import ConvLayer

    np.random.seed(0)
    layer = ConvLayer(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
    x = np.random.randn(4, 2, 7, 9)
    y = layer.forward(x)
    # Output dims: floor((H + 2P - K)/S) + 1
    assert y.shape == (4, 3, (7 + 2 * 1 - 3) // 2 + 1, (9 + 2 * 1 - 3) // 2 + 1)


def test_pooling_layer_forward_shape_and_values():
    from exercise import PoolingLayer

    layer = PoolingLayer(pool_size=2, stride=2)
    x = np.array([[[[1.0, 2.0], [3.0, -1.0]]]])  # (1,1,2,2)
    y = layer.forward(x)
    assert y.shape == (1, 1, 1, 1)
    assert float(y[0, 0, 0, 0]) == 3.0


def test_linear_layer_forward_shape():
    from exercise import LinearLayer

    np.random.seed(0)
    layer = LinearLayer(in_features=5, out_features=2)
    x = np.random.randn(7, 5)
    y = layer.forward(x)
    assert y.shape == (7, 2)


def test_lenet5_forward_shape_small_batch():
    from exercise import LeNet5

    np.random.seed(0)
    model = LeNet5(num_classes=10)
    x = np.random.randn(2, 1, 32, 32)
    y = model.forward(x)
    assert y.shape == (2, 10)


def test_dropout_determinism_with_seed():
    from exercise import AlexNet

    np.random.seed(0)
    model = AlexNet(num_classes=5)
    x = np.ones((2, 10))
    np.random.seed(123)
    y1 = model.dropout(x, rate=0.5)
    np.random.seed(123)
    y2 = model.dropout(x, rate=0.5)
    assert np.array_equal(y1, y2)

