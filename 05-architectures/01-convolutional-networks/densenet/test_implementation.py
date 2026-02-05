import numpy as np


def test_dense_layer_output_shape_and_nonnegativity():
    from exercise import DenseLayer

    np.random.seed(0)
    x = np.random.randn(2, 6, 8, 8)
    layer = DenseLayer(in_channels=6, growth_rate=4, bottleneck_factor=2)
    y = layer.forward(x, training=False)
    assert y.shape == (2, 4, 8, 8)
    assert np.all(y >= 0.0)


def test_dense_block_channel_growth():
    from exercise import DenseBlock

    np.random.seed(0)
    x = np.random.randn(1, 5, 8, 8)
    block = DenseBlock(in_channels=5, num_layers=3, growth_rate=2)
    y = block.forward(x, training=False)
    assert y.shape == (1, 5 + 3 * 2, 8, 8)


def test_transition_layer_compresses_and_downsamples():
    from exercise import TransitionLayer

    np.random.seed(0)
    x = np.random.randn(2, 10, 9, 11)
    layer = TransitionLayer(in_channels=10, compression_factor=0.5)
    y = layer.forward(x, training=False)
    assert y.shape[1] == 5
    assert y.shape[2] == 9 // 2
    assert y.shape[3] == 11 // 2
    assert np.all(y >= 0.0)


def test_densenet_forward_small_config_fast():
    from exercise import DenseNet

    np.random.seed(0)
    model = DenseNet(growth_rate=4, block_config=[2, 2], num_classes=7, compression_factor=0.5, initial_channels=8)
    x = np.random.randn(1, 3, 32, 32)
    y = model.forward(x, training=False)
    assert y.shape == (1, 7)


def test_memory_efficient_densenet_forward_matches_shape():
    from exercise import MemoryEfficientDenseNet

    np.random.seed(0)
    model = MemoryEfficientDenseNet(growth_rate=4, block_config=[2, 2], num_classes=7, compression_factor=0.5, initial_channels=8)
    x = np.random.randn(2, 3, 32, 32)
    y = model.forward(x, training=False)
    assert y.shape == (2, 7)


def test_parameter_counting_returns_positive_int():
    from exercise import DenseNet

    model = DenseNet(growth_rate=4, block_config=[2, 2], num_classes=7, compression_factor=0.5, initial_channels=8)
    n = model.count_parameters()
    assert isinstance(n, int)
    assert n > 0

