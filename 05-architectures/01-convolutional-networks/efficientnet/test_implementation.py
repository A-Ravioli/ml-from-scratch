import numpy as np


def test_depthwise_separable_conv_shapes_small():
    from exercise import DepthwiseSeparableConv

    np.random.seed(0)
    x = np.random.randn(1, 3, 7, 7)
    layer = DepthwiseSeparableConv(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
    y = layer.forward(x, training=False)
    assert y.shape == (1, 4, 7, 7)


def test_depthwise_separable_conv_stride_2_downsamples():
    from exercise import DepthwiseSeparableConv

    np.random.seed(0)
    x = np.random.randn(2, 2, 8, 10)
    layer = DepthwiseSeparableConv(in_channels=2, out_channels=5, kernel_size=3, stride=2, padding=1)
    y = layer.forward(x, training=False)
    assert y.shape[2] == 4
    assert y.shape[3] == 5


def test_squeeze_excitation_preserves_shape_and_changes_values():
    from exercise import SqueezeExcitation

    np.random.seed(0)
    x = np.random.randn(2, 4, 5, 5)
    se = SqueezeExcitation(channels=4, reduction_ratio=2)
    y = se.forward(x)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_swish_basic_properties():
    from exercise import Swish

    swish = Swish()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = swish.forward(x)
    assert abs(float(y[2])) < 1e-8
    # Swish is not globally monotone; it has a small dip for negative inputs.
    assert float(y[0]) < 0.0 and float(y[1]) < 0.0
    assert float(y[4]) > 0.0
    large = swish.forward(np.array([10.0]))
    assert abs(float(large[0]) - 10.0) < 0.1


def test_mbconv_block_shape_and_skip_flag():
    from exercise import MBConvBlock

    np.random.seed(0)
    x = np.random.randn(1, 3, 8, 8)
    block = MBConvBlock(in_channels=3, out_channels=3, stride=1, expansion_ratio=1, se_ratio=0.25)
    y = block.forward(x, training=False)
    assert y.shape == x.shape
    assert block.use_skip is True

    block2 = MBConvBlock(in_channels=3, out_channels=5, stride=2, expansion_ratio=1, se_ratio=0.25)
    y2 = block2.forward(x, training=False)
    assert y2.shape == (1, 5, 4, 4)
    assert block2.use_skip is False


def test_efficientnet_forward_small_config_fast():
    from exercise import EfficientNet

    np.random.seed(0)
    # Tiny config: one stage, one block.
    model = EfficientNet(
        width_coefficient=0.5,
        depth_coefficient=0.5,
        resolution=32,
        dropout_rate=0.0,
        num_classes=7,
        include_top=True,
    )
    x = np.random.randn(2, 3, 32, 32)
    y = model.forward(x, training=False)
    assert y.shape == (2, 7)
