import numpy as np


def test_basic_block_shape_identity_shortcut():
    from exercise import BasicBlock

    np.random.seed(0)
    block = BasicBlock(8, 8, stride=1)
    x = np.random.randn(2, 8, 9, 9)
    y = block.forward(x, training=True)
    assert y.shape == x.shape


def test_basic_block_shape_projection_shortcut():
    from exercise import BasicBlock

    np.random.seed(0)
    block = BasicBlock(4, 6, stride=2)
    x = np.random.randn(2, 4, 10, 10)
    y = block.forward(x, training=True)
    assert y.shape == (2, 6, 5, 5)


def test_bottleneck_block_shape():
    from exercise import Bottleneck

    np.random.seed(0)
    block = Bottleneck(8, 16, stride=2)
    x = np.random.randn(2, 8, 9, 9)
    y = block.forward(x, training=True)
    assert y.shape == (2, 16, 5, 5)


def test_batchnorm_eval_uses_running_stats_shape():
    from exercise import BatchNorm2d

    np.random.seed(0)
    bn = BatchNorm2d(num_features=3, momentum=0.5)
    x = np.random.randn(4, 3, 5, 5)
    _ = bn.forward(x, training=True)
    y = bn.forward(x, training=False)
    assert y.shape == x.shape


def test_resnet18_constructor_layers_exist():
    from exercise import resnet18

    model = resnet18(num_classes=10)
    assert hasattr(model, "layer1") and len(model.layer1) == 2
    assert hasattr(model, "layer2") and len(model.layer2) == 2
    assert hasattr(model, "layer3") and len(model.layer3) == 2
    assert hasattr(model, "layer4") and len(model.layer4) == 2

