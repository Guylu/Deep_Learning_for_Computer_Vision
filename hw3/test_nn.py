import unittest

import torch
import functional

from nn import Conv2d
from nn import MaxPool2d


class TestConv2d(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.conv_layer = Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            dilation=1,
                            groups=1,
                            bias=True)

  @classmethod
  def tearDownClass(cls):
    del cls.conv_layer

  def testContext(self):
    ctx = []
    x = torch.randn(1, 16, 5, 5)
    y = self.conv_layer(x, ctx=ctx)
    assert len(ctx) == 1, 'Conv2d.forward does not use context'
    assert ctx[0][0] is functional.conv2d_backward, 'Conv2d.forward does not use functional.conv2d_backward'

  def testParameters(self):
    # weight
    assert 'weight' in self.conv_layer._parameters, '"weight" is not in self._parameters'
    assert hasattr(self.conv_layer, 'weight'), "self.weight doesn't exist"
    assert self.conv_layer.weight.size() == (
      32, 16, 3, 3), f"self.weight has the wrong size. expected: (32, 16, 3, 3); got: {self.conv_layer.weight.shape}"

    # bias
    assert 'bias' in self.conv_layer._parameters, '"bias" is not in self._parameters'
    assert hasattr(self.conv_layer, 'bias'), "self.bias doesn't exist"
    assert self.conv_layer.bias.size() == (
      32,), f"self.bias has the wrong size. expected: (32,); got: {self.conv_layer.bias.shape}"

    # Conv2D attributes
    # assert hasattr(self.conv_layer, 'padding'), "self.padding doesn't exist"
    # assert hasattr(self.conv_layer, 'stride'), "self.stride doesn't exist"
    # assert hasattr(self.conv_layer, 'dilation'), "self.dilation doesn't exist"
    # assert hasattr(self.conv_layer, 'groups'), "self.groups doesn't exist"

  def testForward(self):
    x = torch.randn(1, 16, 5, 5)
    y = self.conv_layer(x)
    ref = functional.conv2d(x, self.conv_layer.weight, self.conv_layer.bias, padding=1, stride=1, dilation=1, groups=1)
    dbg = f'got: {y}\nexpected: {ref}'
    assert torch.allclose(y, ref), f'wrong output:\n{dbg}'


################################
class TestMaxPool2d(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.max_pool = MaxPool2d(kernel_size=2,
                             padding=0,
                             stride=2,
                             dilation=1)

  @classmethod
  def tearDownClass(cls):
    del cls.max_pool

  def testContext(self):
    ctx = []
    x = torch.randn(1, 16, 8, 8)
    y = self.max_pool(x, ctx=ctx)
    assert len(ctx) == 1, 'MaxPool2d.forward does not use context'
    assert ctx[0][0] is functional.max_pool2d_backward, 'MaxPool2d.forward does not use functional.max_pool2d_backward'

  def testParameters(self):
    # Parameters
    assert len(self.max_pool._parameters) == 0, 'Too many parameters for Max Pooling layer'

    # Attributes
    # assert hasattr(self.max_pool, 'kernel_size'), "self.kernel_size doesn't exist"
    # assert hasattr(self.max_pool, 'padding'), "self.padding doesn't exist"
    # assert hasattr(self.max_pool, 'stride'), "self.stride doesn't exist"
    # assert hasattr(self.max_pool, 'dilation'), "self.dilation doesn't exist"

  def testForward(self):
    x = torch.randn(1, 16, 8, 8)
    y = self.max_pool(x)
    ref = functional.max_pool2d(x, kernel_size=2, padding=0, stride=2, dilation=1)
    dbg = f'got: {y}\nexpected: {ref}'
    assert torch.allclose(y, ref), f'wrong output:\n{dbg}'
