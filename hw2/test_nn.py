import unittest

import torch

import functional
import nn


class TestLinear(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.linear = nn.Linear(in_dim=10, out_dim=20)

  @classmethod
  def tearDownClass(cls):
    del cls.linear

  def testContext(self):
    ctx = []
    x = torch.randn(32, 10)
    y = self.linear(x, ctx=ctx)
    assert len(ctx) == 1, 'Linear.forward does not use context'
    assert ctx[0][0] is functional.linear_backward, 'Linear.forward does not use functional.linear'

  def testShape(self):
    x = torch.randn(32, 10)
    y = self.linear(x)
    expected = (32, 20)
    dbg = f'got: {y.shape}. expected: {expected}.'
    assert y.shape == expected, f'Linear.forward output shape is wrong.\n{dbg}'

  def testParameters(self):
    # weight
    assert 'weight' in self.linear._parameters, '"weight" is not in self._parameters'
    assert hasattr(self.linear, 'weight'), "self.weight doesn't exist"
    assert self.linear.weight.size() == (20, 10), f"self.weight has the wrong size. expected: (20, 10); got: {self.weight.shape}"
    # bias
    assert 'bias' in self.linear._parameters, '"bias" is not in self._parameters'
    assert hasattr(self.linear, 'bias'), "self.bias doesn't exist"
    assert self.linear.bias.size() == (20,), f"self.bias has the wrong size. expected: (20,); got: {self.bias.shape}"

  def testForward(self):
    x = torch.randn(32, 10)
    y = self.linear(x)
    ref = functional.linear(x, self.linear.weight, self.linear.bias)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')
