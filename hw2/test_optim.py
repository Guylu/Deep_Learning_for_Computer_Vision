import unittest

import torch

# import autograd
import optim
from optim import SGD


def test_zero(x):
  if x is None:
    return
  assert torch.all(x == 0), 'x is not zero'


class TestSGD(unittest.TestCase):

  def setUp(self):
    self.parameters = [torch.randn(4, 5), torch.randn(5, 6)]
    self.lr = 4e-3
    self.optimizer = SGD(self.parameters, lr=self.lr)

  def testZeroGrad(self):
    for x in self.parameters:
      x.grad = torch.randn_like(x)
    self.optimizer.zero_grad()
    for x in self.parameters:
      test_zero(x.grad)

  def testStep(self):
    for x in self.parameters:
      x.grad = torch.randn_like(x)
    orig = [(x.clone(), x.grad.clone()) for x in self.parameters]
    ref = [x - self.lr * x.grad for x in self.parameters]
    self.optimizer.step()
    for i, ((x0, g0), x, y) in enumerate(zip(orig, self.parameters, ref)):
      dbg = f'x (before): {x0}\nx.grad (before): {g0}\nx (after): {x}\nexpected: {y}'
      torch.testing.assert_allclose(x, y, msg=f'wrong output (parameter #{i}).\n{dbg}')
