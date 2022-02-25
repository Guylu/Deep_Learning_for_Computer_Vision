import unittest

import torch
import torch.optim

# import autograd
from optim import MomentumSGD


def _test_zero(x):
  assert (x is None) or torch.all(x == 0), 'x is not zero'


class TestMomentumSGD(unittest.TestCase):

  def _test_steps(self, lr, momentum, steps=1):
    param = torch.rand(size=())
    optim = MomentumSGD([param], lr=lr, momentum=momentum)
    ref_param = param.detach().clone()
    ref_optim = torch.optim.SGD([ref_param], lr=lr, momentum=momentum)
    dbg = f'param: {param}   param (ref): {ref_param}\n'
    for step in range(1, steps + 1):
      g = torch.rand_like(param)
      param.grad = g
      ref_param.grad = g.clone()
      optim.step()
      ref_optim.step()
      dbg += f'step: {step}  grad: {g}  param: {param}  param (ref): {ref_param}'
      torch.testing.assert_allclose(param, ref_param, msg=dbg)

  def testZeroGrad(self):
    parameters = [torch.randn(size=(2, 3)) for _ in range(4)]
    optim = MomentumSGD(parameters, lr=1e-2, momentum=0)
    for param in parameters:
      param.grad = torch.rand_like(param)
    optim.zero_grad()
    for param in parameters:
      _test_zero(param.grad)

  def test1StepNoMomentum(self):
    self._test_steps(lr=1e-2, momentum=0, steps=1)

  def test1StepMomentum(self):
    self._test_steps(lr=1e-2, momentum=0.1, steps=1)

  def test2StepsNoMomentum(self):
    self._test_steps(lr=1e-2, momentum=0, steps=2)

  def test2StepsMomentum(self):
    self._test_steps(lr=1e-2, momentum=0.1, steps=2)

  def test10StepsNoMomentum(self):
    self._test_steps(lr=1e-2, momentum=0.0, steps=10)

  def test10StepsMomentum(self):
    self._test_steps(lr=1e-2, momentum=0.1, steps=10)
