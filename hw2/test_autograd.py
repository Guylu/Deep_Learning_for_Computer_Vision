import unittest

import torch

# import autograd
import autograd
from functional import mean
from autograd import backward


def mul_2(x, ctx=None):
  y = 2 * x
  if ctx is not None:
    ctx.append([mul_2_backward, [y, x]])
  return y


def mul_2_backward(y, x):
  x.grad += 0.5 * y.grad


def add_1(x, ctx=None):
  y = x + 1
  if ctx is not None:
    ctx.append([add_1_backward, [y, x]])
  return y


def add_1_backward(y, x):
  x.grad += y.grad


def add(x1, x2, ctx=None):
  assert x1.shape == x2.shape
  y = x1 + x2
  if ctx is not None:
    ctx.append([add_backward, [y, x1, x2]])
  return y


def add_backward(y, x1, x2):
  x1.grad += y.grad
  x2.grad += y.grad


class TestBackward(unittest.TestCase):

  def testMean(self):
    ctx = []
    x = torch.randn(4, 5)
    y = mean(x, ctx=ctx)
    backward(y, ctx)
    ref_x_grad = torch.full_like(x, 1 / x.numel())
    to_test = [
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      assert var.grad is not None, f'{name}.grad is not set by backward'
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')

  def testTwoMeans(self):
    ctx = []
    x = torch.randn(4, 5)
    x1 = mean(x, ctx=ctx)
    y = mean(x1, ctx=ctx)
    backward(y, ctx)
    ref_x1_grad = torch.full_like(x1, 1 / x1.numel())
    ref_x_grad = torch.full_like(x, 1 / x.numel())
    to_test = [
      ('x1', x1, ref_x1_grad),
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      assert var.grad is not None, f'{name}.grad is not set by backward'
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')

  def testManyCalls(self):
    ctx = []
    x = torch.randn(4, 5)
    x1 = add_1(x, ctx=ctx)
    x2 = mul_2(x1, ctx=ctx)
    x3 = add(x1, x2, ctx=ctx)
    x4 = mul_2(x3, ctx=ctx)
    y = mean(x4, ctx=ctx)
    backward(y, ctx)
    ref_x4_grad = torch.full_like(x4, 1 / x4.numel())
    ref_x3_grad = 0.5 * ref_x4_grad
    ref_x2_grad = ref_x3_grad
    ref_x1_grad = 0.5 * ref_x2_grad + ref_x3_grad
    ref_x_grad = ref_x1_grad
    to_test = [
      ('x4', x4, ref_x4_grad),
      ('x3', x3, ref_x3_grad),
      ('x2', x2, ref_x2_grad),
      ('x1', x1, ref_x1_grad),
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      assert var.grad is not None, f'{name}.grad is not set by backward'
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')
