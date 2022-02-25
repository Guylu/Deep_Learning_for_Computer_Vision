import unittest

import torch
import torch.nn.functional as F

# import functional
from functional import view  # , view_backward
from functional import conv2d  # , conv2d_backward
from functional import max_pool2d  # , max_pool2d_backward
from functional import add  # , max_pool2d_backward


def assert_tensor_in(tensor, lst, msg=''):
  assert any(tensor is x for x in lst), msg


class TestView(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-8
    self.rtol = 1e-8
    self.dtype = torch.float64

  def testContext(self):
    size = (3, 4)
    new_size = (3, 2, 2)
    ctx = [None]
    x = torch.rand(size=size, dtype=self.dtype)
    y = view(x, new_size, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")

  def _test_forward(self, size, new_size):
    x = torch.rand(size=size, dtype=self.dtype)
    y = view(x, new_size)
    y_ = x.view(new_size)
    dbg = f'x: {x}\nsize: {new_size}\ngot: {y}\nexpected: {y_}'
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size, new_size):
    ctx = []
    x = torch.rand(size=size, dtype=self.dtype)
    x.grad = torch.zeros_like(x)
    y = view(x, new_size, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone().requires_grad_()
      y_ = x_.view(new_size)
    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = f'x: {x}\nsize: {new_size}\ny.grad: {y.grad}'
    dbg_x = dbg + f'\ngot: {x.grad}\nexpected: {x_.grad}'
    torch.testing.assert_allclose(x.grad, x_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_x)

  def testSize34_322(self):
    size = (3, 4)
    new_size = (3, 2, 2)
    self._test_forward(size, new_size)

  def testSize322_34(self):
    size = (3, 2, 2)
    new_size = (3, 4)
    self._test_forward(size, new_size)

  def testSize322_62(self):
    size = (3, 2, 2)
    new_size = (6, 2)
    self._test_forward(size, new_size)

  def testBackwardSize34_322(self):
    size = (3, 4)
    new_size = (3, 2, 2)
    self._test_backward(size, new_size)

  def testBackwardSize322_34(self):
    size = (3, 2, 2)
    new_size = (3, 4)
    self._test_backward(size, new_size)

  def testBackwardSize322_62(self):
    size = (3, 2, 2)
    new_size = (6, 2)
    self._test_backward(size, new_size)


class TestAdd(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-8
    self.rtol = 1e-8
    self.dtype = torch.float64

  def testContext(self):
    size = (3, 4)

    ctx = [None]
    a = torch.rand(size=size, dtype=self.dtype)
    b = torch.rand(size=size, dtype=self.dtype)
    y = add(a, b, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == add_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(a, backward_args, "a isn't in backward_args")
    assert_tensor_in(b, backward_args, "b isn't in backward_args")

  def _test_forward(self, size):
    a = torch.rand(size=size, dtype=self.dtype)
    b = torch.rand(size=size, dtype=self.dtype)
    y = add(a, b)
    y_ = a + b
    dbg = (f'a: {a}\nb: {b}'
           f'\ngot: {y}\nexpected: {y_}')
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size):
    ctx = []
    a = torch.rand(size=size, dtype=self.dtype)
    b = torch.rand(size=size, dtype=self.dtype)
    a.grad = torch.zeros_like(a)
    b.grad = torch.zeros_like(b)
    y = add(a, b, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      a_ = a.detach().clone().requires_grad_()
      b_ = b.detach().clone().requires_grad_()
      y_ = a_ + b_

    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = f'a: {a}\nb: {b}\ny.grad: {y.grad}'
    dbg_a = dbg + f'\ngot: {a.grad}\nexpected: {a_.grad}'
    dbg_b = dbg + f'\ngot: {b.grad}\nexpected: {b_.grad}'
    torch.testing.assert_allclose(a.grad, a_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_a)
    torch.testing.assert_allclose(b.grad, b_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_b)

  def testSize34(self):
    size = (3, 4)
    self._test_forward(size)

  def testSize345(self):
    size = (3, 4, 5)
    self._test_forward(size)

  def testSize3456(self):
    size = (3, 4, 5, 6)
    self._test_forward(size)

  def testBackwardSize34(self):
    size = (3, 4)
    self._test_backward(size)

  def testBackwardSize345(self):
    size = (3, 4, 5)
    self._test_backward(size)

  def testBackwardSize3456(self):
    size = (3, 4, 5, 6)
    self._test_backward(size)


class TestConv2d(unittest.TestCase):

  def setUp(self):
    self.atol = 1e-8
    self.rtol = 1e-8
    self.dtype = torch.float64

  def testContext(self):
    batch_size = 1
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (1, 1)

    ctx = [None]
    x = torch.rand(size=(batch_size, cin, *input_size), dtype=self.dtype)
    w = torch.rand(size=(cout, cin, *kernel_size), dtype=self.dtype)
    b = torch.rand(size=(cout,), dtype=self.dtype)
    y = conv2d(x, w, b, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == conv2d_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")
    assert_tensor_in(w, backward_args, "w isn't in backward_args")
    assert_tensor_in(b, backward_args, "b isn't in backward_args")

  def _test_forward(self, batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups):
    x = torch.rand(size=(batch_size, cin, *input_size), dtype=self.dtype)
    w = torch.rand(size=(cout, cin // groups, *kernel_size), dtype=self.dtype)
    b = torch.rand(size=(cout,), dtype=self.dtype)
    y = conv2d(x, w, b, padding=padding, stride=stride, dilation=dilation, groups=groups)
    y_ = F.conv2d(x, w, b, padding=padding, stride=stride, dilation=dilation, groups=groups)
    dbg = (f'x: {x}\nw: {w}\nb: {b}'
           f'\npadding: {padding}\nstride: {stride}\ndilation: {dilation}\ngroups: {groups}'
           f'\ngot: {y}\nexpected: {y_}')
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups):
    ctx = []
    x = torch.rand(size=(batch_size, cin, *input_size), dtype=self.dtype)
    w = torch.rand(size=(cout, cin // groups, *kernel_size), dtype=self.dtype)
    b = torch.rand(size=(cout,), dtype=self.dtype)
    x.grad = torch.zeros_like(x)
    w.grad = torch.zeros_like(w)
    b.grad = torch.zeros_like(b)
    y = conv2d(x, w, b, padding=padding, stride=stride, dilation=dilation, groups=groups, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone().requires_grad_()
      w_ = w.detach().clone().requires_grad_()
      b_ = b.detach().clone().requires_grad_()
      y_ = F.conv2d(x_, w_, b_, padding=padding, stride=stride, dilation=dilation, groups=groups)

    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = (f'x: {x}\nw: {w}\nb: {b}\ny.grad: {y.grad}'
           f'\npadding: {padding}\nstride: {stride}\ndilation: {dilation}\ngroups: {groups}')
    dbg_x = dbg + f'\ngot: {x.grad}\nexpected: {x_.grad}'
    dbg_w = dbg + f'\ngot: {w.grad}\nexpected: {w_.grad}'
    dbg_b = dbg + f'\ngot: {b.grad}\nexpected: {b_.grad}'
    torch.testing.assert_allclose(x.grad, x_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_x)
    torch.testing.assert_allclose(w.grad, w_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_w)
    torch.testing.assert_allclose(b.grad, b_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_b)

  def testKernel1NoBatch(self):
    batch_size = 1
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3NoBatch(self):
    batch_size = 1
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3Stride2(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (2, 2)
    dilation = (1, 1)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3Padding1(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (1, 1)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3Dilation2(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (2, 2)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3Padding1Stride2Dilation2(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (1, 1)
    stride = (2, 2)
    dilation = (2, 2)
    groups = 1
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testKernel3Groups2(self):
    batch_size = 5
    cin, cout = 4, 8
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 2
    self._test_forward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel1NoBatch(self):
    batch_size = 1
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3NoBatch(self):
    batch_size = 1
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3Stride2(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (2, 2)
    dilation = (1, 1)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3Padding1(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (1, 1)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3Dilation2(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (2, 2)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3Padding1Stride2Dilation2(self):
    batch_size = 5
    cin, cout = 3, 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (1, 1)
    stride = (2, 2)
    dilation = (2, 2)
    groups = 1
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)

  def testBackwardKernel3Groups2(self):
    batch_size = 5
    cin, cout = 4, 8
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    groups = 2
    self._test_backward(batch_size, cin, cout, input_size, kernel_size, padding, stride, dilation, groups)


class TestMaxPool2d(unittest.TestCase):

  def setUp(self):
    self.atol = 1e-8
    self.rtol = 1e-8
    self.dtype = torch.float64

  def testContext(self):
    batch_size = 1
    cin = 3
    input_size = (15, 15)
    kernel_size = (1, 1)

    ctx = [None]
    x = torch.rand(size=(batch_size, cin, *input_size), dtype=self.dtype)
    y = max_pool2d(x, kernel_size=kernel_size, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == max_pool2d_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")

  def _test_forward(self, batch_size, cin, input_size, kernel_size, padding, stride, dilation):
    x = torch.rand(size=(batch_size, cin, *input_size), dtype=self.dtype)
    y = max_pool2d(x, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
    y_ = F.max_pool2d(x, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
    dbg = (f'x: {x}\nkernel_size: {kernel_size}\npadding: {padding}'
           f'\nstride: {stride}\ndilation: {dilation}'
           f'\ngot: {y}\nexpected: {y_}')
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, batch_size, cin, input_size, kernel_size, padding, stride, dilation):
    ctx = []
    x = torch.rand(size=(batch_size, cin, *input_size), dtype=self.dtype)
    x.grad = torch.zeros_like(x)
    y = max_pool2d(x, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone().requires_grad_()
      y_ = F.max_pool2d(x_, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = (f'x: {x}\ny.grad: {y.grad}'
           f'\nkernel_size: {kernel_size}\npadding: {padding}'
           f'\nstride: {stride}\ndilation: {dilation}')
    dbg_x = dbg + f'\ngot: {x.grad}\nexpected: {x_.grad}'
    torch.testing.assert_allclose(x.grad, x_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_x)

  def testKernel1NoBatch(self):
    batch_size = 1
    cin = 3
    input_size = (15, 15)
    kernel_size = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testKernel3NoBatch(self):
    batch_size = 1
    cin = 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testKernel3(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testKernel2Stride2(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (2, 2)
    padding = (0, 0)
    stride = (2, 2)
    dilation = (1, 1)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testKernel3Padding1(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (1, 1)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testKernel4Dilation2(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (4, 4)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (2, 2)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testKernel4Padding1Stride2Dilation2(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (4, 4)
    padding = (1, 1)
    stride = (2, 2)
    dilation = (2, 2)
    self._test_forward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel1NoBatch(self):
    batch_size = 1
    cin = 3
    input_size = (15, 15)
    kernel_size = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel3NoBatch(self):
    batch_size = 1
    cin = 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel3(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel2Stride2(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (2, 2)
    padding = (0, 0)
    stride = (2, 2)
    dilation = (1, 1)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel3Padding1(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (3, 3)
    padding = (1, 1)
    stride = (1, 1)
    dilation = (1, 1)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel4Dilation2(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (4, 4)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (2, 2)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)

  def testBackwardKernel4Padding1Stride2Dilation2(self):
    batch_size = 5
    cin = 3
    input_size = (15, 15)
    kernel_size = (4, 4)
    padding = (1, 1)
    stride = (2, 2)
    dilation = (2, 2)
    self._test_backward(batch_size, cin, input_size, kernel_size, padding, stride, dilation)
