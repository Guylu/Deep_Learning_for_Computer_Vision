import unittest  # noqa
import torch
from functional import multi_head_attention
from torch.nn.functional import _scaled_dot_product_attention

class TestAttentionFunction(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-8
    self.rtol = 1e-8
    self.dtype = torch.float64

  def _test_forward(self, batch_size, num_heads, sequence_size, num_dims):
    q = torch.rand(batch_size, num_heads, sequence_size, num_dims)
    k = torch.rand(batch_size, num_heads, sequence_size, num_dims)
    v = torch.rand(batch_size, num_heads, sequence_size, num_dims)
    y = multi_head_attention(q, k, v)

    q_ = q.transpose(0, 1)
    k_ = k.transpose(0, 1)
    v_ = v.transpose(0, 1)
    ys_ = []
    for (q_h, k_h, v_h) in zip(q_, k_, v_):
      ys_.append(_scaled_dot_product_attention(q_h, k_h, v_h)[0])
    y_ = torch.cat(ys_, dim=-1)

    dbg = (f'\ngot: {y}\nexpected: {y_}')
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backwards(self, batch_size, num_heads, sequence_size, num_dims):
    q = torch.rand(batch_size, num_heads, sequence_size, num_dims, requires_grad=True)
    k = torch.rand(batch_size, num_heads, sequence_size, num_dims, requires_grad=True)
    v = torch.rand(batch_size, num_heads, sequence_size, num_dims, requires_grad=True)

    q_ = q.transpose(0, 1).detach()
    q_.requires_grad = True
    k_ = k.transpose(0, 1).detach()
    k_.requires_grad = True
    v_ = v.transpose(0, 1).detach()
    v_.requires_grad = True

    y = multi_head_attention(q, k, v)
    y.sum().backward()

    ys_ = []
    for (q_h, k_h, v_h) in zip(q_, k_, v_):
      ys_.append(_scaled_dot_product_attention(q_h, k_h, v_h)[0])
    y_ = torch.cat(ys_, dim=-1)
    y_.sum().backward()

    dbg = f'got: {q.grad}\nexpected: {q_.grad}'
    torch.testing.assert_allclose(q.grad, q_.grad.transpose(0, 1), rtol=self.rtol, atol=self.atol, msg=dbg)
    dbg = f'got: {k.grad}\nexpected: {k_.grad}'
    torch.testing.assert_allclose(k.grad, k_.grad.transpose(0, 1), rtol=self.rtol, atol=self.atol, msg=dbg)
    dbg = f'got: {v.grad}\nexpected: {v_.grad}'
    torch.testing.assert_allclose(v.grad, v_.grad.transpose(0, 1), rtol=self.rtol, atol=self.atol, msg=dbg)

  def testSingleHead(self):
    batch_size = 2
    num_heads = 1
    sequence_size = 8
    num_dims = 16
    self._test_forward(batch_size, num_heads, sequence_size, num_dims)

  def testMultiHead(self):
    batch_size = 2
    num_heads = 4 
    sequence_size = 8
    num_dims = 16
    self._test_forward(batch_size, num_heads, sequence_size, num_dims)

  def testSingleHeadBackward(self):
    batch_size = 2
    num_heads = 1
    sequence_size = 8
    num_dims = 16
    self._test_backwards(batch_size, num_heads, sequence_size, num_dims)

  def testMultiHeadBackward(self):
    batch_size = 2
    num_heads = 4 
    sequence_size = 8
    num_dims = 16
    self._test_backwards(batch_size, num_heads, sequence_size, num_dims)



    

