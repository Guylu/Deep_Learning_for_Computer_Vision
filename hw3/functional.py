import math  # noqa

import torch  # noqa
from torch.nn.functional import one_hot  # noqa
from torch.nn.modules.utils import _pair, _ntuple
from torch.nn.functional import unfold, fold

from hw2_functional import *  # noqa
from hw2_functional import __all__ as __old_all__

__new_all__ = ['view', 'add', 'conv2d', 'max_pool2d']
__all__ = __old_all__ + __new_all__ # 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################################################
# conv2d
#################################################

def conv2d(x, w, b=None, padding=0, stride=1, dilation=1, groups=1, ctx=None):
  """A differentiable convolution of 2d tensors.

  Note: Read this following documentation regarding the output's shape.
  https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

  Backward call:
    backward_fn: conv2d_backward
    args: y, x, w, b, padding, stride, dilation

  Args:
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    w (torch.Tensor): The weight tensor.
      Has shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    b (torch.Tensor): The bias tensor. Has shape `(out_channels,)`.
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    groups (int, Optional): Number of groups. Defaults to 1.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, out_channels, out_height, out_width)`.
  """
  assert w.size(0) % groups == 0, \
    f'expected w.size(0)={w.size(0)} to be divisible by groups={groups}'
  assert x.size(1) % groups == 0, \
    f'expected x.size(1)={x.size(1)} to be divisible by groups={groups}'
  assert x.size(1) // groups == w.size(1), \
    f'expected w.size(1)={w.size(1)} to be x.size(1)//groups={x.size(1)}//{groups}'





  # print(f"input shape = {x.shape}")       

  # print(f"weight shape = {w.shape}")

  # print(f"stride shape = {stride}")

  # print(f"dilation shape = {dilation}")
  

  kernel_size = (w.shape[2],w.shape[3])
  windows = unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
  windows = windows.permute(0, 2, 1)

  # print(f"windows shape = {windows.shape}")
  filters = w.reshape(w.shape[0],w.shape[1]*w.shape[2]*w.shape[3])
  # print(f"filters shape = {filters.shape}")

  convolved = torch.einsum("ijk,kn->ijn",windows,filters.T)

  # print(f"convolved shape = {convolved.shape}")


  res = convolved +b

  # print(f"res shape = {res.shape}")

  t = lambda x: torch.tensor(x)
  
  sh = (  t(x.shape[2:]) +2* t(padding) - t(dilation) * (t(kernel_size) -1) -1)/t(stride) +1

  sh = tuple([int(a) for a in sh.tolist()])

  # print(sh)
  res = res.permute(0,2,1)
  ret = fold(res, output_size =sh ,  kernel_size=(1,1))
  
  # print()
  # print()

  if ctx is not None:
    ctx.append([conv2d_backward, [ret, x, w, b, padding, stride, dilation, groups]])
  return ret

def conv2d_backward(y, x, w, b, padding, stride, dilation, groups):
  """Backward computation of `conv2d`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, `w` and `b` (if `b` is not None),
  and accumulates them in `x.grad`, `w.grad` and `b.grad`, respectively.

  Args:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, out_channels, out_height, out_width)`.
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    w (torch.Tensor): The weight tensor.
      Has shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    b (torch.Tensor): The bias tensor. Has shape `(out_channels,)`.
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    groups (int, Optional): Number of groups. Defaults to 1.
  """
  # BEGIN SOLUTION
  x, y,w,b = x, y, w, b
  y_ufld = unfold(y.grad, (1,1),padding=0,stride = 1)
  filters = w.reshape(w.shape[0],w.shape[1]*w.shape[2]*w.shape[3])
  kernel_size = w.shape[2:]
  x_ufld = unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

  x_ufld_grad = torch.einsum("ijk,kn->ijn",y_ufld.transpose(1,2),filters)
  w_ufld_grad = torch.einsum("ijk,ikw->jw",y_ufld,x_ufld.transpose(1,2))
  b.grad += y_ufld.sum(dim=2).sum(dim=0)
  
  x.grad += fold(x_ufld_grad.transpose(1,2),output_size=(x.shape[2],x.shape[3]) ,kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
  w.grad += w_ufld_grad.reshape(w.shape)
  # END SOLUTION


#################################################
# max_pool2d
#################################################

def max_pool2d(x, kernel_size, padding=0, stride=1, dilation=1, ctx=None):
  """A differentiable convolution of 2d tensors.

  Note: Read this following documentation regarding the output's shape.
  https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

  Backward call:
    backward_fn: max_pool2d_backward
    args: y, x, padding, stride, dilation

  Args:
    x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
    kernel_size (Tuple[int, int] or int): The kernel size in each dimension (height, width).
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, in_channels, out_height, out_width)`.
  """
  # BEGIN SOLUTION
  reparam = lambda z: (z,z) if isinstance(z,int) else z
  kernel_size = reparam(kernel_size)
  padding = reparam(padding)
  stride = reparam(stride)
  dilation = reparam(dilation)
  x_ufld = unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

  x_ufld = x_ufld.unsqueeze(1)
  x_ufld = x_ufld.reshape(x_ufld.shape[0],x.shape[1],int(x_ufld.shape[2]/x.shape[1]),x_ufld.shape[3])

  t = torch.max(x_ufld,dim=2)
  pooled,index = t.values,t.indices



  t = lambda x: torch.tensor(x)
  
  sh = (  t(x.shape[2:]) +2* t(padding) - t(dilation) * (t(kernel_size) -1) -1)/t(stride) +1

  sh = tuple([int(a) for a in sh.tolist()])


  ret = fold(pooled, output_size =sh ,  kernel_size=(1,1))

  if ctx is not None:
    ctx.append([max_pool2d_backward, [ret, x, index, kernel_size, padding, stride, dilation]])
  return ret
  # END SOLUTION


def max_pool2d_backward(y, x, index, kernel_size, padding, stride, dilation):
  """Backward computation of `max_pool2d`.

  Propagates the gradients of `y` (in `y.grad`) to `x` and accumulates it in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, in_channels, out_height, out_width)`.
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    index (torch.Tensor): Auxilary tensor with indices of the maximum elements. You are
      not restricted to a specific format.
    kernel_size (Tuple[int, int] or int): The kernel size in each dimension (height, width).
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
  """
  # BEGIN SOLUTION

  y_grad = unfold(y.grad,(1,1), dilation=dilation, padding=0, stride=1)
  x_unf=unfold(x,kernel_size, dilation=dilation, padding=padding, stride=stride).unsqueeze(1)
  x_unf = x_unf.reshape(x_unf.shape[0],x.shape[1],int(x_unf.shape[2]/x.shape[1]),x_unf.shape[3])

  ret = torch.zeros_like(x_unf,dtype=y_grad.dtype)
  ret.scatter_add_(2,index.unsqueeze(2),y_grad.unsqueeze(2))

  ret = ret.reshape(ret.shape[0],ret.shape[1]*ret.shape[2],ret.shape[3])

  x.grad += fold(ret, (x.shape[2],x.shape[3]),kernel_size,dilation=dilation, padding=padding, stride=stride)
  # END SOLUTION


#################################################
# view
#################################################

def view(x, size, ctx=None):
  """A differentiable view function.

  Backward call:
    backward_fn: view_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    size (torch.Size): The new size (shape).
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. Has shape `size`.
  """
  # BEGIN SOLUTION
  # ret = fold(x.flatten(),size,(1,1))

  ret = x.view(size)
  if ctx is not None:
    ctx.append([view_backward, [ret, x]])
  return ret
  # END SOLUTION


def view_backward(y, x):
  """Backward computation of `view`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  # return
  x.grad = y.grad.view(x.size())
  # END SOLUTION


#################################################
# add
#################################################

def add(a, b, ctx=None):
  """A differentiable addition of two tensors.

  Backward call:
    backward_fn: add_backward
    args: y, a, b

  Args:
    a (torch.Tensor): The first input tensor.
    b (torch.Tensor): The second input tensor. Should have the same shape as `a`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. The sum of `a + b`.
  """
  assert a.size() == b.size(), 'tensors should have the same size'
  # BEGIN SOLUTION
  ret = a+b
  if ctx is not None:
    ctx.append([add_backward, [ret, a,b]])
  return ret
  # END SOLUTION
  

def add_backward(y, a, b):
  """Backward computation of `add`.

  Propagates the gradients of `y` (in `y.grad`) to `a` and `b`, and accumulates them in `a.grad`,
  `b.grad`, respectively.

  Args:
    y (torch.Tensor): The output tensor.
    a (torch.Tensor): The first input tensor.
    b (torch.Tensor): The second input tensor.
  """
  # BEGIN SOLUTION
  a.grad +=y.grad
  b.grad +=y.grad
  # END SOLUTION
