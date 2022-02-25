import math  # noqa

import torch  # noqa
from torch.nn.modules.utils import _pair

from hw2_nn import Module
from functional import conv2d, max_pool2d  # noqa

from hw2_nn import *  # noqa
from hw2_nn import __all__ as __old_all__

__new_all__ = ['Conv2d', 'MaxPool2d']
__all__ = __old_all__ + __new_all__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Conv2d(Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
               groups=1, bias=True):
    """Creates a Conv2d layer.

    In this method you should:
      * Create a weight parameter (call it `weight`).
      * Create a bias parameter (call it `bias`).
      * Add these parameter names to `self._parameters`.
      * Call `init_parameters()` to initialize the parameters.
      * Save the other arguments.

    Args:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      kernel_size (Tuple[int, int] or int): Kernel Size.
      padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
        Defaults to 0.
      stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
        Defaults to 1.
      dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
        Defaults to 1.
      groups (int, Optional): Number of groups. Defaults to 1.
      bias (bool, optional): [description]. Defaults to True.
    """
    assert in_channels % groups == 0, \
      f'in_channels={in_channels} should be divisible by groups={groups}'
    assert out_channels % groups == 0,\
      f'out_channels={out_channels} should be divisible by groups={groups}'
    super().__init__()
    # BEGIN SOLUTION
    reparam = lambda z: (z,z) if isinstance(z,int) else z  
    kernel_size = reparam(kernel_size)
    padding = reparam(padding)
    stride = reparam(stride)
    dilation = reparam(dilation)

    self.in_channels, self.out_channels, self.kernel_size, self.padding, self.stride, self.dilation,self.groups, self.bias = in_channels, out_channels, kernel_size, padding, stride, dilation, groups, bias
    
    self.weight = torch.rand((out_channels, in_channels, kernel_size[0], kernel_size[1]))
    self.isbias = bias
    self.bias = torch.rand(out_channels,) if bias else 0
    
    self._parameters+=["weight","bias"]
    
    self.init_parameters()
    # END SOLUTION

  def init_parameters(self):
    """Initializes the layer's parameters."""
    # BEGIN SOLUTION
    mul = self.kernel_size[0] * self.kernel_size[1]
    den = self.in_channels * mul 
    k = torch.sqrt( torch.tensor(self.groups/den))
    # print(self.weight)
    self.weight = torch.FloatTensor(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k,k)
    self.bias = torch.FloatTensor(self.out_channels,).uniform_(-k,k) if self.isbias else 0
    
    n = self.in_channels
    for k in self.kernel_size:
        n *= k
    stdv = 1. / math.sqrt(n)
    self.weight.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.uniform_(-stdv, stdv)
    
    # print(self.weight)
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the `conv2d` function of that input `x`.

    You should use the `weight` and `bias` parameters of that layer.

    Args:
      x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor. Has shape `(batch_size, out_channels, out_height, out_width)`.
    """
    # BEGIN SOLUTION
    return conv2d(x, self.weight , self.bias, self.padding, self.stride, self.dilation, self.groups, ctx)
    # END SOLUTION


class MaxPool2d(Module):
  def __init__(self, kernel_size, padding=0, stride=1, dilation=1):
    """Creates a MaxPool2d layer.

    In this method you should:
      * Save the layer's arguments.

    Args:
      kernel_size (Tuple[int, int] or int): Kernel Size.
      padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
        Defaults to 0.
      stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
        Defaults to 1.
      dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
        Defaults to 1.
    """
    super().__init__()
    # BEGIN SOLUTION
    reparam = lambda z: (z,z) if isinstance(z,int) else z
    kernel_size = reparam(kernel_size)
    padding = reparam(padding)
    stride = reparam(stride)
    dilation = reparam(dilation)

    self.kernel_size, self.padding, self.stride, self.dilation = kernel_size, padding, stride, dilation
    
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the `max_pool2d` function of that input `x`.

    Args:
      x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor. Has shape `(batch_size, in_channels, out_height, out_width)`.
    """
    # BEGIN SOLUTION
    return max_pool2d(x, self.kernel_size, self.padding, self.stride, self.dilation,ctx)
    # END SOLUTION
