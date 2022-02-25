import torch  # noqa

from functional import linear  # noqa

__all__ = ['Module', 'Linear']

#################################################
# PROVIDED: Module
#################################################

class Module:
  """Module is a base class for differentaible layer or network.

  Has the following attributes:
    * _parameters (List[str]): List of parameter names.
    * _buffers (List[str]): List of buffer names. Buffer is a tensor which is not optimized in SGD.
    * _modules (List[str]): List of submodule names.
    * _train (bool): Whether the module is in "train" mode.

  Has the following methods:
    * forward(*args, ctx=None): The forward computation of the module.
    * __call__(*args, ctx=None): Alias to forward.
    * parameters(): Returns a list of the parameters in the module and its submodules.
    * to(device): Transfer the module to a device, including submodules.
    * train(): Set the module's mode to "train", including submodules.
    * eval(): Set the modules' mode to "eval", including submodules.
  """
  def __init__(self):
    self._parameters = []
    self._buffers = []
    self._modules = []
    self._train = True

  def forward(self, *args, ctx=None):
    """Compute the function of the module.

    Args:
      *args: Inputs to the module.
      ctx ([type], optional): Autograd context. Defaults to None.

    Returns:
      *outs: Outputs of the module.
    """
    raise NotImplementedError  # DO NOT EDIT THIS. IT JUST MEANS THIS METHOD IS ABSTRACT.

  def __call__(self, *args, **kwargs):
    """Alias to forward"""
    return self.forward(*args, **kwargs)

  def to(self, device):
    """Trasfer the module and all submodules to `device`.

    This is an inplace operation.

    Args:
      device (torch.device): The new device.

    Returns:
      module (Module): The module itself.
    """
    device = torch.device(device)
    # transfer all the parameters to device
    for param in self._parameters:
      setattr(self, param, getattr(self, param).to(device=device))
    # transfer all the buffers to device
    for buffer in self._buffers:
      setattr(self, buffer, getattr(self, buffer).to(device=device))
    # transfer all the modules to device
    for module in self._modules:
      setattr(self, module, getattr(self, module).to(device=device))
    return self

  def parameters(self, recurse=True):
    """Returns a list of parameters in the module and its submodules.

    Each parameter appears exactly once in the returned list.

    Args:
      recurse (bool, optional): Whether to recurse into submodules. Defaults to True.

    Returns:
      List[torch.Tensor]: List of parameters.
    """
    # add all parameters
    parameters = []
    parameters += [getattr(self, param) for param in self._parameters]
    if recurse:
      # add all parameters of sub-modules
      for module in self._modules:
        parameters += getattr(self, module).parameters(recurse=True)
    # deduplicate
    return list(set(parameters))

  def train(self, recurse=True):
    """Set module and its submodules to "train" mode.

    Args:
      recurse (bool, optional): Whether to recurse to submodules. Defaults to True.
    """
    self._train = True
    if recurse:
      for module in self._modules:
        getattr(self, module).train(recurse=True)

  def eval(self, recurse=True):
    """Set module and its submodules to "eval" mode.

    Args:
      recurse (bool, optional): Whether to recurse to submodules. Defaults to True.
    """
    self._train = False
    if recurse:
      for module in self._modules:
        getattr(self, module).eval(recurse=True)


#################################################
# Linear
#################################################

class Linear(Module):
  """Linear layer"""

  def __init__(self, in_dim, out_dim):
    """Creates a Linear layer.

    In this method you should:
      * Create a weight parameter (call it `weight`).
      * Create a bias parameter (call it `bias`).
      * Add these parameter names to `self._parameters`.

    Args:
      in_dim (int): The dimension of the input to that layer.
      out_dim (int): The dimension of the output of that layer.
    """
    super().__init__()
    # BEGIN SOLUTION
    self.in_dim = in_dim
    self.out_dim = out_dim

    self.weight = torch.rand(out_dim,in_dim)
    self.bias = torch.rand(out_dim,)
    self._parameters+=["weight","bias"]
    self.init_parameters()
    
    # END SOLUTION

  def init_parameters(self):
    """Initializes the parameters of the Linear layer."""
    # BEGIN SOLUTION
    self.weight = torch.normal(torch.zeros(self.out_dim,self.in_dim),torch.ones(self.out_dim,self.in_dim)*(2/self.in_dim) )
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the Linear function of that input.

    You should use the weight and bias parameters of that layer.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, out_dim)`.
    """
    # BEGIN SOLUTION
    return linear(x,self.weight,self.bias,ctx)
    # END SOLUTION
