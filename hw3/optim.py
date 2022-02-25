import torch  # noqa
from hw2_optim import *  # noqa
from hw2_optim import __all__ as __old_all__

__new_all__ = ['MomentumSGD']
__all__ = __old_all__ + __new_all__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MomentumSGD:
  def __init__(self, parameters, lr, momentum):
    """Creates an SGD optimizer.

    Args:
      parameters (List[torch.Tensor]): List of parameters. Each parameter
        should appear at most once.
      lr (float): The learning rate. Should be positive for gradient
        descent.
      momentum (float): The momentum rate.
    """
    if len(set(parameters)) != len(parameters):
      raise ValueError("can't optimize duplicated parameters!")
    # BEGIN SOLUTION
    self.parameters = parameters
    # for param in self.parameters:
    #   param = param.to(device = device)
    self.lr = lr
    self.momentum = momentum
    self.v = [torch.zeros_like(param) for param in parameters]
    # END SOLUTION

  def zero_grad(self):
    """Zeros the gradients of all the parameters in the network.

    Note: Gradients are zeroed by setting them to `None`, or by
    zeroing all their values.
    """
    # BEGIN SOLUTION
    for param in self.parameters:
      # print(param.grad)
      # print( torch.zeros_like(param.grad).to(device).dtype)
      param.grad = None
    # END SOLUTION

  def step(self):
    """Updates the parameter values according to their gradients
    and the learning rate.

    Note: Parameters should be updated in-place.

    Note: The gradients of some parameters might be `None`. You should
    support that case in your solution.
    """
    # BEGIN SOLUTION
    for i,param in enumerate(self.parameters):
      if param.grad is not None:
        # print(param)
        self.v[i] = self.momentum * self.v[i] + param.grad
        param -= self.lr*self.v[i] 
    # END SOLUTION
