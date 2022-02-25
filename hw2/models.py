import torch  # noqa

from nn import Module, Linear  # noqa
from functional import relu  # noqa

__all__ = ['SoftmaxClassifier', 'MLP']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################################################
# SoftmaxClassifier
#################################################

class SoftmaxClassifier(Module):
  """A simple softmax classifier"""

  def __init__(self, in_dim, num_classes):
    super().__init__()
    # BEGIN SOLUTION
    self.linear = Linear(in_dim, num_classes)
    self.linear.init_parameters()
    self._modules.append("linear")
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    return self.linear.forward(x,ctx)
    # END SOLUTION


#################################################
# MLP
#################################################

class MLP(Module):
  """A multi-layer perceptron"""

  def __init__(self, in_dim, num_classes,c1,c2,c3):  # YOU CAN MODIFY THIS LINE AND ADD ARGUMENTS
    super().__init__()
    # BEGIN SOLUTION
    # hidden_list.insert(0,in_dim)
    # hidden_list.append(num_classes)
    # self.linear_layers = []
    # for layer_num in range(len(hidden_list)-1):
    #   self.linear_layers.append(Linear(hidden_list[layer_num],hidden_list[layer_num+1]))
    #   self._modules.append("linear_layers["+str(layer_num)+"]")



    self.h1 = Linear(in_dim,c1)
    self.h2 = Linear(c1,c2)
    self.h3 = Linear(c2,c3)
    self.h4 = Linear(c3,num_classes)
    
    self.h1.init_parameters()
    self.h2.init_parameters()
    self.h3.init_parameters()
    self.h4.init_parameters()


    self._modules.append("h1")
    self._modules.append("h2")
    self._modules.append("h3")
    self._modules.append("h4")
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    # for layer in hidden_list:
    #   x = layer(x,ctx)

    x = relu(self.h1.forward(x,ctx),ctx)
    x = relu(self.h2.forward(x,ctx),ctx)
    x = relu(self.h3.forward(x,ctx),ctx)
    x = relu(self.h4.forward(x,ctx),ctx)
    return x
    # END SOLUTION
