import torch  # noqa

from torch.nn import Module, Linear  # noqa
from torch.nn.functional import relu  # noqa

__all__ = ['SoftmaxClassifier', 'MLP']


#################################################
# SoftmaxClassifier
#################################################

class SoftmaxClassifier(Module):
  """A simple softmax classifier"""

  def __init__(self, in_dim, num_classes):
    super().__init__()  # This line is important in torch Modules.
                        # It replaces the manual registration of parameters
                        # and sub-modules (to some extent).
    # BEGIN SOLUTION
    self.linear = Linear(in_dim, num_classes)
    # END SOLUTION

  def forward(self, x):
    """Computes the forward function of the network.

    Note: `F.cross_entropy` expects predictions BEFORE applying `F.softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    return relu(self.linear.forward(x))
    # END SOLUTION


#################################################
# MLP
#################################################

class MLP(Module):
  """A multi-layer perceptron"""

  def __init__(self, in_dim, num_classes,hidden_list):  # YOU CAN MODIFY THIS LINE AND ADD ARGUMENTS
    super().__init__()  # This line is important in torch Modules.
                        # It replaces the manual registration of parameters
                        # and sub-modules (to some extent).
    # BEGIN SOLUTION


    self.layers = torch.nn.ModuleList()
    hidden_list.insert(0,in_dim)
    hidden_list.append(num_classes)
    # print(hidden_list)
    for i in range(len(hidden_list)-1):
      l = Linear(hidden_list[i], hidden_list[i + 1])
      # torch.nn.init.kaiming_normal_(l.weight,nonlinearity='relu')
      self.layers.append(l)
    


    # self.h1 = Linear(in_dim,c1)
    # self.h2 = Linear(c1,c2)
    # self.h3 = Linear(c2,c3)
    # self.h4 = Linear(c3,num_classes)




    # END SOLUTION

  def forward(self, x):
    """Computes the forward function of the network.

    Note: `F.cross_entropy` expects predictions BEFORE applying `F.softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION


    # x = relu(self.h1(x))
    # x = relu(self.h2(x))
    # x = relu(self.h3(x))
    # x = relu(self.h4(x))
    # return x


    for layer in self.layers:
      x = relu(layer(x))
    return x
    # END SOLUTION
