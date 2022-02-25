import torch  # noqa

from nn import Module, Linear, Conv2d, MaxPool2d  # noqa
from functional import relu, view,add  # noqa

__all__ = ['ConvNet']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################################################
# ConvNet
#################################################

class ConvNet(Module):
  """A deep convolutional neural network"""

  def __init__(self, in_channels, num_classes):
    super().__init__()
    # BEGIN SOLUTION

    self.conv1 = Conv2d(in_channels, 100, (3,3), 1,1,1)
    self.conv2 = Conv2d(100, 200, (3,3), 1,1,1)
    self.conv3 = Conv2d(200, 300, (3,3), 1,1,1)
    self.conv4 = Conv2d(300, 600, (3,3), 1,1,1)
    self.max_pool1 = MaxPool2d((2,2), 0,2,1)
    self.max_pool2 = MaxPool2d((2,2), 0,2,1)
    self.max_pool3 = MaxPool2d((2,2), 0,2,1)
    self.max_pool4 = MaxPool2d((2,2), 0,2,1)
    self.fc1 = Linear(2400,32)
    self.fc2 = Linear(32,num_classes)
    self._modules = ['conv1', 'conv2','conv3','conv4', 'max_pool1', 'max_pool2','max_pool3','max_pool4','fc1','fc2']
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION

    b = x.shape[0]

    # mu = torch.mean(x,dim=(2,3),keepdim=True)
    # sd = torch.std(x,dim=(2,3),keepdim=True)
    # x = (x - mu)/sd

    x = relu(self.conv1(x,ctx),ctx)
    x = self.max_pool1(x,ctx)
    x = relu(self.conv2(x,ctx),ctx)
    x = self.max_pool2(x,ctx)
    x = relu(self.conv3(x,ctx),ctx)
    x = self.max_pool3(x,ctx)
    x = relu(self.conv4(x,ctx),ctx)
    x = self.max_pool4(x,ctx)


    x = view(x, (b, -1),ctx)
    x = relu(self.fc1(x,ctx),ctx)
    x = self.fc2(x,ctx)
    return x
    # END SOLUTION
