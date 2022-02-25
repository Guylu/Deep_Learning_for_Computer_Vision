from torch import nn
import utils
from functools import partial
import torch.nn.functional as F

##########################################################
# Basic Model
##########################################################
class ZSSRNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    super().__init__()
    self.scale_factor = scale_factor
    self.conv1 = nn.Conv2d(3, 64 , kernel_size,padding=kernel_size//2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv4 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv5 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv6 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv7 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv8 = nn.Conv2d(64, 3 , kernel_size,padding=kernel_size//2)
    
    #END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """    
    # BEGIN SOLUTION

    x= utils.rr_resize(x, self.scale_factor)
   
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))
    x = F.relu(self.conv8(x))
    
    return x
    # END SOLUTION


##########################################################
# Advanced Model
##########################################################



class ZSSRResNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    super().__init__()
    self.scale_factor = scale_factor
    self.conv1 = nn.Conv2d(3, 64 , kernel_size,padding=kernel_size//2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv4 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv5 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv6 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv7 = nn.Conv2d(64, 64, kernel_size,padding=kernel_size//2)
    self.conv8 = nn.Conv2d(64, 3 , kernel_size,padding=kernel_size//2)


    self.bn1 = nn.BatchNorm2d(num_features=64)
    self.bn2 = nn.BatchNorm2d(num_features=64)
    self.bn3 = nn.BatchNorm2d(num_features=64)
    self.bn4 = nn.BatchNorm2d(num_features=64)
    self.bn5 = nn.BatchNorm2d(num_features=64)
    self.bn6 = nn.BatchNorm2d(num_features=64)
    self.bn7 = nn.BatchNorm2d(num_features=64)
    self.bn8 = nn.BatchNorm2d(num_features=64)

    self.relu = F.relu
    # END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Finally, add the CNN's output in a residual manner to the original resized
    image.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """   
    # BEGIN SOLUTION
    # print(x.shape)
    x = utils.rr_resize(x, self.scale_factor)
    x1 = self.relu(self.conv1(x))
    # x1 = self.bn1(x1)
    x2 = self.relu(self.conv2(x1))
    
    # x2 = self.bn2(x2+x1)

    x3 = self.relu(self.conv3(x2))
    # x3 = self.bn3(x3)
    x4 = self.relu(self.conv4(x3))

    # x4 = self.bn4(x4+x2)

    x5 = self.relu(self.conv5(x4))
    # x5 = self.bn5(x5)
    x6 = self.relu(self.conv6(x5))

    # x6 = self.bn6(x6+x4)

    x7 = self.relu(self.conv7(x6))
    # x7 = self.bn7(x7)
    # x7 = x7+x2
    # x7 = self.bn8(x7)
    x8 = self.relu(self.conv8(x7))
    # print(x8.mean())
    return x8 
    # END SOLUTION


##########################################################
# Original Model
##########################################################



class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,l = 1):
        super(ConvolutionalBlock, self).__init__()

        layers = []
        for _ in range(l):
          layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=kernel_size // 2))

          layers.append(nn.BatchNorm2d(num_features=out_channels))

          layers.append(nn.LeakyReLU(0.2))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
       
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output



class ZSSROriginalNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    super().__init__()
    self.scale_factor = scale_factor
    self.r1 = ConvolutionalBlock(8,8,kernel_size,stride=1,l=5)
    self.r2 = ConvolutionalBlock(16,16,kernel_size,stride=1,l=5)
    self.r3 = ConvolutionalBlock(32,32,kernel_size,stride=1,l=5)
    self.r4 = ConvolutionalBlock(64,64,kernel_size,stride=1,l=5)
    self.r5 = ConvolutionalBlock(128,128,kernel_size,stride=1,l=5)
    




    self.up1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_size, stride=1,padding=kernel_size // 2)
    self.up2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=1,padding=kernel_size // 2)
    self.up3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1,padding=kernel_size // 2)
    self.up4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1,padding=kernel_size // 2)
    self.up5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1,padding=kernel_size // 2)
    self.final = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=kernel_size, stride=1,padding=kernel_size // 2)
    # END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Finally, add the CNN's output in a residual manner to the original resized
    image.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """   
    # BEGIN SOLUTION
    x = utils.rr_resize(x, self.scale_factor)

    # print(x.shape)
    x = self.up1(x)
    x1=self.r1(x)
    x = x1 +x
    # print(x.shape)

    x = self.up2(x)
    x1=self.r2(x)
    x = x1 +x
    # print(x.shape)

    x = self.up3(x)
    x1=self.r3(x)
    x = x1 +x
    # print(x.shape)

    x = self.up4(x)
    x1=self.r4(x)
    x = x1 +x
    # print(x.shape)

    x = self.up5(x)
    x1=self.r5(x)
    x = x1 +x
    # print(x.shape)

    x = self.final(x)
    return x
    # END SOLUTION

