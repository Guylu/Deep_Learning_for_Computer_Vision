import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import utils

##########################################################
# Dataset
##########################################################

class BasicZSSRDataset(Dataset):
  """ZSSR dataset. Creates a pair of LR and SR images. 
  The LR image is used for training while the SR image is used solely for
  evaluation."""

  def __init__(self, image_path, scale_factor=2, transform=transforms.ToTensor()):
    """
    Args:            
      image_path (string): Path to image from which to create dataset
      scale_factor (int): Ratio between SR and LR images.
      transform (callable): Transform to be applied on a sample. 
      Default transform turns the image into a tensor.
    """
    # BEGIN SOLUTION
    im_SR = Image.open(image_path)
    self.im_SR = transform(im_SR)
    self.im_LR = utils.rr_resize(self.im_SR, 1/scale_factor)
    self.transform = transform
    
    # print(f"SR shape: {self.im_SR.shape}")
    # print(f"LR shape: {self.im_LR.shape}")
    # END SOLUTION

  def __len__(self):
    """
    Returns:
      the length of the dataset. 
      Our dataset contains a single pair so the length is 1.
    """
    # BEGIN SOLUTION
    return 1
    # END SOLUTION


  def __getitem__(self, idx):
    """
    Args:
      idx (int) - Index of element to fetch. In our case only 1.
    Returns:
      sample (dict) - a dictionary containing two elements:
      Under the key 'SR' a torch.Tensor representing the original image.
      Has shape `(num_channels, height, width)`.
      Under the key 'LR' a torch.Tensor representing the original image.
      Has shape `(num_channels, height / scale_factor, width / scale_factor)`.
      In our case, returns the only element in the dataset.
    """
    # BEGIN SOLUTION
    d= {"SR":self.im_SR,"LR":self.im_LR}
    return d
    # END SOLUTION

class OriginalZSSRDataset(Dataset):
  """Your original ZSSR Dataset. Must include a ground truth SR image and a
  LR image for training"""

  def __init__(self, image_path, scale_factor=2, transform=transforms.ToTensor(),l=1):
    """
    Args:            
      image_path (string): Path to image from which to create dataset
      scale_factor (int): Ratio between SR and LR images.
      transform (callable): Transform to be applied on a sample. 
      Default transform turns the image into a tensor.
    """
    # BEGIN SOLUTION
    self.l = l
    im_SR = Image.open(image_path)

    self.data = []
    for i in range(l):
      # transform = advanced_trans(transforms.ToTensor()(im_SR).shape[1:])
      im_SR1 = transform(im_SR)
      im_LR = utils.rr_resize(im_SR1, 1/scale_factor)
      self.data.append((im_SR1,im_LR))
      # print(f"SR shape: {im_SR1.shape}")
      # print(f"LR shape: {im_LR.shape}")
    self.transform = transform
    
    # print(f"SR shape: {self.im_SR.shape}")
    # print(f"LR shape: {self.im_LR.shape}")
    
 
    # END SOLUTION

  def __len__(self):
    """
    Returns:
      the length of the dataset. 
      Our dataset contains a single pair so the length is 1.
    """
    # BEGIN SOLUTION
    return self.l 
    # END SOLUTION


  def __getitem__(self, idx):
    """
    Args:
      idx (int) - Index of element to fetch.
    Returns:
      sample (dict) - a dictionary containing two elements:
      Under the key 'SR' a torch.Tensor representing the original image.
      Has shape `(num_channels, height, width)`.
      Under the key 'LR' a torch.Tensor representing the original image.
      Has shape `(num_channels, height / scale_factor, width / scale_factor)`.
    """
    # BEGIN SOLUTION
    d= {"SR":self.data[idx][0],"LR":self.data[idx][1]}
    return d
    # END SOLUTION

##########################################################
# Transforms 
##########################################################

class NineCrops:
  """Generate all the possible crops using combinations of
  [90, 180, 270 degrees rotations,  horizontal flips and vertical flips]. 
  In total there are 8 options."""

  def __init__(self):
    pass

  def rotations(img):
    # img.rotate(rt_degr, expand=1)
    h,w,c = img.shape

    img_90  = torch.zeros([h,w,c], dtype=img.dtype)
    img_180 = torch.zeros([h,w,c], dtype=img.dtype)
    img_270 = torch.zeros([h,w,c], dtype=img.dtype)

    for i in range(h):
        for j in range(w):
            img_90[i,j] = img[h-j-1,w-i-1]
            img_90 = img_90[0:h,0:w]

            img_180[i,j] = img[h-i-1,w-j-1]
            img_180 = img_180[0:h,0:w]

            img_270[i,j] = img[j-1,i-1]
            img_270 = img_270[0:h,0:w]

    return img, img_90, img_180, img_270

  
  def __call__(self, sample):
    """
    Args:
      sample (torch.Tensor) - image to be transformed.
      Has shape `(num_channels, height, width)`.
    Returns:
      output (List(torch.Tensor)) - A list of 8 tensors containing the different
      flips and rotations of the original image. Each tensor has the same size as 
      the original image, possibly transposed in the spatial dimensions.
    """
    # BEGIN SOLUTION
    l= [sample]
    for theta in [0,90,180,270]:
      im = torchvision.transforms.functional.rotate(sample,theta)
      l.append(torch.flip(im, [1]))
      l.append(torch.flip(im, [2]))
    res = torch.stack(l, dim=0)
    # print(res.shape)
    return res
    # END SOLUTION


class EightCrops:
  """Generate all the possible crops using combinations of
  [90, 180, 270 degrees rotations,  horizontal flips and vertical flips]. 
  In total there are 8 options."""

  def __init__(self):
    pass
  
  def __call__(self, sample):
    """
    Args:
      sample (torch.Tensor) - image to be transformed.
      Has shape `(num_channels, height, width)`.
    Returns:
      output (List(torch.Tensor)) - A list of 8 tensors containing the different
      flips and rotations of the original image. Each tensor has the same size as 
      the original image, possibly transposed in the spatial dimensions.
    """
    # BEGIN SOLUTION
    l= []
    for theta in [0,90,180,270]:
      im = torchvision.transforms.functional.rotate(sample,theta)
      l.append(torch.flip(im, [1]))
      l.append(torch.flip(im, [2]))
    res = torch.stack(l, dim=0)
    # print(res.shape)
    return res
    # END SOLUTION

##########################################################
# Transforms Compositions
##########################################################
def inference_trans():
  """transforms used for evaluation. Simply convert the images to tensors.
  Returns:
    output (callable) - A transformation that recieves a PIL images and converts
    it to torch.Tensor.
  """
  # BEGIN SOLUTION
  return transforms.ToTensor()
  # END SOLUTION

def default_trans(random_crop_size):
  """transforms used in the basic case for training.
  Args:
    random_crop_size (int / tuple(int, int)) - crop size.
    if int, takes crops of size (random_crop_size x random_crop_size)    
    if tuple, takes crops of size (random_crop_size[0] x random_crop_size[1])
  Returns:
    output (callable) - A transformation that recieves a PIL image, converts it 
    to torch.Tensor and takes a random crop of it. The result's shape is 
    C x random_crop_size x random_crop_size.
  """
  # BEGIN SOLUTION
  # print(random_crop_size)
  random_crop_size = (random_crop_size,random_crop_size) if isinstance(random_crop_size,int) else random_crop_size
  return transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(random_crop_size)])

  # END SOLUTION
def flips():
   return transforms.Compose([
    transforms.ToTensor(),
    NineCrops()])
    
def flip_back(x):
  # [0,90,180,270]
  theta1 = 0
  theta2 = 0
  theta3 = 0
  theta4 = 0
  x[0] = x[0]
  x[1] = torch.flip(torchvision.transforms.functional.rotate(x[0],theta1), [1])
  x[2] = torchvision.transforms.functional.rotate(x[0],theta1)
  x[3] = torchvision.transforms.functional.rotate(x[0],theta2)
  x[4] = torchvision.transforms.functional.rotate(x[0],theta2)
  x[5] = torchvision.transforms.functional.rotate(x[0],theta3)
  x[6] = torchvision.transforms.functional.rotate(x[0],theta3)
  x[7] = torchvision.transforms.functional.rotate(x[0],theta4)
  x[8] = torchvision.transforms.functional.rotate(x[0],theta4)
  return x

def advanced_trans2(random_crop_size):
  """transforms used in the advanced case for training.
  Args:
    random_crop_size (int / tuple(int, int)) - crop size.
    if int, takes crops of size (random_crop_size x random_crop_size)    
    if tuple, takes crops of size (random_crop_size[0] x random_crop_size[1])
  Returns:
    output (callable) - A transformation that recieves a PIL image, converts it 
    to torch.Tensor, takes a random crop of it, and takes the EightCrops of this
    random crop. The result's shape is 8 x C x random_crop_size x random_crop_size.

  Note: you may explore different augmentations for your original implementation.
  """
  # BEGIN SOLUTION
  # print(random_crop_size)
  # print(random_crop_size)
  # random_crop_size = (random_crop_size,random_crop_size) if isinstance(random_crop_size,int) else random_crop_size
  
  random_crop_size = (random_crop_size,random_crop_size) if isinstance(random_crop_size,int) else random_crop_size
  
  i = 2#**torch.randint(0, 3, (1,)).item()

  size = (random_crop_size[0]/i,random_crop_size[1]/i)

  size0 = size[0] if size[0]%2==0 else size[0]-1
  size1 = size[1] if size[1]%2==0 else size[1]-1
  # print(random_crop_size)
  # print((size0,size1))
  # print("hi")
  # print(random_crop_size)
  return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((int(size0),int(size1))),
    # transforms.RandomCrop(random_crop_size),
    NineCrops()])
  # END SOLUTION



def advanced_trans(random_crop_size):
  """transforms used in the advanced case for training.
  Args:
    random_crop_size (int / tuple(int, int)) - crop size.
    if int, takes crops of size (random_crop_size x random_crop_size)    
    if tuple, takes crops of size (random_crop_size[0] x random_crop_size[1])
  Returns:
    output (callable) - A transformation that recieves a PIL image, converts it 
    to torch.Tensor, takes a random crop of it, and takes the EightCrops of this
    random crop. The result's shape is 8 x C x random_crop_size x random_crop_size.

  Note: you may explore different augmentations for your original implementation.
  """
  # BEGIN SOLUTION
  # print(random_crop_size)
  # print(random_crop_size)
  # random_crop_size = (random_crop_size,random_crop_size) if isinstance(random_crop_size,int) else random_crop_size
  
  
  return transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((int(size0),int(size1))),
    transforms.RandomCrop(random_crop_size),
    EightCrops()])
  # END SOLUTION


