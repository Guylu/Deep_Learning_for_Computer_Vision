import unittest
import torch
from torchvision import transforms
from PIL import Image
from data_handling import BasicZSSRDataset, default_trans, advanced_trans, inference_trans, EightCrops

##########################################################
# Dataset
##########################################################

class TestBasicZSSRDataset(unittest.TestCase):
  def setUp(self):
    self.image_path = 'data/train/barbara.png'
    self.scale_factor = 2
    self.transform = transforms.ToTensor()
    self.dataset = BasicZSSRDataset(self.image_path, self.scale_factor, self.transform)

  def testLen(self):    
    self.assertTrue(len(self.dataset) == 1, msg='dataset should be of length 1.')

  def testgetItem(self):
    dataset = BasicZSSRDataset(self.image_path, self.scale_factor, self.transform)
    item = dataset[0]
    self.assertTrue(len(item.keys()) == 2, msg='dataset item dictionary should have only 2 keys.')
    self.assertTrue('LR' in item.keys(), msg="'LR' key doesn't appear in item dictionary")
    self.assertTrue('SR' in item.keys(), msg="'SR' key doesn't appear in item dictionary")
    C, H, W = item['SR'].shape
    lr_shape = (C, H / self.scale_factor, W / self.scale_factor)
    self.assertTrue(item['LR'].shape == lr_shape, msg=f"LR key should be of shape {lr_shape} instead got {item['LR'].shape}")

##########################################################
# transforms
##########################################################

class TestTransforms(unittest.TestCase):
  def setUp(self):
    self.image_path = 'data/train/barbara.png'
    self.image = Image.open(self.image_path)
    self.random_crop_size = 32
  
  def testDefaultTrans(self):
    res = default_trans(self.random_crop_size)(self.image)
    res_size = (3, self.random_crop_size, self.random_crop_size)
    self.assertTrue(res.shape == res_size, msg=f'output shape should be {res_size} but is instead {res.shape}')

  def testInferenceTrans(self):
    res = inference_trans()(self.image)
    res_size = (3, self.image.size[1], self.image.size[0])
    self.assertTrue(res.shape == res_size, msg=f'output shape should be {res_size} but is instead {res.shape}')

  def testAdvancedTrans(self):
    res = advanced_trans(self.random_crop_size)(self.image)
    res_size = (8, 3, self.random_crop_size, self.random_crop_size)
    self.assertTrue(res.shape == res_size, msg=f'output shape should be {res_size} but is instead {res.shape}')

  