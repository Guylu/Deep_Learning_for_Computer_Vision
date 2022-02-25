import unittest
import torch
from torchvision import transforms
from pathlib import Path
from experiment import Experiment
import utils
from models import ZSSRNet, ZSSRResNet, ZSSROriginalNet

##########################################################
# Experiment
##########################################################

class TestModels(unittest.TestCase):
  def setUp(self):
    self.scale_factor = 2
    self.B, self.C, self.H, self.W = 8, 3, 32, 32
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.x = torch.randn(self.B, self.C, self.H, self.W).to(self.device)

  def testZSSRNet(self):
    model = ZSSRNet(scale_factor=self.scale_factor).to(self.device)
    out = model(self.x)
    out_shape = (self.B, self.C, self.scale_factor * self.H, self.scale_factor * self.W)
    self.assertTrue(out.shape == out_shape, msg=f"expected output shape {out_shape} instead got {out.shape}")

  def testZSSRResNet(self):
    model = ZSSRResNet(scale_factor=self.scale_factor).to(self.device)
    out = model(self.x)
    out_shape = (self.B, self.C, self.scale_factor * self.H, self.scale_factor * self.W)
    self.assertTrue(out.shape == out_shape, msg=f"expected output shape {out_shape} instead got {out.shape}")

  def testZSSROriginalNet(self):
    model = ZSSROriginalNet(scale_factor=self.scale_factor).to(self.device)
    out = model(self.x)
    out_shape = (self.B, self.C, self.scale_factor * self.H, self.scale_factor * self.W)
    self.assertTrue(out.shape == out_shape, msg=f"expected output shape {out_shape} instead got {out.shape}")





