import unittest
import torch
from torchvision import transforms
from pathlib import Path
from model import Generator, Discriminator
from config import Config
import numpy as np 

##########################################################
# Experiment
##########################################################

class TestModel(unittest.TestCase):
  def setUp(self):
    self.conf = Config()
    self.latent = torch.randn(self.conf.batch_size, self.conf.latent_dim,1,1, device=self.conf.device)
    self.x = torch.randn(self.conf.batch_size, self.conf.channels, self.conf.img_size, self.conf.img_size).to(self.conf.device)
    self.gen_true_count = 1066880
    self.disc_true_count = 661248

  def testDiscriminatorOutput(self):
    discriminator = Discriminator(self.conf)
    discriminator = discriminator.to(self.conf.device)
    out = discriminator(self.x)
    out_shape = torch.ones(self.conf.batch_size, device=self.conf.device).shape
    self.assertTrue(out.shape == out_shape, msg=f"expected output shape {out_shape} instead got {out.shape}")

  def testGeneratorOutput(self):
    generator = Generator(self.conf)
    generator = generator.to(self.conf.device)
    out = generator(self.latent)
    out_shape = self.x.shape
    self.assertTrue(out.shape == out_shape, msg=f"expected output shape {out_shape} instead got {out.shape}")

  def testGeneratorParams(self):
    generator = Generator(self.conf)
    generator = generator.to(self.conf.device)
    param_count = np.sum([np.prod(p.size()) for p in generator.parameters()])
    self.assertTrue(param_count == self.gen_true_count, msg=f"expected number of model params to be {self.gen_true_count} instead got {param_count}")

  def testDiscriminatorParams(self):
    discriminator = Discriminator(self.conf)
    discriminator = discriminator.to(self.conf.device)
    param_count = np.sum([np.prod(p.size()) for p in discriminator.parameters()])
    self.assertTrue(param_count == self.disc_true_count, msg=f"expected number of model params to be {self.disc_true_count} instead got {param_count}")





