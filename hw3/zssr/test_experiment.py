import unittest
import torch
from torchvision import transforms
from pathlib import Path
from experiment import Experiment
import utils

##########################################################
# Experiment
##########################################################

class TestExperiment(unittest.TestCase):
  def setUp(self):
    self.config = {
        'lr': 0.001,
        'random_crop_size': 16,
        'scale_factor': 2,
        'epochs': 2, #4000
        'show_interval': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbose': False,
        'back_iters' : 80,
        "step_size" : 80
    }
    self.experiment = Experiment(utils.DATA_ROOT, self.config)

  def testRun(self):
    run_df = self.experiment.run()
    self.assertTrue(True)



