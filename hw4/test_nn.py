import unittest  # noqa
from nn import MHSA, PatchEmbedding, TransformerBlock, PositionalEmbedding, CLSToken
import torch
import numpy as np


def init_non_random(tensor_shape):
  tensor_len = np.prod(tensor_shape)
  delta = 0.001 * tensor_len / 2
  x = torch.linspace(-delta, delta, tensor_len)
  return x.reshape(tensor_shape)


def compare(d1, d2, delta):
  return (d1 - d2).abs() < delta


def init_mhsa_layer(mhsa_layer, dim):
  attn_state_dict = {}
  attn_state_dict['lin_qkv.weight'] = init_non_random((dim * 3, dim))
  attn_state_dict['lin_qkv.bias'] = init_non_random((dim * 3,))
  attn_state_dict['lin.weight'] = init_non_random((dim, dim))
  attn_state_dict['lin.bias'] = init_non_random((dim,))
  mhsa_layer.load_state_dict(attn_state_dict)


def init_mlp(mlp, dim_in, dim_out):
  mlp_state_dict = {}
  mlp_state_dict['fc1.weight'] = init_non_random((dim_in * 4, dim_in))
  mlp_state_dict['fc1.bias'] = init_non_random((dim_in * 4,))
  mlp_state_dict['fc2.weight'] = init_non_random((dim_out, dim_in * 4))
  mlp_state_dict['fc2.bias'] = init_non_random((dim_out,))
  mlp.load_state_dict(mlp_state_dict)


def init_norm_layer(norm_layer, dim):
  norm_layer_state_dict = {}
  norm_layer_state_dict['weight'] = init_non_random((dim,))
  norm_layer_state_dict['bias'] = init_non_random((dim,))
  norm_layer.load_state_dict(norm_layer_state_dict)


#################################################
# Test Multi Head Self Attention Layer
#################################################

class TestMHSALayer(unittest.TestCase):
  def setUp(self):
    # Test parameters
    self.dim=32
    self.n_heands = 4
    self.batch_size = 2
    self.sequence_size = 16
    
    # Init layer
    self.mhsa_layer = MHSA(self.dim, self.n_heands)

    # Init non-random weights and input
    init_mhsa_layer(self.mhsa_layer, self.dim)
    
    # Init optimizer
    self.optimizer = torch.optim.Adam(self.mhsa_layer.parameters())

  def testForward(self):
    x = init_non_random((self.batch_size, self.sequence_size, self.dim))
    y = self.mhsa_layer(x) 
    
    assert y.shape == x.shape, "Output tensor shape is incorrect"
    assert compare(y.sum(), 0.2722, 1e-2), "Incorrect output"
  
  def testBackwards(self):
    self.optimizer.zero_grad()
    x = init_non_random((self.batch_size, self.sequence_size, self.dim))
    y = self.mhsa_layer(x)
    loss = y.mean()
    loss.backward()

    assert compare(self.mhsa_layer.lin_qkv.weight.grad.sum(), -0.0062, 1e-2), "Incorrect output"
    assert compare(self.mhsa_layer.lin_qkv.bias.grad.sum(), 0.1172, 1e-2), "Incorrect output"
    assert compare(self.mhsa_layer.lin.weight.grad.sum(), 1.0071, 1e-2), "Incorrect output"
    assert compare(self.mhsa_layer.lin.bias.grad.sum(), 1., 1e-2), "Incorrect output"


#################################################
# Test Transformer Block
#################################################
class TestTransformerBlock(unittest.TestCase):
  def setUp(self):
    # Test parameters
    self.dim=32
    self.n_heands = 4
    self.batch_size = 2
    self.sequence_size = 16
    
    # Init layer
    self.transformer_block = TransformerBlock(self.dim, self.n_heands)

    # Init non-random weights and input
    init_mhsa_layer(self.transformer_block.mhsa, self.dim)
    init_mlp(self.transformer_block.mlp, self.dim, self.dim)
    init_norm_layer(self.transformer_block.norm1, self.dim)
    init_norm_layer(self.transformer_block.norm2, self.dim)

    # Init optimizer
    self.optimizer = torch.optim.Adam(self.transformer_block.parameters())

  def testForward(self):
    x = init_non_random((self.batch_size, self.sequence_size, self.dim))
    y = self.transformer_block(x) 
    
    assert y.shape == x.shape, "Output tensor shape is incorrect"
    assert compare(y.mean(), 0.9731, 1e-2), "Incorrect output"
  
  def testBackwards(self):
    self.optimizer.zero_grad()
    x = init_non_random((self.batch_size, self.sequence_size, self.dim))
    y = self.transformer_block(x) 
    loss = y.mean()
    loss.backward()

    # Test Attention gradients
    assert compare(self.transformer_block.mhsa.lin_qkv.weight.grad.sum(), 7.2423e-07, 1e-9), "Incorrect Attention gradients"
    assert compare(self.transformer_block.mhsa.lin_qkv.bias.grad.sum(), 2.5155e-06, 1e-8), "Incorrect Attention gradients"
    assert compare(self.transformer_block.mhsa.lin.weight.grad.sum(), 10.5837, 1e-2), "Incorrect Attention gradients"
    assert compare(self.transformer_block.mhsa.lin.bias.grad.sum(), 1., 1e-2), "Incorrect Attention gradients"

    # Test MLP gradients
    assert compare(self.transformer_block.mlp.fc1.weight.grad.sum(), 0.4177, 1e-2), "Incorrect MLP gradients"
    assert compare(self.transformer_block.mlp.fc1.bias.grad.sum(), 1.3696, 1e-2), "Incorrect MLP gradients"
    assert compare(self.transformer_block.mlp.fc2.weight.grad.sum(), 7.9017, 1e-2), "Incorrect MLP gradients"
    assert compare(self.transformer_block.mlp.fc2.bias.grad.sum(), 1., 1e-2), "Incorrect MLP gradients"

    # Test LayerNorm gradients
    assert compare(self.transformer_block.norm1.weight.grad.sum(), -1.3411e-06, 1e-8), "Incorrect LayerNorm gradients"
    assert compare(self.transformer_block.norm1.bias.grad.sum(), 2.7972, 1e-2), "Incorrect LayerNorm gradients"
    assert compare(self.transformer_block.norm2.weight.grad.sum(), 0.4048, 1e-2), "Incorrect LayerNorm gradients"
    assert compare(self.transformer_block.norm2.bias.grad.sum(), 89.8179, 1e-2), "Incorrect LayerNorm gradients"



#################################################
# Test Patch Embedding
#################################################

class TestPatchEmbedding(unittest.TestCase):
  def setUp(self):
    # Test parameters
    self.patch_dim = 16
    self.in_chans = 3
    self.dim = 8
    self.batch_size = 2
    self.img_dim = 32
    
    # Init layer
    self.patch_proj = PatchEmbedding(self.patch_dim, self.in_chans,self. dim)

    # Init non-random weights and input
    proj_state_dict = {}
    proj_state_dict['patch_embed.weight'] = init_non_random((self.dim, self.in_chans, self.patch_dim, self.patch_dim))
    proj_state_dict['patch_embed.bias'] = init_non_random((self.dim,))
    self.patch_proj.load_state_dict(proj_state_dict)
    
    # Init optimizer
    self.optimizer = torch.optim.Adam(self.patch_proj.parameters())

  def testForward(self):
    x = init_non_random((self.batch_size, self.in_chans, self.img_dim, self.img_dim))
    y = self.patch_proj(x)   
    sequence_size = PatchEmbedding.get_sequence_size(self.img_dim, self.patch_dim)
  
    assert y.shape == (self.batch_size, sequence_size, self.dim), "Output tensor shape is incorrect"
    assert compare(y.mean(), 142.6364, 1e-2), "Incorrect output"
  
  def testBackwards(self):
    self.optimizer.zero_grad()
    x = init_non_random((self.batch_size, self.in_chans, self.img_dim, self.img_dim))
    y = self.patch_proj(x)
    loss = y.mean()
    loss.backward()

    assert compare(self.patch_proj.patch_embed.weight.grad.sum(), 1.9610e-05, 1e-7), "Incorrect output"
    assert compare(self.patch_proj.patch_embed.bias.grad.sum(), 1., 1e-2), "Incorrect output"


#################################################
# Test Positional Embedding
#################################################
class TestPositionalEmbedding(unittest.TestCase):
  def setUp(self):
    # Test parameters
    self.dim = 16
    self.sequence_size = 8
    self.std = 2
    
    # Init layer
    self.pos_embed = PositionalEmbedding(self.sequence_size, self.dim, self.std)

    # Init optimizer
    self.optimizer = torch.optim.Adam(self.pos_embed.parameters())

  def testForward(self):
    y = self.pos_embed()  
    assert y.min() > -2 and y.max() < 2, "Initialization values are out of bounds "
    assert y.shape == (1, self.sequence_size, self.dim), "Output tensor shape is incorrect"
  
  def testBackwards(self):
    self.optimizer.zero_grad()
    y = self.pos_embed()
    loss = y.mean()
    loss.backward()
    assert next(self.pos_embed.parameters()).grad is not None, "Module parameters have no gradint"
    
    y1 = y.clone()
    self.optimizer.step()
    assert (y1 - self.pos_embed()).abs().sum() > 0, "Failed to update module parameters"


#################################################
# Test CLS Token
#################################################
class TestCLSToken(unittest.TestCase):
  def setUp(self):
    # Test parameters
    self.dim = 16
    self.std = 2

    # Init layer
    self.cls_token = CLSToken(self.dim, self.std)

    # Init optimizer
    self.optimizer = torch.optim.Adam(self.cls_token.parameters())

  def testForward(self):
    y = self.cls_token()
    assert y.min() > -2 and y.max() < 2, "Initialization values are out of bounds "
    assert y.shape == (1, 1, self.dim), "Output tensor shape is incorrect"
  
  def testBackwards(self):
    self.optimizer.zero_grad()
    y = self.cls_token()
    loss = y.mean()
    loss.backward()
    assert next(self.cls_token.parameters()).grad is not None, "Module parameters have no gradint"
    
    y1 = y.clone()
    self.optimizer.step()
    assert (y1 - self.cls_token()).abs().sum() > 0, "Failed to update module parameters"


 
