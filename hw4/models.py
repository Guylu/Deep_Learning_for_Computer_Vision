from nn import *
from torch.nn import Module, Linear, LayerNorm, Sequential, ModuleList
from torch import cat, stack
from vit_helpers import init_vit_weights


__all__ = ['ViT']

#################################################
# Vision Transformer
#################################################
class ViT(Module):
  def __init__(self, img_dim, patch_dim, in_chans, num_classes, dim, depth, num_heads, init_std):
    """Creates a Vision Transformer model

    Args:
      img_dim (int): Image dim, we use only squared images so the total
        image size is (in_chans, img_dim, img_dim).
      patch_dim (int): Patch dim, we use only squared patches so the total
        patch size is (patch_dim, patch_dim).
      in_chans (int): Number of channels in the input image.
      num_classes (int): Number of classes.
      dim (int): The PatchEmbedding output dimension.
      depth (int): The number of transformer blocks in the model.
      num_heads (int): Number of attention heads.
      init_std (float): the standard deviation of the truncated normal distribution used for initialization.
    """
    super().__init__()
    # BEGIN SOLUTION
    
    self.patch_embeddings = PatchEmbedding(patch_dim, in_chans, dim)
    self.sequence_size = PatchEmbedding.get_sequence_size(img_dim, patch_dim)
    self.positional_embedding = PositionalEmbedding(self.sequence_size*2 +1, dim, init_std)()
    self.cls = CLSToken(dim, init_std)
    self.blocks = ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
    self.norm = LayerNorm(dim,self.sequence_size)
    self.fcs = Sequential(Linear(dim,32),Linear(32,num_classes))
    # END SOLUTION

    # Initialize weights
    self.apply(init_vit_weights)

  def forward(self, x):
    """Computes the forward function of a vision transformer model

    Args:
      x (torch.Tensor): The input images.
        Has shape `(batch_size, in_chans, img_dim, img_dim)`, we use only squared images.

    Returns:
      y (torch.Tensor): The output classification tensor.
        Has shape `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    patches = self.patch_embeddings(x)
    # print(self.cls.shape)
    # print(patches.shape)
    cls = cat([self.cls() for _ in range(x.shape[0])], dim=0)
    tokens = cat([cls, patches], dim=1)
    # print(tokens.shape)
    # print(self.positional_embedding.shape)
    tokens = tokens + self.positional_embedding
    # tokens = self.blocks(tokens)
    for block in self.blocks:
      tokens = block(tokens)
    # print(tokens.shape)
    tokens = self.norm(tokens)
    # print(tokens.shape)
    res = self.fcs(tokens[:,1,:])
    # print(tokens.shape)
    # print(res.shape)
    return res

    # END SOLUTION
    
    
    



