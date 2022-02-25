from torch.nn.functional import softmax # noqa
from torch import bmm, cat, transpose, reshape, matmul # noqa

__all__ = ['multi_head_attention']


#################################################
# Multi Head Attention 
#################################################
def re(x):
        batch_size, num_heads, sequence_size, head_emb_dim = x.size()
        return x.reshape(batch_size * num_heads, sequence_size, head_emb_dim)

def multi_head_attention(q, k, v):
  """A differentiable multi head attention function.

  Args:
    q (torch.Tensor): The query embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
    k (torch.Tensor): The key embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
    v (torch.Tensor): The value embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.

  Returns:
    y (torch.Tensor): The multi head attention output.
      Has shape `(batch_size, sequence_size, num_heads * head_emb_dim)`.
  """
  # BEGIN SOLUTION
  batch_size, num_heads, sequence_size, head_emb_dim = q.size()
  qsize = q.size()
  q,k,v = re(q), re(k), re(v)
  y = bmm(softmax((bmm(q,k.transpose(1,2)))/head_emb_dim**0.5,dim=2),v).reshape(*qsize)
  y = y.permute(0,2,1,3).reshape(batch_size,sequence_size,num_heads*head_emb_dim)
  # END SOLUTION
  return y
