import torch  # noqa

__all__ = ['backward']


#################################################
# backward
#################################################

def backward(loss, ctx):
  """Computes and accumulates gradients of tensors w.r.t `loss`.

  Computes gradients of tensors w.r.t `loss` according to the autograd context `ctx`.

  Note: `ctx` should not be used more than once in `backward` (why?). It's recommended to pop
    items from context to prevent another backward pass with the same context.

  Args:
    loss (torch.Tensor): A scalar tensor. Gradients would be computed with respect to it.
    ctx (List, optional): The autograd context. Defaults to None.
  """
  if loss.size() != ():
    raise ValueError("loss should be a scalar")
  # BEGIN SOLUTION
  raise NotImplementedError
  # END SOLUTION


#################################################
# PROVIDED: create_grad_if_necessary
#################################################

def create_grad_if_necessary(*tensors):
  """Creates gradients for tensors that may need it and don't have it.

  Note: having `tensor.grad == None` is considered as not having a gradient.

  Args:
    *tensors: Tensors (possibly) with and without gradients.
  """
  for tensor in tensors:
    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.grad is None:
      tensor.grad = torch.zeros_like(tensor)
