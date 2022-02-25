import torch  # noqa
from torch.nn.functional import one_hot  # noqa

__all__ = ['linear', 'relu', 'softmax', 'cross_entropy', 'cross_entropy_loss']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################################################
# EXAMPLE: mean
#################################################

def mean(x, ctx=None):
  """A differentiable Mean function.

  Backward call:
    backward_fn: mean_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output scalar tensor, the mean of `x`.
  """
  y = x.mean()
  # the backward function with its arguments is appended to `ctx`
  if ctx is not None:
    ctx.append([mean_backward, [y, x]])
  return y


def mean_backward(y, x):
  """Backward computation of `mean`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output scalar tensor.
    x (torch.Tensor): The input tensor.
  """
  # the gradient of `x` is added to `x.grad`
  x.grad += torch.ones_like(x) * (y.grad / x.numel())


#################################################
# linear
#################################################

def linear(x, w, b, ctx=None):
  """A differentiable Linear function. Computes: y = w * x + b

  Backward call:
    backward_fn: linear_backward
    args: y, x, w, b

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(out_dim, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(out_dim,)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, has shape `(batch_size, out_dim)`.
  """
  # VECTORIZATION HINT: torch.mm

  # BEGIN SOLUTION
  y = x@w.T+b

  if ctx is not None:
    ctx.append([linear_backward, [y, x,w,b]])
  return y
  # END SOLUTION


def linear_backward(y, x, w, b):
  """Backward computation of `linear`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, `w` and `b`,
  and accumulates them in `x.grad`, `w.grad` and `b.grad` respectively.

  Args:
    y (torch.Tensor): The output tensor, has shape `(batch_size, out_dim)`.
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(out_dim, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(out_dim,)`.
  """
  # VECTORIZATION HINT: torch.mm

  # BEGIN SOLUTION
  x.grad += y.grad@w
  w.grad += y.grad.T@x
  b.grad += y.grad.sum(dim=0)
  # END SOLUTION


#################################################
# relu
#################################################

def relu(x, ctx=None):
  """A differentiable ReLU function.

  Note: `y` should be a different tensor than `x`. `x` should not be changed.
        Read about Tensor.clone().

  Note: Don't modify the input in-place.

  Backward call:
    backward_fn: relu_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. Has non-negative values.
  """
  # BEGIN SOLUTION
  y = torch.maximum(torch.zeros_like(x),x)

  if ctx is not None:
    ctx.append([relu_backward, [y, x]])
  return y
  # END SOLUTION


def relu_backward(y, x):
  """Backward computation of `relu`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor. Has non-negative values.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  # print(torch.tensor(x>0).long())
  
  x.grad += y.grad*(x>0).float()
  # END SOLUTION


#################################################
# softmax
#################################################

def softmax(x, ctx=None):
  """A differentiable Softmax function.

  Note: make sure to add `x` from the input to the context,
        and not some intermediate tensor.

  Backward call:
    backward_fn: softmax_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, has shape `(batch_size, num_classes)`.
      Each row in `y` is a probability distribution over the classes.
  """
  # BEGIN SOLUTION
  # x = x.to(device)
  y = torch.exp(x-torch.logsumexp(x,1, keepdim=True))
  if ctx is not None:
    ctx.append([softmax_backward, [y,x]])
  return y
  # END SOLUTION


def softmax_backward(y, x):
  """Backward computation of `softmax`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor, has shape `(batch_size, num_classes)`.
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.
  """
  # VECTORIZATION HINT: one_hot, torch.gather, torch.einsum

  # BEGIN SOLUTION
  # dest_matrix = torch.sum(y*y, dim=1)
  # source =  y.T@(1-y)
  # print(dest_matrix)
  # print(torch.sum(y*y, dim=1))
  # dest_matrix[range(len(dest_matrix)), range(len(dest_matrix))] =source
  # x.grad += y.grad @ torch.from_numpy(res)

  J = - y[..., None] @ y[:, None, :] 
  J[:, torch.arange(J[0].shape[0]),torch.arange(J[0].shape[1])] = y * (1. - y)
  # print(J.shape)
  # print(y.grad.shape)
  res = J*y.grad.reshape(y.grad.shape[0],y.grad.shape[1],1)
  x.grad += res.sum(dim=1)
  # END SOLUTION


#################################################
# cross_entropy
#################################################

def cross_entropy(pred, target, ctx=None):
  """A differentiable Cross-Entropy function for hard-labels.

  Backward call:
    backward_fn: cross_entropy
    args: loss, pred, target

  Args:
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Each row is a probability distribution over the classes.
    target (torch.Tensor): The targets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    loss (torch.Tensor): The per-example cross-entropy tensor, has shape `(batch_size,).
      Each value is the cross-entropy loss of that example in the batch.
  """
  # VECTORIZATION HINT: one_hot, torch.gather
  eps = torch.finfo(pred.dtype).tiny
  # BEGIN SOLUTION
  loss = -torch.log(torch.diag(pred[:,target])+eps)
  if ctx is not None:
    ctx.append([cross_entropy_backward, [loss, pred, target]])
  return loss
  # END SOLUTION


def cross_entropy_backward(loss, pred, target):
  """Backward computation of `cross_entropy`.

  Propagates the gradients of `loss` (in `loss.grad`) to `pred`,
  and accumulates them in `pred.grad`.

  Note: `target` is an integer tensor and has no gradients.

  Args:
    loss (torch.Tensor): The per-example cross-entropy tensor, has shape `(batch_size,).
      Each value is the cross-entropy loss of that example in the batch.
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Each row is a probability distribution over the classes.
    target (torch.Tensor): The tragets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
  """
  # VECTORIZATION HINT: one_hot, torch.gather, torch.scatter_add
  eps = torch.finfo(pred.dtype).tiny
  # BEGIN SOLUTION
  res = (-1/(pred+eps)) * one_hot(target,num_classes=pred.shape[1])
  pred.grad += torch.unsqueeze(loss.grad, 1)*res
  # print(pred.grad)
  # END SOLUTION


#################################################
# PROVIDED: cross_entropy_loss
#################################################

def cross_entropy_loss(pred, target, ctx=None):
  """A differentiable Cross-Entropy loss for hard-labels.

  This differentiable function is similar to PyTorch's cross-entropy function.

  Note: Unlike `cross_entropy` this function expects `pred` to be BEFORE softmax.

  Note: You should not implement the backward of that function explicitly, as you use only
        differentiable functions for that. That part of the "magic" in autograd --
        you can simply compose differentiable functions, and it works!

  Args:
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Unlike `cross_entropy`, this prediction IS NOT a probability distribution over
      the classes. It expects to see predictions BEFORE sofmax.
    target (torch.Tensor): The targets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    loss (torch.Tensor): The scalar loss tensor. The mean loss over the batch.
  """
  pred = softmax(pred, ctx=ctx)
  batched_loss = cross_entropy(pred, target, ctx=ctx)
  loss = mean(batched_loss, ctx=ctx)
  return loss
