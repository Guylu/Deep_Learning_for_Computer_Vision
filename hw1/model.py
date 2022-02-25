import torch
from torch.nn.functional import one_hot

__all__ = [
  'softmax',
  'cross_entropy',
  'softmax_classifier',
  'softmax_classifier_backward'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##########################################################
# Softmax
##########################################################

def softmax(x):
  """Softmax activation.

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.

  Returns:
    y (torch.Tensor): The softmax distribution over `x`. Has the same shape as `x`.
      Each row in `y` is a probability over the classes.
  """
  x = x.to(device)
  return torch.exp(x-torch.logsumexp(x,1, keepdim=True))


##########################################################
# Cross Entropy
##########################################################

def cross_entropy(pred, target):
  """Cross-entropy loss for hard-labels.

  Hint: You can use the imported `one_hot` function.

  Args:
    pred (torch.Tensor): The predictions (probability per class), has shape `(batch_size, num_classes)`.
    target (torch.Tensor): The target classes (integers), has shape `(batch_size,)`.

  Returns:
    loss (torch.Tensor): The mean cross-entropy loss over the batch.
  """
  pred = pred.to(device)
  target = target.to(device)
  return -torch.mean(torch.log(torch.diag(pred[:,target])))



##########################################################
# Softmax Classifier
##########################################################

def softmax_classifier(x, w, b):
  """Applies the prediction of the Softmax Classifier.

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.

  Returns:
    pred (torch.Tensor): The predictions, has shape `(batch_size, num_classes)`.
      Each row is a probablity measure over the classes.
  """
  # print("hey")
  # print(b.view(-1,1))
  # print(w.mm(x.T))
  x = x.to(device)
  w = w.to(device)
  b = b.to(device)
  return softmax((w.mm(x.T)+b.view(-1,1)).T)


##########################################################
# Softmax Classifier Backward
##########################################################

def softmax_classifier_backward(x, w, b, pred, target):
  """Computes the gradients of weight in the Softmax Classifier.

  The gradients computed for the parameters `w` and `b` should be stored in
  `w.grad` and `b.grad`, respectively.

  Hint: You can use the imported `one_hot` function.

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    pred (torch.Tensor): The predictions (probability per class), has shape `(batch_size, num_classes)`.
    target (torch.Tensor): The target classes (integers), has shape `(batch_size,)`.
  """
  x = x.to(device)
  w = w.to(device)
  b = b.to(device)
  pred = pred.to(device)
  target = target.to(device)
  c = x.size()[0]
  res = pred -torch.nn.functional.one_hot(target,num_classes=pred.size()[1])
  w.grad,b.grad = res.T.mm(x) / c,torch.sum(res ,0) / c
