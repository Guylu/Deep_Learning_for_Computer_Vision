import torch  # noqa

from model import softmax_classifier
from model import softmax_classifier_backward
from model import cross_entropy
from utils import Metric, accuracy  # noqa

__all__ = ['create_model', 'test_epoch', 'test_epoch', 'train_loop']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################################################
# create_model
#################################################

def create_model():
  """Creates a Softmax Classifier model `(w, b)`.

  Returns:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
  """
  # BEGIN SOLUTION
  w ,b = torch.rand(10, 28*28),torch.rand(10,)
  # END SOLUTION
  return w, b


#################################################
# train_epoch
#################################################

def train_epoch(w, b, lr, loader):
  """Trains over an epoch, and returns the accuracy and loss over the epoch.

  Note: The accuracy and loss are average over the epoch. That's different from
  running the classifier over the data again at the end of the epoch, as the
  weights changed over the iterations. However, it's a common practice, since
  iterating over the training set (again) is time and resource exhustive.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    lr (float): The learning rate.
    loader (torch.utils.data.DataLoader): A data loader. An iterator over the dataset.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  device = w.device

  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION
    # NOTE: In your solution you MUST keep the loss in a tensor called `loss`
    # NOTE: In your solution you MUST keep the acurracy in a tensor called `acc`
    # print("hey")
    # print(x.shape)
    # END SOLUTION
    x = x.view(x.shape[0],w.shape[1])
    pred = softmax_classifier(x,w,b)
    softmax_classifier_backward(x,w,b,pred,y)
    w -= w.grad*lr
    b -= b.grad*lr
    loss = cross_entropy(pred,y)
    acc = accuracy(pred,y)
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# test_epoch
#################################################

def test_epoch(w, b, loader):
  """Evaluating the model at the end of the epoch.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    loader (torch.utils.data.DataLoader): A data loader. An iterator over the dataset.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  device = w.device

  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION
    # NOTE: In your solution you MUST keep the loss in a tensor called `loss`
    # NOTE: In your solution you MUST keep the acurracy in a tensor called `acc`
    # print(x.shape)
    # x = x.view(x.shape[0],w.shape[1])
    # print(x.shape)
    x = x.view(x.shape[0],w.shape[1])
    pred = softmax_classifier(x,w,b)
    loss = cross_entropy(pred,y)
    acc = accuracy(pred,y)
    # END SOLUTION
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# PROVIDED: train
#################################################

def train_loop(w, b, lr, train_loader, test_loader, epochs, test_every=1):
  """Trains the Softmax Classifier model and report the progress.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    lr (float): The learning rate.
    train_loader (torch.utils.data.DataLoader): The training set data loader.
    test_loader (torch.utils.data.DataLoader): The test set data loader.
    epochs (int): Number of training epochs.
    test_every (int): How frequently to report progress on test data.
  """
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(w, b, lr, train_loader)
    print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
          f'Loss: {train_loss.avg:7.4g}',
          f'Accuracy: {train_acc.avg:.3f}',
          sep='   ')
    if epoch % test_every == 0:
      test_loss, test_acc = test_epoch(w, b, test_loader)
      print(' Test', f'Epoch: {epoch:03d} / {epochs:03d}',
            f'Loss: {test_loss.avg:7.4g}',
            f'Accuracy: {test_acc.avg:.3f}',
            sep='   ')
