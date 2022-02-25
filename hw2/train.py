import torch  # noqa

from autograd import backward  # noqa
from utils import Metric, accuracy  # noqa

__all__ = ['test_epoch', 'test_epoch', 'train_loop']


#################################################
# train_epoch
#################################################

def train_epoch(model, criterion, optimizer, loader, device):
  """Trains over an epoch, and returns the accuracy and loss over the epoch.

  Note: The accuracy and loss are average over the epoch. That's different from
  running the classifier over the data again at the end of the epoch, as the
  weights changed over the iterations. However, it's a common practice, since
  iterating over the training set (again) is time and resource exhustive.

  Note: You MUST have `loss` tensor with the loss value, and `acc` tensor with
  the accuracy (you can use the imported `accuracy` method).

  Args:
    model (nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    optimizer (optim.SGD): The optimizer.
    loader (torch.utils.data.DataLoader): The test set data loader.
    device (torch.device): The device to run on.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION
    ctx = []
    x = x.reshape(x.shape[0],x.shape[2]*x.shape[3])
    
    res = model.forward(x,ctx)
    loss = criterion(res,y,ctx)
    acc = accuracy(res,y)
    backward(loss,ctx)
    optimizer.step()
    optimizer.zero_grad()
    # END SOLUTION
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# test_epoch
#################################################

def test_epoch(model, criterion, loader, device):
  """Evaluating the model at the end of the epoch.

  Note: You MUST have `loss` tensor with the loss value, and `acc` tensor with
  the accuracy (you can use the imported `accuracy` method).

  Args:
    model (nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    loader (torch.utils.data.DataLoader): The test set data loader.
    device (torch.device): The device to run on.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION
    x = x.reshape(x.shape[0],x.shape[2]*x.shape[3])
    res = model.forward(x)
    loss = criterion(res,y)
    acc = accuracy(res,y)
    # END SOLUTION
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# PROVIDED: train_loop
#################################################

def train_loop(model, criterion, optimizer, train_loader, test_loader, device, epochs, test_every=1):
  """Trains a model to minimize some loss function and reports the progress.

  Args:
    model (nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    optimizer (optim.SGD): The optimizer.
    train_loader (torch.utils.data.DataLoader): The training set data loader.
    test_loader (torch.utils.data.DataLoader): The test set data loader.
    device (torch.device): The device to run on.
    epochs (int): Number of training epochs.
    test_every (int): How frequently to report progress on test data.
  """
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(model, criterion, optimizer, train_loader, device)
    print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
          f'Loss: {train_loss.avg:7.4g}',
          f'Accuracy: {train_acc.avg:.3f}',
          sep='   ')
    if epoch % test_every == 0:
      test_loss, test_acc = test_epoch(model, criterion, test_loader, device)
      print(' Test', f'Epoch: {epoch:03d} / {epochs:03d}',
            f'Loss: {test_loss.avg:7.4g}',
            f'Accuracy: {test_acc.avg:.3f}',
            sep='   ')
