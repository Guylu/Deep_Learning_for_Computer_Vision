import torch 
import torchvision.utils
import matplotlib.pyplot as plt 
from utils import Metric, show_image  

__all__ = ['train_epoch', 'train_loop']

#################################################
# IMPLEMENT: train_epoch
#################################################

def train_epoch(generator, discriminator, criterion, gen_optimizer, disc_optimizer, loader, conf):
  """Trains over an epoch, and returns the  generator loss metric and  discriminator loss metric over the epoch.

  Note: You MUST have `gen_loss_metric` tensor with the generator loss value, and `disc_loss_metric` tensor with
  the discriminator loss value.

  Args:
    generator (torch.nn.Module): The generator network.
    discriminator (torch.nn.Module): The discriminator network.
    criterion (callable): The loss function. Should return a scalar tensor.
    gen_optimizer (torch.optim.Optimizer): The generator optimizer.
    disc_optimizer (torch.optim.Optimizer): The discriminator optimizer.
    loader (torch.utils.data.DataLoader): The data loader.
    conf (Config): The configuration holding information about hyperparams.

  Returns:
    gen_loss_metric (Metric): The generator loss metric over the epoch.
    disc_loss_metric (Metric): The discriminator loss metric over the epoch.
  """

  disc_loss_metric = Metric()
  gen_loss_metric = Metric()
  # set to training mode
  generator.train()
  discriminator.train()
  
  for imgs, _ in loader:
    
    # Create two tensors, one for the generator loss and one for the discriminator loss. 
    disc_loss = torch.Tensor()
    gen_loss = torch.Tensor()

    # Move the batch of real images to device
    imgs = imgs.to(conf.device)

    # generate a batch of the latent prior (size [batch size, latent dim, 1, 1])
    latent = torch.randn(imgs.size(0), conf.latent_dim,1,1, device=conf.device)
    # Generate a fake batch of images by passing the latent batch through the generator 
    gen_imgs = generator(latent)
    
    #  Adversarial ground truths: create real (ones) and fake (zeros) labels for use in the loss (both of size [batch size])
    label_real = torch.ones(imgs.size(0), device=conf.device)
    label_fake = torch.zeros(imgs.size(0), device=conf.device)
    
    # BEGIN SOLUTION


    # ---------------------
    #  Train Discriminator  (Train discriminator to correctly classify real and fake)
    # ---------------------
    
    # Reset the gradients of the generator optimizer
    disc_optimizer.zero_grad()
    # real_cpu = data[0].to(device)
    # b_size = real_cpu.size(0)
    # label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    # Pass the batch of real images through the discriminator 
    output = discriminator(imgs)
    
    # Calculate the loss for the real batch
    disc_loss1 = criterion(output,label_real)
    
    # Calculate the gradients in a backward pass
    disc_loss1.backward()
    
    # Pass the batch of fake images through the discriminator
    # Note: detach the computation graph of the generator so that gradients are not backpropagated into the generator
    output = discriminator(gen_imgs.detach())
    
    # Calculate the loss for the fake batch
    disc_loss2 = criterion(output,label_fake)
    
    # Calculate the gradients in a backward pass
    disc_loss2.backward()
    disc_loss = disc_loss1+disc_loss2
    
    # Discriminator optim step    
    disc_optimizer.step()

    # -----------------
    #  Train Generator  (Train generator to output an image that is classified as real)
    # -----------------

    # Reset the gradients of the generator optimizer
    gen_optimizer.zero_grad()

    # Pass the fake batch through the discriminator 
    output = discriminator(gen_imgs)
    
    # Calculate the loss.
    # Note: which labels should you assign to the fake images?
    gen_loss = criterion(output,label_real)
    
    # Generator backward + optim step     
    gen_loss.backward()
    
    # Generator optim step  
    gen_optimizer.step()
    
    # END SOLUTION
    
    disc_loss_metric.update(disc_loss.item(), imgs.size(0))
    gen_loss_metric.update(gen_loss.item(), imgs.size(0))
  return gen_loss_metric, disc_loss_metric


#################################################
# PROVIDED: train_loop
#################################################

def train_loop(generator, discriminator, criterion, gen_optimizer, disc_optimizer, dataloader, conf):
  """Trains a model to minimize some loss function and reports the progress.

  Args:
    generator (torch.nn.Module): The generator network.
    discriminator (torch.nn.Module): The discriminator network.
    criterion (callable): The loss function. Should return a scalar tensor.
    gen_optimizer (torch.optim.Optimizer): The generator optimizer.
    disc_optimizer (torch.optim.Optimizer): The discriminator optimizer.
    loader (torch.utils.data.DataLoader): The data loader.
    conf (Config): The configuration holding information about hyperparams.
  """

  g_l_train = []
  d_l_train = []
  for epoch in range(conf.epochs):
    g_l_train.append(0)
    d_l_train.append(0)
    import time
    start_time = time.time()  
    gen_loss, disc_loss = train_epoch(generator, discriminator, criterion, gen_optimizer, disc_optimizer, dataloader, conf)
    
    print('Total execution time of epoch {}:'.format(epoch), '{:5.2f}'.format((time.time() - start_time) / 60), 'minutes')
    print('Train', f'Epoch: {epoch:03d} / {conf.epochs:03d}',
          f'Generator Loss: {gen_loss.avg:7.4g}',
          f'Discriminator Loss: {disc_loss.avg:}',
          sep='   ')
    
    print('Train', f'Epoch: {epoch+1:03d} / {conf.epochs:03d}')
    g_l_train[-1] += gen_loss.avg
    d_l_train[-1] += disc_loss.avg      
    if epoch % conf.test_every == 0:
        if conf.verbose:
            with torch.no_grad():
                generator.eval()
                # sample latent vectors from the standard normal distribution
                image_batch , _ = next(iter(dataloader))
                latent = torch.randn(conf.batch_size, conf.latent_dim,1,1, device=conf.device)
                fake_image_batch = generator(latent)
                fake_image_batch = fake_image_batch.cpu()

                fig, ax = plt.subplots(figsize=(8, 8))
                show_image(torchvision.utils.make_grid(fake_image_batch.data[:100],10,5))
                plt.show()
  return g_l_train, d_l_train

