import torch

__all__ = ['Config']

class Config():
    # Number of epochs
    epochs = 5

    # Batch size
    batch_size=128
    
    # Learning rate
    lr = 0.0002
    
    # Optimizer parameters
    b1=0.5
    b2=0.999
    
    # Size of latent vector (i.e. size of generator input)
    latent_dim=100
    
    # Real image size
    img_size=32

    # Number of channels in the training images. For grayscale images this is 1
    channels=1

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Interactive mode
    verbose=True
    test_every=1

    def __init__(self):
        print('Init config successfully')
        