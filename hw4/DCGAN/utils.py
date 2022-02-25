import numpy as np
import matplotlib.pyplot as plt 

__all__ = ['Metric', 'to_img', 'show_image']

#################################################
# PROVIDED: Metric
#################################################

class Metric:
  def __init__(self):
    self.lst = 0.
    self.sum = 0.
    self.cnt = 0
    self.avg = 0.

  def update(self, val, cnt=1):
    self.lst = val
    self.sum += val * cnt
    self.cnt += cnt
    self.avg = self.sum / self.cnt

#################################################
# PROVIDED: to image
#################################################

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

#################################################
# PROVIDED: show image
#################################################

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))