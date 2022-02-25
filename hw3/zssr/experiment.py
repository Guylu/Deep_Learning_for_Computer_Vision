from pathlib import Path
import pandas as pd
import utils
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_handling import BasicZSSRDataset,OriginalZSSRDataset, default_trans, inference_trans
from data_handling import advanced_trans, EightCrops,flips,flip_back, NineCrops,advanced_trans2
from models import ZSSRNet, ZSSRResNet, ZSSROriginalNet


##########################################################
# Experiment
##########################################################
class Experiment:
  """Trains and evaluates your ZSSR framework on multiple images in a dataset.
  Produces PSNR results per image and total average PSNR. You may use the 
  different components (both basic and advanced) you have implemented so far.
  Eventually come up with your best configuration, training and evaluation
  method.
  
  NOTE: You may add more members to this class as long as you don't change the
  original members.
  """

  def __init__(self, dataset_root, config):
    """
    Args:
      dataset_root (pathlib.Path): Path to the root of the dataset.
      config (dict): Configuration dictionary. 
      Contains parameters from training & evaluation.
    """    
    self.train_paths = sorted([f for f in (dataset_root / 'train').glob('*.png')])
    self.gt_paths = sorted([f for f in (dataset_root / 'gt').glob('*.png')])
    # BEGIN SOLUTION
    self.config = config
    self.device = config["device"]
    self.epochs = config["epochs"]
    self.verbose = config["verbose"]
    self.scale_factor = config["scale_factor"]
    self.show_interval = config["show_interval"]
    self.random_crop_size = config["random_crop_size"]
    self.back_iters = config["back_iters"]
    self.dataset_root = dataset_root
    self.step_size =  config["step_size"]
    # END SOLUTION

  def baseline(self, image_path):
    """ Finds the baseline PSNR for a specific image, 
    when using bicubic interpolation (no learning).

    Args:
      image_path (pathlib.Path): Path to the image to be evaluated.
    Returns:
      image_psnr (float): The PSNR value between the original image and the 
      image created with bicubic interpolation.
    """
    # define dataset
    dataset = BasicZSSRDataset(image_path, self.scale_factor, inference_trans())
    # fetch an instance
    inputs = dataset[0]
    # parse it to gt and input
    gt_image, resize_image = inputs['SR'][None, ...], inputs['LR'][None, ...]
    # resize with bicubic interpolation
    bicubic_image = utils.rr_resize(resize_image, scale_factors=2)
    # compute baseline psnr
    baseline_psnr = utils.psnr(bicubic_image[0], gt_image[0])
    return baseline_psnr.item()

  def train(self, image_path):
    """ Trains a ZSSR model on a specific image.
    Args:
      image_path (pathlib.Path): Path to the image to be trained.

    Returns:
      model (torch.nn.Module): A trained model.

    We hinted specific steps and parts of the training loop and visualization.
    You may change these if you prefer.
    """
    # BEGIN SOLUTION
    # dataset = BasicZSSRDataset(image_path, self.scale_factor, inference_trans())
    adv = True
    trans =  advanced_trans2(self.random_crop_size) if adv else inference_trans()
    dataset = OriginalZSSRDataset(image_path, self.scale_factor, trans,l=5 if adv else 1)

    # define model
    model = ZSSRResNet(self.scale_factor)
    # model = {"reg":ZSSRNet(self.scale_factor),"res":ZSSRResNet(self.scale_factor)}[self.config["model"]]
    model = model.to(device = self.device)
    # define loss
    criterion = torch.nn.functional.l1_loss
    # define optimizer & schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size,gamma=0.9)
    # define visualizer
    if self.verbose:
      visualizer = utils.Visualizer()
      running_loss, running_psnr = 0, 0

    # training loop
    for t in range(1, self.epochs + 1):
      for i, data in enumerate(dataset):
        # fetch dataset instance
        # print(len(dataset))
        
        inputs = data
        # parse it to gt and input
        gt_image, resize_image = inputs['SR'][None, ...].to(device = self.device), inputs['LR'][None, ...].to(device = self.device)
        # forward pass


        res_in = resize_image[0] if adv else resize_image
        gt_image = gt_image[0] if adv else gt_image
        # print(res_in.shape)
        # raise NotImplementedError
        outputs = model(res_in)
        # print(outputs.shape)
        # print(gt_image.shape)      
        # print(resize_image.shape)
        # print(gt_image.shape)
        # print(outputs.shape)

        # calculate the loss
        loss = criterion(outputs,gt_image)
        # zero gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # do optimization step
        optimizer.step()
      scheduler.step()
        # log visualizations if necessary.
      if self.verbose:
        running_loss += loss.item()
        running_psnr += utils.psnr(outputs, gt_image)      
        if t % self.show_interval == 0: 
          val_psnr,psnr2 = self.evaluate(image_path, model)  
          visualizer.update(running_loss / self.show_interval, 
                            running_psnr / self.show_interval,
                            val_psnr)
          running_loss, running_psnr = 0, 0
          print(f'VAL PSNR before back_projection: {psnr2}')

    return model
    # END SOLUTION


  @torch.no_grad()
  def back_projection(self, sr,real,scale_factor):
    sr_lr = utils.rr_resize(sr, 1/scale_factor)
    diff = utils.rr_resize(real-sr_lr,scale_factor)
    # print(real.shape)
    # print(sr_lr[0].shape)
    # print(diff.shape)
    # raise NotImplementedError
    return torch.clip(sr+ diff,0,1)


  @torch.no_grad()
  def evaluate(self, image_path, model):
    """ Evaluates a ZSSR model on a specific image.
    Args:
      image_path (pathlib.Path): Path to the image to be trained.
      model (torch.nn.Module): A trained model.
    Returns:
      image_psnr (float): The PSNR value between the original image and the 
      image created with the model.
    """
    # BEGIN SOLUTION
    # define dataset
   
    adv = True
    trans =  flips() if adv else inference_trans()
    dataset = BasicZSSRDataset(image_path, self.scale_factor,trans)
    # fetch an instance
    inputs =  dataset[0]
    # parse it to gt and input
    gt_image, resize_image = inputs['SR'][None, ...].to(device = self.device), inputs['LR'][None, ...].to(device = self.device)
    # resize with bicubic interpolation

    res_in = resize_image[0] if adv else resize_image
    gt_image = gt_image[0][0] if adv else gt_image


    # print(res_in.shape)
    # plt.imshow(res_in[0].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[1].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[2].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[3].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[4].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[5].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[6].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[7].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res_in[8].cpu().permute(1,2,0)) , plt.show()

    res = model(res_in)
    # raise NotImplementedError   
    # plt.imshow(res[0].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[1].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[2].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[3].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[4].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[5].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[6].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[7].cpu().permute(1,2,0)) , plt.show()  
    # plt.imshow(res[8].cpu().permute(1,2,0)) , plt.show()  
    # raise NotImplementedError
    res = flip_back(res)
    # plt.imshow(res[0].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[1].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[2].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[3].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[4].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[5].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[6].cpu().permute(1,2,0)) , plt.show()
    # plt.imshow(res[7].cpu().permute(1,2,0)) , plt.show()  
    # plt.imshow(res[8].cpu().permute(1,2,0)) , plt.show()  
    # raise NotImplementedError
    res = torch.mean(res,dim=0)
    # plt.imshow(res.cpu().permute(1,2,0)) , plt.show()  
    # raise NotImplementedError
    # print(res.shape)
    # print(res_in[0].shape)
    # raise NotImplementedError
    res_1 = res
    for _ in range(self.back_iters):
      res = self.back_projection(res,res_in[0],self.scale_factor)
    # compute baseline psnr
    psnr = utils.psnr(res, gt_image)
    psnr2 = utils.psnr(res_1, gt_image)
    return psnr.item(),psnr2.item()
    # END SOLUTION

  def run(self):
    """ Run an entire experiment.
    Returns:
      run_df (pd.DataFrame): dataframe of all PSNR values.
    """
    run_list = []
    avg_psnr, avg_baseline_psnr = 0, 0

    # train and evaluate every image
    for train_path, gt_path in tqdm(zip(self.train_paths, self.gt_paths)):
      # make sure train and gt images match
      assert train_path.name == gt_path.name
      print(train_path.name)
      # compute baseline PSNR for comparison
      baseline_psnr = self.baseline(gt_path)
      avg_baseline_psnr += baseline_psnr

      model = self.train(train_path)
      psnr,_ = self.evaluate(gt_path, model)
      run_list.append({'image_path': str(gt_path.name), 
                       'psnr': psnr,
                       'baseline': baseline_psnr
                      })
      avg_psnr += psnr

    # compute average psnr
    avg_psnr = avg_psnr / len(self.gt_paths)
    avg_baseline_psnr = avg_baseline_psnr / len(self.gt_paths)
    run_list.append({'image_path': 'TOTAL_AVG', 
                     'psnr': avg_psnr, 
                     'baseline': avg_baseline_psnr})
    
    # create results file
    run_df = pd.DataFrame(run_list)
    run_df.to_csv(utils.ROOT / f'{self.dataset_root.name}_psnr.csv')
    return run_df


