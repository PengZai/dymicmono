import torch
import torch.nn as nn
from .ssim_loss import SSIM

class ReprojectionLoss(nn.Module):
  def __init__(self, config):
    super(ReprojectionLoss, self).__init__()
    self.config = config

    self.ssim = SSIM()
  
  def forward(self, pred, groundtruth):
    
    """
      Computes reprojection loss between a batch of predicted and target images
    """
    losses = {}
    reprojection_loss = 0
    if 'l1' in self.config.cfg.training.loss_list:
      abs_diff = torch.abs(groundtruth - pred)
      l1_loss = abs_diff.mean(1, True)
      losses['l1'] = l1_loss
    
    if 'ssim' in self.config.cfg.training.loss_list:
      ssim_loss = self.ssim(pred, groundtruth).mean(1, True)
      losses['ssim'] = ssim_loss
    
    for key, loss_value in losses.items():
      reprojection_loss += self.config.cfg.training.loss_list[key]['weight']*loss_value

    return reprojection_loss