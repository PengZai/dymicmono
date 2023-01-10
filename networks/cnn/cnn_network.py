import torch
import torch.nn as nn
from .depth_decoder import DepthDecoder
from .ego_transfer_decoder import EgoTransferDecoder
from .encoder import Encoder
from networks import GeometryTransfer 
from losses import ReprojectionLoss

class CnnDepthEstimator(nn.Module):
  """
  """
  def __init__(self, config):
    super(CnnDepthEstimator, self).__init__()
    
    self.config = config
      
    self.encoder = Encoder(config)

    self.depth_decoder = DepthDecoder(config)
    
    self.ego_transfer_decoder = EgoTransferDecoder(config)
     
    self.geometry_transfer = GeometryTransfer(config)
    
    self.reprojection_loss = ReprojectionLoss(config)
    
    
  
  def forward(self, x):
    
    features = self.encoder(x)
    depth = self.depth_decoder(features)
    return depth
  
  
  def loss(self, input_dict, output_dict):
    
    loss_dict = {}
    reprojection_loss = []
    
    for i, idx in enumerate(self.config.cfg.base.neighbor_frame_idxs[1:]):
      
      pred = output_dict[('wrap_color', idx)]
      target = input_dict['color', idx]
      reprojection_loss.append(self.reprojection_loss(pred, target))
      
    reprojection_loss = torch.cat(reprojection_loss, dim=1)
    
    # reference https://github.com/nianticlabs/monodepth2
    # it is most posibile that there no occlusion at this pair when reprojection loss is small in there
    # that is reason why there use torch.min instead of torch.mean
    reprojection_loss, idxs = torch.min(reprojection_loss, dim=1)
    loss_dict['reprojection_loss'] = reprojection_loss.mean()
    
    
    loss_dict['loss'] = 0
    for value in loss_dict.values():
      loss_dict['loss']+=value
    
    return loss_dict
    

