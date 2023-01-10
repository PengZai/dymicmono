from .kitti import KittiRawDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os


class ValidDataset(object):
  def __init__(self, config):
    
    self.config = config
    
    # kitti
    self.kitti = ValidLoader(config, 
                             KittiRawDataset(config, 'valid'),
                             self.config.cfg.kitti.validation)
    
    
    
class ValidLoader(object):
  
  def __init__(self, config, dataset, validation_config):
    self.config = config
    self.dataset = dataset
    self.sampler = DistributedSampler(self.dataset)
    self.loader = DataLoader(self.dataset, 
                                   validation_config.batch_size, 
                                   num_workers=validation_config.num_worker,
                                   drop_last=False,
                                   pin_memory=True,
                                   shuffle=False,
                                   sampler=self.sampler,
                                   )
    self.vis_loader = DataLoader(self.dataset, 
                                   self.config.cfg.visualization.batch_size, 
                                   num_workers=self.config.cfg.visualization.num_worker,
                                   drop_last=False,
                                   pin_memory=True,
                                   shuffle=False,
                                   )
    self.vis_example = next(iter(self.vis_loader))
    
    
    
    