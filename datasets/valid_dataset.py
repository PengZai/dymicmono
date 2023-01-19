from .kitti import KittiRawDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from .basic_dataset import BasicDataset


class ValidDataset(BasicDataset):
  def __init__(self, config):
    super(ValidDataset, self).__init__(config)

    self.config = config
    
    # kitti
    self.kitti = ValidLoader(config, 
                             KittiRawDataset(config, 'valid'),
                             self.config.cfg.kitti.validation)
    
    
    
class ValidLoader(object):
  
  def __init__(self, config, dataset, validation_config):
    self.config = config
    self.dataset = dataset
    self.sampler = DistributedSampler(self.dataset,
                                      shuffle=False,
                                      )
    self.loader = DataLoader(self.dataset, 
                                   validation_config.batch_size, 
                                   num_workers=validation_config.num_worker,
                                   drop_last=False,
                                   pin_memory=True,
                                   shuffle=False,
                                   sampler=self.sampler,
                                   )
    
    
    
    
    
    