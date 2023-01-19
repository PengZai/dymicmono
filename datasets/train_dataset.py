from .kitti import KittiRawDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from .basic_dataset import BasicDataset


class TrainDataset(BasicDataset):

  def __init__(self, config):
    super(TrainDataset, self).__init__(config)
  
    self.config = config
    
    self.dataset = KittiRawDataset(config, 'train')
    self.sampler = DistributedSampler(self.dataset)
    self.dataloader = DataLoader(self.dataset, 
                                   batch_size = self.config.cfg.training.batch_size, 
                                   shuffle = False,
                                   num_workers=self.config.cfg.training.num_worker,
                                   drop_last=True,
                                   pin_memory=True,
                                   sampler=self.sampler)
    # visualize training data
    # self.vis_loader = DataLoader(self.dataset, 
    #                                batch_size = self.config.cfg.training.batch_size, 
    #                                shuffle = False,
    #                                num_workers=self.config.cfg.training.num_worker,
    #                                drop_last=True,
    #                                pin_memory=True,
    #                                )
    # self.vis_example = next(iter(self.vis_loader))