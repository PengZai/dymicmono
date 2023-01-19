from torch.utils.data import Dataset 
from ..basic_dataset import BasicDataset
import os
from .utils import readlines
import numpy as np
import torch
from torchvision import transforms

class KittiDataset(BasicDataset):
  
  def __init__(self, config, train_or_vaild):
    super(KittiDataset, self).__init__(config)
    
    self.config = config
    self.dataset_config = self.config.cfg.kitti
    self.name = 'kitti'
    self.K = self.config.cfg.kitti.camera.intrinsics
    
    filenames = None
    if train_or_vaild == 'train':
      filenames = readlines(os.path.join(os.path.dirname(__file__), "kitti_splits", self.config.cfg.kitti.split, "train_files.txt"))
      
    elif train_or_vaild == 'valid':
      filenames = readlines(os.path.join(os.path.dirname(__file__), "kitti_splits", self.config.cfg.kitti.split, "val_files.txt"))
    
    self.samples = filenames
    self.train_or_vaild = train_or_vaild

    self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    
  
  
  

    
      
