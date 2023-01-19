import os
import sys
# add your workspaceFolder to python path
sys.path.append(os.getcwd())
from networks import depth_to_real
from utils import set_random_seed
from validator import Validator
import torch
import numpy as np
import argparse
from mmcv import Config
from distribution import DistTools
from torch.utils.tensorboard import SummaryWriter
import logging
from visualization import VisualTools
from networks.cnn import CnnDepthEstimator
from datasets import VisualDataset


class VisualizeConfig(object):
  def __init__(self):
    
    self.preparser = argparse.ArgumentParser(description="dymicmomo valid")
    
    self.preparser.add_argument("--config_options",
                             type=str,
                             default="template_config",
                             help="it is your config file name without suffix .py. \
                                suffix .py will added automatically \
                                it is your project name too. \
                               ",
                             )
    
    self.preparser.add_argument("--modelzoo_dir",
                             type=str,
                             default="work_dirs",
                             help="modelzoo saved directory",
                             )
    
    self.preparser.add_argument("--model_checkpoint_name",
                             type=str,
                             default="e120.pt",
                             help="it is model checkpoint name saved in checkpoint_dir \
                               you must specify one",
                             )
    
    self.preparser.add_argument("--model_structure",
                                type=str,
                                default="cnn",
                                choices=[
                                    "cnn",
                                    "transformer",
                                    ],
                                help=" this option is about model structure \
                                  you can select cnn, transformer \
                                ",
                                )
    
    self.preparser.add_argument("--example_dir",
                             type=str,
                             default="examples",
                             help="put image sequence or video to here",
                             )
    
    
    
    self.args = self.preparser.parse_args()
    self.cfg = Config.fromfile(os.path.join(self.args.modelzoo_dir,
                                            self.args.config_options,
                                            'config',
                                            self.args.config_options) + '.py')
    
    self.create_work_dirs()
    set_random_seed(self.cfg.base.random_seed)
    
  
  def create_work_dirs(self):
      
    # checkpoints
    self.args.checkpoints_path = os.path.join(self.args.modelzoo_dir, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.checkpoints,
                                       self.args.model_checkpoint_name)

    
if __name__ == '__main__':
  
  config = VisualizeConfig()
  
  visual_tools = VisualTools(config)
  visual_dataset = VisualDataset(config)
  
  model_dict = {
                'cnn':CnnDepthEstimator,
                'transformer':""
  }
  models = model_dict[config.args.model_structure](config)
  models.cuda()
      
  if os.path.exists(config.args.checkpoints_path):
    checkpoint_dict = torch.load(config.args.checkpoints_path, map_location={'cuda:0':'cuda:0'})
    models.load_state_dict(checkpoint_dict['checkpoint_state'], strict=False)
    
    print('loaded {0}'.format(config.args.checkpoints_path))
    print('checkpoint metric_result_dict:{0}'.format(checkpoint_dict['metric_result_dict']))
      
  else:
    raise '{}\t there are no any checkpoint'.format(config.args.checkpoints_path)

  models.eval()
  with torch.no_grad():
    
    for exmaple_dir in visual_dataset.example_dirs:
      images_name = visual_dataset.read_images_name(exmaple_dir)
      prev_image = None
      
      if not os.path.exists(os.path.join(exmaple_dir, 'result')):
        os.makedirs(os.path.join(exmaple_dir, 'result'))
        
      for i, imagename in enumerate(images_name):
        image = visual_dataset.readimage(os.path.join(exmaple_dir, imagename))
        image = image.unsqueeze(dim=0).cuda()
        depth = models(image)
        depth_inv, depth = depth_to_real(depth, 
                                     config.cfg.kitti.min_depth, 
                                     config.cfg.kitti.max_depth)
        visual_tools.draw_one(image, depth_inv, os.path.join(exmaple_dir, 'result', imagename))
        prev_image = image