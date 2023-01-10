import os
import sys
# add your workspaceFolder to python path
sys.path.append(os.getcwd())

from utils import set_random_seed
from validator import Validator
import torch
import numpy as np
import argparse
from mmcv import Config
from distribution import Distribution

class ValidatorConfig(object):
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
                             default="modelzoo",
                             help="modelzoo saved directory",
                             )
    
    self.preparser.add_argument("--model_checkpoint_name",
                             type=str,
                             default="e9.pt",
                             help="it is model checkpoint name saved in checkpoint_dir \
                               you must specify one",
                             )
    
    self.preparser.add_argument("--model_structure",
                                type=str,
                                default="cnn",
                                help=" this option is about model structure \
                                  you can select cnn, transformer \
                                ",
                                )
    
    self.preparser.add_argument("--local_rank",
                                default=0,
                                type=int,
                                help="master gpu id"
                                )
    

    self.preparser.add_argument('--validate_dataset_list',  # either of this switches
                                 nargs='+',       # one or more parameters to this switch
                                 type=str,        # /parameters/ are ints
                                 default=['kitti'],      # since we're not specifying required.
                                 help="you can choice kitti cityscapes")          
        
    
    
    self.args = self.preparser.parse_args()

    self.cfg = Config.fromfile(os.path.join('configs', self.args.config_options) + '.py')
    
    
    set_random_seed(self.cfg.base.random_seed)
    
  
    
    
if __name__ == '__main__':
  

  
  config = ValidatorConfig()
  

  validator = Validator(config)
  validator.load_models()
  
  validator.validate()

  