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
from distribution import DistTools
from torch.utils.tensorboard import SummaryWriter
import logging


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
                             default="work_dirs",
                             help="modelzoo saved directory",
                             )
    
    self.preparser.add_argument("--model_checkpoint_name",
                             type=str,
                             default="e149.pt",
                             help="it is model checkpoint name saved in checkpoint_dir \
                               you must specify one",
                             )
    
    self.preparser.add_argument("--model_structure",
                                type=str,
                                default="cnn",
                                choices=[
                                      "cnn",
                                      "transformer"
                                ],
                                help=" this option is about model structure \
                                  you can select cnn, transformer \
                                ",
                                )
    
    self.preparser.add_argument("--master_gpu_id",
                                default=0,
                                type=int,
                                help="gpu id of processing thread"
                                )
    
    self.preparser.add_argument("--visual_all",
                                action='store_true',
                                default=False,
                                help = "visualize all validation data"
                                )

    self.preparser.add_argument('--validate_dataset_list',  # either of this switches
                                 nargs='+',       # one or more parameters to this switch
                                 type=str,        # /parameters/ are ints
                                 default=['kitti'],      # since we're not specifying required.
                                 help="you can choice kitti cityscapes")          
        
    
    
    self.args = self.preparser.parse_args()
    self.args.local_rank = int(os.environ["LOCAL_RANK"])

    self.cfg = Config.fromfile(os.path.join(self.args.modelzoo_dir,
                                            self.args.config_options,
                                            'config',
                                            self.args.config_options) + '.py')
    
    self.create_work_dirs()
    set_random_seed(self.cfg.base.random_seed)
    
  
  def create_work_dirs(self):
    
    
    # logs
    self.args.log_dir = os.path.join(self.args.modelzoo_dir, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.logs,
                                       "validate")
    if not os.path.exists(self.args.log_dir):
      os.makedirs(self.args.log_dir)
      
    # tensorboard
    self.args.tensorboard_dir = os.path.join(self.args.modelzoo_dir, 
                self.args.config_options, 
                self.cfg.base.work_dirs_structure.tensorboard,
                "validate")
    if not os.path.exists(self.args.tensorboard_dir):
      os.makedirs(self.args.tensorboard_dir)
    
    self.init_logs()
    
    # visualization
    self.args.visualization_dir = os.path.join(self.args.modelzoo_dir, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.vis,
                                       "validate")
    if not os.path.exists(self.args.visualization_dir):
      os.makedirs(self.args.visualization_dir)
      
    # checkpoints
    self.args.checkpoints_path = os.path.join(self.args.modelzoo_dir, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.checkpoints,
                                       self.args.model_checkpoint_name)


  
  def init_logs(self,):
    """
      Notice: you must init log system before setup distributed training 
    """
    logging.basicConfig(
      filename=os.path.join(self.args.log_dir,
                            'logging_log.log'),
      level=logging.DEBUG,
    )
    
    
    DistTools.setStaticVarible(args=self.args,
                               master_gpu_id=self.args.master_gpu_id)

    
    
    
if __name__ == '__main__':
  
  config = ValidatorConfig()
  validator = Validator(config)
  validator.load_models()
  
  validator.validate()
  DistTools.dist_print('Done')
  DistTools.cleanup()
  

  