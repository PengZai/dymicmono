import os
import sys
# add your workspaceFolder to python path
sys.path.append(os.getcwd())
from utils import set_random_seed
from trainer import Trainer
import shutil
from distribution import DistTools
import argparse
import logging
from mmcv import Config
from torch.utils.tensorboard import SummaryWriter


class TrainConfig(object):
  def __init__(self):
    
    self.preparser = argparse.ArgumentParser(description="dymicmomo train")
    
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
    
    self.preparser.add_argument("--work_dirs",
                             type=str,
                             default="work_dirs",
                             help="work directory which you save model checkpoint, logs, config",
                             )
    
    self.preparser.add_argument("--model_checkpoint_name",
                             type=str,
                             default="latest",
                             help="it is model checkpoint name saved in work_dirs, \
                               there are option [latest, special name]",
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

    
    # self.preparser.add_argument("--local_rank",
    #                             default=0,
    #                             type=int,
    #                             help="gpu id of processing thread"
    #                             )
    
   
    self.preparser.add_argument("--train_from_scratch",
                                action='store_true',
                                default=False,
                                help = ""
                                )

    self.preparser.add_argument('--validate_dataset_list',  # either of this switches
                                 nargs='+',       # one or more parameters to this switch
                                 type=str,        # /parameters/ are ints
                                 default=['kitti'],      # since we're not specifying required.
                                 help="you can choice kitti cityscapes")          
        
    
    
    self.args = self.preparser.parse_args()
    self.args.local_rank = int(os.environ["LOCAL_RANK"])

    self.cfg = Config.fromfile(os.path.join('configs', self.args.config_options) + '.py')
    
    self.create_work_dirs()
    
    set_random_seed(self.cfg.base.random_seed)
    
  def create_work_dirs(self):
    
    # project 
    if not os.path.exists(os.path.join(self.args.work_dirs, 
                                       self.args.config_options)):
      os.makedirs(os.path.join(self.args.work_dirs, 
                               self.args.config_options))
    
    # config
    if not os.path.exists(os.path.join(self.args.work_dirs, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.config)):
      os.makedirs(os.path.join(self.args.work_dirs, 
                               self.args.config_options, 
                               self.cfg.base.work_dirs_structure.config))
    
    # copy config to work_dirs/${project_name}/config/{config_name}.py
    shutil.copyfile(os.path.join('configs', self.args.config_options + '.py'), 
                    os.path.join(self.args.work_dirs, 
                                 self.args.config_options, 
                                 self.cfg.base.work_dirs_structure.config, 
                                 self.args.config_options+'.py'))

    # logs
    self.args.log_dir = os.path.join(self.args.work_dirs, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.logs,
                                       "train")
    if not os.path.exists(self.args.log_dir):
      os.makedirs(self.args.log_dir)
      
    # tensorboard
    self.args.tensorboard_dir = os.path.join(self.args.work_dirs, 
                self.args.config_options, 
                self.cfg.base.work_dirs_structure.tensorboard,
                "train")
    if not os.path.exists(self.args.tensorboard_dir):
      os.makedirs(self.args.tensorboard_dir)
    
    self.init_logs()
    
    # visualization
    self.args.visualization_dir = os.path.join(self.args.work_dirs, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.vis,
                                       "train")
    if not os.path.exists(self.args.visualization_dir):
      os.makedirs(self.args.visualization_dir)

    
    # checkpoints
    self.args.checkpoints_dir = os.path.join(self.args.work_dirs, 
                                       self.args.config_options, 
                                       self.cfg.base.work_dirs_structure.checkpoints)
    if not os.path.exists(self.args.checkpoints_dir):
      os.makedirs(self.args.checkpoints_dir)

  
  def init_logs(self,):
    """
      Notice: you must init log system before setup distributed training 
    """
    logging.basicConfig(
      filename=os.path.join(self.args.log_dir,
                            'logging_log.log'),
      level=logging.DEBUG,
    )
    
    
    
    DistTools.setStaticVarible( self.args,
                                master_gpu_id=self.args.master_gpu_id)
    


if __name__ == '__main__':
  

  config = TrainConfig()
  
  trainer = Trainer(config)
  trainer.train()
  
  DistTools.dist_print('Done')
  DistTools.cleanup()
