import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import logging


class DistTools(object):
  
  master_gpu_id = None
  # tensorboard writer
  tb_writer = None
  
  
  def __init__(self,):
    pass
    
  
  @staticmethod
  def cleanup():
    dist.destroy_process_group()
  
  @staticmethod
  def setStaticVarible(args, master_gpu_id):
    DistTools.master_gpu_id = master_gpu_id
    
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    
    if DistTools.isMasterGPU():
      DistTools.tb_writer = SummaryWriter(log_dir = args.tensorboard_dir)
    
  
  @staticmethod
  def dist_print(str):
      if DistTools.isDistAvailable() and DistTools.isMasterGPU():
          print(str)
          
  @staticmethod
  def dist_log(str):
      if DistTools.isDistAvailable() and DistTools.isMasterGPU():
          logging.debug(str)
          
  @staticmethod
  def synchronize_op(data_dict):
    """
      data_dict must be a dictionary
    """
    for metric in data_dict:
      dist.all_reduce(data_dict[metric], op=dist.ReduceOp.SUM, async_op=False)
      data_dict[metric] = data_dict[metric].item()/dist.get_world_size()
      
    return data_dict
          
  @staticmethod        
  def isMasterGPU():
      # we set gpu device 0 as log device.
      return dist.get_rank() == DistTools.master_gpu_id
    
  @staticmethod
  def isDistAvailable():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True
    
  
    
    
    
