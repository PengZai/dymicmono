import torch
import torch.nn as nn
from datasets import ValidDataset
from visualization import VisualTools
from tqdm import tqdm
import os
from networks import getModels
from distribution import DistTools
from torch import distributed as dist



class Validator(object):
  
  def __init__(self, config, models=None):
    
    self.config = config
    self.epoch = 0
    self.vaild_dataset = ValidDataset(config)
    self.visual_tools = VisualTools(config)
    
    if models == None:
      self.models = getModels(config)
    else:
      self.models = models
    
  
  def load_models(self,):
    """
      please don't use this function during training
      this function just used to valid or oneshot
    """
    
    model_checkpoint_path = os.path.join(self.config.args.modelzoo_dir, self.config.args.model_checkpoint_name)
    if os.path.exists(model_checkpoint_path):
      checkpoint_dict = torch.load(model_checkpoint_path, map_location={'cuda:0':'cuda:0'})
      self.models.load_state_dict(checkpoint_dict['checkpoint_state'], strict=False)
      
      DistTools.dist_print('loaded {0}'.format(model_checkpoint_path))
      DistTools.dist_print('checkpoint metric_result_dict:{0}'.format(checkpoint_dict['metric_result_dict']))
      
    else:
      raise '{}\t there are no any checkpoint training from scratch'.format(model_checkpoint_path)
  
  
  
  def set_models(self, models):
    """
      update model during training
    """
    self.models = models
    
    
  def set_eval(self):
      """
        Convert all models to testing/evaluation mode
      """
      DistTools.dist_print('='*20 + "validation" + '='*20)
      self.models.eval()
      
  def set_epoch(self, epoch):
    self.epoch = epoch
    self.visual_tools.set_epoch(epoch)
      
      
  def validate(self, epoch = 0):
    
    self.set_epoch(epoch)
    self.set_eval()
    metric_result = {}
    
    with torch.no_grad():
      if 'kitti' in self.config.args.validate_dataset_list:
        metric_result['kitti'] = self.run_epoch(self.vaild_dataset.kitti.loader)
        self.log(metric_result['kitti'], 'kitti: ')
        self.vis(self.vaild_dataset.kitti)
    
    return metric_result
      
  def run_epoch(self, dataloader):

      metric_result = {}
      N = len(dataloader)
      
      if DistTools.isMasterGPU():
        pbar = tqdm(dataloader)
        pbar.set_description(dataloader.dataset.name)
        
      for batch_step, input_dict in enumerate(dataloader):
        
        
        output_dict = self.process_batch(input_dict)
        result_dict = dataloader.dataset.compute_depth_loss(output_dict[('depth', 0)], input_dict['depth_gt'])
        self.update_result(metric_result, result_dict)
        
        # if batch_step > 2:
        #   break
        if DistTools.isMasterGPU():
          pbar.update()
        
      self.synchronize_result(metric_result, N)

      return metric_result
    
  
  def process_batch(self, input_dict):
    
    output_dict = {}
    
    # move input data to cuda
    for i, (key, batch_inp) in enumerate(input_dict.items()):
      input_dict[key] = input_dict[key].cuda()
  
    
    output_dict[('depth', 0)] = self.models(input_dict['color', 0])
    
    return output_dict
    
  def oneshot(self, input):
    
    depth = self.models(input)
    
    return depth
  
  
  def vis(self, validloader):
    """
    Notice: we don't setup with torch.no_grad() and models.eval()
            you should be careful this.
    """
    if DistTools.isMasterGPU():
      input_dict = validloader.vis_example
      output_dict = self.process_batch(input_dict)
      self.visual_tools.draw_batch(input_dict, output_dict, validloader.dataset.name)

    
        

  
  def update_result(self, current_result, next_result):
      
      for i, (key, value) in enumerate(next_result.items()):
        if key in current_result:
          current_result[key]+=value
        else:
          current_result[key]=value
          

  def synchronize_result(self, result, N):
    """
      N is length of dataloader
    """
    # get mean of result
    for i, (key, value) in enumerate(result.items()):
      result[key] = (value/N)
        
    DistTools.synchronize_op(result)
      
        
  def log(self, metric_result, base_str=""):
      
      for key, value in metric_result.items():
        base_str+="{0}:{1:.4}\t".format(key, value)
      
      DistTools.dist_print(base_str)
      DistTools.dist_log(base_str)
        
          
