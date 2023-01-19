import torch
import torch.nn as nn
from datasets import ValidDataset
from visualization import VisualTools
from tqdm import tqdm
import os
from networks import getModels, depth_to_real
from distribution import DistTools
from torch import distributed as dist
from torchvision.utils import make_grid
from utils import normalize_image


class Validator(object):
  
  def __init__(self, config, models=None):
    
    self.config = config
    self.epoch = 0
    self.iter_step = 0
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
    
    if os.path.exists(self.config.args.checkpoints_path):
      checkpoint_dict = torch.load(self.config.args.checkpoints_path, map_location='cuda:{0}'.format(dist.get_rank()))
      self.models.load_state_dict(checkpoint_dict['checkpoint_state'], strict=False)
      self.set_epoch(checkpoint_dict['epoch'] if 'iter_step' in checkpoint_dict else 0)
      self.set_iter_step(checkpoint_dict['iter_step'] if 'iter_step' in checkpoint_dict else 0)
      DistTools.dist_print('loaded {0}'.format(self.config.args.checkpoints_path))
      DistTools.dist_print('checkpoint metric_result_dict:{0}'.format(checkpoint_dict['metric_result_dict']))
      
    else:
      raise '{}\t there are no any checkpoint'.format(self.config.args.checkpoints_path)
  
  
  
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
  
  def set_iter_step(self, iter_step):  
    self.iter_step = iter_step
      
  def validate(self, epoch = 0):
    
    self.set_epoch(epoch)
    self.set_eval()
    metric_result = {}
    
    with torch.no_grad():
      if 'kitti' in self.config.args.validate_dataset_list:
        metric_result['kitti'] = self.run_epoch(self.vaild_dataset.kitti.loader)
        self.log(metric_result['kitti'], 'kitti')
    
    
    return metric_result
      
  def run_epoch(self, dataloader):

      metric_result = {}
      N = len(dataloader)
      
      if DistTools.isMasterGPU():
        pbar = tqdm(dataloader)
        pbar.set_description(dataloader.dataset.name)
        
      for batch_step, input_dict in enumerate(dataloader):
        
        
        output_dict = self.process_batch(input_dict, dataloader.dataset)
        
        # self.visual_tools.draw_one(input_dict[('color', 0)][0].unsqueeze(dim=0), output_dict[('depth', 0)][0].unsqueeze(dim=0), 'test3.jpg')

        result_dict = dataloader.dataset.compute_depth_loss(output_dict[('depth', 0)], input_dict['depth_gt'])
        self.update_result(metric_result, result_dict)
        
        self.vis(batch_step, input_dict, output_dict, dataloader.dataset)
        
        if batch_step > 2:
          break
        if DistTools.isMasterGPU():
          pbar.update()
        
      self.synchronize_result(metric_result, N)

      return metric_result
    
  
  def process_batch(self, input_dict, dataset):
    
    output_dict = {}
    
    # move input data to cuda
    for i, (key, batch_inp) in enumerate(input_dict.items()):
      input_dict[key] = input_dict[key].cuda()
  
    
    output_dict[('depth', 0)] = self.models.module(input_dict[('color', 0)])
    depth_inv, sdepth = depth_to_real(output_dict[('depth', 0)], 
                                         dataset.dataset_config.min_depth, 
                                         dataset.dataset_config.max_depth)
    output_dict[('depth_inv', 0)] = depth_inv
    output_dict[('sdepth', 0)] = sdepth
    
    
    return output_dict
    

  
  def vis(self, batch_step, input_dict, output_dict, dataset):
    """
    """
    if batch_step == 0 or \
      hasattr(self.config.args, "visual_all") and self.config.args.visual_all == True:
      saved_dir = os.path.join(self.config.args.visualization_dir,
                              'e{0}'.format(self.epoch),
                              "valid")
      if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        
      saved_path = os.path.join(saved_dir, dataset.name+'_r{0}_s{1}'.format(dist.get_rank(), batch_step))
      self.visual_tools.vis(input_dict, output_dict, saved_path)
      # tensorboard
      if DistTools.isMasterGPU():
        DistTools.tb_writer.add_image('Validate-{0}/color'.format(dataset.name), make_grid(input_dict[('color', 0)]), self.epoch)
        DistTools.tb_writer.add_image('Validate-{0}/depth_inv'.format(dataset.name), make_grid(normalize_image(output_dict[('depth_inv', 0)])), self.epoch)

  
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
      
      output_str = base_str+'\t'
      if DistTools.isMasterGPU():
        for key, value in metric_result.items():
          output_str+="{0}:{1:.4}\t".format(key, value)
          DistTools.tb_writer.add_scalar('Validate-{0}/{1}'.format(base_str, key), value, self.epoch)
          
        DistTools.dist_print(output_str)
        DistTools.dist_log(output_str)
        
          
