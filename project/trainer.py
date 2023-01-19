import torch
from datasets import TrainDataset
from networks import getModels, transformation_from_parameters
import time
import torch.nn.functional as F
from validator import Validator
from tqdm import tqdm
import numpy as np
import random
import os
import re
from torch import distributed as dist
from distribution import DistTools
from torch.utils.tensorboard import SummaryWriter
from visualization import VisualTools



class Trainer(object):
  def __init__(self, config):
    
    self.config = config

    self.train_dataset = TrainDataset(config)

    self.models = getModels(config)
    self.visual_tools = VisualTools(config)

    self.validator = Validator(config, self.models)  
      

    
    self.model_optimizer = torch.optim.Adam(params = self.models.parameters(), lr = self.config.cfg.training.learning_rate)
    self.model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.model_optimizer, self.config.cfg.training.scheduler_step_size, 0.1)
    
    self.epoch = 0
    self.iter_step = 0
    self.load_models()

  def save_models(self, metric_result_dict, model_name):
    
    if DistTools.isMasterGPU():
      torch.save({
        'epoch':self.epoch,
        'iter_step':self.iter_step,
        'checkpoint_state': self.models.state_dict(),
        'optimizer_state':self.model_optimizer.state_dict(), 
        'lr_scheduler':self.model_lr_scheduler.state_dict(),
        'metric_result_dict':metric_result_dict
      }, os.path.join(self.config.args.checkpoints_dir, 
                      model_name))
    
  def load_models(self,):
    if self.config.args.train_from_scratch == True:
      DistTools.dist_print('training from scratch')
    
    else:
      DistTools.dist_print("loading model from checkpoint")
      if self.config.args.model_checkpoint_name == 'latest':
        checkpoint_list = os.listdir(self.config.args.checkpoints_dir)
        # To find lastest checkpoint
        checkpoint_list = [pt for pt in checkpoint_list if pt.endswith('pt') ]
        checkpoint_list.sort(reverse=True, key=lambda x: int(re.findall(r'\d+', x)[0]))
        if len(checkpoint_list) > 0:
          self.config.args.model_checkpoint_name = checkpoint_list[0]
      
      model_checkpoint_path = os.path.join(self.config.args.checkpoints_dir, 
                                           self.config.args.model_checkpoint_name)
      if os.path.exists(model_checkpoint_path):
        checkpoint_dict = torch.load(model_checkpoint_path, map_location='cuda:{0}'.format(dist.get_rank()))
        self.set_epoch(checkpoint_dict['epoch'] if 'epoch' in checkpoint_dict else 0)
        self.set_iter_step(checkpoint_dict['iter_step'] if 'iter_step' in checkpoint_dict else 0)
        self.models.load_state_dict(checkpoint_dict['checkpoint_state'], strict=False)
        self.model_optimizer.load_state_dict(checkpoint_dict['optimizer_state'])
        self.model_lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        
        # move optimizer parameter to gpu
        for state in self.model_optimizer.state.values():
          for k, v in state.items():
            if isinstance(v, torch.Tensor):
              state[k] = v.cuda()
        DistTools.dist_print('loaded {0}, start from epoch:{1}'.format(model_checkpoint_path, self.epoch))
        DistTools.dist_print('checkpoint metric_result_dict:{0}'.format(checkpoint_dict['metric_result_dict']))
        
      else:
        DistTools.dist_print('{}\t there are no any checkpoint training from scratch'.format(model_checkpoint_path))
    
    # wait until all models are loaded.
    dist.barrier()  
      
  def set_epoch(self, epoch):
    
    self.epoch = epoch
  
  def set_iter_step(self, iter_step):
    self.iter_step = iter_step
  
  def set_train(self,):
    """
      Convert all models to training mode
    """
    DistTools.dist_print('='*20 + "training" + '='*20)
    self.models.train()
      
  def set_eval(self):
      """
        Convert all models to testing/evaluation mode
      """
      DistTools.dist_print('='*20 + "validation" + '='*20)
      self.models.eval()
  
  def train(self):
    
    for i, epoch in enumerate(range(self.epoch, self.config.cfg.training.num_epochs)):
      self.set_epoch(epoch)
      """
        In distributed mode, calling the data_loader.sampler.set_epoch() method 
        at the beginning of each epoch before creating the DataLoader iterator 
        is necessary to make shuffling work properly across multiple epochs. 
        Otherwise, the same ordering will be always used.
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
      """
      self.train_dataset.dataloader.sampler.set_epoch(self.epoch)
      self.run_epoch()
      metric_result = self.val()
      
      self.save_models(metric_result, 'e{0}.pt'.format(self.epoch))
      
        
      
  def val(self):

    self.validator.set_models(self.models)
    metric_result = self.validator.validate(epoch=self.epoch)
    
    return metric_result
    
  def vis(self, batch_step, input_dict, output_dict, dataset):
    """
    """
    if batch_step == 0:
      
      saved_dir = os.path.join(self.config.args.visualization_dir,
                              'e{0}'.format(self.epoch),
                              "train")
      if DistTools.isMasterGPU():
        if not os.path.exists(saved_dir):
          os.makedirs(saved_dir)
      dist.barrier()
        
      saved_path = os.path.join(saved_dir, dataset.name+'_r{0}_s{1}'.format(dist.get_rank(), batch_step))
      self.visual_tools.vis(input_dict, output_dict, saved_path)
       
      
  def run_epoch(self,):
    
    self.model_lr_scheduler.step()
    self.set_train()

    for batch_step, input_dict in enumerate(self.train_dataset.dataloader):
      
      
      before_op_time = time.time()
      output_dict, loss_dict = self.process_batch(input_dict)
      self.model_optimizer.zero_grad()
      loss_dict['loss'].backward()
      self.model_optimizer.step()
      duration = time.time() - before_op_time
      
      # batch vaild during training
      if 'depth_gt' in input_dict:
        loss_dict.update(self.train_dataset.dataset.compute_depth_loss(output_dict[('depth', 0)], input_dict['depth_gt']))
        
      if batch_step % 10 == 0:
        base_str = "iter_step:{0}\tepoch:{1}\tbatch_step:{2}/{3}\t".format(self.iter_step, self.epoch, len(self.train_dataset.dataloader), batch_step)
        self.log(loss_dict, base_str=base_str)

      # self.vis(batch_step, input_dict, output_dict, self.train_dataset.dataloader.dataset)
      
      self.set_iter_step(self.iter_step+1)
      if batch_step > 1:
        break

      
  def process_batch(self, input_dict):
    
    output_dict = {}
    loss_dict = {}
    
    # move input data to cuda
    for i, (key, batch_inp) in enumerate(input_dict.items()):
      input_dict[key] = input_dict[key].cuda()
      
    frames_color = torch.cat([input_dict[('color', idx)] for idx in self.config.cfg.base.neighbor_frame_idxs])
    frames_feature_list = self.models.module.encoder(frames_color)
    frames_depth = self.models.module.depth_decoder(frames_feature_list)
    frames_depth_list = torch.split(frames_depth, self.config.cfg.training.batch_size)
    for i, idx in enumerate(self.config.cfg.base.neighbor_frame_idxs):
      output_dict[('depth', idx)] = frames_depth_list[i]
    
    frames_feature_list = [torch.split(f, self.config.cfg.training.batch_size) for f in frames_feature_list]
    for i, k in enumerate(self.config.cfg.base.neighbor_frame_idxs):
        output_dict[('frame_feature_list', k)] = [f[i] for f in frames_feature_list]
                
    # DistTools.dist_print('depth')
    # DistTools.dist_print(frames_depth_list[0])
    
    self.generate_wrap_images(input_dict, output_dict)
    losses = self.models.module.loss(input_dict, output_dict)
    loss_dict.update(losses)
    
    
    return output_dict, loss_dict
  
  def generate_wrap_images(self, input_dict, output_dict):
    
    # source_idx represents the offset idx of current frame
    # target_idx represents the offset idx of frame to be aligend
    for i, target_idx in enumerate(self.config.cfg.base.neighbor_frame_idxs[1:]):
        
        axisangle, translation = self.models.module.ego_transfer_decoder([output_dict[('frame_feature_list', self.config.cfg.base.source_idx)], output_dict[('frame_feature_list', target_idx)]])
        # DistTools.dist_print('transform_matrix')
        # DistTools.dist_print(transform_matrix[-1])
        # pix_coords_s_to_t, pix_coords_t_to_s = self.models.module.geometry_transfer(output_dict[('depth', self.config.cfg.base.source_idx)], output_dict[('depth', target_idx)], transform_matrix)
        
        output_dict[("axisangle", target_idx)] = axisangle
        output_dict[("translation", target_idx)] = translation
        output_dict[("transfer_matrix", target_idx)] = transformation_from_parameters(axisangle.unsqueeze(dim=1), translation.unsqueeze(dim=1), invert = False)
        
        output_dict[("wrap_coords", target_idx)] = self.models.module.geometry_project_transfer(
          depth = output_dict[("depth", self.config.cfg.base.source_idx)],
          K = input_dict["K"],
          T = output_dict[("transfer_matrix", target_idx)]
        )
        
        output_dict[("wrap_color", target_idx)] = F.grid_sample(
                                                    input_dict[("color", target_idx)],
                                                    output_dict[("wrap_coords", target_idx)],
                                                    padding_mode="border")
        
  def log(self, loss_dict, base_str=""):
    
    output_str = base_str+'\t'
    if DistTools.isMasterGPU():
      for key, value in loss_dict.items():
        output_str+="{0}:{1:.4}\t".format(key, value.item())
        # tensorboard
        DistTools.tb_writer.add_scalar('Train/{0}'.format(key), value.item(), self.iter_step)
      
      
      DistTools.dist_print(output_str)
    
    
    
    
        
        

    
    
    
      
      
    
      
    