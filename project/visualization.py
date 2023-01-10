import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torch
from PIL import Image



class VisualTools(object):
  
  def __init__(self, config):
    
    self.config = config
    self.epoch = 0
    
    
  def set_epoch(self, epoch):
    self.epoch = epoch
  
  
  
  def draw_batch(self, input_dict, output_dict, saved_name):
    """
      draw figure row by row
      image1 image2 ... imageN
      depth1 depth2 ... depthN
    """
    
    def get_concat_v(im1, im2):
      dst = Image.new('RGB', (im1.width, im1.height + im2.height))
      dst.paste(im1, (0, 0))
      dst.paste(im2, (0, im1.height))
      return dst
    
    def merge_grid(grid, images):
      images_grid = F.to_pil_image(make_grid(images, nrow=self.config.cfg.visualization.batch_size))
      images_grid = images_grid.convert('RGB')
      grid = get_concat_v(grid, images_grid)
      return grid
    
    N = len(input_dict['color', 0])
    saved_dir = os.path.join(self.config.args.work_dirs, 
                                       self.config.args.config_options, 
                                       self.config.cfg.base.work_dirs_structure.vis,
                                       'e{0}'.format(self.epoch))
    if not os.path.exists(saved_dir):
      os.makedirs(saved_dir)
    
    
    images = input_dict['color', 0]
    grid = F.to_pil_image(make_grid(images, nrow=self.config.cfg.visualization.batch_size))
    
    grid = merge_grid(grid, output_dict[('depth', 0)])
    
    grid.save(os.path.join(saved_dir, saved_name+'_example.jpg'))
    
  
 

  
      
    
    
    
    
    
    
  
  
    
    