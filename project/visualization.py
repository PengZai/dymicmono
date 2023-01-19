import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import os
from torchvision.utils import make_grid
import torchvision.transforms
import torchvision.transforms.functional as F
import torch
from PIL import Image
import numpy as np
import PIL.Image as pil
from torch import distributed as dist

class VisualTools(object):
  
  def __init__(self, config):
    
    self.config = config
    self.epoch = 0
    self.pil_to_tensor = torchvision.transforms.ToTensor()

    
  def set_epoch(self, epoch):
    self.epoch = epoch
  
  def vis(self, input_dict, output_dict, saved_path):
    """
    """
    self.draw_batch(input_dict, output_dict, saved_path) 
  
  def depth_to_rgb(self, depths):
    
    depths = depths.detach().squeeze().cpu().numpy()
    # vmax = np.percentile(depths, 95)
    
    depth_list = []
    
    for b in range(len(depths)):
      depth = depths[b]
      colormapped_im = self.gray2rgb(depth)
      # colormapped_im.save('test.jpg')
      depth_list.append(self.pil_to_tensor(colormapped_im).unsqueeze(dim=0))

      
    depths = torch.cat(depth_list)
    
    return depths
  
  def gray2rgb(self, image):
    
    """
     input: image shape must be (H, W)
     output: PIL Image in uint8 range from 0 to 255
     Convert gray image to rgb with magma cmap
    """
    
    if torch.is_tensor(image):
      image = image.numpy()
    
    normalizer = mpl.colors.Normalize(vmin=image.min(), vmax=image.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(image)[:, :, :3])
    colormapped_im = pil.fromarray((colormapped_im*255).astype(np.uint8))

    return colormapped_im
  
    
  def get_concat_h(self, im1, im2):
      dst = Image.new('RGB', (im1.width + im2.width, im1.height))
      dst.paste(im1, (0, 0))
      dst.paste(im2, (im1.width, 0))
      return dst
    
  def get_concat_v(self, im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
  
  
  def draw_one(self, image, depth, saved_path):
    """

    """
    image = F.to_pil_image(image.squeeze(dim=0).cpu())
    depth = depth.detach().squeeze().cpu().numpy()
    
    depth_inv = self.gray2rgb(depth_inv)
    # depth_inv.save('test2.jpg')
    results = self.get_concat_h(image, depth_inv)
    
    depth_inv.save(saved_path+'_depth.jpg')
    results.save(saved_path+'_results.jpg')
  
  def draw_batch(self, input_dict, output_dict, saved_path):
    """
      draw figure row by row
      image1 image2 ... imageN
      depth1 depth2 ... depthN
    """
    
    def merge_grid(grid, images):
      images_grid = F.to_pil_image(make_grid(images, nrow=self.config.cfg.visualization.batch_size))
      images_grid = images_grid.convert('RGB')
      grid = self.get_concat_v(grid, images_grid)
      return grid
    
    
    images = input_dict[('color', 0)].cpu()
    images = F.to_pil_image(make_grid(images, nrow=self.config.cfg.visualization.batch_size))

    depth_inv = self.depth_to_rgb(output_dict[('depth_inv', 0)])
    
    depth_inv = F.to_pil_image(make_grid(depth_inv, nrow=self.config.cfg.visualization.batch_size))
    results = self.get_concat_v(images, depth_inv)

    results.save(saved_path+'_example.jpg')
  
 

  
      
    
    
    
    
    
    
  
  
    
    