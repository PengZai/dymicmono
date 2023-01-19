import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GeometryTransfer(nn.Module):
  """
    transform a view to other view by transform matrix
  """
  def __init__(self, config):
    super(GeometryTransfer, self).__init__()
    
    self.config = config
    self.eps = 1e-7
    self.height = self.config.cfg.base.image_heigh
    self.width = self.config.cfg.base.image_width
    
    meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
    self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                  requires_grad=False)
    self.ones = nn.Parameter(torch.ones(self.config.cfg.training.batch_size, 1, self.height * self.width),
                                 requires_grad=False)
    
    self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
    self.pix_coords = self.pix_coords.repeat(self.config.cfg.training.batch_size, 1, 1)
    self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                    requires_grad=False)
  
  
  
  """
    
    we do pixel coordinate norm like that
          (z*u)     (u)
          (z*v) => z(v) => (u,v)
          (z*1)     (1)
          (1)       (1/z)
          
    and offset the range of coordinate to [-1, 1], for example
    
  """
  def coord_norm(self, pix_coords):
    
    # that is reason why we only need source image depth and don't need target image depth
    # if our just do transform from source image to target image
    pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :].unsqueeze(1) + self.eps)
    
    pix_coords = pix_coords.view(self.config.cfg.training.batch_size, 2, self.height, self.width)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= self.width - 1
    pix_coords[..., 1] /= self.height - 1
    pix_coords = (pix_coords - 0.5) * 2
    
    return pix_coords
  
  
  
        
  
  def forward(self, sdepth, tdepth, transform_matrix):
    """
      sdepth: depth of source image
      tdepth: depth of target image 
    """

    transform_matrix_inv = torch.linalg.pinv(transform_matrix)
    pix_coords_with_sdepth = F.pad(input = sdepth.view(self.config.cfg.training.batch_size, 1, -1) * self.pix_coords, pad=(0, 0, 0, 1), mode='constant', value=1.0)
    pix_coords_s_to_t = torch.matmul(transform_matrix, pix_coords_with_sdepth)
    pix_coords_s_to_t = self.coord_norm(pix_coords_s_to_t)
    pix_coords_with_tdepth = F.pad(input = tdepth.view(self.config.cfg.training.batch_size, 1, -1) * self.pix_coords, pad=(0, 0, 0, 1), mode='constant', value=1.0)
    pix_coords_t_to_s = torch.matmul(transform_matrix_inv, pix_coords_with_tdepth)
    pix_coords_t_to_s = self.coord_norm(pix_coords_t_to_s)
    # pix_coords_s_to_t = sdepth.view(self.config.cfg.training.batch_size, 1, -1) * torch.matmul(transform_matrix, self.pix_coords)
    # pix_coords_s_to_t = self.coord_norm(pix_coords_s_to_t)
    # pix_coords_t_to_s = tdepth.view(self.config.cfg.training.batch_size, 1, -1) * torch.matmul(transform_matrix_inv, self.pix_coords)
    # pix_coords_t_to_s = self.coord_norm(pix_coords_t_to_s)

    return pix_coords_s_to_t, pix_coords_t_to_s

