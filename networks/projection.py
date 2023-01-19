import torch
import torch.nn as nn
import numpy as np


class GeometryProjectTransfer(nn.Module):

  def __init__(self, config):
    super(GeometryProjectTransfer, self).__init__()
    
    self.config = config
    self.eps = 1e-7
    self.batch_size = self.config.cfg.training.batch_size
    self.height = self.config.cfg.base.image_heigh
    self.width = self.config.cfg.base.image_width

    meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
    self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                  requires_grad=False)

    self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                              requires_grad=False)

    self.pix_coords = torch.unsqueeze(torch.stack(
        [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
    self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)
    self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                    requires_grad=False)
  
  def backproject(self, depth, inv_K):
    """
    Layer to transform a depth image into a point cloud
    """
    cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
    cam_points = depth.view(self.batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, self.ones], 1)

    return cam_points
    
  
  def project(self, points, K, T):
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
    pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= self.width - 1
    pix_coords[..., 1] /= self.height - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords
    
  def forward(self, depth, K, T):
    
    inv_K = torch.linalg.pinv(K)
    cam_points = self.backproject(depth, inv_K)
    pix_coords = self.project(cam_points, K, T)
    
    return pix_coords
    
  
    
    
    
        
