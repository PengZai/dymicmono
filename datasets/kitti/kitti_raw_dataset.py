import os
import torch
from .kitti_dataset import KittiDataset
from .utils import generate_depth_map
import skimage.transform
import torch.nn.functional as F
import numpy as np
from metric import compute_depth_errors

class KittiRawDataset(KittiDataset):
  
  def __init__(self, *args, **kwargs):
    super(KittiRawDataset, self).__init__(*args, **kwargs)
    

  def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.config.cfg.kitti.img_ext)
        image_path = os.path.join(
            self.config.cfg.kitti.raw_root, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
      
  def get_depth(self, folder, frame_index, side):
        calib_path = os.path.join(self.config.cfg.kitti.raw_root, folder.split("/")[0])

        velo_filename = os.path.join(
            self.config.cfg.kitti.raw_root,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.config.cfg.kitti.full_res_shape, order=0, preserve_range=True, mode='constant')

        return depth_gt
      
  def compute_depth_loss(self, depth_pred, depth_gt):
    """
      Compute depth metrics, to allow monitoring during training

      This isn't particularly accurate as it averages over the entire batch,
      so is only used to give an indication of validation performance
    """
    
    loss_dict = {}
    mask = depth_gt > 0
    
    depth_pred = depth_pred.detach()
    depth_pred = torch.clamp(F.interpolate(
            depth_pred, self.config.cfg.kitti.full_res_shape, mode="bilinear", align_corners=False), 1e-3, 80)
    
    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    for i, (key, value) in enumerate(depth_errors.items()):
        loss_dict[key] = value

    return loss_dict
  
  
  def __len__(self):
    return len(self.samples)

  # abst
  def __getitem__(self, idx):
    """
      Return a single trainining item as a dictionary
      
      folder : record data folder, there are gray and color image sequence, oxts, and velodyne points(point cloud)
      frame_idx : index of frame
      side : left or right image
    """
    
    input_dict = {}
    line = self.samples[idx].split()
    folder = line[0]
    frame_idx = int(line[1])
    side = line[2]
    
    K = self.K.copy()
    # fx is related to image width
    K[0, :] *= self.config.cfg.base.image_width
    # fy is related to image height
    K[1, :] *= self.config.cfg.base.image_heigh
    inv_K = np.linalg.pinv(K)
    input_dict["K"] = torch.from_numpy(K)
    input_dict["inv_K"] = torch.from_numpy(inv_K)

    for nidx in self.config.cfg.base.neighbor_frame_idxs:
      image = self.pil_loader(self.get_image_path(folder, frame_idx+nidx, side))
      
      input_dict[('color', nidx)] = self.preprocess_pipline(image)
    
    
    depth_gt = self.get_depth(folder, frame_idx, side)
    input_dict["depth_gt"] = np.expand_dims(depth_gt, 0)
    input_dict["depth_gt"] = torch.from_numpy(input_dict["depth_gt"].astype(np.float32))

    
    
    return input_dict
  
  
  # def collate_fn(self, input_dict):
  #   return input_dict