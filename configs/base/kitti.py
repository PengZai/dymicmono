import numpy as np
import os

# dataset config
kitti = dict(
  # kitti raw dataset root
  root = os.path.join('/mnt/hdd2/pengzai/dataset', 'kitti', 'raw', 'kitti_data'),
  
  # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
  # To normalize you need to scale the first row by 1 / image_width and the second row
  # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
  # If your principal point is far from the center you might need to disable the horizontal
  # flip augmentation.
  camera = dict(
    intrinsics = np.array(
      [[0.58, 0, 0.5, 0],
       [0, 1.92, 0.5, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1]], dtype=np.float32),
  ),

  validation = dict(
    batch_size = 16,
    num_worker = 0
  ),
  
  # which training split to use, ["eigen_zhou", "eigen_full", "odom", "benchmark"]
  split = 'eigen_zhou',
        
  # you can select .jpg or .png, we set .jpg as default
  img_ext = '.jpg',
    
  full_res_shape = (375, 1242),
  
  min_depth = 1e-3,
  max_depth = 80,
  
)
