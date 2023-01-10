"""
  this is a config template. You can copy and modify
"""
import os
import numpy as np

_base_ = [
  'base_config.py'
]


# root path of all dataset
dataset_root = '/mnt/hdd2/pengzai/dataset'

# dataset config
kitti = dict(
  root= os.path.join(dataset_root, 'kitti'),
  # kitti raw dataset root
  raw_root = os.path.join(dataset_root, 'kitti', 'raw', 'kitti_data'),
  
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
  
)

cityscapes = dict(
  root=os.path.join(dataset_root, 'cityscapes')
)
megadepth = dict(
  root=os.path.join(dataset_root, 'megadepth')
)
nyuv2 = dict(
  root=os.path.join(dataset_root, 'nyuv2')
)


# training config
training = dict(
  # learning rate of optimizer
  learning_rate = 1e-5,
  # step size of the scheduler
  scheduler_step_size = 15,
  # max number of epoch
  num_epochs = 200,
  
  batch_size = 4,
  # the numbers of thread
  num_worker = 0,
  
  # loss list you can choice, the sum of their weight must be 1
  loss_list = dict(
    l1 = dict(weight = 0.15), 
    ssim = dict(weight = 0.85),
  ),
)

visualization = dict(
  batch_size = 8,
  num_worker = 0,
)

# cnn structure
cnn = dict(
  num_layers = 50, 
  isPretrain = True
)