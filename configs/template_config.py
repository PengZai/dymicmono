"""
  this is a config template. You can copy and modify
"""
import os
import numpy as np



_base_ = [
  os.path.join(os.getcwd(), 'configs/base/base.py'),
  os.path.join(os.getcwd(), 'configs/base/kitti.py'),
  os.path.join(os.getcwd(), 'configs/base/cityscapes.py'),
  os.path.join(os.getcwd(), 'configs/base/megadepth.py'),
  os.path.join(os.getcwd(), 'configs/base/nyuv2.py'),
]


# training config
training = dict(
  # learning rate of optimizer
  learning_rate = 1e-5,
  # step size of the scheduler
  scheduler_step_size = 15,
  # max number of epoch
  num_epochs = 200,
  
  batch_size = 12,
  # the numbers of thread
  num_worker = 0,
  
  # loss list you can choice, the sum of their weight must be 1
  loss_list = dict(
    l1 = dict(weight = 0.15), 
    ssim = dict(weight = 0.85),
  ),
)


# cnn structure
cnn = dict(
  num_layers = 50, 
  isPretrain = True
)