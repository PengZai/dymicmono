

_base_ = []


base = dict(
  
  # (H, W)
  image_heigh = 192,
  image_width = 640,

  # 0, -1, 1 represent the offset idx of current frame, previous frame and next frame respectively.
  neighbor_frame_idxs = [0,-1,1],

  # source_idx represents the offset idx of current frame 
  source_idx = 0,

  # all seting about random
  random_seed = 42,
  
  work_dirs_structure = dict(
    config = "config",
    checkpoints = "checkpoints",
    logs = "logs",
    vis = "vis",
  )
  
)



    