import torch
from torch import distributed as dist
from networks.cnn import CnnDepthEstimator
from torch.nn.parallel import DistributedDataParallel as DDP


def getModels(config):
    """
        the model output is difference between distributed and non-distributed computation.
        there are some problems if you want to visualize your result during distributed computation
    """
    model_dict = {
                          'cnn':CnnDepthEstimator,
                          'transformer':""
                        }
    models = model_dict[config.args.model_structure](config)
    models.cuda()
    models=torch.nn.SyncBatchNorm.convert_sync_batchnorm(models)
    models=DDP(module=models,device_ids=[config.args.local_rank], output_device=config.args.local_rank)
    
    
    return models
  
def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M
  
def get_translation_matrix(translation_vector):
  """Convert a translation vector into a 4x4 transformation matrix
  """
  T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

  t = translation_vector.contiguous().view(-1, 3, 1)

  T[:, 0, 0] = 1
  T[:, 1, 1] = 1
  T[:, 2, 2] = 1
  T[:, 3, 3] = 1
  T[:, :3, 3, None] = t

  return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def depth_to_real(depth, min_depth, max_depth):
    """
    depth: depth has not real world scale
    sdepth: scale depth has real world scale
    sdepth_inv: scale depth invert for visualization
    
    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    max_depth_invert = 1 / max_depth
    min_depth_invert = 1 / min_depth
    sdepth_inv = min_depth_invert + (max_depth_invert - min_depth_invert) * depth
    sdepth = 1 / sdepth_inv
    return sdepth_inv, sdepth