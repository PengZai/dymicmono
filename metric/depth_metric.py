import torch

def compute_depth_errors(gt, pred):
    """
      Computation of error metrics between predicted and ground truth depths
    """
    
    result_dict = {}
    
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    result_dict['a1'] = a1
    result_dict['a2'] = a2
    result_dict['a3'] = a3

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    result_dict['rmse'] = rmse

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    result_dict['rmse_log'] = rmse_log
    
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    result_dict['abs_rel'] = abs_rel
    
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    result_dict['sq_rel'] = sq_rel
    
    return result_dict