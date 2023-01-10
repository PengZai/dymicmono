import torch
from torch import distributed as dist
from networks.cnn import CnnDepthEstimator
from torch.nn.parallel import DistributedDataParallel as DDP


def getModels(config):

    model_dict = {
                          'cnn':CnnDepthEstimator,
                          'transformer':""
                        }
    models = model_dict[config.args.model_structure](config)
    models.cuda()
    models=torch.nn.SyncBatchNorm.convert_sync_batchnorm(models)
    models=DDP(module=models,device_ids=[config.args.local_rank], output_device=config.args.local_rank)
    
    
    return models