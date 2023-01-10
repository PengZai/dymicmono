import numpy as np
import torch
import random
from torch import distributed as dist


def set_random_seed(random_seed):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    

         
    
    