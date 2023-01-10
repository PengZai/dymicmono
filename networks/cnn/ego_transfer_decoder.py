import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EgoTransferDecoder(nn.Module):
  """
  
  """
  def __init__(self, config):
    super(EgoTransferDecoder, self).__init__()
    
    self.config = config
    
    # self.linear1 = nn.Linear()  
    self.conv = nn.Conv2d(4096, 512, 3)
    self.conv_nonlin = nn.ELU(inplace=True)
    self.maxpool = nn.AdaptiveMaxPool2d(1)
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(512, 12)

    
    
  def forward(self, feature_list):
    
    last_feature_list = [f[-1] for f in feature_list]
    features = torch.cat(last_feature_list, dim=1)
    x = self.conv(features)
    x = self.conv_nonlin(x)
    x = self.maxpool(x)
    x = self.flatten(x)
    x = self.linear(x)
    matrix = x.view(-1, 3, 4)
    matrix = F.pad(input=matrix, pad=(0, 0, 0, 1), mode='constant', value=0.0)
    matrix[:, :, -1][:, -1] = 1.0
    
    return matrix