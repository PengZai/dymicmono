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
    self.conv1 = nn.Conv2d(4096, 512, 3)
    self.conv2 = nn.Conv2d(512, 128, 3)
    self.conv3 = nn.Conv2d(128, 6, 1)
    self.conv_nonlin = nn.ELU(inplace=True)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten()
    

    
    
  def forward(self, feature_list):
    
    last_feature_list = [f[-1] for f in feature_list]
    features = torch.cat(last_feature_list, dim=1)
    x = self.conv1(features)
    x = self.conv_nonlin(x)
    
    x = self.conv2(x)
    x = self.conv_nonlin(x)
    
    x = self.conv3(x)
    x = self.conv_nonlin(x)

    x = self.avgpool(x)
    x = self.flatten(x)
    matrix = 0.01*x.view(-1, 6)
    axisangle = matrix[..., :3]
    translation = matrix[..., 3:]
    
    return axisangle, translation