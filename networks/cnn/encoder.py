import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
  """
  
  """
  def __init__(self, config):
    super(Encoder, self).__init__()
    
    self.config = config
    
    resnets = {
      18: models.resnet18,
      34: models.resnet34,
      50: models.resnet50,
      101: models.resnet101,
      152: models.resnet152,
      }
      
    if self.config.cfg.cnn.num_layers not in resnets:
        raise ValueError("{} is not a valid number of resnet layers".format(self.config.cfg.cnn.num_layers))

    self.encoder = resnets[self.config.cfg.cnn.num_layers](weights=self.config.cfg.cnn.isPretrain)
    
  def forward(self, input_image):
    
    features = []
    x = self.encoder.conv1(input_image)
    x = self.encoder.bn1(x)
    x = self.encoder.relu(x)
    features.append(x)
    x = self.encoder.maxpool(x)
    x = self.encoder.layer1(x)
    features.append(x)
    x = self.encoder.layer2(x)
    features.append(x)
    x = self.encoder.layer3(x)
    features.append(x)
    x = self.encoder.layer4(x)
    features.append(x)
    
    return features



