import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
      
class UpSample(nn.Module):
  
  def __init__(self,):
    super(UpSample, self).__init__()
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)
  
  def forward(self, x):
      """Upsample input tensor by a factor of 2
      """
      
      return self.pixel_shuffle(x)


class DepthDecoder(nn.Module):
  """
  """
  def __init__(self, config):
    super(DepthDecoder, self).__init__()

    self.config = config
    self.use_skip = True
    
    self.num_in_channels = [64, 256, 512, 1024, 2048]
    self.num_out_channels= [128, 256, 1024, 2048, 4096]
    self.N_gain = len(self.num_in_channels)
    
    self.upconvs = []
    for i in range(self.N_gain):
      num_in_channel = self.num_in_channels[i]
      num_out_channel = self.num_out_channels[i]
      
      self.upconvs.append(ConvBlock(num_in_channel, num_out_channel))
      
    self.upconvs = nn.ModuleList(self.upconvs)
    
    self.dispconv = ConvBlock(32, 1)
    
    self.upsample = UpSample()
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, input_features):
    
    x = input_features[-1]
   
    for i in range(self.N_gain-1, -1, -1):
      x = self.upconvs[i](x)
      x = self.upsample(x)
      if i > 0:
        x += input_features[i-1]
    
    
    x = self.dispconv(x)
    
    disp = self.sigmoid(x)
    
    return disp


