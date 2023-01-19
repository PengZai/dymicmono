from torch.utils.data import Dataset 
from PIL import Image  # using pillow-simd for increased speed
import torch
from torchvision import transforms


class BasicDataset(Dataset):
  
  def __init__(self, config):
    self.config = config
    
    self.torch_transforms = transforms.Compose([
      transforms.Resize((self.config.cfg.base.image_heigh, self.config.cfg.base.image_width)),
      transforms.ToTensor()] 
    )
  
  def pil_loader(self, path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
          
  
  def preprocess_pipline(self, x):
    
    x = self.torch_transforms(x)
    return x
          