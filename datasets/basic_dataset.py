from torch.utils.data import Dataset 
from PIL import Image  # using pillow-simd for increased speed



class BasicDataset(Dataset):
  
  def __init__(self, config):
    self.config = config
  
  def pil_loader(self, path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
          