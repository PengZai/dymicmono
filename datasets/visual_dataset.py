import os
import glob
import re
from .basic_dataset import BasicDataset

class VisualDataset(BasicDataset):
  
  def __init__(self, config):
    super(VisualDataset, self).__init__(config)

    self.config = config
    
    self.example_dirs = os.listdir(self.config.args.example_dir)
    self.example_dirs.sort(reverse=True, key=lambda x: int(re.findall(r'\d+', x)[0]))
    self.example_dirs = [os.path.join(self.config.args.example_dir, d) for d in self.example_dirs]
    
  def read_images_name(self, dir):
    """
      Only read image name from folder
      we sort image name with his digit part. 
    """
    images_name = os.listdir(dir)
    images_name = list(filter(lambda filename: os.path.isdir(os.path.join(dir, filename)) == False, images_name))
    if len(images_name) > 1:
      images_name.sort(reverse=False, key=lambda x: int(re.findall(r'\d+', x)[0]))

    return images_name
  
  def readimage(self, imagename):
    
    pil_image = self.pil_loader(imagename)
    image = self.preprocess_pipline(pil_image)
    return image