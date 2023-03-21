import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2

# NOTE: resolutions not divisible by 8, 16 can cause problems
class PathPlanner(nn.Module):
  def __init__(self):
    super(PathPlanner, self).__init__()
    self.vision = efficientnet_b2(pretrained=True)
    del self.vision.classifier
    """
     (classifier): Sequential(                                                                            
      (0): Dropout(p=0.3, inplace=True)                                                                  
      (1): Linear(in_features=1408, out_features=1000, bias=True)
    """
    # TODO: ConvNet for now, RNN later on
    self.policy = nn.Sequential()

  def forward(self, x):
    x = self.vision(x)
    return x
