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
    # TODO: start with simple ConvNet, RNN later on (maybe GRU or LSTM)
    # TODO: more than image for input (desire, state, etc)
    # TODO: multimodal (multiple paths with probabilities) output (check out mixture density networks)
    # meaning output is M future paths (for now) <xi,yi> for i in range(2*H)
    # along with each path's probabilities, these probabilities are passed through a softmax layer
    # TODO: L2 Loss for each(i) path predicted
    # TODO: for Multimodal Loss Function do not use Mixture of Experts (ME) Loss,
    # but a custom Multiple-Trajectory Prediction Loss:
    # get the mode/path m that is closest to the groundtruth
    self.policy = nn.Sequential()

  def forward(self, x):
    x = self.vision(x)
    return x

def save_model(path, model):
 torch.save(model.state_dict(), path)
 print("Model saved at", path)

def load_model(path, model):
  model.load_state_dict(torch.load(path))
  print("Loaded model from", path)
  return model
