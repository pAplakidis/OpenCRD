import torch
import torch.nn as nn
import torch.nn.functional as F


class PathPlanner(nn.Module):
  def __init__(self):
    super(PathPlanner, self).__init__()
