import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# TODO: predict crossroad distance as well? (will make the project actually useful, it might not matter if we can output a good path (car will slow down to turn in the curve))
# test out different numbers of layers (conv2d + fc) and neurons per layer
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
   
    # image size
    self.W = W
    self.H = H

    # Convolutional layers (TODO: maybe add another one and have bigger output channels?)
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # Fully connected layers
    self.fc1 = nn.Linear(16 * W * H, 120) # TODO: this consumes a lot of emory, maybe change W and H to smaller values
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 1)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    #x = F.sigmoid(self.fc3(x))  # sigmoid for binary classification
    x = self.fc3(x) # NOTE: we might not need to apply sigmoid (TODO: try out softmax as well)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

