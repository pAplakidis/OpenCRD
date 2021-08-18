import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

# model for binary image classification (0: crossroad, 1: no-crossroad)
class CRDetector(nn.Module):
  def __init__(self):
    super(CRDetector, self).__init__()

    # Convolutional Layers
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.conv2_bn1 = nn.BatchNorm2d(16)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.conv2_bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, 5)
    self.conv2_bn3 = nn.BatchNorm2d(64)

    # Fully connected layers
    self.fc1 = nn.Linear(64 * 16 * 36, 120) # for 320x160 image 64 channels
    self.bn1 = nn.BatchNorm1d(num_features=120)
    self.fc2 = nn.Linear(120, 84)
    self.bn2 = nn.BatchNorm1d(num_features=84)
    self.fc3 = nn.Linear(84, 1)

  def forward(self, x):
    x = self.pool(F.relu(self.conv2_bn1(self.conv1(x))))
    x = self.pool(F.relu(self.conv2_bn2(self.conv2(x))))
    x = self.pool(F.relu(self.conv2_bn3(self.conv3(x))))
    #print(x.shape)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.bn1(self.fc1(x)))
    x = F.relu(self.bn2(self.fc2(x)))
    x = torch.sigmoid(self.fc3(x))
    return x

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


# ResNet block
class ResBlock(nn.Module):
  def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
    super(ResBlock, self).__init__()

    self.num_layers = num_layers
    if self.num_layers > 34:
      self.expansion = 4
    else:
      self.expansion =1

    # ResNet50, 101 and 152 include additional layer of 1x1 kernels
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    if self.num_layers > 34:
      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    else:
      # for ResNet18 and 34, connect input directly to 3x3 kernel (skip first 1x1)
      self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
    self.elu = nn.ELU()
    self.identity_downsample = identity_downsample

  def forward(self, x):
    identity = x
    if self.num_layers > 34:
      x = self.elu(self.bn1(self.conv1(x)))
    x = self.elu(self.bn2(self.conv2(x)))
    x = self.bn3(self.conv3(x))
    
    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)
    x += identity
    x = self.elu(x)
    return x

# ResNet CRDetector model
class ResCRDetector(nn.Module):
  def __init__(self, num_layers, block, image_channels):
    assert num_layers in [18, 34, 50, 101, 152], "Unknown ResNet architecture, number of layers must be 18, 34, 50, 101 or 152"
    super(ResCRDetector, self).__init__()

    # for output layer (polylines shape)
    self.n_coords = 2  # 2 coordinates: x,y
    self.n_points = 4  # number of points of each polyline
    self.max_n_lines = 6 # max number of polylines per frame

    if num_layers < 50:
      self.expansion = 1
    else:
      self.expansion = 4
    if num_layers == 18:
      layers = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
      layers = [3, 4, 23, 3]
    elif num_layers == 101:
      layers = [3, 8, 23, 3]
    else:
      layers = [3, 8, 36, 3]

    self.in_channels = 16
    self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=7, stride=2, padding=3)  # TODO: maybe kernel 5x5
    self.bn1 = nn.BatchNorm2d(16)
    self.elu = nn.ELU()
    self.avgpool1 = nn.AvgPool2d(3, 2, padding=1)

    # ResNet Layers
    self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=32, stride=1)
    self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=64, stride=2)
    self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=128, stride=2)
    self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=256, stride=2)

    self.avgpool2 = nn.AvgPool2d(1, 1)

    # Fully Connected Layers
    self.fc1 = nn.Linear(256*5*10, 1024) # NOTE: this works only with ResNet18
    self.fc_bn1 = nn.BatchNorm1d(1024)
    self.fc2 = nn.Linear(1024, 128)
    self.fc_bn2 = nn.BatchNorm1d(128)
    self.fc3 = nn.Linear(128, 84)
    self.fc_bn3 = nn.BatchNorm1d(84)
    self.fc4 = nn.Linear(84, 1)

  def forward(self, x):
    x = self.avgpool1(self.elu(self.bn1(self.conv1(x))))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool2(x)
    #print(x.shape)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc_bn1(self.fc1(x)))
    x = F.relu(self.fc_bn2(self.fc2(x)))
    x = F.relu(self.fc_bn3(self.fc3(x)))
    x = torch.sigmoid(self.fc4(x))
    return x

  def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
    layers = []
    identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                        nn.BatchNorm2d(intermediate_channels*self.expansion))
    layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
    self.in_channels = intermediate_channels*self.expansion
    for i in range(num_residual_blocks - 1):
      layers.append(block(num_layers, self.in_channels, intermediate_channels))
    return nn.Sequential(*layers)

  def num_flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

#====================================================================================================

# model for road edge detection
class REDetector(nn.Module):
  def __init__(self):
    super(REDetector, self).__init__()

    # output polylines attributes
    self.n_coords = 2  # 2 coordinates: x,y
    self.n_points = 4  # number of points of each polyline
    self.max_n_lines = 6 # max number of polylines per frame

    # Convolutional Layers 
    # NOTE: in order to multitask-learn with CRDetector (conv layers will be the same since they sohould detect the same features), this part should not be changed
    # TODO: change this architecture (ResNet)
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.conv2d_bn1 = nn.BatchNorm2d(16)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.conv2d_bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, 5)
    self.conv4 = nn.Conv2d(64, 128, 3)
    self.conv5 = nn.Conv2d(128, 256, 3)

    # Fully Connected Layers
    self.dropout1 = nn.Dropout(0.1)
    self.fc1 = nn.Linear(256*14*34, 2048) # for 320x160 image 256 channels
    self.fc2 = nn.Linear(2048, 1024)
    self.fc3 = nn.Linear(1024, 256)
    self.fc4 = nn.Linear(256, 96)
    self.dropout2 = nn.Dropout(0.5)
    self.fc5 = nn.Linear(96, self.n_coords*self.n_points*self.max_n_lines)

  def forward(self, x):
    x = self.pool(F.relu(self.conv2d_bn1(self.conv1(x))))
    x = self.pool(F.relu(self.conv2d_bn2(self.conv2(x))))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool(F.relu(self.conv5(x)))
    #print(x.shape)
    # TODO: relu doesnt allow negative values, try to mix it up with tanh (first or last FC layers) as well as relu
    x = x.view(-1, self.num_flat_features(x))
    x = self.dropout1(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = torch.tanh(self.fc4(x))
    x = self.dropout2(x)
    x = self.fc5(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


# ResNet Bayesian model for road edge detection
@variational_estimator
class ResREDetector(nn.Module):
  def __init__(self, num_layers, block, image_channels):
    assert num_layers in [18, 34, 50, 101, 152], "Unknown ResNet architecture, number of layers must be 18, 34, 50, 101 or 152"
    super(ResREDetector, self).__init__()

    # for output layer (polylines shape)
    self.n_coords = 2  # 2 coordinates: x,y
    self.n_points = 4  # number of points of each polyline
    self.max_n_lines = 6 # max number of polylines per frame

    if num_layers < 50:
      self.expansion = 1
    else:
      self.expansion = 4
    if num_layers == 18:
      layers = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
      layers = [3, 4, 23, 3]
    elif num_layers == 101:
      layers = [3, 8, 23, 3]
    else:
      layers = [3, 8, 36, 3]

    self.in_channels = 16
    self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=7, stride=2, padding=3)  # TODO: maybe kernel 5x5
    self.bn1 = nn.BatchNorm2d(16)
    self.elu = nn.ELU()
    self.avgpool1 = nn.AvgPool2d(3, 2, padding=1)

    # ResNet Layers
    self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
    self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
    self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
    self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

    self.avgpool2 = nn.AvgPool2d(1, 1)

    # TODO: try 2 layers before changing the loss function, or all batch normalized or more layers (maybe 2 more)
    # Fully Connected Layers
    self.fc1 = nn.Linear(512*5*10, 2048) # NOTE: this works only with ResNet18
    #self.fc1 = nn.Linear(256*self.expansion, 2048)
    self.fc_bn1 = nn.BatchNorm1d(2048)

    # Bayesian Layers
    self.blinear1 = BayesianLinear(2048, 512)
    self.b_bn1 = nn.BatchNorm1d(512)
    self.blinear2 = BayesianLinear(512, 128)
    self.b_bn2 = nn.BatchNorm1d(128)
    self.blinear3 = BayesianLinear(128, self.n_coords*self.n_points*self.max_n_lines)

  def forward(self, x):
    x = self.avgpool1(self.elu(self.bn1(self.conv1(x))))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool2(x)
    #print(x.shape)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc_bn1(self.fc1(x)))
    x = F.relu(self.b_bn1(self.blinear1(x)))
    x = F.relu(self.b_bn2(self.blinear2(x)))
    x = self.blinear3(x)
    return x

  def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
    layers = []
    identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                        nn.BatchNorm2d(intermediate_channels*self.expansion))
    layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
    self.in_channels = intermediate_channels*self.expansion
    for i in range(num_residual_blocks - 1):
      layers.append(block(num_layers, self.in_channels, intermediate_channels))
    return nn.Sequential(*layers)

  def num_flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

#====================================================================================================

# Multitask Model
class ComboModel(nn.Module):
  def __init__(self, num_layers=18, block=ResBlock, image_channels=3):
    assert num_layers in [18, 34, 50, 101, 152], "Unknown ResNet architecture, number of layers must be 18, 34, 50, 101 or 152"
    super(ComboModel, self).__init__()

    # polylines' shape
    self.n_coords = 2  # 2 coordinates: x,y
    self.n_points = 4  # number of points of each polyline
    self.max_n_lines = 6 # max number of polylines per frame

    if num_layers < 50:
      self.expansion = 1
    else:
      self.expansion = 4
    if num_layers == 18:
      layers = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
      layers = [3, 4, 23, 3]
    elif num_layers == 101:
      layers = [3, 8, 23, 3]
    else:
      layers = [3, 8, 36, 3]

    self.in_channels = 16
    self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=7, stride=2, padding=3)  # TODO: maybe kernel 5x5
    self.bn1 = nn.BatchNorm2d(16)
    self.elu = nn.ELU()
    self.avgpool1 = nn.AvgPool2d(3, 2, padding=1)

    # ResNet Layers
    self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
    self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
    self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
    self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

    self.avgpool2 = nn.AvgPool2d(1, 1)

    # Fully Connected Layers
    self.cr_head = get_cr_head()
    self.re_head = get_re_head()

  def forward(self, x):
    x = self.avgpool1(self.elu(self.bn1(self.conv1(x))))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool2(x)
    #print(x.shape)
    x = x.view(-1, self.num_flat_features(x))
    cr = torch.sigmoid(self.cr_head(x))
    re = self.re_head(x)
    return cr, re

  def get_cr_head(self):
    relu = nn.ReLU()
    fc1 = nn.Linear(256*5*10, 1024) # NOTE: this works only with ResNet18
    bn1 = nn.BatchNorm1d(1024)
    fc2 = nn.Linear(1024, 128)
    bn2 = nn.BatchNorm1d(128)
    fc3 = nn.Linear(128, 84)
    bn3 = nn.BatchNorm1d(84)
    fc4 = nn.Linear(84, 1)

    head = nn.Sequential(fc1, bn1, relu, fc2, bn2, relu, fc3, bn3, relu, fc4)
    return head

  def get_re_head(self):
    l_relu = nn.LeakyReLU()
    fc1 = nn.Linear(512*5*10, 8192) # NOTE: this works only with ResNet18
    bn1 = nn.BatchNorm1d(8192)
    fc2 = nn.Linear(8192, 4096)
    bn2 = nn.BatchNorm1d(4096)
    fc3 = nn.Linear(4096, 2048)
    bn3 = nn.BatchNorm1d(2048)
    fc4 = nn.Linear(2048, 1024)
    bn4 = nn.BatchNorm1d(1024)
    fc5 = nn.Linear(1024, 512)
    bn5 = nn.BatchNorm1d(512)
    fc6 = nn.Linear(512, 256)
    bn6 = nn.BatchNorm1d(256)
    fc7 = nn.Linear(256, 128)
    bn7 = nn.BatchNorm1d(128)
    fc8 = nn.Linear(128, 64)
    bn8 = nn.BatchNorm1d(64)
    fc9 = nn.Linear(64, self.n_coords*self.n_points*self.max_n_lines)

    head = nn.Sequential(fc1, bn1, l_relu, fc2, bn2, l_relu, fc3, bn3, l_relu,
                         fc4, bn4, l_relu, fc5, bn5, l_relu, fc6, bn6, l_relu,
                         fc8, bn8, l_relu, fc9)
    return head

#----------------------------------------------------------------------------------------------

# Custom Loss functions

# negative log likelihood loss
def neg_log_likelihood(output, target, sigma=1.0):
  dist = torch.distributions.normal.Normal(output, sigma)
  return torch.sum(-dist.log_prob(target))

