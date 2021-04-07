import torch
import torch.nn as nn
import torch.nn.functional as F

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


# mulitask learning model
class ComboModel(nn.Module):
  def __init__(self):
    super(ComboModel, self).__init__()

