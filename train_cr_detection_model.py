#!/usr/bin/env python3
import sys
import numpy as np

import cv2
import pims
import matplotlib.pyplot as plt
from plt import plot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# resolution
W = 1280
H = 960
LABEL_DICT = {0: "no crossroad", 1: "crossroad"}

def get_data(video_path, log_path):
  with open(log_path, "r") as log_file:
    labels = log_file.read().split("\n")[:-1]
    log_file.close()
  print("Log file read")

  # make a Video Object instead of loading all frames in memory
  # to access frames just do: frame = frames[n]
  print("Extracting frames ...")
  frames = pims.Video(video_path, format="mp4")
  print("Done extracting frames")

  return frames, np.array(labels).astype(np.int)

class ConvNet(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
   
    # image size
    self.W = W
    self.H = H

    # Convolutional layers
    # TODO: decide layers architecture depending on the images to be used
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # Fully connected layers
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 5)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def train(frames, Y_train):
  model = ConvNet()
  
  loss_function = nn.NLLLoss(reduction='none')  # check if this loss is better (or try nn.CrossEntropyLoss())
  optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0) # TODO: check if momentum is correct
  BS = 128
  losses, accuracies = [], []

  for i in (t := trange(1000)):
    samp = np.random.randint(0, len(frames), size=(BS))

    # TODO: this is very slow and memory consuming, try to improve it's performance
    X_train = []
    print("Extracting frames ...")
    cnt = 0
    for idx in samp:
      print("%d,%d"%(cnt,idx))
      X_train.append(cv2.resize(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), (W,H)))
      cnt += 1
    X_train = np.array(X_train)

    X = torch.tensor(X_train).float()
    Y = torch.tensor(Y_train[samp]).long()
    model.zero_grad()
    out = model(X)
    cat = torch.round(out)  # TODO: in the deployment need to print out the probability of crossroad (rounded label + nonrounded value for explainability) + in deployment don't use round, instead use a threshold higher than 50% (maybe 80%) (the last part might be applied in the training as well, test to see)
    accuracy = (cat == Y).float().mean()  # TODO: this might give AttributeError (FIX IT) (or it may not since we are dealing with tensors)
    loss = loss_function(out, Y)
    loss = loss.mean()
    loss.backward()
    optim.step()
    loss, accuracy = loss.item(), accuracy.item()
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

  # plot losses and accuracies
  plt.ylim(-0.1, 1.1)
  plot(losses)
  plot(accuracies)
  
  return model

# TODO: evaluate the net + compute gradients in torch (need to split data to train and test first)
def evaluate(model):
  pass


if __name__ == '__main__':
  video_path = sys.argv[1]
  log_path = video_path[:-4] + ".txt"

  frames, labels = get_data(video_path, log_path)

  # NOTE: DEBUGGING CODE/EXAMPLE FOR PROCESSING/SHOWING FRAME
  """
  print("Frame 300")
  frame = frames[300]
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = cv2.resize(frame, (W,H))
  cv2.imshow('frame', frame)
  print("Label:", LABEL_DICT[labels[300]])
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  """

  model = train(frames, labels)
  evaluate(model)

  model_path = "models/cr_detection_conv_model.pt"
  torch.save(model, model_path)

