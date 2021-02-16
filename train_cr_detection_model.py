#!/usr/bin/env python3
import sys
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# resolution
W = 1280
H = 960

def get_data(video_path, log_path):
  with open(log_path, "r") as log_file:
    labels = log_file.read().split("\n")
    log_file.close()
  print("Log file read")

  frames = []
  cap = cv2.VideoCapture(video_path)
  idx = 0

  # TODO: this comsumes a lot of memory!!!
  print("Extracting frames ...")
  vframes, aframes, info = torchvision.io.read_video(video_path)
  print(vframes)
  print(aframes)
  print(info)
  """
  reader = torchvision.io.VideoReader(video_path, "video")
  reader.seek(2)
  idx = 0
  for frame in reader:
    print("Reading frame", idx)
    frames.append(frame['data'])
  print("Frames read")
  """
  """
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      frame = cv2.resize(frame, (W,H))
      print("Reading frame %d ..." % idx)
      frames.append(frame)
      #cv2.imshow('frame', frame)
      idx += 1
      if cv2.waitKey(1) & 0xff == ord('q'):
        break 
    else:
      break
  cap.release()
  cv2.destroyAllWindows()
  """

  print("Done extracting frames")
    
  return np.array(frames), np.array(labels) 

class Net(nn.Module):
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

    # Convolutional layers
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 5)
    # types of crossroads: no_crossroad, regular cross, T shaped, T (no left), T (no right)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

if __name__ == '__main__':
  video_path = sys.argv[1]
  log_path = video_path[:-4] + ".txt"

  frames, labels = get_data(video_path, log_path)
  print("Frames:")
  print(frames)
  print("Labels:")
  print(labels)
  #model = Net()

